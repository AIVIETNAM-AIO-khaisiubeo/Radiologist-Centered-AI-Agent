"""Reader-adaptive decision policies (selective prediction + learning to defer).

A policy decides, per case and per reader:
* the **suggested label**: "tumour" if the calibrated probability >= threshold;
* whether to **show** the suggestion or **defer** - withhold it and ask for an
  independent read - when the case's uncertainty is >= the deferral cut-off;
  cases that failed QC are always deferred.

Policies compared
-----------------
``universal``               one threshold (0.5) for every reader, always shown
``universal_defer``         one threshold + deferral cut-off for everyone, learned from pooled feedback
``reader_threshold``        reader-specific threshold, always shown
``reader_threshold_defer``  reader-specific threshold + reader-specific deferral

How a reader-specific policy is learned (model-based, CoDoC-inspired)
---------------------------------------------------------------------
From the reader's feedback (sequential protocol, so we observe their independent
impression ``o`` and their final decision) we estimate
* a **trust curve** pi(c): probability of switching to the AI when they disagree, as a
  logistic function of the AI's confidence c, fitted by MAP with a weak Gaussian prior
  (so a handful of events does not produce an extreme curve);
* a **lapse rate**: how often they abandon an agreeing call.
For any candidate (threshold, deferral cut-off) the *expected* loss on the reader's
past cases follows in closed form - their independent impressions are policy-free -
using correctness from the *reference annotation*:
    loss = w_fn * FN + w_fp * FP + burden_weight * review_actions
The grid point with the lowest expected loss (plus a small shrinkage towards the
universal threshold) is adopted only if the reader has >= ``min_feedback`` events and
it beats the universal policy by >= ``min_improvement``; otherwise the reader keeps
the universal policy. This guards against overfitting tiny feedback logs.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class ReaderPolicy:
    name: str
    reader_id: str | None
    threshold: float = 0.5
    defer_uncertainty: float | None = None
    personalised: bool = False
    trust_alpha: float | None = None
    trust_beta: float | None = None
    p_lapse: float | None = None
    n_feedback: int = 0
    n_disagreements: int = 0
    expected_loss: float | None = None
    universal_loss: float | None = None
    notes: list = field(default_factory=list)

    def decide(self, p: float, u: float, qc_passed: bool = True) -> dict:
        if self.name == "no_ai":
            return {"decision": "defer", "ai_label": None, "confidence": None, "threshold": self.threshold,
                    "reason": "no AI assistance (control arm)", "personalised": False}
        if not qc_passed:
            return {"decision": "defer", "ai_label": None, "confidence": None, "threshold": self.threshold,
                    "reason": "quality control failed", "personalised": self.personalised}
        if self.defer_uncertainty is not None and u >= self.defer_uncertainty:
            return {"decision": "defer", "ai_label": None, "confidence": None, "threshold": self.threshold,
                    "reason": f"uncertainty {u:.2f} >= {self.defer_uncertainty:.2f}", "personalised": self.personalised}
        label = int(p >= self.threshold)
        return {"decision": "show", "ai_label": label, "confidence": float(p if label else 1 - p),
                "threshold": self.threshold, "reason": f"p {p:.2f} vs threshold {self.threshold:.2f}",
                "personalised": self.personalised}

    def trust(self, conf):
        a = 0.0 if self.trust_alpha is None else self.trust_alpha
        b = 4.0 if self.trust_beta is None else self.trust_beta
        return 1.0 / (1.0 + np.exp(-(a + b * (np.asarray(conf, dtype=float) - 0.75))))

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        d = "never" if self.defer_uncertainty is None else f"if uncertainty >= {self.defer_uncertainty:.2f}"
        tag = "personalised" if self.personalised else "universal"
        return f"{tag}: suggest tumour if p >= {self.threshold:.2f}; defer {d}"


def universal_policy(cfg: dict) -> ReaderPolicy:
    return ReaderPolicy(name="universal", reader_id=None, threshold=float(cfg["policy"]["universal_threshold"]))


def no_ai_policy() -> ReaderPolicy:
    return ReaderPolicy(name="no_ai", reader_id=None)


# --------------------------------------------------------------------------- fitting
def fit_trust_curve(conf: np.ndarray, switched: np.ndarray, prior: dict, iters: int = 30) -> tuple[float, float]:
    """MAP logistic regression  logit pi = alpha + beta * (c - 0.75)  with Gaussian priors."""
    mu = np.array([prior["alpha_mean"], prior["beta_mean"]], dtype=float)
    prec = np.diag([1.0 / prior["alpha_sd"] ** 2, 1.0 / prior["beta_sd"] ** 2])
    theta = mu.copy()
    if len(conf) == 0:
        return float(theta[0]), float(theta[1])
    X = np.stack([np.ones(len(conf)), np.asarray(conf, dtype=float) - 0.75], axis=1)
    y = np.asarray(switched, dtype=float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ theta)))
        grad = X.T @ (y - p) - prec @ (theta - mu)
        H = -(X.T * (p * (1 - p))) @ X - prec
        step = np.linalg.solve(H, grad)
        theta = theta - step
        if np.abs(step).max() < 1e-6:
            break
    return float(theta[0]), float(theta[1])


def _arrays(events: list[dict], cases: dict[str, dict]) -> dict:
    ev = [e for e in events if e.get("initial_read") is not None and e["case_id"] in cases]
    return {
        "p": np.array([cases[e["case_id"]]["p"] for e in ev], dtype=float),
        "u": np.array([cases[e["case_id"]]["u"] for e in ev], dtype=float),
        "qc": np.array([cases[e["case_id"]]["qc_passed"] for e in ev], dtype=bool),
        "g": np.array([int(cases[e["case_id"]]["gt_present"]) for e in ev], dtype=int),
        "o": np.array([int(e["initial_read"]) for e in ev], dtype=int),
        "events": ev,
    }


def _loss(final: np.ndarray, g: np.ndarray, w_fn: float, w_fp: float) -> np.ndarray:
    return w_fn * ((g == 1) & (final == 0)) + w_fp * ((g == 0) & (final == 1))


def expected_loss(arr: dict, threshold: float, defer_u: float | None, alpha: float, beta: float,
                  p_lapse: float, pcfg: dict, per_case: bool = False):
    p, u, g, o, qc = arr["p"], arr["u"], arr["g"], arr["o"], arr["qc"]
    if len(p) == 0:
        return float("nan")
    w_fn, w_fp, lam = pcfg["w_fn"], pcfg["w_fp"], pcfg["burden_weight"]
    cost = pcfg["action_cost"]
    a = (p >= threshold).astype(int)
    c = np.where(a == 1, p, 1 - p)
    trust = 1.0 / (1.0 + np.exp(-(alpha + beta * (c - 0.75))))
    p_keep_ai = np.where(o == a, 1.0 - p_lapse, trust)
    show_loss = (p_keep_ai * _loss(a, g, w_fn, w_fp) + (1 - p_keep_ai) * _loss(1 - a, g, w_fn, w_fp)
                 + lam * (p_keep_ai * cost["accept"] + (1 - p_keep_ai) * cost["reject"]))
    defer_loss = _loss(o, g, w_fn, w_fp) + lam * cost["manual"]
    defer = ~qc
    if defer_u is not None:
        defer = defer | (u >= defer_u)
    losses = np.where(defer, defer_loss, show_loss)
    return losses if per_case else float(losses.mean())


def fit_policy(events: list[dict], cases: dict[str, dict], cfg: dict, reader_id: str | None,
               variant: str = "reader_threshold_defer", name: str | None = None) -> ReaderPolicy:
    """Learn a policy from feedback events (see module docstring).

    ``cases[case_id]`` must provide p (calibrated), u, gt_present, qc_passed.
    """
    pcfg = cfg["policy"]
    t_univ = float(pcfg["universal_threshold"])
    name = name or variant
    # a deferred decision ("second opinion") carries no final finding -> not evidence about trust
    events = [e for e in events if e.get("action") != "defer" and e.get("final_label") is not None]
    arr = _arrays(events, cases)
    ev = arr["events"]
    n = len(ev)
    pol = ReaderPolicy(name=name, reader_id=reader_id, threshold=t_univ, n_feedback=n)
    # lapse: agreeing impressions that were nevertheless abandoned
    agree = [e for e in ev if e.get("shown") and e.get("ai_label") is not None and e["initial_read"] == e["ai_label"]]
    k_lapse = sum(1 for e in agree if e["final_label"] != e["ai_label"])
    pol.p_lapse = (k_lapse + 1.0) / (len(agree) + 20.0)          # Beta(1, 19) prior, mean 0.05
    dis = [e for e in ev if e.get("shown") and e.get("ai_label") is not None and e["initial_read"] != e["ai_label"]]
    pol.n_disagreements = len(dis)
    conf = np.array([e["ai_confidence"] for e in dis], dtype=float)
    sw = np.array([e["final_label"] == e["ai_label"] for e in dis], dtype=float)
    pol.trust_alpha, pol.trust_beta = fit_trust_curve(conf, sw, pcfg["trust_prior"])
    if n == 0:
        pol.notes.append("no feedback yet - universal policy")
        return pol
    pol.universal_loss = expected_loss(arr, t_univ, None, pol.trust_alpha, pol.trust_beta, pol.p_lapse, pcfg)
    defer_grid = pcfg["defer_grid"] if variant.endswith("defer") else [None]
    t_grid = [t_univ] if variant == "universal_defer" else pcfg["threshold_grid"]
    best = (math.inf, t_univ, None)
    for t in t_grid:
        for du in defer_grid:
            loss = expected_loss(arr, float(t), du, pol.trust_alpha, pol.trust_beta, pol.p_lapse, pcfg)
            reg = loss + pcfg["shrink_rho"] * abs(float(t) - t_univ)
            if reg < best[0] - 1e-12:
                best = (reg, float(t), du)
    _, t_best, du_best = best
    best_loss = expected_loss(arr, t_best, du_best, pol.trust_alpha, pol.trust_beta, pol.p_lapse, pcfg)
    pol.expected_loss = best_loss
    if n < pcfg["min_feedback"]:
        pol.notes.append(f"only {n} feedback events (< {pcfg['min_feedback']}) - keeping the universal policy")
        return pol
    # paired per-case difference: is the improvement distinguishable from noise?
    d = (expected_loss(arr, t_univ, None, pol.trust_alpha, pol.trust_beta, pol.p_lapse, pcfg, per_case=True)
         - expected_loss(arr, t_best, du_best, pol.trust_alpha, pol.trust_beta, pol.p_lapse, pcfg, per_case=True))
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("inf")
    gain = float(d.mean())
    need = max(pcfg["min_improvement"], pcfg["improvement_se"] * se)
    if gain < need:
        pol.notes.append(f"best policy improves expected loss by {gain:.3f} (< {need:.3f} = max(floor, "
                         f"{pcfg['improvement_se']} x SE)) - keeping the universal policy")
        return pol
    pol.threshold, pol.defer_uncertainty, pol.personalised = t_best, du_best, True
    pol.notes.append(f"expected loss {pol.universal_loss:.3f} -> {best_loss:.3f} on {n} past reads")
    return pol


def case_table(records: list[dict]) -> dict[str, dict]:
    """Records -> the small per-case dict policies and simulations need."""
    out = {}
    for r in records:
        ref = r.get("reference", {})
        out[r["case_id"]] = {
            "case_id": r["case_id"], "split": r.get("split"),
            "p": float(r["confidence"]["tumour_probability"]), "u": float(r["confidence"]["uncertainty"]),
            "qc_passed": bool(r["quality"]["passed"]),
            "gt_present": bool(ref.get("tumour_present", False)),
            "gt_tumour_ml": float(ref.get("tumour_volume_ml", 0.0)),
            "gt_contrast_hu": float(ref.get("tumour_contrast_hu", 25.0)),
            "poor_outline": bool(ref.get("tumour_present", False) and ref.get("dice_tumour", 0.0) < 0.5),
            "priority": float(r.get("triage", {}).get("priority", 0.0)),
        }
    return out
