"""Confidence calibration and its metrics.

A score of 0.9 should mean "right about 90 % of the time" across comparable cases. Raw
scores are usually over-confident, so they are mapped to a probability on a *separate
calibration split*. Two scores exist per case:

* ``p_seg`` - the segmenter's case score (top tumour probabilities inside the liver);
* ``p_vlm`` - an optional second reader's P("yes") to "is there a liver tumour?" (missing by
  default: the project uses one model, so only ``identity``, ``temperature`` and ``platt`` apply).

Candidate maps (``z`` = logit):

* ``identity``     the segmenter score as is
* ``temperature``  ``sigma(z_seg / T)``
* ``platt``        ``sigma(a z_seg + b)``
* ``vlm``          ``sigma(va z_vlm + vb)``              (falls back to ``platt`` without a VLM read)
* ``fused``        ``sigma(w0 + w1 z_seg + w2 z_vlm)``    (logistic fusion; same fallback)

``auto`` picks the one with the lowest stratified k-fold cross-validated NLL *on the
calibration split only* - the test split is never touched - so the data decide whether
a second reader adds anything over the segmenter.

Metrics: Brier score, expected calibration error (equal-width and equal-mass bins), NLL,
AUROC, reliability-diagram tables and bootstrap confidence intervals.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

EPS = 1e-4
METHODS = ("identity", "temperature", "platt", "vlm", "fused")


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def nll(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _as_vlm(p_vlm, n: int) -> np.ndarray:
    if p_vlm is None:
        return np.full(n, np.nan)
    v = np.asarray(p_vlm, dtype=np.float64).reshape(-1)
    return np.full(n, v[0]) if v.size == 1 and n > 1 else v


@dataclass
class Calibrator:
    method: str = "identity"
    T: float = 1.0
    a: float = 1.0
    b: float = 0.0
    va: float = 1.0
    vb: float = 0.0
    w: list = field(default_factory=lambda: [0.0, 1.0, 1.0])
    has_vlm: bool = False
    n_fit: int = 0
    note: str = ""

    # -- component maps
    def seg_only(self, p_raw):
        z = logit(p_raw)
        if self.method == "identity":
            return np.clip(np.asarray(p_raw, dtype=np.float64), 0.0, 1.0)
        if self.method == "temperature":
            return sigmoid(z / self.T)
        return sigmoid(self.a * z + self.b)

    def vlm_only(self, p_vlm):
        v = np.asarray(p_vlm, dtype=np.float64)
        return sigmoid(self.va * logit(v) + self.vb) if self.has_vlm else np.clip(v, 0.0, 1.0)

    def __call__(self, p_raw, p_vlm=None):
        scalar = np.ndim(p_raw) == 0
        pr = np.atleast_1d(np.asarray(p_raw, dtype=np.float64))
        pv = _as_vlm(p_vlm, len(pr))
        out = self.seg_only(pr)
        ok = np.isfinite(pv)
        if self.method == "vlm" and ok.any():
            out = np.where(ok, sigmoid(self.va * logit(np.where(ok, pv, 0.5)) + self.vb), out)
        elif self.method == "fused" and ok.any():
            z = self.w[0] + self.w[1] * logit(pr) + self.w[2] * logit(np.where(ok, pv, 0.5))
            out = np.where(ok, sigmoid(z), out)
        return float(out[0]) if scalar else out

    def describe(self) -> str:
        if self.method == "temperature":
            return f"temperature scaling (T = {self.T:.3f})"
        if self.method == "platt":
            return f"Platt scaling of the segmenter score (a = {self.a:.3f}, b = {self.b:.3f})"
        if self.method == "vlm":
            return f"Platt scaling of the second reader's P(yes) (a = {self.va:.3f}, b = {self.vb:.3f})"
        if self.method == "fused":
            return (f"fused logistic calibration: segmenter x {self.w[1]:.2f} + second reader x {self.w[2]:.2f} "
                    f"(bias {self.w[0]:.2f})")
        return "uncalibrated (identity)"

    def to_dict(self) -> dict:
        return {"method": self.method, "T": self.T, "a": self.a, "b": self.b, "va": self.va, "vb": self.vb,
                "w": list(self.w), "has_vlm": self.has_vlm, "n_fit": self.n_fit, "note": self.note,
                "description": self.describe()}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(method=d.get("method", "identity"), T=float(d.get("T", 1.0)), a=float(d.get("a", 1.0)),
                   b=float(d.get("b", 0.0)), va=float(d.get("va", 1.0)), vb=float(d.get("vb", 0.0)),
                   w=[float(x) for x in d.get("w", [0.0, 1.0, 1.0])], has_vlm=bool(d.get("has_vlm", False)),
                   n_fit=int(d.get("n_fit", 0)), note=d.get("note", ""))


def fit_temperature(p_raw, y) -> float:
    z, y = logit(p_raw), np.asarray(y, dtype=np.float64)
    res = minimize_scalar(lambda lt: nll(sigmoid(z / np.exp(lt)), y), bounds=(-3.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def fit_platt(p_raw, y, l2: float = 1e-3) -> tuple[float, float]:
    z, y = logit(p_raw), np.asarray(y, dtype=np.float64)

    def obj(w):
        return nll(sigmoid(w[0] * z + w[1]), y) + l2 * (w[0] - 1.0) ** 2 + l2 * w[1] ** 2

    res = minimize(obj, x0=np.array([1.0, 0.0]), method="L-BFGS-B", bounds=[(1e-3, 50.0), (-50.0, 50.0)])
    return float(res.x[0]), float(res.x[1])


def fit_fused(p_seg, p_vlm, y, l2: float = 1e-2) -> list:
    """Logistic regression on (logit p_seg, logit p_vlm) with a weak ridge towards (0, 1, 1)."""
    z1, z2, y = logit(p_seg), logit(p_vlm), np.asarray(y, dtype=np.float64)

    def obj(w):
        return nll(sigmoid(w[0] + w[1] * z1 + w[2] * z2), y) + l2 * (w[0] ** 2 + (w[1] - 1) ** 2 + (w[2] - 1) ** 2)

    res = minimize(obj, x0=np.array([0.0, 1.0, 1.0]), method="L-BFGS-B",
                   bounds=[(-50.0, 50.0), (-20.0, 20.0), (-20.0, 20.0)])
    return [float(x) for x in res.x]


def _fit(method: str, p, v, y) -> Calibrator:
    n = len(y)
    if method == "identity":
        return Calibrator("identity", n_fit=n)
    if method == "temperature":
        return Calibrator("temperature", T=fit_temperature(p, y), n_fit=n)
    a, b = fit_platt(p, y)
    cal = Calibrator("platt", a=a, b=b, n_fit=n)
    ok = np.isfinite(v)
    if ok.sum() >= 4 and np.unique(y[ok]).size == 2:
        cal.va, cal.vb = fit_platt(v[ok], y[ok])
        cal.has_vlm = True
        if method == "vlm":
            cal.method = "vlm"
        elif method == "fused":
            cal.method, cal.w = "fused", fit_fused(p[ok], v[ok], y[ok])
    return cal


def _stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return [np.array(sorted(f)) for f in folds if f]


def fit_calibrator(p_raw, y, method: str = "auto", n_folds: int = 5, seed: int = 0,
                   p_vlm=None) -> tuple[Calibrator, dict]:
    p_raw, y = np.asarray(p_raw, dtype=np.float64), np.asarray(y, dtype=np.int64)
    v = _as_vlm(p_vlm, len(y))
    has_vlm = int(np.isfinite(v).sum())
    info: dict = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "n_with_vlm": has_vlm}
    if len(y) < 4 or y.min() == y.max():
        cal = Calibrator("identity", n_fit=len(y), note="calibration split has a single class - left uncalibrated")
        info["selected"] = "identity"
        info["warning"] = cal.note
        return cal, info
    if method != "auto":
        cal = _fit(method, p_raw, v, y)
        info["selected"] = cal.method
        return cal, info
    candidates = ["identity", "temperature", "platt"]
    if has_vlm >= 4:
        candidates += ["vlm", "fused"]
    k = int(min(n_folds, max(2, min(int(y.sum()), int(len(y) - y.sum())))))
    folds = _stratified_folds(y, k, seed)
    cv = {m: 0.0 for m in candidates}
    used = 0
    for f in folds:
        tr = np.setdiff1d(np.arange(len(y)), f)
        if y[tr].min() == y[tr].max():
            continue
        used += len(f)
        for m in candidates:
            cal = _fit(m, p_raw[tr], v[tr], y[tr])
            cv[m] += nll(cal(p_raw[f], v[f]), y[f]) * len(f)
    scores = {m: cv[m] / max(1, used) for m in candidates}
    best = min(scores, key=scores.get)
    info["cv_nll"] = {m: round(s, 4) for m, s in scores.items()}
    cal = _fit(best, p_raw, v, y)
    info["selected"] = cal.method
    cal.note = f"selected by {len(folds)}-fold stratified CV NLL on the calibration split"
    return cal, info


# --------------------------------------------------------------------------- metrics
def brier(p, y) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return float(((p - y) ** 2).mean())


def reliability_table(p, y, n_bins: int = 10, strategy: str = "uniform") -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([0.0, 1.0])
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    conf, acc, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        cnt.append(int(m.sum()))
        conf.append(float(p[m].mean()) if m.any() else float("nan"))
        acc.append(float(y[m].mean()) if m.any() else float("nan"))
    return {"edges": edges.tolist(), "confidence": conf, "frequency": acc, "count": cnt}


def ece(p, y, n_bins: int = 10, strategy: str = "uniform") -> float:
    t = reliability_table(p, y, n_bins, strategy)
    n = max(1, sum(t["count"]))
    return float(sum(c * abs(a - f) for a, f, c in zip(t["confidence"], t["frequency"], t["count"]) if c > 0) / n)


def auroc(p, y) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def bootstrap_ci(fn, p, y, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    p, y = np.asarray(p), np.asarray(y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(len(y), size=len(y))
        try:
            v = fn(p[i], y[i])
        except Exception:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def calibration_report(p, y, n_bins: int = 10, n_boot: int = 1000, seed: int = 0) -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "brier": brier(p, y), "ece": ece(p, y, n_bins),
           "ece_adaptive": ece(p, y, n_bins, "quantile"), "nll": nll(p, y), "auroc": auroc(p, y)}
    out["brier_ci"] = bootstrap_ci(brier, p, y, n_boot, seed)
    out["ece_ci"] = bootstrap_ci(lambda a, b: ece(a, b, n_bins), p, y, n_boot, seed)
    out["auroc_ci"] = bootstrap_ci(auroc, p, y, n_boot, seed) if 0 < y.sum() < len(y) else (float("nan"),) * 2
    return out


def binary_entropy(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
