"""Controlled reader experiment: universal vs reader-specific vs deferral policies.

Protocol (all on held-out cases, patient-level splits):
1. **Burn-in / feedback collection** - each simulated reader reads the *calibration*
   cases (random order per seed) with the universal policy, using the sequential
   protocol; every action is logged as a feedback event.
2. **Policy fitting** - reader-specific policies are fitted on that reader's events;
   the pooled ``universal_defer`` policy on everyone's events.
3. **Evaluation** - every policy (plus the no-AI control) is applied to the *same test
   cases* with the *same random draws* (common random numbers), so differences are
   due to the policy, not to simulation noise.
4. Repeated over ``readers.seeds`` seeds; we report means with 95 % intervals.
5. **Learning curve** - step 2-3 repeated using only the first n feedback events
   (how much feedback is needed? is the learned policy stable?).

Correctness always comes from the reference annotation, never from a reader.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from radagent.policy import case_table, fit_policy, no_ai_policy, universal_policy
from radagent.readers.feedback import FeedbackEvent
from radagent.readers.profiles import draws_for, load_profiles, simulate_read

POLICY_ORDER = ["no_ai", "universal", "universal_defer", "reader_threshold", "reader_threshold_defer"]
POLICY_LABELS = {
    "no_ai": "Reader alone (no AI)",
    "universal": "Universal threshold",
    "universal_defer": "Universal + deferral",
    "reader_threshold": "Reader-specific threshold",
    "reader_threshold_defer": "Reader-specific + deferral",
}


def simulate_stream(profile, cases: list[dict], policy, draws: np.ndarray, source: str = "simulated") -> list[dict]:
    events = []
    for i, case in enumerate(cases):
        d = policy.decide(case["p"], case["u"], case["qc_passed"])
        r = simulate_read(profile, case, d, draws[i])
        ev = FeedbackEvent(reader_id=profile.reader_id, case_id=case["case_id"], policy=policy.name,
                           shown=r["shown"], ai_label=d["ai_label"], ai_probability=case["p"],
                           ai_confidence=d["confidence"], uncertainty=case["u"], threshold=policy.threshold,
                           initial_read=r["initial_read"], action=r["action"], final_label=r["final_label"],
                           source=source).to_dict()
        ev["switched"] = r["switched"]
        events.append(ev)
    return events


def summarize(events: list[dict], cases: dict[str, dict], cfg: dict, threshold: float) -> dict:
    pcfg = cfg["policy"]
    g = np.array([int(cases[e["case_id"]]["gt_present"]) for e in events])
    f = np.array([int(e["final_label"]) for e in events])
    p = np.array([cases[e["case_id"]]["p"] for e in events])
    shown = np.array([bool(e["shown"]) for e in events])
    ai = np.array([-1 if e["ai_label"] is None else int(e["ai_label"]) for e in events])
    ai_wrong = shown & (ai != g)
    ai_right = shown & (ai == g)
    accepted_ai = shown & (f == ai)
    cost = pcfg["action_cost"]
    acts = np.array([cost[e["action"]] for e in events], dtype=float)
    n = len(events)
    n_pos, n_neg = int(g.sum()), int((1 - g).sum())
    loss = (pcfg["w_fn"] * ((g == 1) & (f == 0)) + pcfg["w_fp"] * ((g == 0) & (f == 1))).mean()
    out = {
        "n": n,
        "accuracy": float((f == g).mean()),
        "sensitivity": float(((f == 1) & (g == 1)).sum() / n_pos) if n_pos else float("nan"),
        "specificity": float(((f == 0) & (g == 0)).sum() / n_neg) if n_neg else float("nan"),
        "weighted_error": float(loss),
        "deferral_rate": float((~shown).mean()),
        "actions_per_case": float(acts.mean()),
        "harmful_acceptance_rate": float((ai_wrong & accepted_ai).sum() / n),
        "over_reliance": float((ai_wrong & accepted_ai).sum() / ai_wrong.sum()) if ai_wrong.any() else float("nan"),
        "correct_rejection_rate": float((ai_wrong & ~accepted_ai).sum() / n),
        "wrong_rejection_rate": float((ai_right & ~accepted_ai).sum() / n),
        "accept_rate": float(np.mean([e["action"] == "accept" for e in events])),
        "edit_rate": float(np.mean([e["action"] == "edit" for e in events])),
        "reject_rate": float(np.mean([e["action"] == "reject" for e in events])),
        "ai_shown_accuracy": float(ai_right.sum() / shown.sum()) if shown.any() else float("nan"),
        "ai_alone_accuracy": float(((p >= threshold).astype(int) == g).mean()),
        "reader_alone_accuracy": float((np.array([e["initial_read"] for e in events]) == g).mean()),
    }
    return out


def _policies_for(reader_id: str, burn: list[dict], pooled_defer, cases, cfg) -> dict:
    return {
        "no_ai": no_ai_policy(),
        "universal": universal_policy(cfg),
        "universal_defer": pooled_defer,
        "reader_threshold": fit_policy(burn, cases, cfg, reader_id, "reader_threshold"),
        "reader_threshold_defer": fit_policy(burn, cases, cfg, reader_id, "reader_threshold_defer"),
    }


def run_reader_experiment(records: list[dict], cfg: dict, seeds: int | None = None, verbose: bool = True) -> dict:
    cases = case_table(records)
    calib = [c for c in cases.values() if c["split"] == "calib"]
    test = sorted([c for c in cases.values() if c["split"] == "test"], key=lambda c: c["case_id"])
    if not calib or not test:
        raise ValueError("need case records for both the calib and the test split")
    profiles = load_profiles(cfg)
    S = int(seeds or cfg["readers"]["seeds"])
    lc_n = sorted({min(n, len(calib)) for n in cfg["readers"]["learning_curve_n"]} | {len(calib)})
    rows, lc_rows, pol_rows, burn_rows = [], [], [], []
    t0 = time.time()
    for seed in range(S):
        order = np.random.default_rng([seed, 999]).permutation(len(calib))
        calib_seq = [calib[i] for i in order]
        burn = {}
        for ri, (rid, prof) in enumerate(profiles.items()):
            burn[rid] = simulate_stream(prof, calib_seq, universal_policy(cfg), draws_for(seed, ri, 0, len(calib_seq)))
            for k, e in enumerate(burn[rid]):
                burn_rows.append({"seed": seed, "reader": rid, "order": k, **{x: e[x] for x in
                                  ("case_id", "ai_label", "ai_confidence", "initial_read", "action", "final_label", "switched")}})
        pooled = [e for evs in burn.values() for e in evs]
        pooled_defer = fit_policy(pooled, cases, cfg, None, "universal_defer")
        for ri, (rid, prof) in enumerate(profiles.items()):
            d_test = draws_for(seed, ri, 1, len(test))
            pols = _policies_for(rid, burn[rid], pooled_defer, cases, cfg)
            for pname, pol in pols.items():
                ev = simulate_stream(prof, test, pol, d_test)
                m = summarize(ev, cases, cfg, pol.threshold)
                rows.append({"seed": seed, "reader": rid, "policy": pname, "threshold": pol.threshold,
                             "defer_uncertainty": pol.defer_uncertainty, "personalised": pol.personalised, **m})
                if pname in ("universal_defer", "reader_threshold", "reader_threshold_defer"):
                    pol_rows.append({"seed": seed, "reader": rid, "policy": pname, "threshold": pol.threshold,
                                     "defer_uncertainty": pol.defer_uncertainty, "personalised": pol.personalised,
                                     "trust_alpha": pol.trust_alpha, "trust_beta": pol.trust_beta, "p_lapse": pol.p_lapse,
                                     "n_feedback": pol.n_feedback, "n_disagreements": pol.n_disagreements,
                                     "expected_loss": pol.expected_loss, "universal_loss": pol.universal_loss})
            for n in lc_n:
                pol = fit_policy(burn[rid][:n], cases, cfg, rid, "reader_threshold_defer")
                ev = simulate_stream(prof, test, pol, d_test)
                m = summarize(ev, cases, cfg, pol.threshold)
                lc_rows.append({"seed": seed, "reader": rid, "n_feedback": n, "threshold": pol.threshold,
                                "defer_uncertainty": pol.defer_uncertainty if pol.defer_uncertainty is not None else np.nan,
                                "personalised": pol.personalised, **m})
        if verbose and (seed % 5 == 0 or seed == S - 1):
            print(f"[readers] seed {seed + 1}/{S} done ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows)
    return {"results": res, "learning_curve": pd.DataFrame(lc_rows), "policies": pd.DataFrame(pol_rows),
            "burnin": pd.DataFrame(burn_rows), "summary": aggregate(res), "benefit_spread": benefit_spread(res),
            "meta": {"n_calib": len(calib), "n_test": len(test), "seeds": S, "readers": list(profiles),
                     "test_positives": int(sum(c["gt_present"] for c in test))}}


def _ci(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    if len(x) < 2:
        return (float("nan"), float("nan"))
    return float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))


METRICS = ["accuracy", "sensitivity", "specificity", "weighted_error", "deferral_rate", "actions_per_case",
           "harmful_acceptance_rate", "over_reliance", "wrong_rejection_rate", "threshold"]


def aggregate(res: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (reader, policy), g in res.groupby(["reader", "policy"], sort=False):
        row = {"reader": reader, "policy": policy}
        for m in METRICS:
            row[m] = float(g[m].mean())
            lo, hi = _ci(g[m])
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        row["personalised_frac"] = float(g["personalised"].mean())
        out.append(row)
    df = pd.DataFrame(out)
    df["policy"] = pd.Categorical(df["policy"], POLICY_ORDER, ordered=True)
    return df.sort_values(["reader", "policy"]).reset_index(drop=True)


def benefit_spread(res: pd.DataFrame) -> pd.DataFrame:
    """RQ1: does adaptation reduce the variation in benefit across readers?

    Benefit = accuracy with the policy - accuracy reading alone, paired per seed (common random
    numbers). Each reader's *expected* benefit is its mean over seeds; heterogeneity is the
    spread (max - min) and SD of those expected benefits across readers.
    """
    base = res[res.policy == "no_ai"].set_index(["seed", "reader"])["accuracy"]
    rows = []
    for pol, g in res[res.policy != "no_ai"].groupby("policy"):
        b = g.set_index(["seed", "reader"])["accuracy"] - base
        per_reader = b.groupby(level="reader").mean()
        rows.append({"policy": pol, "mean_benefit": float(per_reader.mean()),
                     "min_benefit": float(per_reader.min()), "max_benefit": float(per_reader.max()),
                     "spread_max_minus_min": float(per_reader.max() - per_reader.min()),
                     "std_across_readers": float(per_reader.std(ddof=0))})
    df = pd.DataFrame(rows)
    df["policy"] = pd.Categorical(df["policy"], POLICY_ORDER, ordered=True)
    return df.sort_values("policy").reset_index(drop=True)


def benefit_by_reader(res: pd.DataFrame) -> pd.DataFrame:
    base = res[res.policy == "no_ai"].set_index(["seed", "reader"])["accuracy"]
    out = []
    for (pol, reader), g in res[res.policy != "no_ai"].groupby(["policy", "reader"]):
        b = g.set_index(["seed", "reader"])["accuracy"] - base.loc[g.set_index(["seed", "reader"]).index]
        lo, hi = _ci(b)
        out.append({"policy": pol, "reader": reader, "benefit": float(b.mean()), "lo": lo, "hi": hi})
    return pd.DataFrame(out)
