"""Figures for the evaluation (matplotlib, static PNG for the repo and the app).

Colour is assigned by job: the four AI policies take the first four categorical slots
of a CVD-validated palette in a fixed order; the no-AI control is the muted neutral;
readers are distinguished by marker shape (never by a second colour scheme).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from radagent.experiments import POLICY_LABELS, POLICY_ORDER  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POLICY_COLORS = {"no_ai": MUTED, "universal": SERIES[0], "universal_defer": SERIES[1],
                 "reader_threshold": SERIES[2], "reader_threshold_defer": SERIES[3]}
READER_MARKERS = {"reader_a": "o", "reader_b": "s", "reader_c": "^"}


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    })


def _reader_name(cfg: dict, rid: str) -> str:
    return cfg["readers"]["profiles"].get(rid, {}).get("name", rid)


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- calibration
def reliability_figure(metrics: dict, path: Path, split: str = "test") -> str | None:
    style()
    if split not in metrics:
        return None
    m = metrics[split]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for j, (key, title) in enumerate((("before", "Raw score (uncalibrated)"), ("after", "Calibrated"))):
        t = m[f"reliability_{key}"]
        rep = m[key]
        ax, axh = axes[0, j], axes[1, j]
        edges = np.array(t["edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        conf = np.array(t["confidence"], dtype=float)
        freq = np.array(t["frequency"], dtype=float)
        cnt = np.array(t["count"])
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.0, zorder=1)
        ok = cnt > 0
        for c, f in zip(conf[ok], freq[ok]):
            ax.plot([c, c], [c, f], color=SERIES[7] if f < c else SERIES[0], lw=1.2, alpha=0.6, zorder=2)
        ax.plot(conf[ok], freq[ok], color=SERIES[0], lw=2, zorder=3)
        ax.scatter(conf[ok], freq[ok], s=20 + 12 * cnt[ok], color=SERIES[0], edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, loc="left")
        ax.set_ylabel("Observed tumour frequency")
        ax.text(0.03, 0.97, f"Brier {rep['brier']:.3f}\nECE {rep['ece']:.3f}\nNLL {rep['nll']:.3f}\nAUROC {rep['auroc']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK2)
        axh.bar(centers, cnt, width=(edges[1] - edges[0]) * 0.8, color=MUTED)
        axh.set_ylabel("Cases")
        axh.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        axh.set_xlabel("Predicted probability")
    fig.suptitle(f"Reliability of the case-level tumour probability ({split} split, n={m['after']['n']})",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------------------- policies
def policy_comparison_figure(summary: pd.DataFrame, cfg: dict, path: Path) -> str:
    style()
    panels = [("accuracy", "Final-decision accuracy", None),
              ("harmful_acceptance_rate", "Harmful acceptances per case", None),
              ("deferral_rate", "Deferral rate", None),
              ("actions_per_case", "Review actions per case", None)]
    readers = list(dict.fromkeys(summary["reader"]))
    pols = [p for p in POLICY_ORDER if p in set(summary["policy"].astype(str))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    width = 0.8 / len(pols)
    for ax, (metric, title, _) in zip(axes.ravel(), panels):
        for k, pol in enumerate(pols):
            sub = summary[summary["policy"].astype(str) == pol].set_index("reader").reindex(readers)
            x = np.arange(len(readers)) + (k - (len(pols) - 1) / 2) * width
            y = sub[metric].to_numpy(dtype=float)
            lo, hi = sub[f"{metric}_lo"].to_numpy(dtype=float), sub[f"{metric}_hi"].to_numpy(dtype=float)
            if metric == "deferral_rate" and pol == "no_ai":
                y = np.full_like(y, np.nan)
            ax.bar(x, y, width=width * 0.86, color=POLICY_COLORS[pol], label=POLICY_LABELS[pol], zorder=2)
            err = np.vstack([np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)])
            ok = np.isfinite(y)
            ax.errorbar(x[ok], y[ok], yerr=err[:, ok], fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=3)
        ax.set_xticks(np.arange(len(readers)))
        ax.set_xticklabels([_reader_name(cfg, r) for r in readers], fontsize=9)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        if metric == "accuracy":
            lo_y = np.nanmin(summary["accuracy_lo"].to_numpy(dtype=float))
            ax.set_ylim(max(0.0, lo_y - 0.05), 1.0)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(pols), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Test split - bars: mean over simulation seeds, whiskers: 2.5-97.5 percentile over seeds",
                 y=0.958, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, path)


def benefit_figure(res: pd.DataFrame, cfg: dict, path: Path) -> str:
    from radagent.experiments import benefit_by_reader

    style()
    b = benefit_by_reader(res)
    pols = [p for p in POLICY_ORDER if p != "no_ai" and p in set(b["policy"])]
    readers = list(dict.fromkeys(res["reader"]))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, pol in enumerate(pols):
        sub = b[b.policy == pol].set_index("reader").reindex(readers)
        ys = sub["benefit"].to_numpy(dtype=float)
        ax.plot([i, i], [np.nanmin(ys), np.nanmax(ys)], color=POLICY_COLORS[pol], lw=6, alpha=0.25, solid_capstyle="round")
        for j, r in enumerate(readers):
            ax.errorbar(i + (j - 1) * 0.12, sub.loc[r, "benefit"],
                        yerr=[[max(0, sub.loc[r, "benefit"] - sub.loc[r, "lo"])], [max(0, sub.loc[r, "hi"] - sub.loc[r, "benefit"])]],
                        fmt=READER_MARKERS.get(r, "o"), color=POLICY_COLORS[pol], mec=SURFACE, mew=1.5, ms=8,
                        ecolor=INK2, elinewidth=1)
        ax.text(i + 0.22, np.nanmean(ys), f"spread\n{np.nanmax(ys) - np.nanmin(ys):.3f}", ha="left", va="center",
                fontsize=8.5, color=INK2)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(len(pols)))
    ax.set_xticklabels([POLICY_LABELS[p] for p in pols], fontsize=9)
    ax.set_ylabel("Accuracy benefit vs reading alone")
    ax.set_xlim(-0.5, len(pols) - 0.3)
    ax.set_title("RQ1 - Does adaptation reduce the variation in benefit across readers?", loc="left")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D

    proxies = [Line2D([], [], marker=READER_MARKERS.get(r, "o"), color=INK2, linestyle="None", markersize=8,
                      label=_reader_name(cfg, r)) for r in readers]
    ax.legend(handles=proxies, loc="lower right", ncol=3)
    fig.tight_layout()
    return _save(fig, path)


def learning_curve_figure(lc: pd.DataFrame, res: pd.DataFrame, cfg: dict, path: Path) -> str:
    """Top: change in weighted error vs the universal policy, paired per seed (common random numbers).
    Bottom: the learned suggestion threshold and the share of seeds where personalisation was adopted."""
    style()
    readers = list(dict.fromkeys(lc["reader"]))
    uni = res[res.policy == "universal"].set_index(["seed", "reader"])["weighted_error"]
    noai = res[res.policy == "no_ai"].set_index(["seed", "reader"])["weighted_error"]
    fig, axes = plt.subplots(2, len(readers), figsize=(4.0 * len(readers), 6.4), sharex=True)
    axes = np.atleast_2d(axes)
    for j, r in enumerate(readers):
        sub = lc[lc.reader == r].copy()
        sub["delta"] = sub["weighted_error"].to_numpy() - uni.loc[list(zip(sub.seed, sub.reader))].to_numpy()
        g = sub.groupby("n_feedback")
        n = np.array(sorted(sub["n_feedback"].unique()))
        mean = g["delta"].mean().reindex(n).to_numpy()
        q = g["delta"].quantile([0.25, 0.75]).unstack().reindex(n)
        ax = axes[0, j]
        ax.axhline(0, color=POLICY_COLORS["universal"], lw=1.5)
        d_noai = float((noai.xs(r, level="reader") - uni.xs(r, level="reader")).mean())
        ax.axhline(d_noai, color=POLICY_COLORS["no_ai"], lw=1.5)
        ax.fill_between(n, q[0.25].to_numpy(), q[0.75].to_numpy(), color=POLICY_COLORS["reader_threshold_defer"],
                        alpha=0.15, lw=0)
        ax.plot(n, mean, color=POLICY_COLORS["reader_threshold_defer"], marker="o", ms=6, mec=SURFACE, mew=2)
        ax.set_title(_reader_name(cfg, r), loc="left")
        if j == 0:
            ax.set_ylabel("Change in weighted error\nvs universal policy")
        ax2 = axes[1, j]
        thr = g["threshold"].mean().reindex(n)
        tq = g["threshold"].quantile([0.25, 0.75]).unstack().reindex(n)
        ax2.fill_between(n, tq[0.25].to_numpy(), tq[0.75].to_numpy(), color=SERIES[0], alpha=0.12, lw=0)
        ax2.plot(n, thr.to_numpy(), color=SERIES[0], marker="o", ms=6, mec=SURFACE, mew=2)
        pers = g["personalised"].mean().reindex(n)
        ax2.plot(n, pers.to_numpy(), color=MUTED, marker="s", ms=6, mec=SURFACE, mew=2)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Feedback events from this reader")
        if j == 0:
            ax2.set_ylabel("Learned policy")
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=POLICY_COLORS["reader_threshold_defer"], marker="o", label="Reader-specific + deferral"),
               Line2D([], [], color=POLICY_COLORS["universal"], label="Universal threshold (reference = 0)"),
               Line2D([], [], color=POLICY_COLORS["no_ai"], label="Reader alone"),
               Line2D([], [], color=SERIES[0], marker="o", label="learned suggestion threshold"),
               Line2D([], [], color=MUTED, marker="s", label="share of seeds personalised")]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.995), fontsize=8.5)
    fig.suptitle("RQ3 - How much feedback is needed, and is the learned policy stable?  "
                 "(line = mean over seeds, band = interquartile range)", y=0.955, fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


# --------------------------------------------------------------------------- AI-only analyses
def risk_coverage_figure(records: list[dict], path: Path, threshold: float = 0.5) -> str | None:
    style()
    recs = [r for r in records if r.get("split") == "test" and "reference" in r]
    if len(recs) < 3:
        return None
    u = np.array([r["confidence"]["uncertainty"] for r in recs])
    p = np.array([r["confidence"]["tumour_probability"] for r in recs])
    g = np.array([int(r["reference"]["tumour_present"]) for r in recs])
    err = ((p >= threshold).astype(int) != g).astype(float)
    order = np.argsort(u)
    cov = np.arange(1, len(recs) + 1) / len(recs)
    risk = np.cumsum(err[order]) / np.arange(1, len(recs) + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(err.mean(), color=MUTED, lw=1.5)
    ax.text(0.02, err.mean(), " no abstention", va="bottom", fontsize=8.5, color=INK2)
    ax.plot(cov, risk, color=SERIES[0], marker="o", ms=4, mec=SURFACE, mew=1)
    ax.set_xlabel("Coverage (fraction of cases the AI keeps, most certain first)")
    ax.set_ylabel("AI error rate on kept cases")
    ax.set_title("Selective prediction: risk-coverage on the test split", loc="left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, max(0.05, risk.max() * 1.15))
    fig.tight_layout()
    return _save(fig, path)


def triage_lift_figure(records: list[dict], path: Path) -> str | None:
    style()
    recs = [r for r in records if r.get("split") == "test" and "reference" in r]
    if len(recs) < 3:
        return None
    g = np.array([int(r["reference"]["tumour_present"]) for r in recs])
    if g.sum() == 0:
        return None
    pr = np.array([r["triage"]["priority"] for r in recs])
    order = np.argsort(-pr)
    x = np.arange(0, len(recs) + 1) / len(recs)
    found = np.concatenate([[0], np.cumsum(g[order])]) / g.sum()
    oracle = np.concatenate([[0], np.cumsum(np.sort(g)[::-1])]) / g.sum()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, label="Random order")
    ax.plot(x, oracle, color=AXIS, lw=1.5, label="Perfect order")
    ax.plot(x, found, color=SERIES[0], label="Agent priority")
    ax.set_xlabel("Fraction of the worklist read")
    ax.set_ylabel("Fraction of tumour cases found")
    ax.set_title("Worklist triage on the test split", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def segmentation_figure(records: list[dict], path: Path) -> str | None:
    style()
    recs = [r for r in records if r.get("split") == "test" and "reference" in r]
    if not recs:
        return None
    dl = np.array([r["reference"]["dice_liver"] for r in recs])
    dt = np.array([r["reference"]["dice_tumour"] for r in recs if r["reference"]["tumour_present"]])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rng = np.random.default_rng(0)
    for i, (vals, name) in enumerate(((dl, "Liver (all cases)"), (dt, "Tumour (cases with tumour)"))):
        if len(vals) == 0:
            continue
        ax.scatter(vals, i + rng.uniform(-0.12, 0.12, len(vals)), s=36, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.plot([np.median(vals)] * 2, [i - 0.25, i + 0.25], color=INK, lw=2, zorder=4)
        ax.text(np.median(vals), i + 0.3, f"median {np.median(vals):.3f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Liver (all cases)", "Tumour (cases with tumour)"])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("Dice (original resolution, per case)")
    ax.set_title("Segmentation quality on the test split", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def _roc(score: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    keep = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]       # one point per distinct threshold
    tp, fp = np.cumsum(yy)[keep], np.cumsum(1 - yy)[keep]
    return np.r_[0, fp / max(1, (1 - y).sum())], np.r_[0, tp / max(1, y.sum())]


SEG_NAMES = {"vista3d": "NVIDIA VISTA3D"}


def tool_names(cfg: dict) -> tuple[str, str]:
    seg = cfg["segmenter"]["backend"]
    vlm = cfg.get("vlm", {})
    vname = str(vlm.get("plugin") or vlm.get("backend") or "none")
    return SEG_NAMES.get(seg, str(cfg["segmenter"].get("plugin") or seg).split(":")[-1]), vname.split(":")[-1]


def ai_tools_figure(records: list[dict], path: Path, metrics: dict | None = None,
                    names: tuple[str, str] = ("NVIDIA VISTA3D", "second reader")) -> str | None:
    """Segmenter vs an optional second reader vs the score the agent used, on the test split."""
    style()
    recs = [r for r in records if r.get("split") == "test" and "reference" in r
            and r["confidence"].get("vlm_probability_raw") is not None]
    if len(recs) < 3:
        return None
    y = np.array([int(r["reference"]["tumour_present"]) for r in recs])
    if y.min() == y.max():
        return None
    seg = np.array([r["confidence"]["tumour_probability_raw"] for r in recs])
    vlm = np.array([r["confidence"]["vlm_probability_raw"] for r in recs])
    used = np.array([r["confidence"]["tumour_probability"] for r in recs])
    tools = (metrics or {}).get("tools_test", {})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    a1.plot([0, 1], [0, 1], color=MUTED, lw=1.2)
    for (name, score, key), col in zip(((f"Segmenter: {names[0]}", seg, "segmenter"), (names[1], vlm, "second_reader"),
                                        ("Used by the agent", used, None)), (SERIES[0], SERIES[1], INK)):
        fx, ty = _roc(score, y)
        auc = tools.get(key, {}).get("auroc") if key else None
        if auc is None:
            from radagent.calibration import auroc
            auc = auroc(score, y)
        a1.plot(fx, ty, color=col, lw=2.2 if key is None else 1.8,
                ls="--" if key is None else "-", label=f"{name} (AUROC {auc:.2f})")
    a1.set_xlabel("False-positive rate")
    a1.set_ylabel("True-positive rate")
    a1.set_title("Case-level tumour detection", loc="left")
    a1.legend(loc="lower right")
    a1.set_xlim(-0.02, 1.02)
    a1.set_ylim(-0.02, 1.02)
    for g, col, lab, mk in ((1, SERIES[1], "Tumour (reference)", "o"), (0, SERIES[0], "No tumour (reference)", "s")):
        m = y == g
        a2.scatter(seg[m], vlm[m], s=44, color=col, marker=mk, edgecolor=SURFACE, linewidth=1.2, label=lab, zorder=3)
    dis = np.array([bool(r["confidence"].get("tools_disagree")) for r in recs])
    if dis.any():
        a2.scatter(seg[dis], vlm[dis], s=120, facecolor="none", edgecolor=INK, linewidth=1.2,
                   label="Flagged: tools disagree", zorder=4)
    a2.axhline(0.5, color=AXIS, lw=1)
    a2.axvline(0.5, color=AXIS, lw=1)
    a2.set_xlim(-0.02, 1.02)
    a2.set_ylim(-0.02, 1.02)
    a2.set_xlabel("Segmenter score (raw)")
    a2.set_ylabel(f"{names[1]} P(yes) (raw)")
    a2.set_title("Do the two AI tools agree?", loc="left")
    a2.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, path)


def make_all(store, cfg: dict) -> list[str]:
    from radagent.store import load_json

    figs = []
    fd = store.p("figures")
    cm = store.p("calibration", "calibration_metrics.json")
    metrics = load_json(cm) if cm.exists() else None
    if metrics:
        figs.append(reliability_figure(metrics, fd / "calibration_reliability.png"))
    records = store.records()
    figs.append(ai_tools_figure(records, fd / "ai_tools.png", metrics, tool_names(cfg)))
    figs += [segmentation_figure(records, fd / "segmentation_quality.png"),
             risk_coverage_figure(records, fd / "risk_coverage.png", cfg["triage"]["suggest_threshold"]),
             triage_lift_figure(records, fd / "triage_lift.png")]
    ed = store.p("experiments")
    if (ed / "summary.csv").exists():
        summary = pd.read_csv(ed / "summary.csv")
        res = pd.read_csv(ed / "results.csv")
        lc = pd.read_csv(ed / "learning_curve.csv")
        figs += [policy_comparison_figure(summary, cfg, fd / "policy_comparison.png"),
                 benefit_figure(res, cfg, fd / "benefit_by_reader.png"),
                 learning_curve_figure(lc, res, cfg, fd / "learning_curve.png")]
    return [f for f in figs if f]
