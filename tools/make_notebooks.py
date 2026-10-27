"""Generate the two notebooks from source (keeps them reviewable as Python).

    python tools/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def nb(cells, colab=False):
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if colab:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


EVAL = [
    md("""
# ReaderAdapt-Liver - reproducible evaluation

This notebook re-computes every number of the evaluation from a finished run directory
(`outputs/<run>/`, produced by `python -m radagent all --config ...`). It covers:

| Section | Guidance step | Question |
|---|---|---|
| 1 | Data | What was evaluated, with which patient-level split? |
| 2 | Step 1 | How good are segmentation and measurement? |
| 3 | Step 1-2 | How well does the pretrained segmenter's case score (NVIDIA VISTA3D) separate tumour from no-tumour cases? |
| 3b | Step 2 | Is the case-level confidence calibrated (Brier, ECE, reliability)? |
| 4 | Step 2/4 | Does uncertainty identify AI errors (selective prediction)? |
| 5 | Step 3 | Does the triage order surface tumour cases first? |
| 6 | Step 4/5 | Universal vs reader-specific vs deferral policies with simulated readers (RQ1-RQ3) |
| 7 | Agent | What did the agent actually do? |

> **Research prototype.** Simulated readers show that the method behaves as designed; they are *not*
> evidence of clinical benefit or of reading-time savings.
>
> **Segmenter evaluation caveat.** The one model used, NVIDIA VISTA3D, was trained on data that include MSD
> Task 03 (built from the LiTS training volumes). Segmentation and AUROC numbers on LiTS are therefore
> optimistic; they are not a held-out evaluation of VISTA3D.

Set `RUN` below to the run you want to evaluate - by default `outputs/lits`, the run made by
`notebooks/colab_lits_pipeline.ipynb` on LiTS17 (unzip `lits_run.zip` into `outputs/`).
"""),
    code("""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
from radagent.store import RunStore, load_json
from radagent.config import load_config
from radagent import plots
from radagent.experiments import run_reader_experiment, POLICY_LABELS, benefit_by_reader

RUN = ROOT / "outputs" / "lits"          # the run folder produced by the Colab notebook
store = RunStore(RUN)
cfg = load_config(RUN / "config_used.yaml")
records = store.records()
test = [r for r in records if r["split"] == "test"]
calib = [r for r in records if r["split"] == "calib"]
pd.set_option("display.precision", 3)
print(f"run: {RUN.name} | data: LiTS17 | segmenter: {cfg['segmenter']['backend']} | "
      f"second reader: {cfg['vlm']['backend']} | "
      f"records: {len(records)} (calib {len(calib)}, test {len(test)})")
"""),
    md("## 1. Data and splits\nSplits are by patient/case; the calibrator and reader policies never see a test case."),
    code("""
man = store.manifest()
split_counts = man.groupby("split").size().reindex(["calib", "test"])
pos = {s: int(sum(r["reference"]["tumour_present"] for r in records if r["split"] == s)) for s in ("calib", "test")}
pd.DataFrame({"cases": split_counts,
              "with tumour (reference)": pd.Series(pos)}).fillna("-")
"""),
    md("""
## 2. Segmentation and measurement (Step 1)
Dice is computed at the **original resolution** against the reference masks. Volumes use the voxel spacing
from each NIfTI header. One case record looks like this (the structured output of the agent's first read):
"""),
    code("""
ex = next(r for r in test if r["reference"]["tumour_present"])
{k: ex["confidence"].get(k) for k in ("tumour_probability_raw", "vlm_probability_raw", "tumour_probability",
                                     "uncertainty", "tools_disagree")} | {
    "case_id": ex["case_id"], "predicted_tumour_volume_ml": ex["measurements"]["predicted_tumour_volume_ml"],
    "n_lesions": ex["measurements"]["n_lesions"], "triage": ex["triage"]["category"], "review_status": ex["review_status"]}
"""),
    code("""
seg = pd.DataFrame([{"case": r["case_id"], "tumour_present": r["reference"]["tumour_present"],
                     "dice_liver": r["reference"]["dice_liver"], "dice_tumour": r["reference"]["dice_tumour"],
                     "pred_tumour_ml": r["measurements"]["predicted_tumour_volume_ml"],
                     "ref_tumour_ml": r["reference"]["tumour_volume_ml"],
                     "pred_liver_ml": r["measurements"]["liver_volume_ml"], "ref_liver_ml": r["reference"]["liver_volume_ml"],
                     "ref_lesions": r["reference"]["n_lesions"], "detected": r["reference"]["lesions_detected"],
                     "fp_lesions": r["reference"]["fp_lesions"]} for r in test])
posc = seg[seg.tumour_present]
summary = pd.Series({
    "liver Dice (mean)": seg.dice_liver.mean(),
    "liver volume error (median |%|)": (100 * (seg.pred_liver_ml - seg.ref_liver_ml).abs() / seg.ref_liver_ml).median(),
    "tumour Dice, cases with tumour (mean)": posc.dice_tumour.mean(),
    "tumour Dice, cases with tumour (median)": posc.dice_tumour.median(),
    "lesion-level sensitivity": posc.detected.sum() / max(1, posc.ref_lesions.sum()),
    "false-positive lesions per case": seg.fp_lesions.mean(),
})
summary.to_frame("test split")
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
plots.style()
axes[0].scatter(seg.ref_liver_ml, seg.pred_liver_ml, color=plots.SERIES[0], edgecolor=plots.SURFACE, s=40)
lim = [seg[["ref_liver_ml", "pred_liver_ml"]].min().min() * 0.95, seg[["ref_liver_ml", "pred_liver_ml"]].max().max() * 1.05]
axes[0].plot(lim, lim, color=plots.AXIS, lw=1); axes[0].set_xlabel("Reference liver volume (ml)"); axes[0].set_ylabel("Predicted (ml)")
axes[0].set_title("Liver volume", loc="left")
axes[1].scatter(seg.ref_tumour_ml + 0.01, seg.pred_tumour_ml + 0.01, color=plots.SERIES[0], edgecolor=plots.SURFACE, s=40)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].plot([0.01, 100], [0.01, 100], color=plots.AXIS, lw=1)
axes[1].set_xlabel("Reference tumour volume + 0.01 (ml)"); axes[1].set_ylabel("Predicted + 0.01 (ml)")
axes[1].set_title("Tumour volume (log scale)", loc="left")
plt.tight_layout()
"""),
    md("""
## 3. The AI tool: NVIDIA VISTA3D's case score
VISTA3D segments the liver (label 1) and hepatic tumours (label 26) in 3D. Its **case score** is the mean of the
top-k smoothed tumour probabilities inside the liver (k = 0.1 ml). It is calibrated on the calibration split and
evaluated on the test split. (If a research second reader is plugged in, it appears here as well.)
"""),
    code("""
cm = load_json(RUN / "calibration" / "calibration_metrics.json")
tools = cm.get("tools_test", {})
pd.DataFrame([{"tool": k, "n": r["n"], "with tumour": r["n_pos"], "AUROC": r["auroc"],
               "AUROC 95% CI": f"{r['auroc_ci'][0]:.3f}-{r['auroc_ci'][1]:.3f}" if r.get("auroc_ci") else "-",
               "Brier": r["brier"], "ECE": r["ece"], "NLL": r["nll"]} for k, r in tools.items()]).set_index("tool")
"""),
    code("""
v = [r for r in test if r["confidence"].get("vlm_probability_raw") is not None]
if not v:
    print("one model, no second reader - nothing to compare")
agree = pd.DataFrame([{"case": r["case_id"], "tumour": r["reference"]["tumour_present"],
                       "segmenter": r["confidence"]["tumour_probability_raw"],
                       "second_reader": r["confidence"]["vlm_probability_raw"],
                       "used": r["confidence"]["tumour_probability"],
                       "disagree": r["confidence"].get("tools_disagree", False),
                       "final_answer": (r.get("vlm") or {}).get("guard", {}).get("final_answer"),
                       "unverified_sizes": len((r.get("vlm") or {}).get("guard", {}).get("unverified_sizes", []))}
                      for r in v])
if len(agree):
    print(f"{len(v)}/{len(test)} test cases have a second read; tools disagree on {int(agree.disagree.sum())}")
    err = (agree.used >= cfg["triage"]["suggest_threshold"]) != agree.tumour
    print("AI error rate where the tools agree / disagree:", err.groupby(agree.disagree).mean().round(3).to_dict())
    print("Second-reader texts that quote sizes the measurement tool did not produce:", int((agree.unverified_sizes > 0).sum()))
agree.round(3)
"""),
    code("""
if plots.ai_tools_figure(records, RUN / "figures" / "ai_tools.png", cm, plots.tool_names(cfg)):
    plt.imshow(plt.imread(RUN / "figures" / "ai_tools.png")); plt.axis("off"); plt.gcf().set_size_inches(12, 5)
"""),
    md("""
## 3b. Confidence calibration (Step 2)
The score the agent uses is selected on the **calibration split** by 5-fold cross-validated NLL among: no
calibration, temperature scaling and Platt scaling of the segmenter score (plus the second-reader and fused models
when a research second reader is plugged in). The winner is refitted on the full calibration split and evaluated once on the test split.
Intervals are 95 % bootstrap intervals over test cases.
"""),
    code("""
print("calibrator:", cm["calibrator"]["description"], "| selection:", cm["fit"].get("cv_nll"))
rows = []
for stage in ("before", "after"):
    m = cm["test"][stage]
    rows.append({"stage": "raw score" if stage == "before" else "calibrated", "n": m["n"], "positives": m["n_pos"],
                 "Brier": m["brier"], "Brier 95% CI": f"{m['brier_ci'][0]:.3f}-{m['brier_ci'][1]:.3f}",
                 "ECE": m["ece"], "ECE 95% CI": f"{m['ece_ci'][0]:.3f}-{m['ece_ci'][1]:.3f}",
                 "ECE (equal-mass)": m["ece_adaptive"], "NLL": m["nll"], "AUROC": m["auroc"]})
pd.DataFrame(rows).set_index("stage")
"""),
    code("""
plots.reliability_figure(cm, RUN / "figures" / "calibration_reliability.png")
plt.imshow(plt.imread(RUN / "figures" / "calibration_reliability.png")); plt.axis("off"); plt.gcf().set_size_inches(10, 7)
"""),
    md("""
Calibration of a single score changes *what a probability means*, not the ranking: AUROC is unchanged by
temperature scaling and by Platt scaling with a positive slope (the fused model can change the ranking). With few test cases the ECE interval is wide - the bootstrap interval is
reported so the improvement is not over-read.

## 4. Does uncertainty point at the AI's mistakes? (selective prediction)
"""),
    code("""
from sklearn.metrics import roc_auc_score
t = cfg["triage"]["suggest_threshold"]
p = np.array([r["confidence"]["tumour_probability"] for r in test])
u = np.array([r["confidence"]["uncertainty"] for r in test])
g = np.array([int(r["reference"]["tumour_present"]) for r in test])
ai = (p >= t).astype(int)
err = (ai != g).astype(int)
out = {"AI-alone accuracy": (ai == g).mean(), "sensitivity": ai[g == 1].mean(), "specificity": 1 - ai[g == 0].mean(),
       "AUROC of uncertainty for AI error": roc_auc_score(err, u) if 0 < err.sum() < len(err) else np.nan,
       "error rate, most certain 50%": err[np.argsort(u)[: len(u) // 2]].mean(),
       "error rate, least certain 50%": err[np.argsort(u)[len(u) // 2:]].mean()}
pd.Series(out).to_frame("test split")
"""),
    code("""
plots.risk_coverage_figure(records, RUN / "figures" / "risk_coverage.png", t)
plt.imshow(plt.imread(RUN / "figures" / "risk_coverage.png")); plt.axis("off"); plt.gcf().set_size_inches(7, 4.6)
"""),
    md("## 5. Worklist triage (Step 3)"),
    code("""
pr = np.array([r["triage"]["priority"] for r in test])
order = np.argsort(-pr)
k = max(1, len(test) // 4)
cats = pd.Series([r["triage"]["category"] for r in test]).value_counts()
print("triage categories on the test worklist:", cats.to_dict())
pd.Series({"tumour prevalence": g.mean(), f"tumour rate in the top {k} cases": g[order[:k]].mean(),
           f"tumour rate in the bottom {k} cases": g[order[-k:]].mean()}).to_frame("test split")
"""),
    code("""
plots.triage_lift_figure(records, RUN / "figures" / "triage_lift.png")
plt.imshow(plt.imread(RUN / "figures" / "triage_lift.png")); plt.axis("off"); plt.gcf().set_size_inches(7, 4.6)
"""),
    md("""
## 6. Reader adaptation and deferral (Steps 4-5)
Re-run the controlled experiment here (deterministic seeds - it reproduces `experiments/*.csv`).

* **Burn-in:** each simulated reader reads the calibration cases with the universal policy (sequential protocol).
* **Fit:** reader-specific threshold / deferral from that reader's feedback (trust curve + lapse rate); a pooled
  deferral rule for `Universal + deferral`.
* **Evaluate:** all policies on the same test cases with the same random draws.
Correctness always comes from the reference annotation.
"""),
    code("""
exp = run_reader_experiment(records, cfg, verbose=False)
s = exp["summary"].copy()
s["policy"] = s["policy"].map(POLICY_LABELS)
cols = ["reader", "policy", "accuracy", "sensitivity", "specificity", "weighted_error", "harmful_acceptance_rate",
        "over_reliance", "deferral_rate", "actions_per_case", "threshold", "personalised_frac"]
s[cols]
"""),
    code("""
plots.policy_comparison_figure(exp["summary"], cfg, RUN / "figures" / "policy_comparison.png")
plt.imshow(plt.imread(RUN / "figures" / "policy_comparison.png")); plt.axis("off"); plt.gcf().set_size_inches(12, 8)
"""),
    md("### RQ1 - Does per-reader adaptation reduce the variation in benefit across readers?"),
    code("""
b = exp["benefit_spread"].copy(); b["policy"] = b["policy"].map(POLICY_LABELS); b
"""),
    code("""
plots.benefit_figure(exp["results"], cfg, RUN / "figures" / "benefit_by_reader.png")
plt.imshow(plt.imread(RUN / "figures" / "benefit_by_reader.png")); plt.axis("off"); plt.gcf().set_size_inches(10, 5)
"""),
    md("### RQ2 - Does learning to defer reduce the harm caused by incorrect suggestions?"),
    code("""
r = exp["summary"]
piv = r.pivot_table(index="reader", columns="policy", values="harmful_acceptance_rate", observed=False)
piv = piv[[c for c in ["universal", "universal_defer", "reader_threshold", "reader_threshold_defer"] if c in piv]]
piv.columns = [POLICY_LABELS[c] for c in piv.columns]
piv.round(4)
"""),
    md("### RQ3 - How much feedback is needed, and is the learned policy stable?"),
    code("""
lc = exp["learning_curve"]
tab = lc.groupby(["reader", "n_feedback"]).agg(weighted_error=("weighted_error", "mean"),
                                              threshold=("threshold", "mean"),
                                              threshold_sd=("threshold", "std"),
                                              personalised=("personalised", "mean")).round(3)
tab.unstack(0)
"""),
    code("""
plots.learning_curve_figure(lc, exp["results"], cfg, RUN / "figures" / "learning_curve.png")
plt.imshow(plt.imread(RUN / "figures" / "learning_curve.png")); plt.axis("off"); plt.gcf().set_size_inches(13, 7)
"""),
    md("## 7. What did the agent do?\nEvery first read is a traced plan of tool calls with preconditions enforced by "
       "the runner."),
    code("""
steps = []
for f in sorted((RUN / "traces").glob("*.json")):
    tr = json.loads(f.read_text())
    for s_ in tr["steps"]:
        steps.append({"case": tr["case_id"], "planner": tr["planner"], "tool": s_["tool"], "ms": s_.get("duration_ms", 0),
                      "override": bool(s_.get("guardrail"))})
steps = pd.DataFrame(steps)
print(f"{steps.case.nunique()} traces, planner(s): {steps.planner.unique().tolist()}, guardrail overrides: {int(steps.override.sum())}")
steps.groupby("tool", sort=False).agg(calls=("ms", "size"), median_ms=("ms", "median")).round(1)
"""),
    md("""
## Limitations
* **Single dataset, seen by the segmenter.** All numbers are on LiTS17 (131 CTs from several centres, mostly with
  tumour), and VISTA3D's training data include these volumes (MSD Task 03). Segmentation quality and AUROC here
  are optimistic; an external test set is needed to evaluate the segmenter. Calibration and the reader
  experiments are still split correctly (calib vs test).
* **One pretrained model, used as published.** VISTA3D is not fine-tuned here; its weights are for
  non-commercial research use (NVIDIA NCLS v1).
* **LiTS case-level imbalance.** Most LiTS volumes contain tumour, so case-level specificity and calibration of the
  negative class rest on few cases; report intervals, and consider `confidence.target_min_tumour_ml` for an
  "actionable burden" definition.
* **Simulated readers** follow a parametric behaviour model (own read -> trust curve -> lapses). Real readers will
  differ - the next step is the pilot with mentor radiologists.
* **Small calibration/burn-in sets** (about 65 cases) make learned thresholds noisy; the policy therefore only
  personalises when the expected improvement clears a margin, and the learning curve shows how this stabilises.
"""),
]

COLAB = [
    md("""
# ReaderAdapt-Liver on LiTS17 with one pretrained model: NVIDIA VISTA3D (Colab, GPU)

**Runtime -> Change runtime type -> GPU** (T4 works; L4 / A100 is faster).

One Colab secret (key icon on the left, *Notebook access* on):

| Secret | Why | Where to get it |
|---|---|---|
| `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` + `KAGGLE_KEY`) | download the two LiTS17 datasets | kaggle.com -> Settings -> API -> Create New Token |

The segmentation model, **NVIDIA VISTA3D** (MONAI), is downloaded from Hugging Face (`MONAI/VISTA3D-HF`,
~870 MB) without a token and used as published - **nothing is trained**. It segments the liver and hepatic
tumours in 3D. Its weights are for non-commercial research use (NVIDIA NCLS v1).

Pipeline: download LiTS (both Kaggle parts) -> calib / test split -> fetch VISTA3D -> segment + measure every
CT -> calibration -> agentic first read + reports -> reader experiment -> figures -> **website**.

Note: VISTA3D's training data include MSD Task 03, built from the LiTS training volumes, so its segmentation
quality on LiTS is optimistic (not a held-out evaluation).

Rough times on a T4: download 10-15 min; VISTA3D about 1-2 min per CT with the flip pass (~2-3 h for all 131).
Set `QUICK = True` below for a ~40-case first look without the flip pass.
"""),
    code("""
# 1. Code + dependencies
REPO_URL = "https://github.com/AIVIETNAM-AIO-khaisiubeo/ReaderAdapt-Liver.git"
BRANCH = "vista3d-single-model"   # use "main" once the pull requests are merged
import os
if not os.path.exists("ReaderAdapt-Liver"):
    !git clone -q -b $BRANCH $REPO_URL
%cd ReaderAdapt-Liver
!git pull -q
!pip install -q -r requirements.txt -r requirements-deep.txt
"""),
    code("""
# 2. Secrets -> environment (KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY; HF_TOKEN optional)
import os
from google.colab import userdata
for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "HF_TOKEN"):
    try:
        v = userdata.get(k)
        if v:
            os.environ[k] = v
    except Exception:
        pass
print({k: (k in os.environ) for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY")})
import torch
print(torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - change the runtime!")
"""),
    code("""
# 3. Run configuration
import yaml, torch
QUICK = False                     # True: proportional ~40-case subset, no flip pass, for a fast first look
cfg = yaml.safe_load(open("configs/lits.yaml"))
big_gpu = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 30e9
cfg["segmenter"]["sw_batch_size"] = 4 if big_gpu else 1
if QUICK:
    cfg["run_name"], cfg["data"]["max_cases"], cfg["inference"]["tta_flip"] = "lits_quick", 40, False
yaml.safe_dump(cfg, open("configs/lits_colab.yaml", "w"), sort_keys=False)
RUN = f"outputs/{cfg['run_name']}"
print("run:", RUN, "| segmenter: NVIDIA VISTA3D (pretrained) | flip pass:", cfg["inference"]["tta_flip"])
"""),
    code("""
# 4. LiTS17 from Kaggle (part 1 = segmentations + volume_pt1-5, part 2 = the remaining volumes), then the split
!python -m radagent download --config configs/lits_colab.yaml
!python -m radagent data --config configs/lits_colab.yaml
"""),
    code("# 5. Fetch the pretrained NVIDIA VISTA3D weights (~870 MB, cached) and write the model card - no training\n"
         "!python -m radagent model --config configs/lits_colab.yaml"),
    code("# 6. Segment + measure every CT with VISTA3D (liver + hepatic tumour; CT and its mirror image)\n"
         "!python -m radagent infer --config configs/lits_colab.yaml"),
    code("""
# 7. Calibration, agentic first read + reports, reader experiment, figures
!python -m radagent calibrate --config configs/lits_colab.yaml
!python -m radagent first-read --config configs/lits_colab.yaml
!python -m radagent simulate --config configs/lits_colab.yaml
!python -m radagent figures --config configs/lits_colab.yaml
"""),
    code("""
# 8. One agent trace and its report
!python -m radagent agent --config configs/lits_colab.yaml
import glob
print(open(sorted(glob.glob(f"{RUN}/reports/*.txt"))[0]).read())
"""),
    code("""
from IPython.display import Image, display
for f in ("calibration_reliability.png", "segmentation_quality.png", "policy_comparison.png"):
    if os.path.exists(f"{RUN}/figures/{f}"):
        display(Image(f"{RUN}/figures/{f}", width=900))
"""),
    md("""
## 9. Open the website

**Option A - straight from Colab (everything works, including *Recompute from the CT file*, because the CTs, the
GPU and the model are here).** The cell starts the website and prints a public `trycloudflare.com` link (a free
Cloudflare quick tunnel; anyone with the link can open it while the cell runs - stop the cell to close it).
"""),
    code("""
import subprocess, re, time
if not os.path.exists("cloudflared"):
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    !chmod +x cloudflared
app = subprocess.Popen(["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", "8501",
                        "--server.headless", "true", "--", "--run", RUN],
                       stdout=open("streamlit.log", "w"), stderr=subprocess.STDOUT)
time.sleep(5)
tunnel = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in tunnel.stdout:
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        print("Website:", m.group(0))
        break
try:
    while app.poll() is None:
        time.sleep(30)
except KeyboardInterrupt:
    pass
finally:
    tunnel.terminate(); app.terminate()
"""),
    md("""
**Option B - on your own computer.** Download the run, unzip it into the `outputs/` folder of your local copy of
the repository and start `python -m streamlit run app/streamlit_app.py -- --run outputs/lits`. Everything except
*Recompute from the CT file* works without a GPU (recompute needs the LiTS files, PyTorch, MONAI and the VISTA3D
weights; a GPU is recommended).
"""),
    code("""
!cd outputs && zip -qr ../lits_run.zip {cfg['run_name']}
!ls -lh lits_run.zip
from google.colab import files
files.download("lits_run.zip")
# Optional: keep a copy in Google Drive
# from google.colab import drive; drive.mount("/content/drive"); !cp lits_run.zip /content/drive/MyDrive/
"""),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    (out / "evaluation.ipynb").write_text(json.dumps(nb(EVAL), indent=1))
    (out / "colab_lits_pipeline.ipynb").write_text(json.dumps(nb(COLAB, colab=True), indent=1))
    print("wrote notebooks/evaluation.ipynb and notebooks/colab_lits_pipeline.ipynb")


if __name__ == "__main__":
    main()
