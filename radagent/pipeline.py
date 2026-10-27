"""End-to-end pipeline steps (each step is also a CLI sub-command).

download -> data -> model (fetch the pretrained VISTA3D weights, write the model card) -> infer
(segment + measure) -> [vlm: optional second reader, off by default] -> calibrate -> first-read
-> simulate -> figures. ``all`` runs them in order; every step caches its results. Nothing is trained.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from radagent.agent.planner import RulePlanner, make_planner
from radagent.agent.runner import run_agent
from radagent.agent.tools import (CaseContext, load_study, measure_findings, model_signature,
                                  segment_liver_tumour, vlm_second_read)
from radagent.calibration import _fit, calibration_report, fit_calibrator, reliability_table
from radagent.config import run_dir, save_config
from radagent.store import RunStore, dump_json, load_json
from radagent.triage import rank_worklist
from radagent.vlm import vlm_enabled, vlm_signature

_MODELS: dict = {}


def _store(cfg) -> RunStore:
    s = RunStore(run_dir(cfg))
    s.root.mkdir(parents=True, exist_ok=True)
    return s


def release_models(kind: str | None = None) -> None:
    """Free cached models (e.g. the segmenter before a second reader is loaded on the same GPU)."""
    for k in [k for k in _MODELS if kind is None or k[0] == kind]:
        del _MODELS[k]
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _segmenter_loader(cfg):
    def get():
        key = ("seg", model_signature(cfg))
        if key not in _MODELS:
            from radagent.models import load_segmenter

            _MODELS[key] = load_segmenter(cfg)
        return _MODELS[key]

    return get


def _vlm_loader(cfg):
    def get():
        key = ("vlm", vlm_signature(cfg))
        if key not in _MODELS:
            from radagent.vlm import load_vlm

            _MODELS[key] = load_vlm(cfg)
        return _MODELS[key]

    return get


def _contexts(cfg: dict, splits: list[str], case_ids: list[str] | None = None):
    st = _store(cfg)
    man = st.manifest()
    rows = man[man["split"].isin(splits)]
    if case_ids:
        rows = rows[rows["case_id"].isin(case_ids)]
    cal = st.calibrator()
    get_seg, get_vlm = _segmenter_loader(cfg), _vlm_loader(cfg)
    from radagent.data.lits import locate

    for _, row in rows.iterrows():
        row = row.to_dict()
        for k in ("ct_path", "mask_path"):
            row[k] = locate(row.get(k, ""), cfg)
        yield CaseContext(case_id=row["case_id"], row=row, cfg=cfg, store=st, calibrator=cal,
                          get_segmenter=get_seg, get_vlm=get_vlm)


# --------------------------------------------------------------------------- data
def step_download(cfg: dict) -> list[str]:
    from radagent.data.lits import resolve_lits_dirs

    dirs = resolve_lits_dirs(cfg, allow_download=True)
    print("[download] LiTS folders:\n  " + "\n  ".join(dirs))
    return dirs


def step_data(cfg: dict) -> pd.DataFrame:
    from radagent.data.lits import build_manifest

    df = build_manifest(cfg)
    st = _store(cfg)
    st.save_manifest(df)
    save_config(cfg, st.p("config_used.yaml"))
    print(f"[data] {len(df)} LiTS cases -> {st.manifest_path} | " +
          ", ".join(f"{k}: {v}" for k, v in df["split"].value_counts().reindex(["calib", "test"]).fillna(0).astype(int).items()))
    return df


def step_model(cfg: dict) -> dict:
    """Fetch the pretrained segmenter (nothing is trained) and record what it is in the run's model card."""
    from radagent.models import model_card_path

    sc = cfg["segmenter"]
    if sc["backend"] == "vista3d":
        from radagent.models.vista3d import HF_FILE, HF_REPO, LABEL_LIVER, LABEL_TUMOUR, LICENSE, weights_path

        path = weights_path(cfg)
        card = {"backend": "vista3d", "name": "NVIDIA VISTA3D", "trained_here": False,
                "source": f"https://huggingface.co/{sc.get('hf_repo', HF_REPO)}", "file": sc.get("hf_file", HF_FILE),
                "revision": sc.get("hf_revision") or "main", "weights_path": str(path), "license": LICENSE,
                "reference": "He et al., VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging, "
                             "CVPR 2025 (arXiv:2406.05285)",
                "label_prompts": {"liver": LABEL_LIVER, "hepatic tumour": LABEL_TUMOUR},
                "preprocessing": f"RAS, {sc.get('resample_mm', 1.5)} mm isotropic, HU [-963.8, 1053.7] -> [0, 1], "
                                 f"sliding window {sc.get('roi_size', [128, 128, 128])}",
                "training_data_overlap": "VISTA3D was trained on 11,454 CTs from 15 sources, including MSD Task 03 "
                                         "(Liver), which is built from the LiTS training volumes - the cases used "
                                         "here. Metrics on LiTS are therefore optimistic and are not a held-out "
                                         "evaluation of the segmenter."}
        print(f"[model] VISTA3D weights: {path}")
    else:
        card = {"backend": sc["backend"], "plugin": sc.get("plugin"), "trained_here": False}
    p = model_card_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    dump_json(card, p)
    return card


# --------------------------------------------------------------------------- perception
def step_infer(cfg: dict, force: bool = False) -> None:
    """Segmenter tools (load -> segment -> measure) on the evaluation splits, cached per case."""
    t0 = time.time()
    ctxs = list(_contexts(cfg, cfg["inference"]["eval_splits"]))
    for k, ctx in enumerate(ctxs, start=1):
        ctx.force_recompute = force
        t1 = time.time()
        load_study(ctx)
        s = segment_liver_tumour(ctx)
        m = measure_findings(ctx)
        ref = ctx.analysis.get("reference", {})
        print(f"[infer] {k:3d}/{len(ctxs)} {ctx.case_id} ({ctx.row['split']}) {s['source']:>5} | "
              f"liver {m['liver_ml']} ml, tumour {m['tumour_ml']} ml ({m['n_lesions']} lesions) | "
              f"raw score {ctx.analysis['segmentation']['tumour_score_raw']:.3f} | "
              f"ref tumour {ref.get('tumour_present')} Dice L {ref.get('dice_liver', float('nan')):.3f} "
              f"T {ref.get('dice_tumour', float('nan')):.3f} | {time.time() - t1:.1f}s", flush=True)
        ctx.volume = ctx.seg = None
    print(f"[infer] done in {time.time() - t0:.0f}s")


def step_vlm(cfg: dict, force: bool = False) -> None:
    """Optional second reader on the evaluation splits (off by default: the project uses one model)."""
    if not vlm_enabled(cfg):
        print("[vlm] no second reader configured (vlm.backend: none) - skipping")
        return
    release_models("seg")
    reader = _vlm_loader(cfg)()
    if hasattr(reader, "_load"):
        reader._load()  # fail early and clearly
    t0 = time.time()
    ctxs = list(_contexts(cfg, cfg["inference"]["eval_splits"]))
    n_err = 0
    for k, ctx in enumerate(ctxs, start=1):
        if not ctx.store.has_analysis(ctx.case_id):
            print(f"[vlm] {ctx.case_id}: no segmenter result - run `infer` first")
            continue
        load_study(ctx)
        segment_liver_tumour(ctx)
        measure_findings(ctx)
        ctx.force_recompute, ctx.vlm_mode = force, "compute"
        v = vlm_second_read(ctx)
        ref = ctx.analysis.get("reference", {})
        if v.get("status") == "ok":
            print(f"[vlm] {k:3d}/{len(ctxs)} {ctx.case_id} {v.get('source', ''):>5} | P(yes) {v['p_yes']:.3f} | "
                  f"final answer {v.get('final_answer')} | ref tumour {ref.get('tumour_present')} | "
                  f"{v.get('seconds') or 0:.1f}s", flush=True)
        else:
            n_err += 1
            print(f"[vlm] {k:3d}/{len(ctxs)} {ctx.case_id} {v.get('status')}: {v.get('reason')}", flush=True)
        ctx.volume = None
    print(f"[vlm] done in {time.time() - t0:.0f}s ({n_err} without a read)")


# --------------------------------------------------------------------------- calibration
def _scores(st: RunStore, man: pd.DataFrame, split: str):
    ids = man.loc[man["split"] == split, "case_id"].tolist()
    p, v, y, keep = [], [], [], []
    for cid in ids:
        if not st.has_analysis(cid):
            continue
        a = st.load_analysis(cid)
        if "reference" not in a or "segmentation" not in a:
            continue
        p.append(a["segmentation"]["tumour_score_raw"])
        vr = a.get("vlm") or {}
        v.append(vr["p_yes"] if vr.get("status") == "ok" else np.nan)
        y.append(int(a["reference"]["tumour_present"]))
        keep.append(cid)
    return np.array(p, dtype=float), np.array(v, dtype=float), np.array(y, dtype=int), keep


def step_calibrate(cfg: dict) -> dict:
    st = _store(cfg)
    man = st.manifest()
    p_cal, v_cal, y_cal, _ = _scores(st, man, "calib")
    p_test, v_test, y_test, _ = _scores(st, man, "test")
    if len(p_cal) == 0:
        raise RuntimeError("no analysed calibration cases - run `infer` first")
    cal, info = fit_calibrator(p_cal, y_cal, cfg["confidence"]["method"], seed=cfg["seed"], p_vlm=v_cal)
    nb = cfg["confidence"]["n_bins"]
    out = {"calibrator": cal.to_dict(), "fit": info, "n_bins": nb}
    for split, (p, v, y) in {"calib": (p_cal, v_cal, y_cal), "test": (p_test, v_test, y_test)}.items():
        if len(p) == 0:
            continue
        pc = cal(p, v)
        out[split] = {"before": calibration_report(p, y, nb, seed=cfg["seed"]),
                      "after": calibration_report(pc, y, nb, seed=cfg["seed"]),
                      "reliability_before": reliability_table(p, y, nb),
                      "reliability_after": reliability_table(pc, y, nb)}
    # AI tools compared on the test split: each one calibrated on the calib split alone
    if len(p_test) and len(np.unique(y_cal)) == 2:
        comp = {}
        seg = _fit("platt", p_cal, v_cal, y_cal)
        comp["segmenter"] = calibration_report(seg.seg_only(p_test), y_test, nb, n_boot=300, seed=cfg["seed"])
        ok_c, ok_t = np.isfinite(v_cal), np.isfinite(v_test)
        if ok_c.sum() >= 4 and ok_t.sum() >= 2 and len(np.unique(y_cal[ok_c])) == 2:
            vl = _fit("vlm", p_cal, v_cal, y_cal)
            fu = _fit("fused", p_cal, v_cal, y_cal)
            comp["second_reader"] = calibration_report(vl.vlm_only(v_test[ok_t]), y_test[ok_t], nb, n_boot=300, seed=cfg["seed"])
            comp["fused"] = calibration_report(fu(p_test, v_test), y_test, nb, n_boot=300, seed=cfg["seed"])
        out["tools_test"] = comp
    dump_json(cal.to_dict(), st.calibrator_path())
    dump_json(out, st.p("calibration", "calibration_metrics.json"))
    t = out.get("test", {})
    print(f"[calibrate] {cal.describe()} (selected: {info.get('selected')}, CV NLL {info.get('cv_nll')})")
    if t:
        b, a = t["before"], t["after"]
        print(f"[calibrate] TEST n={b['n']} (pos {b['n_pos']}): Brier {b['brier']:.3f} -> {a['brier']:.3f} | "
              f"ECE {b['ece']:.3f} -> {a['ece']:.3f} | NLL {b['nll']:.3f} -> {a['nll']:.3f} | AUROC {a['auroc']:.3f}")
    for name, r in (out.get("tools_test") or {}).items():
        print(f"[calibrate]   {name:10s} AUROC {r['auroc']:.3f} | Brier {r['brier']:.3f} | ECE {r['ece']:.3f} (n={r['n']})")
    return out


# --------------------------------------------------------------------------- first read
def step_first_read(cfg: dict, planner_name: str | None = None) -> list[dict]:
    st = _store(cfg)
    if planner_name:
        cfg = {**cfg, "agent": {**cfg["agent"], "planner": planner_name}}
    planner = make_planner(cfg)
    if not st.calibrator_path().exists():
        print("[first-read] WARNING: no calibrator - confidences are raw scores. Run `calibrate` first.")
    records, t0 = [], time.time()
    ctxs = list(_contexts(cfg, cfg["inference"]["eval_splits"]))
    for k, ctx in enumerate(ctxs, start=1):
        ctx.record = {"case_id": ctx.case_id, "split": ctx.row["split"], "source": "LiTS17",
                      "review_status": "pending"}
        trace = run_agent(ctx, planner, cfg["agent"]["max_steps"])
        if trace["status"] != "complete":
            print(f"[first-read] {ctx.case_id}: agent status {trace['status']} - {trace['steps'][-1].get('summary')}")
            continue
        records.append(ctx.record)
        ctx.volume = ctx.seg = None
        if k % 10 == 0 or k == len(ctxs):
            print(f"[first-read] {k}/{len(ctxs)} cases ({time.time() - t0:.0f}s, planner {trace['planner']}, "
                  f"overrides {trace['n_guardrail_overrides']})")
    for split in {r["split"] for r in records}:
        rank_worklist([r for r in records if r["split"] == split])
    from radagent.report import draft_report

    for r in records:
        r.pop("report_text", None)
        st.save_record(r)
        st.report_path(r["case_id"]).write_text(draft_report(r))
    cats = pd.Series([r["triage"]["category"] for r in records if r["split"] == "test"]).value_counts().to_dict()
    n_v = sum(1 for r in records if (r.get("vlm") or {}).get("status") == "ok")
    extra = f" ({n_v} with a second read)" if vlm_enabled(cfg) else ""
    print(f"[first-read] {len(records)} case records written{extra} | test-split triage: {cats}")
    return records


def run_single_case(cfg: dict, case_id: str, planner_name: str = "rule", reader_policy=None, save: bool = False,
                    recompute: bool = False) -> tuple[dict, dict]:
    """Run the agent on one case.

    Default: reuse cached perception (fast, no GPU). ``recompute=True`` re-runs the segmenter (and a
    second reader, if one is configured) on the CT file, updates the case's stored analysis, viewer
    images, record and report.
    """
    cfg2 = {**cfg, "agent": {**cfg["agent"], "planner": planner_name}}
    ctx = next(_contexts(cfg2, ["calib", "test"], [case_id]))
    ctx.reader_policy = reader_policy
    ctx.persist = save or recompute
    if recompute:
        ctx.force_recompute, ctx.vlm_mode = True, "compute"
    ctx.record = {"case_id": case_id, "split": ctx.row["split"], "source": "LiTS17", "review_status": "pending"}
    planner = make_planner(cfg2) if planner_name == "llm" else RulePlanner()
    trace = run_agent(ctx, planner, cfg["agent"]["max_steps"], save=save or recompute)
    if recompute and trace["status"] == "complete":
        st = ctx.store
        old = st.load_record(case_id) if st.case_path(case_id).exists() else {}
        if old.get("triage"):
            ctx.record["triage"]["rank"] = old["triage"].get("rank")
            ctx.record["triage"]["of"] = old["triage"].get("of")
        ctx.record.pop("report_text", None)
        ctx.record.pop("policy_decisions", None)
        st.save_record(ctx.record)
        from radagent.report import draft_report

        st.report_path(case_id).write_text(draft_report(ctx.record))
    return ctx.record, trace


# --------------------------------------------------------------------------- readers
def step_simulate(cfg: dict, seeds: int | None = None) -> dict:
    from radagent.experiments import run_reader_experiment

    st = _store(cfg)
    records = st.records(["calib", "test"])
    out = run_reader_experiment(records, cfg, seeds=seeds)
    d = st.p("experiments")
    d.mkdir(parents=True, exist_ok=True)
    for k in ("results", "learning_curve", "policies", "burnin", "summary", "benefit_spread"):
        out[k].to_csv(d / f"{k}.csv", index=False)
    dump_json(out["meta"], d / "meta.json")
    s = out["summary"]
    print("[simulate] test-split means over seeds (accuracy | harmful acceptance | deferral | actions/case):")
    for _, r in s.iterrows():
        print(f"   {r['reader']:9s} {str(r['policy']):24s} acc {r['accuracy']:.3f} | harm {r['harmful_acceptance_rate']:.3f} | "
              f"defer {r['deferral_rate']:.2f} | actions {r['actions_per_case']:.2f} | thr {r['threshold']:.2f}")
    print("[simulate] benefit spread across readers (RQ1):")
    for _, r in out["benefit_spread"].iterrows():
        print(f"   {str(r['policy']):24s} mean benefit {r['mean_benefit']:+.3f} | spread {r['spread_max_minus_min']:.3f}")
    return out


def step_figures(cfg: dict) -> list[str]:
    from radagent import plots

    st = _store(cfg)
    return plots.make_all(st, cfg)


def step_all(cfg: dict) -> None:
    t0 = time.time()
    step_data(cfg)
    step_model(cfg)
    step_infer(cfg)
    step_vlm(cfg)
    release_models()
    step_calibrate(cfg)
    step_first_read(cfg)
    step_simulate(cfg)
    files = step_figures(cfg)
    print(f"[all] finished in {(time.time() - t0) / 60:.1f} min; {len(files)} figures in {run_dir(cfg) / 'figures'}")
    print(f"[all] open the app:  python -m streamlit run app/streamlit_app.py -- --run {run_dir(cfg)}")


__all__ = ["step_download", "step_data", "step_model", "step_infer", "step_vlm", "step_calibrate", "step_first_read",
           "step_simulate", "step_figures", "step_all", "run_single_case", "release_models", "load_json"]
