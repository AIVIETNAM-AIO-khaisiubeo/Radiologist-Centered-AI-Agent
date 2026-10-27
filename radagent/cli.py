"""Command line:  python -m radagent <command> --config configs/lits.yaml

Commands
  download    download both LiTS parts from Kaggle with kagglehub (cached afterwards)
  data        pair CTs with masks and write the case manifest + split
  model       fetch the pretrained NVIDIA VISTA3D weights and write the model card (nothing is trained)
  infer       run VISTA3D (segment + measure) on the calib/test cases (GPU recommended)
  vlm         optional second reader (off by default: vlm.backend none)
  calibrate   fit the confidence calibrator on the calib split, evaluate on test
  first-read  run the agent (plan -> tools -> report) on every evaluation case
  simulate    reader experiment: universal vs reader-specific vs deferral policies
  figures     make all plots
  all         everything above, in order
  agent       run the agent on one case and print its trace (--planner llm, --recompute)
"""
from __future__ import annotations

import argparse
import json

from radagent.config import load_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="radagent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "data", "model", "infer", "vlm", "calibrate", "first-read",
                                        "simulate", "figures", "all", "agent"])
    ap.add_argument("--config", default="configs/lits.yaml")
    ap.add_argument("--force", action="store_true", help="infer / vlm: recompute cached results")
    ap.add_argument("--seeds", type=int, default=None, help="simulate: number of seeds")
    ap.add_argument("--planner", choices=["rule", "llm"], default=None)
    ap.add_argument("--case", default=None, help="agent: case id (default: first test case)")
    ap.add_argument("--recompute", action="store_true", help="agent: re-run the segmenter on the CT file")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    from radagent import pipeline as P

    if args.command == "download":
        P.step_download(cfg)
    elif args.command == "data":
        P.step_data(cfg)
    elif args.command == "model":
        P.step_model(cfg)
    elif args.command == "infer":
        P.step_infer(cfg, force=args.force)
    elif args.command == "vlm":
        P.step_vlm(cfg, force=args.force)
    elif args.command == "calibrate":
        P.step_calibrate(cfg)
    elif args.command == "first-read":
        P.step_first_read(cfg, args.planner)
    elif args.command == "simulate":
        P.step_simulate(cfg, args.seeds)
    elif args.command == "figures":
        for f in P.step_figures(cfg):
            print(f)
    elif args.command == "all":
        P.step_all(cfg)
    elif args.command == "agent":
        from radagent.config import run_dir
        from radagent.store import RunStore

        man = RunStore(run_dir(cfg)).manifest()
        case = args.case or man.loc[man["split"] == "test", "case_id"].iloc[0]
        rec, trace = P.run_single_case(cfg, case, args.planner or cfg["agent"]["planner"], recompute=args.recompute)
        for s in trace["steps"]:
            g = f"  [guardrail: {s['guardrail']}]" if s.get("guardrail") else ""
            print(f"{s['step']:2d}. {s['tool']:22s} by {s['planned_by']:26s} {s.get('duration_ms', 0):8.1f} ms  "
                  f"{json.dumps(s['summary'], default=str)[:110]}{g}")
            if s.get("reason"):
                print(f"      reason: {s['reason']}")
        print(f"status: {trace['status']}")


if __name__ == "__main__":
    main()
