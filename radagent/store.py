"""Run directory layout and JSON/NPZ persistence.

outputs/<run_name>/
    manifest.csv                 case list with paths and split
    config_used.yaml
    model/                       segmenter checkpoint + model_card.json
    cache/analysis/<case>.json   perception results (segmentation summary, measurements, reference)
    viewer/<case>.npz            compact arrays for the review UI
    calibration/                 calibrator.json, metrics, reliability tables
    cases/<case>.json            final case records (what the UI and experiments read)
    traces/<case>.json           agent traces
    reports/<case>.txt           report drafts
    experiments/                 reader-simulation results (csv/json)
    figures/                     plots
    feedback/<reader>.jsonl      feedback logged from the UI
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, cls=_NpEncoder, allow_nan=True))
    tmp.replace(path)


def load_json(path: str | Path):
    return json.loads(Path(path).read_text())


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    # ---- paths
    def p(self, *parts) -> Path:
        return self.root.joinpath(*parts)

    @property
    def manifest_path(self) -> Path:
        return self.p("manifest.csv")

    def analysis_path(self, case_id: str) -> Path:
        return self.p("cache", "analysis", f"{case_id}.json")

    def viewer_path(self, case_id: str) -> Path:
        return self.p("viewer", f"{case_id}.npz")

    def case_path(self, case_id: str) -> Path:
        return self.p("cases", f"{case_id}.json")

    def trace_path(self, case_id: str) -> Path:
        return self.p("traces", f"{case_id}.json")

    def report_path(self, case_id: str) -> Path:
        return self.p("reports", f"{case_id}.txt")

    def calibrator_path(self) -> Path:
        return self.p("calibration", "calibrator.json")

    # ---- io
    def manifest(self) -> pd.DataFrame:
        return pd.read_csv(self.manifest_path)

    def save_manifest(self, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.manifest_path, index=False)

    def has_analysis(self, case_id: str) -> bool:
        return self.analysis_path(case_id).exists()

    def load_analysis(self, case_id: str) -> dict:
        return load_json(self.analysis_path(case_id))

    def save_analysis(self, case_id: str, d: dict) -> None:
        dump_json(d, self.analysis_path(case_id))

    def save_viewer(self, case_id: str, arrays: dict) -> None:
        self.viewer_path(case_id).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.viewer_path(case_id), **arrays)

    def load_viewer(self, case_id: str) -> dict:
        with np.load(self.viewer_path(case_id)) as d:
            return {k: d[k] for k in d.files}

    def save_record(self, rec: dict) -> None:
        dump_json(rec, self.case_path(rec["case_id"]))

    def load_record(self, case_id: str) -> dict:
        return load_json(self.case_path(case_id))

    def records(self, splits: list[str] | None = None) -> list[dict]:
        out = []
        for f in sorted(self.p("cases").glob("*.json")):
            r = load_json(f)
            if splits is None or r.get("split") in splits:
                out.append(r)
        return out

    def calibrator(self):
        from radagent.calibration import Calibrator

        if self.calibrator_path().exists():
            return Calibrator.from_dict(load_json(self.calibrator_path()))
        return Calibrator("identity", note="no calibrator fitted yet")
