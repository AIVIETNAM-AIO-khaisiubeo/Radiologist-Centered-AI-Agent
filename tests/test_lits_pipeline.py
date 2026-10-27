"""Tests for the LiTS17 data layer, the VISTA3D adapter, the optional second-reader hook and calibration.

None of these need a GPU or a download: the LiTS files are small phantoms written in the Kaggle
layout, the pipeline runs with the stand-in plugins in tests/fixtures, and the VISTA3D adapter is
checked with a stand-in network (geometry, class assembly, flip test-time augmentation). The real
VISTA3D weights are exercised by the Colab notebook.

Run:  pytest -q
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radagent.calibration import Calibrator, fit_calibrator, nll, sigmoid  # noqa: E402
from radagent.config import load_config  # noqa: E402
from radagent.data.lits import build_manifest, case_number, discover_pairs, locate, random_split  # noqa: E402
from radagent.io.nifti import _to_compact  # noqa: E402
from radagent.vlm.ct_prep import liver_z_range, select_slices, window_rgb  # noqa: E402
from radagent.vlm.guard import check_text, parse_final_answer, unverified_measurements  # noqa: E402
from tests.fixtures.phantoms import write_lits_like  # noqa: E402


# --------------------------------------------------------------------------- LiTS data layer
def test_split_has_no_training_cases():
    split = random_split(list(range(131)), {"calib": 0.5, "test": 0.5})
    counts = {k: sum(v == k for v in split.values()) for k in ("train", "val", "calib", "test")}
    assert counts == {"train": 0, "val": 0, "calib": 65, "test": 66}  # nothing is trained in this project
    assert split == random_split(list(range(131)), {"calib": 0.5, "test": 0.5})      # deterministic


def test_kaggle_layout_discovery_manifest_and_locate(tmp_path):
    root = write_lits_like(tmp_path / "lits", n_cases=6, size=96, seed=3)
    pairs = discover_pairs([root])
    assert sorted(pairs) == list(range(6))
    for n, (ct, seg) in pairs.items():
        assert case_number(ct) == n == case_number(seg)
        assert Path(ct).name == f"volume-{n}.nii" and Path(seg).name == f"segmentation-{n}.nii"
    assert {Path(ct).parent.parent.name for ct, _ in pairs.values()} == {"part1", "part2"}

    cfg = load_config(None)
    cfg["data"].update({"lits_dirs": [str(root)]})
    man = build_manifest(cfg, allow_download=False)
    assert list(man["case_id"]) == [f"lits_{i:03d}" for i in range(6)]
    assert set(man["split"]) == {"calib", "test"}
    assert all(Path(p).exists() for p in man["ct_path"])

    # a run made elsewhere (e.g. on Colab) points at a path that does not exist here
    moved = "/content/kaggle/liver-tumor-segmentation/volume_pt1/" + Path(pairs[0][0]).name
    assert Path(locate(moved, cfg)).resolve() == Path(pairs[0][0]).resolve()
    assert locate("/nowhere/volume-999.nii", cfg) == "/nowhere/volume-999.nii"


def test_float_ct_is_stored_compactly():
    ints = [np.array([[-1024.0, 0.0], [12.0, 3071.0]], dtype=np.float32)]
    a = _to_compact(ints, (2, 2))
    assert a.dtype == np.int16 and a[1, 1] == 3071
    frac = [np.array([[0.5, 1.0]], dtype=np.float64)]
    assert _to_compact(frac, (1, 2)).dtype == np.float32


# --------------------------------------------------------------------------- second-reader hook pieces
def test_ct_window_rgb_and_slice_selection():
    hu = np.array([[-2000, -135, 40, 215, 3000]], dtype=np.int16)
    rgb = window_rgb(hu, [[-1024, 1024], [-135, 215], [0, 80]])
    assert rgb.shape == (1, 5, 3) and rgb.dtype == np.uint8
    assert list(rgb[0, :, 1]) == [0, 0, 128, 255, 255]                # soft-tissue window
    assert list(rgb[0, :, 2]) == [0, 0, 128, 255, 255]                # 0..80 HU window
    zs = select_slices(10, 90, 16)
    assert len(zs) == 16 and zs[0] == 90 and zs[-1] == 10 and zs == sorted(zs, reverse=True)
    assert select_slices(5, 7, 16) == [7, 6, 5]
    assert liver_z_range({"measurements": {"liver_slice_range": [3, 40]}}) == (3, 40)
    assert liver_z_range({"measurements": {}, "segmentation": {"found_liver": False}}) is None


def test_guard_flags_sizes_the_measurement_tool_did_not_produce():
    meas = {"lesions": [{"max_axial_diameter_mm": 24.0, "volume_ml": 6.2}], "predicted_tumour_volume_ml": 6.2,
            "liver_volume_ml": 1500.0}
    assert unverified_measurements("lesion of 2.5 cm and 6 ml", meas) == []
    assert unverified_measurements("a 9 cm mass", meas) == ["9 cm"]
    assert parse_final_answer("blah. Final Answer: **Yes**") == 1 and parse_final_answer("nothing") is None
    g = check_text("No lesion seen. Final Answer: no", meas, p_yes=0.9)
    assert g["final_answer"] == 0 and any("disagrees" in n for n in g["notes"])


# --------------------------------------------------------------------------- fused calibration
def test_fused_calibrator_learns_from_the_second_reader():
    rng = np.random.default_rng(5)
    n = 120
    y = (rng.random(n) < 0.5).astype(int)
    p_seg = sigmoid(rng.normal(0, 1, n) + 0.6 * (2 * y - 1))         # weak segmenter
    p_vlm = sigmoid(rng.normal(0, 1, n) + 2.0 * (2 * y - 1))         # informative second reader
    cal, info = fit_calibrator(p_seg, y, "auto", p_vlm=p_vlm)
    assert info["selected"] in ("vlm", "fused") and info["n_with_vlm"] == n
    assert nll(cal(p_seg, p_vlm), y) < nll(cal.seg_only(p_seg), y)
    # no stored second read for a case -> the segmenter-only map is used for it
    assert math.isclose(cal(float(p_seg[0])), float(cal.seg_only(p_seg[0])))
    mixed = np.where(np.arange(n) % 2 == 0, p_vlm, np.nan)
    out = cal(p_seg, mixed)
    assert np.allclose(out[1::2], cal.seg_only(p_seg[1::2]))
    assert np.allclose(Calibrator.from_dict(cal.to_dict())(p_seg, p_vlm), cal(p_seg, p_vlm))
    cal2, info2 = fit_calibrator(p_seg, y, "auto")                      # without a VLM: old behaviour
    assert info2["selected"] in ("identity", "temperature", "platt") and not cal2.has_vlm


# --------------------------------------------------------------------------- whole pipeline
def _e2e_cfg(tmp_path):
    root = write_lits_like(tmp_path / "lits", n_cases=12, size=112, seed=11)   # both classes in calib
    cfg = load_config(None)
    cfg["run_name"], cfg["output_root"] = "e2e", str(tmp_path / "outputs")
    cfg["data"].update({"lits_dirs": [str(root)], "split_fractions": {"calib": 0.42, "test": 0.58}})
    cfg["preprocess"]["img_size"] = 112
    cfg["segmenter"].update({"backend": "plugin", "plugin": "tests.fixtures.stubs:ThresholdSegmenter"})
    cfg["vlm"].update({"backend": "plugin", "plugin": "tests.fixtures.stubs:StubVLM", "n_slices": 6})
    cfg["inference"].update({"viewer_size": 96, "viewer_max_slices": 32})
    cfg["readers"].update({"seeds": 2, "learning_curve_n": [0, 2]})
    cfg["policy"]["min_feedback"] = 2
    return cfg


def test_end_to_end_pipeline_with_stub_models(tmp_path):
    from radagent import pipeline as P
    from radagent.store import RunStore

    cfg = _e2e_cfg(tmp_path)
    cwd = os.getcwd()
    os.chdir(ROOT)                                    # plugins are imported as tests.fixtures.*
    try:
        P.step_data(cfg)
        card = P.step_model(cfg)
        assert card["trained_here"] is False
        P.step_infer(cfg)
        P.step_vlm(cfg)
        cal = P.step_calibrate(cfg)
        P.step_first_read(cfg, "rule")
        P.step_simulate(cfg, 2)
        figs = P.step_figures(cfg)
        st = RunStore(Path(cfg["output_root"]) / "e2e")
        recs = st.records(["calib", "test"])
        assert recs and all(r["vlm"]["status"] == "ok" for r in recs)
        assert all(0.0 <= r["confidence"]["tumour_probability"] <= 1.0 for r in recs)
        assert all("vlm_probability" in r["confidence"] for r in recs)
        assert sorted(r["triage"]["rank"] for r in recs if r["split"] == "test") == \
            list(range(1, 1 + sum(r["split"] == "test" for r in recs)))
        rep = st.report_path(recs[0]["case_id"]).read_text()
        assert "SECOND READ" in rep and "stub-vlm" in rep
        calib = [bool(r["reference"]["tumour_present"]) for r in recs if r["split"] == "calib"]
        assert len(calib) == 5 and len(set(calib)) == 2   # seed 11: both classes -> the tools are compared
        assert {"segmenter", "second_reader", "fused"} <= set(cal["tools_test"])
        assert figs and all(Path(f).exists() for f in figs)
        arrs = st.load_viewer(recs[0]["case_id"])
        assert "vlm_rgb" in arrs and arrs["vlm_rgb"].ndim == 4 and arrs["vlm_rgb"].shape[-1] == 3

        case = next(r["case_id"] for r in recs if r["split"] == "test")
        rec, trace = P.run_single_case(cfg, case, "rule")                      # cached perception
        assert trace["status"] == "complete" and rec["vlm"]["source"] == "cache"
        rec2, trace2 = P.run_single_case(cfg, case, "rule", recompute=True)    # from the CT file
        assert trace2["status"] == "complete" and rec2["vlm"]["source"] == "model"
        assert [s["tool"] for s in trace2["steps"]][:5] == ["load_study", "segment_liver_tumour", "measure_findings",
                                                            "quality_check", "vlm_second_read"]
    finally:
        P.release_models()
        os.chdir(cwd)


# --------------------------------------------------------------------------- VISTA3D adapter
def test_vista3d_adapter_geometry_classes_and_flip_tta():
    """Stand-in network with VISTA3D's calling convention: an off-centre 'liver' block and a 'tumour' cube
    must come back at the same place in the radiological view, with anisotropic spacing and flip TTA."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")
    from radagent.inference import segment_volume
    from radagent.models.vista3d import A_MAX, A_MIN, LABEL_LIVER, LABEL_TUMOUR, Vista3DSegmenter

    def scaled(hu):
        return (hu - A_MIN) / (A_MAX - A_MIN)

    class FakeVista(torch.nn.Module):
        def forward(self, x, patch_coords=None, point_coords=None, point_labels=None, class_vector=None,
                    prompt_class=None, transpose=False, **kw):
            assert point_coords is None and class_vector.flatten().tolist() == [LABEL_LIVER, LABEL_TUMOUR]
            liver = (x > scaled(40.0)) & (x < scaled(90.0))
            tum = x > scaled(110.0)
            logits = torch.cat([torch.where(liver | tum, 8.0, -8.0), torch.where(tum, 8.0, -8.0)], dim=1)
            return logits                                          # [1, 2, h, w, d] (already "transposed")

    Z, H, W = 40, 96, 80
    view = np.full((Z, H, W), -1000.0, dtype=np.float32)
    view[:, 8:88, 6:74] = 0.0                                      # body (soft tissue, not liver)
    view[10:30, 20:60, 10:40] = 60.0                               # liver on the patient's right (view cols 10..40)
    view[18:24, 30:40, 20:28] = 140.0                              # tumour inside it
    cfg = load_config(None)
    cfg["preprocess"]["img_size"] = 64
    cfg["segmenter"].update({"resample_mm": 3.0, "roi_size": [32, 32, 32], "device": "cpu"})
    seg = Vista3DSegmenter(cfg, network=FakeVista())
    assert seg.n_passes == 2 and seg.describe()["backend"] == "vista3d"
    spacing = (2.5, 1.5, 1.5)
    probs = seg.predict_volume(view, spacing, 64)
    assert probs.shape == (2, Z, 3, 64, 64) and probs.dtype == np.float16
    assert np.allclose(probs.astype(np.float32).sum(axis=2), 1.0, atol=2e-2)
    assert np.abs(probs[0].astype(np.float32) - probs[1].astype(np.float32)).max() < 0.2   # flip undone
    lab = probs.astype(np.float32).mean(0).argmax(1)               # (Z, 64, 64)
    r = lambda a: int(round(a * 64 / H))                          # noqa: E731 - row scale
    c = lambda a: int(round(a * 64 / W))                          # noqa: E731 - col scale
    assert (lab[20, r(25):r(55), c(12):c(38)] > 0).mean() > 0.9            # liver (or tumour) where it was put
    assert (lab[20, r(25):r(55), c(50):c(70)] == 0).mean() > 0.95          # not mirrored to the left side
    assert (lab[21, r(32):r(38), c(21):c(27)] == 2).mean() > 0.8           # tumour
    assert (lab[2] == 0).all() and (lab[-2] == 0).all()                    # outside the liver's z-range
    out = segment_volume(view, spacing, seg, cfg)
    assert out.found_liver and out.n_passes == 2 and out.tumour.any() and 0 < out.z0 < 10 and 30 < out.z1 <= Z
    assert out.p_raw > 0.9 and len(out.pass_scores) == 2
