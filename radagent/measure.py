"""Measurements at the *original* image resolution, from the header voxel spacing.

The mean class probabilities are resized back to the original slice size (whole
slice, then cropped to the liver bounding box, so pixel centres stay aligned), the
per-voxel argmax is taken, and every volume is ``n_voxels x dz x drow x dcol``.
Lesions are 3D connected components of the tumour mask; for each we report volume,
the longest in-plane diameter on its largest axial cross-section (RECIST-like),
slice range, centroid, mean attenuation and mean model probability.

Reference comparison (research mode only) - Dice and lesion detection against the
dataset annotation - is computed here because it needs the full-resolution masks,
but it is stored separately and is never shown to the planner or written into a
report.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from radagent.inference import SegOutput, fill_holes_slicewise, filter_lesions, largest_component
from radagent.preprocess import upsample_slice, window_u8


def _max_diameter_mm(rows: np.ndarray, cols: np.ndarray, d_row: float, d_col: float) -> float:
    pts = np.stack([rows * d_row, cols * d_col], axis=1).astype(np.float64)
    if len(pts) < 3:
        return float(max(d_row, d_col))
    try:
        from scipy.spatial import ConvexHull

        hull = pts[ConvexHull(pts).vertices]
    except Exception:  # collinear points
        hull = pts
    if len(hull) > 400:
        hull = hull[np.linspace(0, len(hull) - 1, 400).astype(int)]
    diff = hull[:, None, :] - hull[None, :, :]
    return float(np.sqrt((diff ** 2).sum(-1)).max() + 0.5 * (d_row + d_col))   # + one pixel of extent


def _bbox_orig(liver_model: np.ndarray, S: int, H: int, W: int, pad: int = 3) -> tuple[int, int, int, int]:
    rows = np.where(liver_model.any(axis=(0, 2)))[0]
    cols = np.where(liver_model.any(axis=(0, 1)))[0]
    r0, r1 = max(0, rows.min() - pad), min(S, rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(S, cols.max() + pad + 1)
    return (int(math.floor(r0 * H / S)), int(math.ceil(r1 * H / S)), int(math.floor(c0 * W / S)), int(math.ceil(c1 * W / S)))


def measure(view_hu: np.ndarray, spacing: tuple, seg: SegOutput, cfg: dict) -> dict:
    """Return dict(measurements=..., maps=...) where maps hold original-res crops for later steps."""
    Z, H, W = view_hu.shape
    dz, dr, dc = spacing
    vox_ml = dz * dr * dc / 1000.0
    if not seg.found_liver:
        return {"measurements": {"liver_found": False, "liver_volume_ml": 0.0, "predicted_tumour_volume_ml": 0.0,
                                 "n_lesions": 0, "lesions": [], "tumour_burden_pct": 0.0},
                "maps": None}
    S = seg.img_size
    R0, R1, C0, C1 = _bbox_orig(seg.liver, S, H, W)
    n = seg.z1 - seg.z0
    lab_o = np.zeros((n, R1 - R0, C1 - C0), dtype=np.uint8)
    tprob_o = np.zeros((n, R1 - R0, C1 - C0), dtype=np.float32)
    for i in range(n):
        ups = np.stack([upsample_slice(seg.mean[i, c], (H, W))[R0:R1, C0:C1] for c in range(3)])
        lab_o[i] = ups.argmax(0)
        tprob_o[i] = ups[2]
    liver_o = fill_holes_slicewise(largest_component(lab_o > 0))
    min_ml = float(cfg["inference"]["min_lesion_ml"])
    min_comp = max(1, int(round(min_ml / vox_ml)))
    tumour_o = filter_lesions((lab_o == 2) & liver_o, tprob_o, min_comp, float(cfg["inference"]["min_lesion_peak_prob"]))
    outside = int(((lab_o == 2) & ~liver_o).sum())
    tumour_class = int((lab_o == 2).sum())

    hu_crop = view_hu[seg.z0:seg.z1, R0:R1, C0:C1]
    lesion_lab, n_les = ndi.label(tumour_o)
    lesions = []
    if n_les:
        objs = ndi.find_objects(lesion_lab)
        for j, sl in enumerate(objs, start=1):
            comp = lesion_lab[sl] == j
            vox = int(comp.sum())
            areas = comp.sum(axis=(1, 2))
            zi = int(np.argmax(areas))
            rr, cc = np.nonzero(comp[zi])
            zz, r_, c_ = np.nonzero(comp)
            cz = float(zz.mean() + sl[0].start)
            crow = float(r_.mean() + sl[1].start)
            ccol = float(c_.mean() + sl[2].start)
            lesions.append({
                "id": j,
                "volume_ml": round(vox * vox_ml, 3),
                "max_axial_diameter_mm": round(_max_diameter_mm(rr, cc, dr, dc), 1),
                "slice_range": [int(seg.z0 + sl[0].start), int(seg.z0 + sl[0].stop - 1)],
                "key_slice": int(seg.z0 + sl[0].start + zi),
                "centroid_view": [round(seg.z0 + cz, 1), round(R0 + crow, 1), round(C0 + ccol, 1)],
                "centroid_mm": [round((seg.z0 + cz) * dz, 1), round((R0 + crow) * dr, 1), round((C0 + ccol) * dc, 1)],
                "mean_hu": round(float(hu_crop[sl][comp].mean()), 1),
                "mean_probability": round(float(tprob_o[sl][comp].mean()), 3),
            })
        lesions.sort(key=lambda d: -d["volume_ml"])
        # relabel by size so lesion 1 is the largest
        order = {les["id"]: k for k, les in enumerate(lesions, start=1)}
        remap = np.zeros(n_les + 1, dtype=np.uint8)
        for old, new in order.items():
            remap[old] = min(new, 255)
        lesion_lab = remap[lesion_lab]
        for k, les in enumerate(lesions, start=1):
            les["id"] = k

    liver_ml = float(liver_o.sum()) * vox_ml
    lz = np.where(liver_o.any(axis=(1, 2)))[0]
    tumour_ml = float(tumour_o.sum()) * vox_ml
    meas = {
        "liver_found": True,
        "liver_volume_ml": round(liver_ml, 1),
        "liver_slice_range": [int(seg.z0 + lz.min()), int(seg.z0 + lz.max())] if lz.size else None,
        "liver_mean_hu": round(float(hu_crop[liver_o & ~tumour_o].mean()) if (liver_o & ~tumour_o).any() else float("nan"), 1),
        "predicted_tumour_volume_ml": round(tumour_ml, 3),
        "tumour_burden_pct": round(100.0 * tumour_ml / liver_ml, 3) if liver_ml > 0 else 0.0,
        "n_lesions": len(lesions),
        "largest_lesion_diameter_mm": lesions[0]["max_axial_diameter_mm"] if lesions else 0.0,
        "lesions": lesions,
        "tumour_voxels_outside_liver_frac": round(outside / max(1, tumour_class), 4),
        "tumour_volume_outside_liver_ml": round(outside * vox_ml, 3),
        "measured_at": "original resolution",
        "voxel_spacing_mm": [round(float(dz), 4), round(float(dr), 4), round(float(dc), 4)],
    }
    maps = {"bbox": (R0, R1, C0, C1), "liver": liver_o, "tumour": tumour_o, "lesion_label": lesion_lab}
    return {"measurements": meas, "maps": maps}


def reference_metrics(gt_view: np.ndarray, spacing: tuple, seg: SegOutput, maps: dict | None,
                      target_min_tumour_ml: float = 0.0, ct_view: np.ndarray | None = None) -> dict:
    """Research-only comparison with the dataset annotation (labels 0/1/2)."""
    dz, dr, dc = spacing
    vox_ml = dz * dr * dc / 1000.0
    gt = np.round(gt_view).astype(np.uint8)
    gt_liver, gt_tum = gt > 0, gt == 2
    gt_tumour_ml = float(gt_tum.sum()) * vox_ml
    gt_lab, n_gt = ndi.label(gt_tum)
    out = {"tumour_present": bool(gt_tumour_ml > max(0.0, target_min_tumour_ml) and gt_tum.any()),
           "tumour_volume_ml": round(gt_tumour_ml, 3), "liver_volume_ml": round(float(gt_liver.sum()) * vox_ml, 1),
           "n_lesions": int(n_gt)}
    if n_gt:
        sizes = ndi.sum(gt_tum, gt_lab, range(1, n_gt + 1)) * vox_ml
        out["largest_lesion_ml"] = round(float(sizes.max()), 3)
        if ct_view is not None and (gt_liver & ~gt_tum).any():
            # conspicuity: mean attenuation difference tumour vs surrounding liver (used by the reader model)
            out["tumour_contrast_hu"] = round(float(ct_view[gt_tum].mean() - ct_view[gt_liver & ~gt_tum].mean()), 1)
    if maps is None:
        out.update({"dice_liver": 0.0 if gt_liver.any() else 1.0, "dice_tumour": 0.0 if gt_tum.any() else 1.0,
                    "lesions_detected": 0, "fp_lesions": 0})
        return out
    R0, R1, C0, C1 = maps["bbox"]
    sl = (slice(seg.z0, seg.z1), slice(R0, R1), slice(C0, C1))
    p_l, p_t = maps["liver"], maps["tumour"]

    def dice(pred_crop, gt_full):
        inter = float((pred_crop & gt_full[sl]).sum())
        denom = float(pred_crop.sum() + gt_full.sum())
        return 1.0 if denom == 0 else 2.0 * inter / denom

    out["dice_liver"] = round(dice(p_l, gt_liver), 4)
    out["dice_tumour"] = round(dice(p_t, gt_tum), 4)
    out["tumour_overlap_voxels"] = int((p_t & gt_tum[sl]).sum())
    detected = 0
    if n_gt:
        hit = np.unique(gt_lab[sl][p_t])
        detected = int((hit > 0).sum())
    pl = maps["lesion_label"]
    fp = 0
    for j in range(1, int(pl.max()) + 1 if pl.size else 1):
        comp = pl == j
        if comp.any() and not (gt_tum[sl][comp]).any():
            fp += 1
    out["lesions_detected"] = detected
    out["fp_lesions"] = fp
    return out


def viewer_arrays(view_hu: np.ndarray, seg: SegOutput, maps: dict | None, cfg: dict,
                  gt_view: np.ndarray | None = None) -> dict:
    """Compact arrays for the review UI (liver slab, viewer resolution, uint8)."""
    Sv = int(cfg["inference"]["viewer_size"])
    Z, H, W = view_hu.shape
    lo, hi = cfg["preprocess"]["hu_window"]
    if not seg.found_liver:
        zs = np.linspace(0, Z - 1, min(Z, cfg["inference"]["viewer_max_slices"])).astype(int)
    else:
        zs = np.arange(seg.z0, seg.z1)
        maxn = int(cfg["inference"]["viewer_max_slices"])
        if len(zs) > maxn:
            keep = set(np.linspace(seg.z0, seg.z1 - 1, maxn).astype(int).tolist())
            if maps is not None and maps["tumour"].any():
                keep.add(int(seg.z0 + np.argmax(maps["tumour"].sum(axis=(1, 2)))))
            zs = np.array(sorted(keep))
    import cv2

    def rs(img, nearest=False):
        return cv2.resize(np.ascontiguousarray(img), (Sv, Sv),
                          interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    ct = np.stack([rs(window_u8(view_hu[z], lo, hi)) for z in zs])
    liver = np.zeros((len(zs), Sv, Sv), dtype=np.uint8)
    lesions = np.zeros_like(liver)
    tprob = np.zeros_like(liver)
    entropy = np.zeros_like(liver)
    epi = np.zeros_like(liver)
    if seg.found_liver:
        R0, R1, C0, C1 = maps["bbox"] if maps else (0, H, 0, W)
        pstd = seg.pass_tumour.astype(np.float32).std(axis=0) if seg.pass_tumour is not None else None
        for i, z in enumerate(zs):
            j = z - seg.z0
            full = np.zeros((H, W), dtype=np.uint8)
            if maps is not None:
                full[R0:R1, C0:C1] = maps["liver"][j]
                liver[i] = rs(full, nearest=True)
                full[:] = 0
                full[R0:R1, C0:C1] = maps["lesion_label"][j]
                lesions[i] = rs(full, nearest=True)
            tprob[i] = rs(np.clip(seg.mean[j, 2] * 255, 0, 255).astype(np.uint8))
            entropy[i] = rs(np.clip(seg.entropy[j] * 255, 0, 255).astype(np.uint8))
            if pstd is not None:
                epi[i] = rs(np.clip(pstd[j] / 0.5 * 255, 0, 255).astype(np.uint8))
    out = {"z": zs.astype(np.int32), "ct": ct, "liver": liver, "lesions": lesions, "tumour_prob": tprob,
           "entropy": entropy, "epistemic": epi}
    if gt_view is not None:
        gt = np.round(gt_view).astype(np.uint8)
        out["reference"] = np.stack([rs(gt[z], nearest=True) for z in zs])
    return out
