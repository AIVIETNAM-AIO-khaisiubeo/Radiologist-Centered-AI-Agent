"""Test fixture: small CT phantoms written exactly like the Kaggle LiTS mirror.

Only for automated tests - the project itself runs on LiTS17. The phantoms let the whole
pipeline (manifest -> segmenter -> MedGemma tool -> calibration -> agent -> website) be
exercised in seconds on a CPU, with stand-in plugins from :mod:`tests.fixtures.stubs`.

Layout written (same as the two Kaggle datasets)::

    <root>/part1/segmentations/segmentation-<n>.nii
    <root>/part1/volume_pt1/volume-<n>.nii        (first half of the cases)
    <root>/part2/volume_pt6/volume-<n>.nii        (second half)

CTs are stored in LPS voxel order like LiTS; every third one as float32 (LiTS has such files).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from radagent.io.nifti import save_nifti


def _smooth_field(rng: np.random.Generator, shape: tuple[int, int, int], grid: int) -> np.ndarray:
    """Zero-mean, unit-std low-frequency random field."""
    coarse = rng.standard_normal((grid, grid, grid)).astype(np.float32)
    zoom = [s / grid for s in shape]
    f = ndi.zoom(coarse, zoom, order=3, mode="reflect")[: shape[0], : shape[1], : shape[2]]
    pad = [(0, max(0, s - f.shape[i])) for i, s in enumerate(shape)]
    f = np.pad(f, pad, mode="edge")
    f -= f.mean()
    f /= f.std() + 1e-6
    return f


def _ellipsoid(z, r, c, center, radii):
    return ((z - center[0]) / radii[0]) ** 2 + ((r - center[1]) / radii[1]) ** 2 + ((c - center[2]) / radii[2]) ** 2


def _stamp_tube(vol_mask: np.ndarray, pts_mm: np.ndarray, radii_mm: np.ndarray,
                spacing: tuple[float, float, float]) -> None:
    """Rasterise a tube (sequence of spheres) given points in voxel-mm space."""
    Z, H, W = vol_mask.shape
    sz, sr, sc = spacing
    for (pz, pr, pc), rad in zip(pts_mm, radii_mm):
        kz, kr, kc = int(np.ceil(rad / sz)) + 1, int(np.ceil(rad / sr)) + 1, int(np.ceil(rad / sc)) + 1
        iz, ir, ic = int(round(pz / sz)), int(round(pr / sr)), int(round(pc / sc))
        z0, z1 = max(0, iz - kz), min(Z, iz + kz + 1)
        r0, r1 = max(0, ir - kr), min(H, ir + kr + 1)
        c0, c1 = max(0, ic - kc), min(W, ic + kc + 1)
        if z0 >= z1 or r0 >= r1 or c0 >= c1:
            continue
        gz = (np.arange(z0, z1) * sz - pz)[:, None, None]
        gr = (np.arange(r0, r1) * sr - pr)[None, :, None]
        gc = (np.arange(c0, c1) * sc - pc)[None, None, :]
        vol_mask[z0:z1, r0:r1, c0:c1] |= (gz ** 2 + gr ** 2 + gc ** 2) <= rad ** 2


def make_phantom(rng: np.random.Generator, size: int = 144, tumour_prob: float = 0.55) -> dict:
    """Return dict(ct, label, spacing, meta) in the radiological view V[z, row, col]."""
    ps = float(rng.uniform(2.3, 2.7))                 # in-plane pixel spacing (mm)
    st = float(rng.choice([3.0, 4.0, 5.0]))            # slice thickness (mm)
    Z = int(round(rng.uniform(190, 230) / st))
    H = W = size
    spacing = (st, ps, ps)

    # physical coordinates (mm), origin at the volume centre
    z = ((np.arange(Z) - Z / 2) * st)[:, None, None].astype(np.float32)
    r = ((np.arange(H) - H / 2) * ps)[None, :, None].astype(np.float32)   # - anterior, + posterior
    c = ((np.arange(W) - W / 2) * ps)[None, None, :].astype(np.float32)   # - patient right, + left
    shape = (Z, H, W)
    full = np.zeros(shape, dtype=bool)

    hu = np.full(shape, -1000.0, dtype=np.float32)
    label = np.zeros(shape, dtype=np.uint8)

    # ---- body
    a_c, a_r = rng.uniform(140, 170), rng.uniform(100, 125)
    zscale = 1.0 + 0.04 * np.sin(z / 60.0)
    body_q = (c / (a_c * zscale)) ** 2 + (r / (a_r * zscale)) ** 2
    body = (body_q <= 1.0) | full
    fat_t, mus_t = rng.uniform(10, 25), 8.0
    inner_fat = body_q <= (1 - fat_t / a_r) ** 2
    inner_mus = body_q <= (1 - (fat_t + mus_t) / a_r) ** 2
    hu[body] = -100
    hu[inner_fat] = 50
    hu[inner_mus] = -70                                 # visceral fat background
    # diaphragm dome: thorax above, abdomen below
    z_diaph = rng.uniform(40, 70)
    dome = z_diaph - 25 * ((c + 55) / 90.0) ** 2 - 12 * ((r - 10) / 80.0) ** 2
    thorax = (z > dome) & inner_mus
    heart = thorax & (_ellipsoid(z, r, c, (z_diaph + 45, -25, 25), (55, 50, 60)) <= 1)
    lungs = thorax & ~heart
    hu[lungs] = -850
    hu[heart] = 42

    # ---- spine, aorta, IVC
    spine_r = a_r * 0.55
    spine = ((r - spine_r) ** 2 + c ** 2 <= 18 ** 2) | full
    cortex = spine & ((r - spine_r) ** 2 + c ** 2 >= 15 ** 2)
    hu[spine] = 350
    hu[cortex] = 800
    canal = ((r - spine_r - 26) ** 2 + c ** 2 <= 7 ** 2) | full
    hu[canal] = 20
    aorta = ((r - a_r * 0.32) ** 2 + (c - 15) ** 2 <= 11 ** 2) | full
    ivc = ((r - a_r * 0.25) ** 2 + (c + 22) ** 2 <= 11 ** 2) & (z < dome + 10)
    enh = rng.uniform(140, 200)
    hu[aorta] = enh
    hu[ivc] = enh - 40
    blocked = spine | canal | aorta | ivc

    # ---- liver: deformed right + left lobe below the dome
    zc = z_diaph - rng.uniform(65, 80)
    f_liver = _smooth_field(rng, shape, 6)
    right = _ellipsoid(z, r, c, (zc, -5 + rng.uniform(-8, 8), -55 + rng.uniform(-8, 8)),
                       (rng.uniform(62, 76), rng.uniform(78, 92), rng.uniform(68, 82)))
    left = _ellipsoid(z, r, c, (zc + 18, -45, 22 + rng.uniform(-8, 8)),
                      (rng.uniform(32, 44), rng.uniform(30, 40), rng.uniform(45, 62)))
    liver_q = np.minimum(right, left) + 0.18 * f_liver
    liver = (liver_q <= 1.0) & (z < dome - 3) & inner_mus & ~blocked
    liver_hu = float(rng.uniform(50, 72))
    phase_noise = float(rng.uniform(-5, 5))

    # ---- other organs
    spleen = (_ellipsoid(z, r, c, (zc + 8, 40, 95), (50, 55, 33)) + 0.2 * f_liver <= 1) & inner_mus & ~liver & ~blocked
    stomach = (_ellipsoid(z, r, c, (zc + 25, -30, 58), (45, 40, 44)) <= 1) & inner_mus & ~liver & ~blocked & (z < dome)
    gas = stomach & (r < -30 - 18)
    kid_l = _ellipsoid(z, r, c, (zc - 62, 55, 72), (55, 25, 30))
    kid_r = _ellipsoid(z, r, c, (zc - 70, 55, -72), (55, 25, 30))
    kidney = ((kid_l <= 1) | (kid_r <= 1)) & inner_mus & ~liver & ~blocked
    medulla = ((kid_l <= 0.35) | (kid_r <= 0.35)) & kidney
    hu[spleen] = liver_hu + rng.uniform(-8, 4)
    hu[stomach] = 15
    hu[gas] = -900
    hu[kidney] = enh - 20
    hu[medulla] = enh - 70
    # bowel loops in the lower anterior abdomen
    lower = (z < zc - 30) & inner_mus & ~kidney & ~blocked
    for _ in range(int(rng.integers(8, 14))):
        bz, br, bc = rng.uniform(z.min() + 10, zc - 30), rng.uniform(-a_r * 0.7, 30), rng.uniform(-a_c * 0.6, a_c * 0.6)
        rad = rng.uniform(14, 26)
        loop = (_ellipsoid(z, r, c, (bz, br, bc), (rad * 1.4, rad, rad)) <= 1) & lower
        hu[loop] = rng.uniform(15, 45)
        air = (_ellipsoid(z, r, c, (bz, br - rad * 0.5, bc), (rad * 0.5, rad * 0.35, rad * 0.5)) <= 1) & loop
        hu[air] = -700
    gallbladder = np.zeros(shape, dtype=bool)
    if rng.random() < 0.7:
        gb = _ellipsoid(z, r, c, (zc - 38, -45, -28), (28, 20, 15))
        gallbladder = (gb <= 1) & inner_mus & ~blocked
        hu[gallbladder] = rng.uniform(0, 15)
        liver &= ~gallbladder

    hu[liver] = liver_hu
    label[liver] = 1

    # ---- vessel tree inside the liver
    vessels = np.zeros(shape, dtype=bool)
    lz, lr, lc = np.nonzero(liver)
    offs = np.array([Z / 2 * st, H / 2 * ps, W / 2 * ps])
    hilum = np.array([zc, 12.0, -30.0]) + offs
    for _ in range(int(rng.integers(5, 9))):
        k = int(rng.integers(len(lz)))
        target = np.array([lz[k] * st, lr[k] * ps, lc[k] * ps])
        ctrl = (hilum + target) / 2 + rng.normal(0, 15, 3)
        t = np.linspace(0, 1, 50)[:, None]
        pts = (1 - t) ** 2 * hilum + 2 * (1 - t) * t * ctrl + t ** 2 * target
        _stamp_tube(vessels, pts, np.linspace(rng.uniform(3.5, 5.5), 1.5, len(pts)), spacing)
    vessels &= liver
    hu[vessels] = enh - rng.uniform(10, 50)

    # ---- tumours and cysts
    edt = ndi.distance_transform_edt(liver, sampling=spacing)
    f_t = _smooth_field(rng, shape, 18)
    tumours, cysts = [], []
    is_pos = rng.random() < tumour_prob
    n_t = int(min(5, 1 + rng.geometric(0.5) - 1)) if is_pos else 0
    for _ in range(n_t):
        rad = float(np.exp(rng.uniform(np.log(2.5), np.log(22))))
        cand = np.argwhere(edt >= 0.6 * rad)
        if len(cand) == 0:
            cand = np.argwhere(edt >= edt.max() * 0.5)
        iz, ir, ic = cand[int(rng.integers(len(cand)))]
        center = (z[iz, 0, 0], r[0, ir, 0], c[0, 0, ic])
        radii = tuple(rad * rng.uniform(0.75, 1.25, 3))
        q = _ellipsoid(z, r, c, center, radii) + 0.25 * f_t
        tum = (q <= 1.0) & liver
        if not tum.any():
            continue
        kind = rng.choice(["hypo", "hypo", "hypo", "hyper"], p=[0.3, 0.3, 0.3, 0.1])
        delta = -rng.uniform(15, 45) if kind == "hypo" else rng.uniform(25, 50)
        hu[tum] = liver_hu + delta
        rim = bool(rng.random() < 0.3)
        if rim:
            hu[tum & (q > 0.7)] = liver_hu + delta + 25
        necrotic = bool(rad > 12 and rng.random() < 0.4)
        if necrotic:
            hu[tum & (q < 0.3)] = liver_hu + delta - 20
        label[tum] = 2
        tumours.append({"radius_mm": round(rad, 2), "contrast_hu": round(float(delta), 1), "kind": str(kind),
                        "rim": rim, "necrotic": necrotic, "voxels": int(tum.sum()),
                        "volume_ml": round(float(tum.sum() * st * ps * ps / 1000.0), 3)})
    if rng.random() < 0.35:
        for _ in range(int(rng.integers(1, 4))):
            rad = float(rng.uniform(3, 9))
            cand = np.argwhere((edt >= rad) & (label == 1))
            if len(cand) == 0:
                continue
            iz, ir, ic = cand[int(rng.integers(len(cand)))]
            cy = (_ellipsoid(z, r, c, (z[iz, 0, 0], r[0, ir, 0], c[0, 0, ic]), (rad, rad, rad)) <= 1) & (label == 1)
            hu[cy] = rng.uniform(0, 12)
            cysts.append({"radius_mm": round(rad, 2)})

    # ---- texture, partial volume, noise
    texture = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), (0.5, 1.2, 1.2)) * 6
    hu[body] += texture[body] + phase_noise
    hu = ndi.gaussian_filter(hu, sigma=(0.35, 0.5, 0.5))
    noise_sd = float(rng.uniform(6, 15))
    noise = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), (0, 0.7, 0.7))
    noise *= noise_sd / (noise.std() + 1e-6)
    hu = hu + noise
    hu[~body] = -1000 + rng.normal(0, 4, size=int((~body).sum()))
    ct = np.clip(np.round(hu), -1024, 3071).astype(np.int16)

    meta = {"pixel_spacing_mm": round(ps, 3), "slice_thickness_mm": st, "shape_zyx": [Z, H, W],
            "liver_hu": round(liver_hu, 1), "noise_sd_hu": round(noise_sd, 1),
            "n_tumours": len(tumours), "tumours": tumours, "n_cysts": len(cysts),
            "gallbladder": bool(gallbladder.any()),
            "tumour_volume_ml": round(float((label == 2).sum() * st * ps * ps / 1000.0), 3),
            "liver_volume_ml": round(float((label > 0).sum() * st * ps * ps / 1000.0), 1)}
    return {"ct": ct, "label": label, "spacing": spacing, "meta": meta}




def view_to_lps_storage(view: np.ndarray) -> np.ndarray:
    """Radiological view V[z,row,col] -> array stored like LiTS (L, P, S axes)."""
    return np.ascontiguousarray(np.transpose(view, (2, 1, 0)))


def lps_affine(spacing, shape_xyz) -> np.ndarray:
    dz, dr, dc = spacing
    aff = np.diag([-dc, -dr, dz, 1.0])
    aff[:3, 3] = [dc * shape_xyz[0] / 2, dr * shape_xyz[1] / 2, -dz * shape_xyz[2] / 2]
    return aff


def write_lits_like(root, n_cases: int = 12, size: int = 144, seed: int = 7) -> Path:
    root = Path(root)
    for i in range(n_cases):
        ph = make_phantom(np.random.default_rng([seed, i]), size=size, tumour_prob=0.6)
        ct = view_to_lps_storage(ph["ct"])
        if i % 3 == 2:
            ct = ct.astype(np.float32)
        aff = lps_affine(ph["spacing"], ct.shape)
        part = "part1/volume_pt1" if i < n_cases // 2 else "part2/volume_pt6"
        save_nifti(ct, aff, root / part / f"volume-{i}.nii")
        save_nifti(view_to_lps_storage(ph["label"]), aff, root / "part1" / "segmentations" / f"segmentation-{i}.nii")
    return root
