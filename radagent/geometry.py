"""3D medical-image geometry.

Conventions used everywhere in the project
-------------------------------------------
* World space is NIfTI RAS+ in millimetres: +x points to the patient's Right,
  +y Anterior, +z Superior.
* Files are first re-ordered to the *closest canonical* (RAS+) voxel order - the same
  operation as ``nibabel.as_closest_canonical`` - so every case has the same axis
  semantics regardless of how the scanner/exporter stored it (LiTS mixes these).
* For models and display we then use a **radiological axial view**
  ``V[z, row, col]``: ``z`` increases superiorly, ``row 0`` is the most anterior
  row and ``col 0`` is the patient's right (shown on the viewer's left).
* Voxel spacing is always carried alongside the array as ``(dz, drow, dcol)`` mm,
  taken from the image header (never assumed), and volumes are
  ``n_voxels * dz * drow * dcol / 1000`` millilitres.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXIS_CODES = (("L", "R"), ("P", "A"), ("I", "S"))


def ornt_from_affine(affine: np.ndarray) -> list[tuple[int, int]]:
    """For each voxel axis j return (world_axis, sign).

    The world axis is the one the voxel axis is most aligned with; ``sign`` is +1 if
    increasing the voxel index moves towards R/A/S. Ties/obliqueness are resolved
    greedily in order of alignment strength (as in nibabel.io_orientation).
    """
    RZS = np.asarray(affine, dtype=np.float64)[:3, :3]
    zooms = np.sqrt((RZS ** 2).sum(axis=0))
    zooms[zooms == 0] = 1.0
    R = np.abs(RZS / zooms)
    out: list[tuple[int, int] | None] = [None, None, None]
    used_w, used_j = set(), set()
    for _ in range(3):
        best, bw, bj = -1.0, -1, -1
        for w in range(3):
            if w in used_w:
                continue
            for j in range(3):
                if j in used_j:
                    continue
                if R[w, j] > best:
                    best, bw, bj = R[w, j], w, j
        used_w.add(bw)
        used_j.add(bj)
        out[bj] = (bw, 1 if RZS[bw, bj] >= 0 else -1)
    return out  # type: ignore[return-value]


def axcodes(affine: np.ndarray) -> tuple[str, str, str]:
    return tuple(AXIS_CODES[w][1 if s > 0 else 0] for w, s in ornt_from_affine(affine))  # type: ignore


def to_canonical(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Re-order/flip voxel axes so that the array is RAS+ (closest canonical)."""
    ornt = ornt_from_affine(affine)
    perm = [None, None, None]
    for j, (w, _s) in enumerate(ornt):
        perm[w] = j
    arr = np.transpose(data, perm)
    new_aff = np.eye(4)
    new_aff[:3, :3] = np.asarray(affine)[:3, perm]
    new_aff[:3, 3] = np.asarray(affine)[:3, 3]
    for w in range(3):
        j = perm[w]
        if ornt[j][1] < 0:
            n = arr.shape[w]
            arr = np.flip(arr, axis=w)
            new_aff[:3, 3] = new_aff[:3, 3] + (n - 1) * new_aff[:3, w]
            new_aff[:3, w] = -new_aff[:3, w]
    return np.ascontiguousarray(arr), new_aff


def zooms_from_affine(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))


def canonical_to_view(arr_ras: np.ndarray) -> np.ndarray:
    """RAS+ array [x, y, z] -> radiological axial view V[z, row, col].

    row 0 = most anterior (max y), col 0 = patient's right (max x).
    """
    return np.ascontiguousarray(np.transpose(arr_ras[::-1, ::-1, :], (2, 1, 0)))


def view_to_canonical(view: np.ndarray) -> np.ndarray:
    """Inverse of :func:`canonical_to_view`."""
    return np.ascontiguousarray(np.transpose(view, (2, 1, 0))[::-1, ::-1, :])


@dataclass
class Volume:
    """A CT (or mask) in the radiological view with its physical metadata."""

    view: np.ndarray                      # V[z, row, col]
    spacing: tuple[float, float, float]   # (dz, drow, dcol) in mm
    affine_canonical: np.ndarray           # RAS+ affine of the canonical array
    orientation_on_disk: tuple[str, str, str]
    shape_on_disk: tuple[int, ...]
    meta: dict = field(default_factory=dict)

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))

    def volume_ml(self, mask: np.ndarray) -> float:
        return float(np.count_nonzero(mask)) * self.voxel_volume_mm3 / 1000.0


def load_volume(path, use_nibabel: bool | None = None) -> Volume:
    from radagent.io.nifti import load_nifti

    data, affine = load_nifti(path, use_nibabel=use_nibabel)
    codes = axcodes(affine)
    ras, aff_c = to_canonical(data, affine)
    zx, zy, zz = zooms_from_affine(aff_c)
    view = canonical_to_view(ras)
    return Volume(view=view, spacing=(float(zz), float(zy), float(zx)), affine_canonical=aff_c,
                  orientation_on_disk=codes, shape_on_disk=tuple(int(s) for s in data.shape))


def physical_extent_mm(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(float(n * s) for n, s in zip(shape, spacing))  # type: ignore[return-value]
