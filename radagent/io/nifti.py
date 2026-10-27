"""NIfTI-1 reading/writing.

``nibabel`` is used when it is installed (it is in ``requirements.txt``). A small,
dependency-free NIfTI-1 reader/writer is kept as a fallback so that the geometry
code and the test fixtures work in minimal environments. LiTS stores some volumes as
float32/float64; ``load_nifti`` reads those chunk-wise and keeps integral HU as int16.
Both paths return the same thing: the raw voxel array (as stored on disk) and the
4x4 voxel-to-world affine (RAS+ world, millimetres).
"""
from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

try:  # pragma: no cover - exercised implicitly
    import nibabel as nib  # type: ignore

    HAVE_NIBABEL = True
except Exception:  # pragma: no cover
    nib = None
    HAVE_NIBABEL = False

_DTYPES = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64,
           256: np.int8, 512: np.uint16, 768: np.uint32}
_CODES = {np.dtype(v): k for k, v in _DTYPES.items()}


def _to_compact(blocks, shape) -> np.ndarray:
    """Assemble float blocks (along the last axis) into int16 when the values are integral HU/labels.

    LiTS stores some CTs and masks as float32/float64. Converting block by block keeps the
    peak memory at one block instead of a full float64 copy of a 512 x 512 x 900 volume.
    """
    out16 = np.empty(shape, dtype=np.int16)
    out32 = None
    z = 0
    for blk in blocks:
        n = blk.shape[-1]
        if out32 is None:
            fin = np.isfinite(blk)
            r = np.round(np.where(fin, blk, 0))
            if fin.all() and np.abs(r).max(initial=0) < 32767 and np.allclose(r, blk, atol=1e-3):
                out16[..., z:z + n] = r.astype(np.int16)
            else:  # not integral -> keep float32 from here on
                out32 = out16.astype(np.float32)
                del out16
                out32[..., z:z + n] = blk.astype(np.float32)
        else:
            out32[..., z:z + n] = blk.astype(np.float32)
        z += n
    return out16 if out32 is None else out32


def load_nifti(path: str | Path, use_nibabel: bool | None = None, chunk: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(data, affine)``. Integral data (HU, labels) comes back as int16 to save memory."""
    path = str(path)
    if use_nibabel is None:
        use_nibabel = HAVE_NIBABEL
    if use_nibabel:
        img = nib.load(path)
        affine = np.asarray(img.affine, dtype=np.float64)
        shape = tuple(int(x) for x in img.shape)
        if len(shape) == 4 and shape[3] == 1:
            shape = shape[:3]
        proxy = img.dataobj
        disk = np.dtype(img.get_data_dtype())
        slope, inter = getattr(proxy, "slope", 1.0), getattr(proxy, "inter", 0.0)
        trivial_scale = (slope in (1.0, None) or np.isnan(slope)) and (inter in (0.0, None) or np.isnan(inter))
        if disk.kind in "iub" and trivial_scale:
            data = np.asanyarray(proxy)
            if data.ndim == 4 and data.shape[3] == 1:
                data = data[..., 0]
            if data.dtype.itemsize > 2 or data.dtype == np.uint16:
                if data.size and np.abs(data).max() < 32767:
                    data = data.astype(np.int16)
            return data, affine

        def blocks():
            for z0 in range(0, shape[2], chunk):
                blk = np.asarray(proxy[:, :, z0:min(shape[2], z0 + chunk)] if len(img.shape) == 3
                                 else proxy[:, :, z0:min(shape[2], z0 + chunk), 0])
                yield blk

        return _to_compact(blocks(), shape), affine
    data, affine = _read_nifti1(path)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.dtype.kind == "f" and data.size:
        data = _to_compact((data[..., z0:z0 + chunk] for z0 in range(0, data.shape[2], chunk)), data.shape)
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path, use_nibabel: bool | None = None) -> None:
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if use_nibabel is None:
        use_nibabel = HAVE_NIBABEL
    if use_nibabel:
        img = nib.Nifti1Image(data, affine)
        img.header.set_xyzt_units("mm", "sec")
        img.set_qform(affine, code=1)
        img.set_sform(affine, code=1)
        nib.save(img, path)
    else:
        _write_nifti1(data, affine, path)


# --------------------------------------------------------------------------- fallback
def _open(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode, compresslevel=1) if "w" in mode else gzip.open(path, mode)
    return open(path, mode)


def _quat_to_affine(b, c, d, qx, qy, qz, pixdim) -> np.ndarray:
    a = 1.0 - (b * b + c * c + d * d)
    a = np.sqrt(a) if a > 1e-7 else 0.0
    if a == 0.0:  # normalise (b, c, d) as in the NIfTI reference implementation
        n = np.sqrt(b * b + c * c + d * d)
        b, c, d = b / n, c / n, d / n
    R = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
    ])
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    zooms = np.array([pixdim[1], pixdim[2], pixdim[3] * qfac])
    aff = np.eye(4)
    aff[:3, :3] = R * zooms[None, :]
    aff[:3, 3] = [qx, qy, qz]
    return aff


def _read_nifti1(path: str) -> tuple[np.ndarray, np.ndarray]:
    with _open(path, "rb") as f:
        raw = f.read()
    endian = "<" if struct.unpack("<i", raw[:4])[0] == 348 else ">"
    if struct.unpack(endian + "i", raw[:4])[0] != 348:
        raise ValueError(f"{path}: not a NIfTI-1 file")
    dim = struct.unpack(endian + "8h", raw[40:56])
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    pixdim = struct.unpack(endian + "8f", raw[76:108])
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    scl_slope, scl_inter = struct.unpack(endian + "2f", raw[112:120])
    qform_code, sform_code = struct.unpack(endian + "2h", raw[252:256])
    qb, qc, qd, qx, qy, qz = struct.unpack(endian + "6f", raw[256:280])
    srow = np.array(struct.unpack(endian + "12f", raw[280:328]), dtype=np.float64).reshape(3, 4)
    if datatype not in _DTYPES:
        raise ValueError(f"{path}: unsupported NIfTI datatype {datatype}")
    ndim = dim[0]
    shape = tuple(int(x) for x in dim[1:1 + ndim])
    dtype = np.dtype(_DTYPES[datatype]).newbyteorder(endian)
    n = int(np.prod(shape))
    data = np.frombuffer(raw, dtype=dtype, count=n, offset=vox_offset).reshape(shape, order="F")
    data = data.astype(dtype.newbyteorder("="), copy=True)
    if scl_slope not in (0.0, 1.0) or scl_inter != 0.0:
        slope = scl_slope if scl_slope != 0.0 else 1.0
        data = data.astype(np.float32) * slope + scl_inter
    if sform_code > 0:
        affine = np.eye(4)
        affine[:3, :] = srow
    elif qform_code > 0:
        affine = _quat_to_affine(qb, qc, qd, qx, qy, qz, pixdim)
    else:
        affine = np.diag([pixdim[1], pixdim[2], pixdim[3], 1.0])
    return data, affine


def _affine_to_quat(affine: np.ndarray) -> tuple[float, float, float, float, float, float, float, np.ndarray]:
    """Return (qfac, b, c, d, qx, qy, qz, zooms) for a rigid+scaling affine."""
    M = affine[:3, :3]
    zooms = np.sqrt((M ** 2).sum(axis=0))
    R = M / zooms[None, :]
    qfac = 1.0
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
        qfac = -1.0
    # matrix -> quaternion (a >= 0)
    a = 0.5 * np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2]))
    if a > 1e-6:
        b = 0.25 * (R[2, 1] - R[1, 2]) / a
        c = 0.25 * (R[0, 2] - R[2, 0]) / a
        d = 0.25 * (R[1, 0] - R[0, 1]) / a
    else:
        xd = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        yd = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2]))
        zd = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2]))
        if xd >= yd and xd >= zd:
            b = 0.5 * xd
            c = 0.25 * (R[0, 1] + R[1, 0]) / b
            d = 0.25 * (R[0, 2] + R[2, 0]) / b
        elif yd >= zd:
            c = 0.5 * yd
            b = 0.25 * (R[0, 1] + R[1, 0]) / c
            d = 0.25 * (R[1, 2] + R[2, 1]) / c
        else:
            d = 0.5 * zd
            b = 0.25 * (R[0, 2] + R[2, 0]) / d
            c = 0.25 * (R[1, 2] + R[2, 1]) / d
        a = 0.0
        if b < 0:
            b, c, d = -b, -c, -d
    return qfac, float(b), float(c), float(d), *[float(x) for x in affine[:3, 3]], zooms


def _write_nifti1(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    data = np.asarray(data)
    if data.dtype == np.bool_:
        data = data.astype(np.uint8)
    if data.dtype not in _CODES:
        data = data.astype(np.float32)
    code = _CODES[data.dtype]
    qfac, b, c, d, qx, qy, qz, zooms = _affine_to_quat(np.asarray(affine, dtype=np.float64))
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    hdr[38] = ord("r")
    dims = [data.ndim] + list(data.shape) + [1] * (7 - data.ndim)
    struct.pack_into("<8h", hdr, 40, *dims)
    struct.pack_into("<h", hdr, 70, code)
    struct.pack_into("<h", hdr, 72, data.dtype.itemsize * 8)
    pixdim = [qfac, *zooms.tolist()] + [1.0] * (7 - 3)
    struct.pack_into("<8f", hdr, 76, *pixdim)
    struct.pack_into("<f", hdr, 108, 352.0)            # vox_offset
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)        # scl_slope, scl_inter
    hdr[123] = 2 | 8                                    # mm, sec
    descrip = b"ReaderAdapt-Liver"
    hdr[148:148 + len(descrip)] = descrip
    struct.pack_into("<2h", hdr, 252, 1, 1)             # qform_code, sform_code
    struct.pack_into("<6f", hdr, 256, b, c, d, qx, qy, qz)
    struct.pack_into("<12f", hdr, 280, *np.asarray(affine, dtype=np.float64)[:3, :].ravel().tolist())
    hdr[344:348] = b"n+1\x00"
    # bytes 348..351: extension flag = 0
    with _open(path, "wb") as f:
        f.write(bytes(hdr))
        f.write(data.tobytes(order="F"))
