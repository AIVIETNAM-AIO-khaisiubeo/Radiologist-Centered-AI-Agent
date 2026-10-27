"""NVIDIA VISTA3D - the project's single pretrained segmentation model.

VISTA3D (Versatile Imaging SegmenTation and Annotation, NVIDIA / MONAI; He et al., CVPR 2025,
arXiv:2406.05285) is a 3D foundation model for CT that segments 127 structures automatically from a
class prompt. This project asks it for two classes:

* label ``1``  - liver
* label ``26`` - hepatic tumour

Weights: Hugging Face ``MONAI/VISTA3D-HF`` (``vista3d_pretrained_model/model.safetensors``, ~870 MB,
downloaded once and cached; no token needed). Code is Apache-2.0; the **weights are under NVIDIA's
non-commercial licence (NCLS v1)** - research use only, like this prototype.

Important for evaluation: VISTA3D's training data include MSD Task 03 (Liver), which is built from
the LiTS training volumes - the same 131 CTs as the Kaggle LiTS mirror used here. Results on LiTS are
therefore *not* a held-out evaluation of the segmenter; see README section "Model".

Preprocessing follows the published bundle: RAS orientation, 1.5 mm isotropic resampling,
intensity scaling of [-963.8, 1053.7] HU to [0, 1], sliding-window inference on 128^3 patches. The
network returns one sigmoid map per prompted class. Test-time augmentation (left-right flip) gives a
second pass, whose disagreement with the first is the epistemic-uncertainty signal used downstream.
"""
from __future__ import annotations

import math
import time

import numpy as np

from radagent.geometry import canonical_to_view, view_to_canonical
from radagent.models.base import VolumeSegmenter

HF_REPO = "MONAI/VISTA3D-HF"
HF_FILE = "vista3d_pretrained_model/model.safetensors"
LABEL_LIVER = 1
LABEL_TUMOUR = 26
A_MIN, A_MAX = -963.8247715525971, 1053.678477684517   # intensity range of the VISTA3D bundle (HU)
LICENSE = "code Apache-2.0; weights NVIDIA NCLS v1 (non-commercial research use)"


def weights_path(cfg: dict) -> str:
    """Local path of the VISTA3D weights (downloads them on first use)."""
    sc = cfg["segmenter"]
    if sc.get("weights"):
        return str(sc["weights"])
    from huggingface_hub import hf_hub_download

    return hf_hub_download(sc.get("hf_repo", HF_REPO), sc.get("hf_file", HF_FILE), revision=sc.get("hf_revision"))


def build_network():
    from monai.networks.nets import vista3d132

    return vista3d132(encoder_embed_dim=48, in_channels=1)


def load_network(path: str):
    """VISTA3D network with the published weights (the HF checkpoint prefixes every key with ``network.``)."""
    from safetensors.torch import load_file

    net = build_network()
    sd = {k[len("network."):] if k.startswith("network.") else k: v for k, v in load_file(path).items()}
    net.load_state_dict(sd, strict=True)
    return net


def _device(name: str):
    import torch

    if name and name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Vista3DSegmenter(VolumeSegmenter):
    name = "vista3d"
    version = "VISTA3D 1.0 (MONAI/VISTA3D-HF)"

    def __init__(self, cfg: dict, network=None):
        sc = cfg["segmenter"]
        self.cfg = cfg
        self.repo = sc.get("hf_repo", HF_REPO)
        self.resample_mm = float(sc.get("resample_mm", 1.5))
        self.roi = tuple(int(x) for x in sc.get("roi_size", [128, 128, 128]))
        self.overlap = float(sc.get("overlap", 0.25))
        self.sw_batch = int(sc.get("sw_batch_size", 1))
        self.tta = bool(cfg["inference"].get("tta_flip", True))
        self.device = _device(sc.get("device", "auto"))
        self.amp = bool(sc.get("amp", True)) and self.device.type == "cuda"
        self.net = (network if network is not None else load_network(weights_path(cfg))).to(self.device).eval()
        self.timings: dict = {}

    @property
    def n_passes(self) -> int:
        return 2 if self.tta else 1

    def describe(self) -> dict:
        return {"backend": self.name, "version": self.version, "passes": self.n_passes,
                "weights": f"{self.repo}/{HF_FILE}", "license": LICENSE,
                "label_prompts": {"liver": LABEL_LIVER, "hepatic tumour": LABEL_TUMOUR},
                "resample_mm": self.resample_mm}

    def _sliding_window(self, x):
        """x: [1, 1, X, Y, Z] scaled image -> sigmoid probabilities [1, 2, X, Y, Z] (liver, tumour)."""
        import torch
        from monai.inferers import SlidingWindowInfererAdapt

        cv = torch.tensor([[LABEL_LIVER], [LABEL_TUMOUR]], device=x.device)
        inferer = SlidingWindowInfererAdapt(roi_size=self.roi, sw_batch_size=self.sw_batch, overlap=self.overlap,
                                            with_coord=True, padding_mode="replicate")
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=self.amp):
            logits = inferer(x, self.net, transpose=True, point_coords=None, point_labels=None,
                             class_vector=cv, prompt_class=cv)
        return torch.sigmoid(logits.float())

    def predict_volume(self, view_hu: np.ndarray, spacing: tuple, size: int) -> np.ndarray:
        """CT view ``V[z, row, col]`` (HU) -> class probabilities ``(P, Z, 3, size, size)`` float16.

        Classes: 0 background, 1 liver (without tumour), 2 tumour - the project's convention.
        """
        import torch
        import torch.nn.functional as F

        t0 = time.time()
        dz, drow, dcol = (float(s) for s in spacing)
        ras = view_to_canonical(np.asarray(view_hu, dtype=np.float32))            # [X, Y, Z], RAS+
        X, Y, Z = ras.shape
        grid = [max(8, int(round(n * s / self.resample_mm))) for n, s in zip((X, Y, Z), (dcol, drow, dz))]
        x = torch.from_numpy(np.ascontiguousarray(ras))[None, None].to(self.device)
        x = F.interpolate(x, size=grid, mode="trilinear", align_corners=False)
        x = ((x - A_MIN) / (A_MAX - A_MIN)).clamp_(0.0, 1.0)
        # crop to the body (> -500 HU) + 10 voxels, as the bundle crops the foreground
        body = x[0, 0] > ((-500.0 - A_MIN) / (A_MAX - A_MIN))
        box = []
        for ax in range(3):
            idx = torch.nonzero(body.any(dim=tuple(a for a in range(3) if a != ax))).flatten()
            lo, hi = (int(idx.min()), int(idx.max()) + 1) if idx.numel() else (0, grid[ax])
            box.append(slice(max(0, lo - 10), min(grid[ax], hi + 10)))
        xc = x[(..., *box)]
        del x
        out = np.zeros((self.n_passes, Z, 3, size, size), dtype=np.float16)
        for k, flip in enumerate([False, True][: self.n_passes]):
            inp = torch.flip(xc, dims=[2]) if flip else xc                          # left-right flip
            p = self._sliding_window(inp)
            if flip:
                p = torch.flip(p, dims=[2])
            full = torch.zeros((1, 2, *grid), device=p.device)
            full[(..., *box)] = p
            del p
            r = F.interpolate(full, size=(size, size, Z), mode="trilinear", align_corners=False)[0].cpu().numpy()
            del full
            liver_any = np.maximum(r[0], r[1])                    # VISTA3D's liver label excludes the tumour
            tumour = np.clip(r[1], 0.0, 1.0)
            cls = np.stack([1.0 - liver_any, liver_any - tumour, tumour]).clip(0.0, 1.0)   # [3, S, S, Z]
            out[k] = np.stack([canonical_to_view(c) for c in cls], axis=1).astype(np.float16)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.timings = {"vista3d_s": round(time.time() - t0, 2), "grid_1p5mm": grid,
                        "windows": int(np.prod([math.ceil(max(1, b.stop - b.start - 1) / (r * (1 - self.overlap)))
                                                for b, r in zip(box, self.roi)]))}
        return out
