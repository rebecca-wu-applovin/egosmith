"""AnyCalib-based focal estimation — a learned replacement for the W/2 focal guess.

AnyCalib (https://github.com/javrtg/AnyCalib, Apache-2.0) regresses camera intrinsics
from a single image. Egocentric cameras are fixed per clip, so we estimate the focal on
a handful of evenly-spaced frames and take the median (single-frame perspective focal is
noisy; the median is markedly tighter). Only the focal is used — it is written to
``est_focal.txt`` and flows through ``resolve_calibration`` to DPVO/Any4D/HaWoR unchanged.

Validated vs GT focal (median error): W/2 guess -> AnyCalib = taco 30.3%->9.0%,
hot3d 15.6%->3.6%.
"""
from __future__ import annotations

import numpy as np

from lib.pipeline.proc.logging_setup import vprint

_MODEL_CACHE: dict = {}


def _get_model(model_id: str, device: str):
    key = (model_id, device)
    if key not in _MODEL_CACHE:
        from anycalib import AnyCalib  # lazy: only import when actually calibrating
        import torch  # noqa: F401
        _MODEL_CACHE[key] = AnyCalib(model_id=model_id).to(device)
    return _MODEL_CACHE[key]


def estimate_focal(
    frame_source,
    *,
    num_frames: int = 5,
    model_id: str = "anycalib_gen",
    device: str = "cuda",
) -> float | None:
    """Median AnyCalib pinhole focal over `num_frames` evenly-spaced frames, or None on failure."""
    import torch

    n = len(frame_source)
    if n == 0:
        return None
    idxs = np.unique(np.linspace(0, n - 1, min(num_frames, n)).astype(int))
    try:
        model = _get_model(model_id, device)
    except Exception as e:  # noqa: BLE001 — never let calibration crash the stage
        vprint(f"anycalib: model load failed ({e}); falling back to default focal")
        return None

    focals = []
    for i in idxs:
        try:
            img = frame_source.get_frame(int(i), rgb=True)  # HWC uint8 RGB
            image = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1) / 255
            with torch.no_grad():
                out = model.predict(image, cam_id="pinhole")
            fx, fy = float(out["intrinsics"][0]), float(out["intrinsics"][1])
            f = 0.5 * (fx + fy)
            if np.isfinite(f) and f > 0:
                focals.append(f)
        except Exception as e:  # noqa: BLE001
            vprint(f"anycalib: frame {i} failed ({e})")
            continue
    if not focals:
        return None
    return float(np.median(focals))
