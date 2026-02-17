"""
Depth Anything 3 backend.

Repo:    https://github.com/ByteDance-Seed/Depth-Anything-3
Install: git clone ... && pip install -e .

Models:
  DA3NESTED-GIANT-LARGE  – multi-view, outputs meters directly
  DA3METRIC-LARGE        – monocular metric (needs focal*depth/300 conversion)
  DA3-LARGE / DA3-BASE   – relative depth (unitless)
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from base_runner import BaseDepthRunner

logger = logging.getLogger("depthify")

DEFAULT_MODEL_ID = "depth-anything/DA3METRIC-LARGE"


class DA3Runner(BaseDepthRunner):
    backend_name = "da3"

    def __init__(self, device="cuda", model_id=DEFAULT_MODEL_ID,
                 focal_length: Optional[float] = None, **kw):
        super().__init__(device=device, **kw)
        self.model_id = model_id
        self.focal_length = focal_length
        self._model = None
        self._tmpdir: Optional[str] = None
        # Tracks whether metric conversion actually happened during inference.
        # None = not run yet, True = converted, False = no conversion available.
        self._metric_converted: Optional[bool] = None

    def __del__(self):
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _get_tmpdir(self) -> str:
        if self._tmpdir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="depthify_da3_")
        return self._tmpdir

    def load_model(self):
        if self._model is not None:
            return
        import torch
        from depth_anything_3.api import DepthAnything3
        logger.info(f"Loading DA3: {self.model_id}")
        self._model = DepthAnything3.from_pretrained(self.model_id)
        self._model = self._model.to(device=torch.device(self.device))

    def infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if self._model is None:
            self.load_model()

        # DA3 expects file paths – write to reusable temp dir as JPEG (fast)
        tmpdir = self._get_tmpdir()
        paths = []
        for i, img in enumerate(images):
            p = Path(tmpdir) / f"{i:06d}.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(str(p))

        pred = self._model.inference(paths)
        raw = pred.depth  # (N, H, W) float32

        # Metric conversion for DA3METRIC-* (not NESTED which is already meters)
        if "METRIC" in self.model_id.upper() and "NESTED" not in self.model_id.upper():
            if self.focal_length is not None:
                raw = self.focal_length * raw / 300.0
                self._metric_converted = True
            elif hasattr(pred, "intrinsics") and pred.intrinsics is not None:
                fx = pred.intrinsics[:, 0, 0]
                for i in range(raw.shape[0]):
                    raw[i] = float(fx[i]) * raw[i] / 300.0
                self._metric_converted = True
            else:
                logger.warning("DA3METRIC: no focal length – depth in raw model units.")
                self._metric_converted = False

        # Clean temp files (keep dir)
        for p in paths:
            try: Path(p).unlink()
            except OSError: pass

        depths = [raw[i] for i in range(raw.shape[0])]
        for i, (img, d) in enumerate(zip(images, depths)):
            if d.shape[:2] != img.shape[:2]:
                depths[i] = cv2.resize(d, (img.shape[1], img.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        return depths

    def depth_units(self) -> str:
        mid = self.model_id.upper()
        if "NESTED" in mid:
            return "meters"
        if "METRIC" in mid:
            # After inference: reflect what actually happened
            if self._metric_converted is True:
                return "best_effort_meters"
            if self._metric_converted is False:
                return "model_units"
            # Before inference: predict based on config
            if self.focal_length is not None:
                return "best_effort_meters"
            return "model_units"
        return "model_units"

    def model_metadata(self) -> dict:
        return {
            "backend": "da3", "model_id": self.model_id,
            "focal_length": self.focal_length,
            "depth_units": self.depth_units(),
            "repo": "https://github.com/ByteDance-Seed/Depth-Anything-3",
        }
