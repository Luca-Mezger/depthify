"""
UniDepthV2 backend.

Repo:    https://github.com/lpiccinelli-eth/UniDepth
Install: git clone ... && pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

Models: lpiccinelli/unidepth-v2-vitl14, -vitb14, -vits14
Output:  metric depth in meters.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from base_runner import BaseDepthRunner

logger = logging.getLogger("depthify")

DEFAULT_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"


class UniDepthRunner(BaseDepthRunner):
    backend_name = "unidepthv2"

    def __init__(self, device="cuda", model_id=DEFAULT_MODEL_ID, **kw):
        super().__init__(device=device, **kw)
        self.model_id = model_id
        self._model = None

    def load_model(self):
        if self._model is not None:
            return
        import torch
        from unidepth.models import UniDepthV2
        logger.info(f"Loading UniDepthV2: {self.model_id}")
        self._model = UniDepthV2.from_pretrained(self.model_id)
        self._model = self._model.to(torch.device(self.device))
        self._model.eval()

    def infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        if self._model is None:
            self.load_model()

        depths = []
        for img in images:
            rgb = torch.from_numpy(img).permute(2, 0, 1).to(self.device)
            with torch.no_grad():
                pred = self._model.infer(rgb)
            depth = pred["depth"]
            if hasattr(depth, "cpu"):
                depth = depth.cpu().numpy()
            depth = np.squeeze(depth)
            if depth.shape[:2] != img.shape[:2]:
                depth = cv2.resize(depth, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)
            depths.append(depth.astype(np.float32))
        return depths

    def depth_units(self):
        return "meters"

    def model_metadata(self):
        return {
            "backend": "unidepthv2", "model_id": self.model_id,
            "depth_units": "meters",
            "repo": "https://github.com/lpiccinelli-eth/UniDepth",
        }
