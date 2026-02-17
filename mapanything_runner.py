"""
MapAnything backend.

Repo:    https://github.com/facebookresearch/map-anything
Install: git clone ... && pip install -e .

Models:  facebook/map-anything (CC-BY-NC), facebook/map-anything-apache (Apache 2.0)
Output:  metric Z-depth in meters.

Uses file-based load_images by default (proven path). Falls back to direct
view dicts only if a self-test at load time confirms compatibility.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from base_runner import BaseDepthRunner

logger = logging.getLogger("depthify")

DEFAULT_MODEL_ID = "facebook/map-anything"


class MapAnythingRunner(BaseDepthRunner):
    backend_name = "mapanything"

    def __init__(self, device="cuda", model_id=DEFAULT_MODEL_ID,
                 use_amp=True, amp_dtype="bf16", **kw):
        super().__init__(device=device, **kw)
        self.model_id = model_id
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self._model = None
        self._use_direct_views: bool | None = None

    def load_model(self):
        if self._model is not None:
            return
        import torch
        from mapanything.models import MapAnything
        logger.info(f"Loading MapAnything: {self.model_id}")
        self._model = MapAnything.from_pretrained(self.model_id)
        self._model = self._model.to(torch.device(self.device))
        self._model.eval()

        # Probe whether direct view dicts work with this version
        self._use_direct_views = self._probe_direct_views()
        logger.info(f"MapAnything loaded (direct_views={self._use_direct_views})")

    def _probe_direct_views(self) -> bool:
        try:
            import torch
            from mapanything.utils.image import preprocess_inputs
            dummy = torch.zeros(4, 4, 3, dtype=torch.uint8, device=self.device)
            preprocess_inputs([{"img": dummy}])
            return True
        except Exception:
            return False

    def infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        import torch
        if self._model is None:
            self.load_model()

        if self._use_direct_views:
            from mapanything.utils.image import preprocess_inputs
            views = preprocess_inputs([
                {"img": torch.from_numpy(img).to(self.device)} for img in images
            ])
        else:
            views = self._load_via_files(images)

        preds = self._model.infer(
            views, memory_efficient_inference=True,
            use_amp=self.use_amp, amp_dtype=self.amp_dtype,
        )

        depths = []
        for pred in preds:
            dz = pred["depth_z"]
            if hasattr(dz, "cpu"):
                dz = dz.cpu().numpy()
            dz = np.squeeze(dz)
            depths.append(dz.astype(np.float32))

        for i, (img, d) in enumerate(zip(images, depths)):
            if d.shape[:2] != img.shape[:2]:
                depths[i] = cv2.resize(d, (img.shape[1], img.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        return depths

    def _load_via_files(self, images):
        from mapanything.utils.image import load_images
        tmpdir = tempfile.mkdtemp(prefix="depthify_map_")
        try:
            for i, img in enumerate(images):
                p = Path(tmpdir) / f"{i:06d}.jpg"
                cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            return load_images(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def depth_units(self):
        return "meters"

    def model_metadata(self):
        return {
            "backend": "mapanything", "model_id": self.model_id,
            "depth_units": "meters", "use_amp": self.use_amp,
            "repo": "https://github.com/facebookresearch/map-anything",
        }
