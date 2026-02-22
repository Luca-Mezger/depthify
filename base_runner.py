"""Abstract base class for depth estimation backends."""

from __future__ import annotations

import abc
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from common import (
    append_manifest_line, compute_depth_stats, compute_temporal_jitter,
    find_videos, get_video_info, iter_video_frames, now_iso,
    load_depth_npy, save_depth_npy, save_preview_png, write_json,
)

logger = logging.getLogger("depthify")

# Max frames kept in memory for temporal jitter QC.
_JITTER_WINDOW = 30


class BaseDepthRunner(abc.ABC):

    backend_name: str = "base"

    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self._manifest_seen: dict[str, set[str]] = {}

    @abc.abstractmethod
    def load_model(self) -> None:
        """Load model weights."""

    @abc.abstractmethod
    def infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """images: list of (H,W,3) uint8 RGB -> list of (H,W) float32 depth."""

    def infer_one(self, image: np.ndarray) -> np.ndarray:
        return self.infer_batch([image])[0]

    def model_metadata(self) -> dict:
        return {"backend": self.backend_name}

    def depth_units(self) -> str:
        """'meters', 'model_units', or 'best_effort_meters'."""
        return "model_units"

    def _manifest_cache_key(self, manifest_path: Path) -> str:
        return str(manifest_path.resolve())

    def _get_manifest_seen(self, manifest_path: Path) -> set[str]:
        key = self._manifest_cache_key(manifest_path)
        if key in self._manifest_seen:
            return self._manifest_seen[key]
        seen: set[str] = set()
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    depth_path = rec.get("depth_path")
                    if isinstance(depth_path, str):
                        seen.add(depth_path)
        self._manifest_seen[key] = seen
        return seen

    def _append_manifest_unique(self, manifest_path: Path, record: dict) -> None:
        depth_path = record.get("depth_path")
        if not isinstance(depth_path, str):
            append_manifest_line(manifest_path, record)
            return
        seen = self._get_manifest_seen(manifest_path)
        if depth_path in seen:
            return
        append_manifest_line(manifest_path, record)
        seen.add(depth_path)

    def _prepare_output_dir(self, out_dir: Path, write_preview: bool, overwrite: bool) -> None:
        if overwrite and out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        (out_dir / "depth").mkdir(parents=True, exist_ok=True)
        if write_preview:
            (out_dir / "preview").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # High-level: process one video
    # ------------------------------------------------------------------

    def run_on_video(
        self, video_path: Path, out_base: Path,
        stride: int = 1, max_frames: Optional[int] = None,
        resize_to: Optional[Tuple[int, int]] = None,
        chunk_size: int = 1, write_preview: bool = False,
        preview_max: Optional[int] = None,
        depth_dtype: str = "fp16",
        skip_existing: bool = False,
        overwrite: bool = False,
    ) -> dict:
        video_name = Path(video_path).stem
        out_dir = out_base / self.backend_name / video_name
        self._prepare_output_dir(out_dir, write_preview=write_preview, overwrite=overwrite)
        depth_dir = out_dir / "depth"

        manifest_path = out_base / self.backend_name / "manifest.jsonl"
        video_info = get_video_info(video_path)

        buf: list[Tuple[int, np.ndarray]] = []
        all_stats, recent_depths = [], []
        n_previews = 0
        n_inferred = 0
        n_reused = 0
        n_selected = 0

        def _flush():
            nonlocal n_inferred, n_previews
            if not buf:
                return
            indices = [i for i, _ in buf]
            images = [img for _, img in buf]
            try:
                depths = self._safe_infer_batch(images)
            except Exception as e:
                logger.error(f"Inference failed at frame {indices[0]}: {e}")
                buf.clear()
                return

            for fidx, depth in zip(indices, depths):
                fname = f"{fidx:06d}"
                dp = depth_dir / f"{fname}.npy"
                save_depth_npy(dp, depth, dtype=depth_dtype)

                stats = compute_depth_stats(depth)
                stats["frame_index"] = fidx
                all_stats.append(stats)

                h, w = depth.shape
                self._append_manifest_unique(manifest_path, {
                    "video": Path(video_path).name, "frame_index": fidx,
                    "depth_path": str(dp.relative_to(out_base)),
                    "width": w, "height": h,
                    "backend": self.backend_name,
                    "model_id": self.model_metadata().get("model_id", self.backend_name),
                    "depth_units": self.depth_units(),
                    "stride": stride,
                    "resize_to": list(resize_to) if resize_to else None,
                })

                if write_preview and (preview_max is None or n_previews < preview_max):
                    save_preview_png(out_dir / "preview" / f"{fname}.png", depth)
                    n_previews += 1

                recent_depths.append(depth)
                if len(recent_depths) > _JITTER_WINDOW:
                    recent_depths.pop(0)
                n_inferred += 1
            buf.clear()

        for fidx, rgb in iter_video_frames(video_path, stride=stride,
                                            max_frames=max_frames, resize_to=resize_to):
            n_selected += 1
            if skip_existing:
                dp = depth_dir / f"{fidx:06d}.npy"
                if dp.exists():
                    depth = load_depth_npy(dp)
                    stats = compute_depth_stats(depth)
                    stats["frame_index"] = fidx
                    all_stats.append(stats)

                    h, w = depth.shape
                    self._append_manifest_unique(manifest_path, {
                        "video": Path(video_path).name, "frame_index": fidx,
                        "depth_path": str(dp.relative_to(out_base)),
                        "width": w, "height": h,
                        "backend": self.backend_name,
                        "model_id": self.model_metadata().get("model_id", self.backend_name),
                        "depth_units": self.depth_units(),
                        "stride": stride,
                        "resize_to": list(resize_to) if resize_to else None,
                    })

                    if write_preview and (preview_max is None or n_previews < preview_max):
                        save_preview_png(out_dir / "preview" / f"{fidx:06d}.png", depth)
                        n_previews += 1

                    recent_depths.append(depth)
                    if len(recent_depths) > _JITTER_WINDOW:
                        recent_depths.pop(0)
                    n_reused += 1
                    continue
            buf.append((fidx, rgb))
            if len(buf) >= chunk_size:
                _flush()
        _flush()

        n_processed = n_inferred + n_reused
        jitter = compute_temporal_jitter(recent_depths) if len(recent_depths) > 1 else []

        meta = {
            "backend": self.backend_name, "model": self.model_metadata(),
            "depth_units": self.depth_units(),
            "video_path": str(video_path), "video_info": video_info,
            "stride": stride, "max_frames": max_frames,
            "resize_to": resize_to, "chunk_size": chunk_size,
            "depth_dtype": depth_dtype,
            "skip_existing": bool(skip_existing),
            "n_frames_selected": n_selected,
            "n_frames_inferred": n_inferred,
            "n_frames_reused": n_reused,
            "n_frames_processed": n_processed,
            "camera_intrinsics": None,  # fill with known K later
            "timestamp": now_iso(),
        }
        write_json(out_dir / "meta.json", meta)

        write_json(out_dir / "stats.json", {
            "per_frame": all_stats,
            "temporal_jitter_mad": jitter,
            "temporal_jitter_window": min(n_processed, _JITTER_WINDOW),
            "temporal_jitter_note":
                f"Computed over last {min(n_processed, _JITTER_WINDOW)} frames "
                f"(rolling window of {_JITTER_WINDOW}), not the full video.",
        })

        logger.info(
            f"[{self.backend_name}] {video_name}: {n_processed} frames "
            f"({n_inferred} inferred, {n_reused} reused)"
        )
        return meta

    # ------------------------------------------------------------------
    # High-level: process a folder of videos
    # ------------------------------------------------------------------

    def run_on_folder(self, video_dir, out_dir, **kwargs) -> list[dict]:
        video_dir, out_dir = Path(video_dir), Path(out_dir)
        videos = find_videos(video_dir)
        if not videos:
            logger.warning(f"No videos in {video_dir}")
            return []
        logger.info(f"Found {len(videos)} videos in {video_dir}")
        self.load_model()
        results = []
        for vp in videos:
            try:
                results.append(self.run_on_video(vp, out_dir, **kwargs))
            except Exception as e:
                logger.error(f"Failed {vp}: {e}", exc_info=True)
                results.append({"video_path": str(vp), "error": str(e)})
        return results

    # ------------------------------------------------------------------
    # OOM-safe inference with recursive batch halving
    # ------------------------------------------------------------------

    def _safe_infer_batch(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Try full batch; on OOM split in half recursively."""
        try:
            return self.infer_batch(images)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower() or len(images) <= 1:
                raise
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            mid = len(images) // 2
            logger.warning(f"OOM at batch {len(images)}, halving -> {mid}+{len(images)-mid}")
            return self._safe_infer_batch(images[:mid]) + self._safe_infer_batch(images[mid:])
