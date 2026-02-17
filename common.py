"""Common utilities for video decoding, depth I/O, preview generation, and metadata."""

from __future__ import annotations

import json
import logging
import datetime
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("depthify")


# ---------------------------------------------------------------------------
# Video frame iterator
# ---------------------------------------------------------------------------

def iter_video_frames(
    video_path: str | Path,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize_to: Optional[Tuple[int, int]] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """Yield (frame_index, rgb_uint8) from a video.

    frame_index is the *original* 0-based index in the video,
    not a sequential count of yielded frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    yielded = 0
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if resize_to is not None:
                    rgb = cv2.resize(rgb, resize_to, interpolation=cv2.INTER_LINEAR)
                yield frame_idx, rgb
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()


def get_video_info(video_path: str | Path) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Depth I/O
# ---------------------------------------------------------------------------

def save_depth_npy(path: str | Path, depth: np.ndarray, dtype: str = "fp16") -> None:
    """Save a depth map as .npy. dtype: 'fp16' (default, smaller) or 'fp32' (exact)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_dtype = np.float16 if dtype == "fp16" else np.float32
    np.save(str(path), depth.astype(out_dtype))


def load_depth_npy(path: str | Path) -> np.ndarray:
    """Load a depth map saved by save_depth_npy."""
    return np.load(str(path)).astype(np.float32)


# ---------------------------------------------------------------------------
# Preview images
# ---------------------------------------------------------------------------

def depth_to_preview(depth: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Convert a float depth map to 8-bit BGR preview using robust percentiles."""
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo = np.percentile(finite, 2)
    hi = np.percentile(finite, 98)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    normed = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    normed = (normed * 255).astype(np.uint8)
    return cv2.applyColorMap(normed, colormap)


def save_preview_png(path: str | Path, depth: np.ndarray) -> None:
    """Save a depth preview as an 8-bit PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), depth_to_preview(depth))


# ---------------------------------------------------------------------------
# Metadata / manifest
# ---------------------------------------------------------------------------

def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def append_manifest_line(manifest_path: str | Path, record: dict) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# Depth statistics (QC)
# ---------------------------------------------------------------------------

def compute_depth_stats(depth: np.ndarray) -> dict:
    finite = depth[np.isfinite(depth)]
    total = depth.size
    n_finite = finite.size
    if n_finite == 0:
        return {"min": None, "median": None, "max": None,
                "mean": None, "std": None, "frac_nonfinite": 1.0}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "frac_nonfinite": 1.0 - n_finite / total,
    }


def compute_temporal_jitter(depths: list[np.ndarray], downsample: int = 4) -> list[float]:
    """Mean absolute difference between consecutive depth frames."""
    jitter = []
    for i in range(1, len(depths)):
        a = depths[i - 1][::downsample, ::downsample]
        b = depths[i][::downsample, ::downsample]
        if a.shape != b.shape:
            jitter.append(float("nan"))
            continue
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() == 0:
            jitter.append(float("nan"))
        else:
            jitter.append(float(np.mean(np.abs(a[mask] - b[mask]))))
    return jitter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def find_videos(video_dir: str | Path, extensions=(".mp4", ".avi", ".mov", ".mkv")) -> list[Path]:
    video_dir = Path(video_dir)
    videos = []
    for ext in extensions:
        videos.extend(video_dir.rglob(f"*{ext}"))
    return sorted(set(videos))
