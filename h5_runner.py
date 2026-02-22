"""H5 trajectory depth-caching pipeline for Depthify backends."""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np

from common import (
    compute_depth_stats,
    compute_temporal_jitter,
    load_depth_npy,
    now_iso,
    save_depth_npy,
    save_preview_png,
    write_json,
)

logger = logging.getLogger("depthify")

_JITTER_WINDOW = 30


def _sorted_traj_keys(keys: Sequence[str]) -> list[str]:
    def key_fn(raw: str) -> tuple[int, int | str]:
        if raw.startswith("traj_"):
            suffix = raw[5:]
            if suffix.isdigit():
                return (0, int(suffix))
            return (1, suffix)
        return (2, raw)

    return sorted(keys, key=key_fn)


def _source_id(source_h5: Path, source_root: Optional[Path]) -> str:
    if source_root is not None:
        try:
            rel = source_h5.resolve().relative_to(source_root.resolve())
            base = str(rel.with_suffix("")).replace("\\", "__").replace("/", "__")
        except ValueError:
            base = source_h5.stem
    else:
        base = source_h5.stem
    base = base.replace(":", "_")
    digest = hashlib.sha1(str(source_h5.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{base}--{digest}"


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3:
        raise ValueError(f"Unexpected RGB frame shape: {rgb.shape}")
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            maxv = float(np.max(rgb)) if rgb.size > 0 else 1.0
            if maxv <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def run_runner_on_h5_files(
    runner,
    h5_files: Sequence[Path],
    out_dir: str | Path,
    camera_key: str = "base_camera",
    traj_keys: Optional[Sequence[str]] = None,
    source_root: Optional[Path] = None,
    stride: int = 1,
    max_frames: Optional[int] = None,
    resize_to: Optional[Tuple[int, int]] = None,
    chunk_size: int = 1,
    write_preview: bool = False,
    preview_max: Optional[int] = None,
    depth_dtype: str = "fp16",
    skip_existing: bool = False,
    overwrite: bool = False,
) -> list[dict]:
    h5_files = [Path(p) for p in h5_files]
    out_base = Path(out_dir)
    manifest_path = out_base / runner.backend_name / "manifest.jsonl"
    results: list[dict] = []

    if not h5_files:
        logger.warning("No H5 files provided.")
        return results

    runner.load_model()
    for h5_path in h5_files:
        sid = _source_id(h5_path, source_root)
        with h5py.File(h5_path, "r") as h5:
            all_traj = _sorted_traj_keys([k for k in h5.keys() if k.startswith("traj_")])
            if traj_keys:
                requested = set(traj_keys)
                use_traj = [k for k in all_traj if k in requested]
                for miss in sorted(requested - set(use_traj)):
                    logger.warning(f"{h5_path}: requested traj '{miss}' not found.")
            else:
                use_traj = all_traj

            for traj_key in use_traj:
                rgb_path = f"{traj_key}/obs/sensor_data/{camera_key}/rgb"
                if rgb_path not in h5:
                    logger.warning(f"{h5_path}: missing dataset {rgb_path}; skipping.")
                    continue
                rgb_ds = h5[rgb_path]
                n_rgb = int(rgb_ds.shape[0])

                traj_out = out_base / runner.backend_name / "h5" / sid / traj_key
                if overwrite and traj_out.exists():
                    shutil.rmtree(traj_out, ignore_errors=True)
                depth_dir = traj_out / "depth"
                depth_dir.mkdir(parents=True, exist_ok=True)
                if write_preview:
                    (traj_out / "preview").mkdir(parents=True, exist_ok=True)

                selected_indices = list(range(0, n_rgb, stride))
                if max_frames is not None:
                    selected_indices = selected_indices[:max_frames]

                manifest_context = {
                    "input_kind": "h5",
                    "source_h5": str(h5_path),
                    "source_id": sid,
                    "traj_key": traj_key,
                    "camera_key": camera_key,
                    "stride": stride,
                    "resize_to": list(resize_to) if resize_to else None,
                }

                buf_idx: list[int] = []
                buf_img: list[np.ndarray] = []
                all_stats: list[dict] = []
                recent_depths: list[np.ndarray] = []
                n_preview = 0
                n_selected = 0
                n_inferred = 0
                n_reused = 0

                def _append_recent(depth: np.ndarray) -> None:
                    recent_depths.append(depth)
                    if len(recent_depths) > _JITTER_WINDOW:
                        recent_depths.pop(0)

                def _write_record(frame_index: int, depth: np.ndarray, save_depth: bool) -> None:
                    nonlocal n_preview
                    fname = f"{frame_index:06d}"
                    depth_path = depth_dir / f"{fname}.npy"
                    if save_depth:
                        save_depth_npy(depth_path, depth, dtype=depth_dtype)

                    stats = compute_depth_stats(depth)
                    stats["frame_index"] = frame_index
                    all_stats.append(stats)

                    h, w = depth.shape
                    runner._append_manifest_unique(
                        manifest_path,
                        {
                            **manifest_context,
                            "frame_index": frame_index,
                            "depth_path": str(depth_path.relative_to(out_base)),
                            "width": int(w),
                            "height": int(h),
                            "backend": runner.backend_name,
                            "model_id": runner.model_metadata().get(
                                "model_id", runner.backend_name
                            ),
                            "depth_units": runner.depth_units(),
                        },
                    )

                    if write_preview and (preview_max is None or n_preview < preview_max):
                        save_preview_png(traj_out / "preview" / f"{fname}.png", depth)
                        n_preview += 1
                    _append_recent(depth)

                def _flush() -> None:
                    nonlocal n_inferred
                    if not buf_img:
                        return
                    try:
                        pred_depths = runner._safe_infer_batch(buf_img)
                    except Exception as exc:
                        logger.error(
                            f"Inference failed at {h5_path}:{traj_key}:frame={buf_idx[0]}: {exc}"
                        )
                        buf_idx.clear()
                        buf_img.clear()
                        return
                    for frame_index, depth in zip(buf_idx, pred_depths):
                        _write_record(frame_index, depth, save_depth=True)
                        n_inferred += 1
                    buf_idx.clear()
                    buf_img.clear()

                for frame_index in selected_indices:
                    n_selected += 1
                    depth_path = depth_dir / f"{frame_index:06d}.npy"
                    if skip_existing and depth_path.exists():
                        depth = load_depth_npy(depth_path)
                        _write_record(frame_index, depth, save_depth=False)
                        n_reused += 1
                        continue

                    rgb = _as_rgb_uint8(rgb_ds[frame_index])
                    if resize_to is not None:
                        rgb = cv2.resize(rgb, resize_to, interpolation=cv2.INTER_LINEAR)
                    buf_idx.append(frame_index)
                    buf_img.append(rgb)
                    if len(buf_img) >= chunk_size:
                        _flush()
                _flush()

                n_processed = n_inferred + n_reused
                jitter = (
                    compute_temporal_jitter(recent_depths) if len(recent_depths) > 1 else []
                )
                meta = {
                    "backend": runner.backend_name,
                    "model": runner.model_metadata(),
                    "depth_units": runner.depth_units(),
                    "input_kind": "h5",
                    "source_h5": str(h5_path),
                    "source_id": sid,
                    "traj_key": traj_key,
                    "camera_key": camera_key,
                    "rgb_steps_total": n_rgb,
                    "stride": stride,
                    "max_frames": max_frames,
                    "resize_to": resize_to,
                    "chunk_size": chunk_size,
                    "depth_dtype": depth_dtype,
                    "skip_existing": bool(skip_existing),
                    "n_frames_selected": n_selected,
                    "n_frames_inferred": n_inferred,
                    "n_frames_reused": n_reused,
                    "n_frames_processed": n_processed,
                    "camera_intrinsics": None,
                    "timestamp": now_iso(),
                }
                write_json(traj_out / "meta.json", meta)
                write_json(
                    traj_out / "stats.json",
                    {
                        "per_frame": all_stats,
                        "temporal_jitter_mad": jitter,
                        "temporal_jitter_window": min(n_processed, _JITTER_WINDOW),
                        "temporal_jitter_note": (
                            f"Computed over last {min(n_processed, _JITTER_WINDOW)} frames "
                            f"(rolling window of {_JITTER_WINDOW}), not the full trajectory."
                        ),
                    },
                )
                logger.info(
                    f"[{runner.backend_name}] {h5_path.name}:{traj_key}: {n_processed} frames "
                    f"({n_inferred} inferred, {n_reused} reused)"
                )
                results.append(meta)

    return results

