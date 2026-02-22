#!/usr/bin/env python3
"""
run_depth.py – Run depth estimation on a folder of videos.

Usage:
  python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache
  python run_depth.py --help
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

logger = logging.getLogger("depthify")


def _parse_traj_keys(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    keys = [x.strip() for x in raw.split(",") if x.strip()]
    return keys or None


def _discover_h5_files(h5_glob: str | None, h5_manifest: str | None) -> list[Path]:
    files: list[Path] = []
    if h5_manifest:
        manifest_path = Path(h5_manifest).expanduser()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        for raw in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if not p.exists():
                raise FileNotFoundError(f"Manifest path does not exist: {p}")
            files.append(p)

    if h5_glob:
        for pattern in [x.strip() for x in h5_glob.split(",") if x.strip()]:
            for match in glob.glob(pattern, recursive=True):
                files.append(Path(match).resolve())

    return sorted({p.resolve() for p in files})


def main():
    p = argparse.ArgumentParser(
        description="Offline depth caching for Any3D-VLA.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache --device cuda --chunk_size 16 --write_preview
  python run_depth.py --backend unidepthv2 --video_dir ./videos --out_dir ./depth_cache --device cuda --write_preview
  python run_depth.py --backend mapanything --video_dir ./videos --out_dir ./depth_cache --device cuda --chunk_size 4 --map_use_amp
  python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache --device cpu --max_frames 10 --resize_w 320 --resize_h 240 --write_preview
  python run_depth.py --backend da3 --h5_glob "../data/maniskill2/demos/*/motionplanning/trajectory.rgbd*.h5" --out_dir ./depth_cache --device cuda
  python run_depth.py --backend da3 --h5_glob "../data/maniskill2/demos/*/motionplanning/trajectory.rgbd*.h5" --out_dir ./depth_cache --skip_existing
""",
    )

    p.add_argument("--backend", required=True, choices=["da3", "unidepthv2", "mapanything"])
    p.add_argument("--video_dir", default=None, help="Folder with MP4 videos.")
    p.add_argument("--h5_glob", default=None, help="Glob for H5 sources (e.g. data/**/*.h5).")
    p.add_argument("--h5_manifest", default=None, help="Text file with one H5 path per line.")
    p.add_argument("--h5_camera_key", default="base_camera", help="Camera key inside H5 sensor_data.")
    p.add_argument("--h5_traj_keys", default=None, help="Optional CSV of trajectory keys, e.g. traj_0,traj_5")
    p.add_argument("--h5_source_root", default=None, help="Optional root used to derive stable H5 source IDs.")
    p.add_argument("--out_dir", required=True, help="Where to write depth maps.")

    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--chunk_size", type=int, default=1, help="Batch size.")
    p.add_argument("--stride", type=int, default=1, help="Process every N-th frame.")
    p.add_argument("--max_frames", type=int, default=None, help="Cap frames per video.")
    p.add_argument("--resize_w", type=int, default=None)
    p.add_argument("--resize_h", type=int, default=None)
    p.add_argument("--write_preview", action="store_true", help="Save 8-bit PNG previews.")
    p.add_argument("--preview_max", type=int, default=None)
    p.add_argument("--depth_dtype", default="fp16", choices=["fp16", "fp32"],
                   help="Depth save dtype: fp16 (smaller) or fp32 (exact).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse existing .npy depth files if present (resume-safe).")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete per-item output directories before processing.")

    # DA3
    p.add_argument("--da3_model_id", default="depth-anything/DA3METRIC-LARGE")
    p.add_argument("--da3_focal_length", type=float, default=None)
    # UniDepth
    p.add_argument("--unidepth_model_id", default="lpiccinelli/unidepth-v2-vitl14")
    # MapAnything
    p.add_argument("--map_model_id", default="facebook/map-anything")
    p.add_argument("--map_use_amp", action="store_true")
    p.add_argument("--map_amp_dtype", default="bf16", choices=["bf16", "fp16"])

    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resize
    resize_to = None
    if args.resize_w and args.resize_h:
        resize_to = (args.resize_w, args.resize_h)
    elif args.resize_w or args.resize_h:
        logger.error("Need both --resize_w and --resize_h.")
        sys.exit(1)

    use_video = bool(args.video_dir)
    use_h5 = bool(args.h5_glob or args.h5_manifest)
    if use_video and use_h5:
        logger.error("Use either --video_dir OR (--h5_glob/--h5_manifest), not both.")
        sys.exit(1)
    if not use_video and not use_h5:
        logger.error("Provide one input source: --video_dir or --h5_glob/--h5_manifest.")
        sys.exit(1)

    # Build runner
    if args.backend == "da3":
        from da3_runner import DA3Runner
        runner = DA3Runner(device=args.device, model_id=args.da3_model_id,
                           focal_length=args.da3_focal_length)
    elif args.backend == "unidepthv2":
        from unidepth_runner import UniDepthRunner
        runner = UniDepthRunner(device=args.device, model_id=args.unidepth_model_id)
    elif args.backend == "mapanything":
        from mapanything_runner import MapAnythingRunner
        runner = MapAnythingRunner(device=args.device, model_id=args.map_model_id,
                                   use_amp=args.map_use_amp, amp_dtype=args.map_amp_dtype)

    run_kwargs = dict(
        stride=args.stride,
        max_frames=args.max_frames,
        resize_to=resize_to,
        chunk_size=args.chunk_size,
        write_preview=args.write_preview,
        preview_max=args.preview_max,
        depth_dtype=args.depth_dtype,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )

    if use_video:
        results = runner.run_on_folder(
            video_dir=args.video_dir,
            out_dir=args.out_dir,
            **run_kwargs,
        )
    else:
        try:
            h5_files = _discover_h5_files(args.h5_glob, args.h5_manifest)
        except Exception as e:
            logger.error(str(e))
            sys.exit(1)
        if not h5_files:
            logger.error("No H5 files found from --h5_glob/--h5_manifest.")
            sys.exit(1)
        logger.info(f"Found {len(h5_files)} H5 files.")

        from h5_runner import run_runner_on_h5_files

        results = run_runner_on_h5_files(
            runner=runner,
            h5_files=h5_files,
            out_dir=args.out_dir,
            camera_key=args.h5_camera_key,
            traj_keys=_parse_traj_keys(args.h5_traj_keys),
            source_root=(Path(args.h5_source_root).resolve() if args.h5_source_root else None),
            **run_kwargs,
        )

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    logger.info(f"Done. {n_ok} OK, {n_err} failed.")
    if n_err:
        for r in results:
            if "error" in r:
                src = r.get("video_path") or r.get("source_h5") or "unknown"
                logger.error(f"  FAILED: {src}: {r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
