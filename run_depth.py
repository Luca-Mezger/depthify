#!/usr/bin/env python3
"""
run_depth.py – Run depth estimation on a folder of videos.

Usage:
  python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache
  python run_depth.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger("depthify")


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
""",
    )

    p.add_argument("--backend", required=True, choices=["da3", "unidepthv2", "mapanything"])
    p.add_argument("--video_dir", required=True, help="Folder with MP4 videos.")
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

    results = runner.run_on_folder(
        video_dir=args.video_dir, out_dir=args.out_dir,
        stride=args.stride, max_frames=args.max_frames,
        resize_to=resize_to, chunk_size=args.chunk_size,
        write_preview=args.write_preview, preview_max=args.preview_max,
        depth_dtype=args.depth_dtype,
    )

    n_ok = sum(1 for r in results if "error" not in r)
    n_err = sum(1 for r in results if "error" in r)
    logger.info(f"Done. {n_ok} OK, {n_err} failed.")
    if n_err:
        for r in results:
            if "error" in r:
                logger.error(f"  FAILED: {r['video_path']}: {r['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
