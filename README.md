# Depthify

Offline depth caching from MP4 videos for Any3D-VLA training.

Three backends: **Depth Anything 3**, **UniDepthV2**, **MapAnything**.

## Setup

```bash
pip install numpy opencv-python

# Then install whichever backend(s) you need:
git clone https://github.com/ByteDance-Seed/Depth-Anything-3 && cd Depth-Anything-3 && pip install -e .
git clone https://github.com/lpiccinelli-eth/UniDepth && cd UniDepth && pip install -e .
git clone https://github.com/facebookresearch/map-anything && cd map-anything && pip install -e .
```

Don't pip-install torch separately — let each model repo handle its own torch version to avoid conflicts.

## Usage

```bash
cd depthify/

# DA3 (GPU)
python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache --device cuda --chunk_size 16 --write_preview

# UniDepth (GPU)
python run_depth.py --backend unidepthv2 --video_dir ./videos --out_dir ./depth_cache --device cuda --write_preview

# MapAnything (GPU, slower)
python run_depth.py --backend mapanything --video_dir ./videos --out_dir ./depth_cache --device cuda --chunk_size 4 --map_use_amp

# Smoke test (CPU, 10 frames)
python run_depth.py --backend da3 --video_dir ./videos --out_dir ./depth_cache --device cpu --max_frames 10 --resize_w 320 --resize_h 240 --write_preview
```

Run `python run_depth.py --help` for all flags.

## Output

```
depth_cache/
  da3/
    foo/                  # one folder per video
      meta.json           # model, stride, resize, timestamp, camera_intrinsics (null placeholder)
      stats.json          # per-frame stats + temporal jitter (last 30 frames, documented)
      depth/
        000000.npy        # float16, named by original frame index
        000005.npy
      preview/
        000000.png        # 8-bit colormap (if --write_preview)
    manifest.jsonl        # one line per frame with backend, model_id, depth_units, stride, resize
```

## OOM handling

On CUDA OOM the batch is recursively halved (not dropped to single-frame), so you keep good throughput.

## Tests

```bash
pip install pytest
cd depthify/
python -m pytest test_common.py -v
```
