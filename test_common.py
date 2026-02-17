"""Tests for common.py – run with: python -m pytest test_common.py -v"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from common import (
    iter_video_frames, save_depth_npy, load_depth_npy, save_preview_png,
    write_json, append_manifest_line, compute_depth_stats,
    compute_temporal_jitter, find_videos,
)


def _make_video(path, n=20, w=64, h=48):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), (i * 7) % 256, dtype=np.uint8)
        out.write(f)
    out.release()


class TestVideoIter:
    def test_all(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v)
        frames = list(iter_video_frames(v))
        assert len(frames) == 20
        assert [i for i, _ in frames] == list(range(20))

    def test_stride(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v)
        assert [i for i, _ in iter_video_frames(v, stride=5)] == [0, 5, 10, 15]

    def test_max(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v)
        assert len(list(iter_video_frames(v, max_frames=3))) == 3

    def test_stride_max(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v)
        assert [i for i, _ in iter_video_frames(v, stride=3, max_frames=2)] == [0, 3]

    def test_resize(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v, n=2)
        _, f = next(iter(iter_video_frames(v, resize_to=(32, 24))))
        assert f.shape == (24, 32, 3)

    def test_deterministic(self, tmp_path):
        v = tmp_path / "t.mp4"; _make_video(v, n=5)
        a = [(i, f.copy()) for i, f in iter_video_frames(v)]
        b = [(i, f.copy()) for i, f in iter_video_frames(v)]
        for (i1, f1), (i2, f2) in zip(a, b):
            assert i1 == i2
            np.testing.assert_array_equal(f1, f2)


class TestDepthIO:
    def test_roundtrip(self, tmp_path):
        d = np.random.rand(48, 64).astype(np.float32) * 10
        save_depth_npy(tmp_path / "d.npy", d)
        np.testing.assert_allclose(load_depth_npy(tmp_path / "d.npy"), d, atol=0.01)

    def test_float16(self, tmp_path):
        save_depth_npy(tmp_path / "d.npy", np.ones((4, 4), dtype=np.float32))
        assert np.load(str(tmp_path / "d.npy")).dtype == np.float16


class TestPreview:
    def test_creates_file(self, tmp_path):
        save_preview_png(tmp_path / "p.png", np.random.rand(48, 64).astype(np.float32))
        assert (tmp_path / "p.png").exists()


class TestMetadata:
    def test_json(self, tmp_path):
        write_json(tmp_path / "m.json", {"k": 1})
        assert json.loads((tmp_path / "m.json").read_text())["k"] == 1

    def test_manifest(self, tmp_path):
        p = tmp_path / "m.jsonl"
        append_manifest_line(p, {"a": 1})
        append_manifest_line(p, {"b": 2})
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2


class TestStats:
    def test_basic(self):
        s = compute_depth_stats(np.array([[1, 2], [3, 4]], dtype=np.float32))
        assert s["min"] == 1.0 and s["max"] == 4.0

    def test_nonfinite(self):
        s = compute_depth_stats(np.array([[np.nan, np.inf], [1, 2]], dtype=np.float32))
        assert s["frac_nonfinite"] == 0.5

    def test_jitter(self):
        d1 = np.ones((16, 16), dtype=np.float32)
        d2 = d1 * 2
        assert compute_temporal_jitter([d1, d2], downsample=1) == [1.0]


class TestFindVideos:
    def test_finds(self, tmp_path):
        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.mp4").touch()
        names = [v.name for v in find_videos(tmp_path)]
        assert "a.mp4" in names and "c.mp4" in names and "b.txt" not in names
