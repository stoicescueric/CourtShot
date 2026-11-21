"""Tests for where the YOLO weights are kept (no download is made)."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtshot import weights


def test_folder_comes_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSHOT_WEIGHTS_DIR", str(tmp_path))
    (tmp_path / "yolo26m.pt").write_bytes(b"weights")
    assert weights.get_weight_path("yolo26m.pt") == tmp_path / "yolo26m.pt"


def test_default_folder(monkeypatch):
    monkeypatch.delenv("COURTSHOT_WEIGHTS_DIR", raising=False)
    assert weights.weights_dir() == Path.home() / ".courtshot" / "weights"


def test_failed_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("COURTSHOT_WEIGHTS_DIR", str(tmp_path))

    def fail(url, dest):
        dest.write_bytes(b"part")
        raise TimeoutError("timed out")

    monkeypatch.setattr(weights, "_download_with_progress", fail)
    with pytest.raises(RuntimeError, match="timed out"):
        weights.get_weight_path("yolo26m-pose.pt")
    assert list(tmp_path.iterdir()) == []
