import pytest
from paralleldomain import Scene


def test_lazy_cloud_loading(scene: Scene):
    frames = scene.frames
    assert len(frames) > 0