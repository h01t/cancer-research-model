"""
Tests for data augmentation transforms.
"""

import pytest
import torch
from PIL import Image

from src.data.transforms import (
    StrongAugmentation,
    EvalTransforms,
    WeakAugmentation,
    get_transforms,
)


@pytest.fixture
def pil_image():
    """Create a test PIL image."""
    return Image.new("RGB", (128, 128), (128, 128, 128))


class TestWeakAugmentation:
    def test_output_shape(self, pil_image):
        transform = WeakAugmentation(image_size=64)
        result = transform(pil_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    def test_output_normalized(self, pil_image):
        transform = WeakAugmentation(image_size=64)
        result = transform(pil_image)
        # After ImageNet normalization, values should be roughly in [-2.5, 2.5]
        assert result.min() > -3.0
        assert result.max() < 3.0

    def test_config_driven(self, pil_image):
        config = {
            "random_horizontal_flip": False,
            "random_vertical_flip": False,
            "random_rotation": 0,
            "color_jitter": 0.2,
        }
        transform = WeakAugmentation(image_size=64, config=config)
        result = transform(pil_image)
        assert result.shape == (3, 64, 64)


class TestStrongAugmentation:
    def test_output_shape(self, pil_image):
        transform = StrongAugmentation(image_size=64, n=2, m=10)
        result = transform(pil_image)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)


class TestEvalTransforms:
    def test_deterministic(self, pil_image):
        transform = EvalTransforms(image_size=64)
        result1 = transform(pil_image)
        result2 = transform(pil_image)
        assert torch.equal(result1, result2)

    def test_output_shape(self, pil_image):
        transform = EvalTransforms(image_size=64)
        result = transform(pil_image)
        assert result.shape == (3, 64, 64)


class TestGetTransforms:
    def test_factory_weak(self):
        t = get_transforms("weak", image_size=32)
        assert isinstance(t, WeakAugmentation)

    def test_factory_strong(self):
        t = get_transforms("strong", image_size=32)
        assert isinstance(t, StrongAugmentation)

    def test_factory_test(self):
        t = get_transforms("test", image_size=32)
        assert isinstance(t, EvalTransforms)

    def test_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown augmentation type"):
            get_transforms("unknown", image_size=32)
