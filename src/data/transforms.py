import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image


class WeakAugmentation:
    """Weak augmentation for labeled and unlabeled data (flips, small shifts)."""

    def __init__(self, image_size=512):
        self.transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class StrongAugmentation:
    """Strong augmentation for unlabeled data (RandAugment + cutout)."""

    def __init__(self, image_size=512, n=2, m=10, cutout_prob=0.5, cutout_size=0.25):
        self.image_size = image_size
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size

        # Base transforms: resize + RandAugment + cutout
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandAugment(num_ops=n, magnitude=m),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(
                    p=cutout_prob,
                    scale=(cutout_size * 0.5, cutout_size),
                    ratio=(0.3, 3.3),
                    value=0,
                ),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


class TestTransforms:
    """Transform for test/validation data (resize + normalize)."""

    def __init__(self, image_size=512):
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


def get_transforms(aug_type="weak", image_size=512, **kwargs):
    """Factory function to get augmentation transforms.

    Args:
        aug_type: 'weak', 'strong', or 'test'
        image_size: target image size
        **kwargs: additional arguments for specific augmentations

    Returns:
        transform callable
    """
    if aug_type == "weak":
        return WeakAugmentation(image_size=image_size)
    elif aug_type == "strong":
        return StrongAugmentation(image_size=image_size, **kwargs)
    elif aug_type == "test":
        return TestTransforms(image_size=image_size)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
