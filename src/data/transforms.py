"""
Data augmentation transforms for SSL mammography.

Design:
- All transforms expect PIL images as input
- All transforms return tensors (post ToTensor + Normalize)
- Config-driven: parameters read from config dict
- Factory function `get_transforms()` is the public API
"""

import torchvision.transforms as T


# ImageNet normalization (used for pretrained EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WeakAugmentation:
    """Weak augmentation: flips, small affine, optional color jitter.

    Used for:
    - Supervised training
    - FixMatch weak view (pseudo-label generation)
    """

    def __init__(self, image_size: int = 512, config: dict | None = None):
        aug_config = config or {}

        transforms = []

        if aug_config.get("random_horizontal_flip", True):
            transforms.append(T.RandomHorizontalFlip(p=0.5))

        if aug_config.get("random_vertical_flip", True):
            transforms.append(T.RandomVerticalFlip(p=0.5))

        rotation = aug_config.get("random_rotation", 5)
        if rotation > 0:
            transforms.append(T.RandomAffine(degrees=rotation, translate=(0.05, 0.05)))

        color_jitter = aug_config.get("color_jitter", 0.0)
        if color_jitter > 0:
            transforms.append(
                T.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter * 0.5,
                )
            )

        transforms.extend(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.transform = T.Compose(transforms)

    def __call__(self, img):
        return self.transform(img)


class StrongAugmentation:
    """Strong augmentation: RandAugment + cutout.

    Used for FixMatch strong view (consistency target).
    """

    def __init__(
        self,
        image_size: int = 512,
        n: int = 2,
        m: int = 10,
        cutout_prob: float = 0.5,
        cutout_size: float = 0.25,
    ):
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandAugment(num_ops=n, magnitude=m),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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


class EvalTransforms:
    """Deterministic transforms for validation and test data."""

    def __init__(self, image_size: int = 512):
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, img):
        return self.transform(img)


def get_transforms(
    aug_type: str = "weak",
    image_size: int = 512,
    config: dict | None = None,
    **kwargs,
):
    """Factory function to get augmentation transforms.

    Args:
        aug_type: 'weak', 'strong', or 'test'
        image_size: target image size
        config: augmentation config dict (from YAML augmentation.weak/strong section)
        **kwargs: additional arguments for specific augmentations

    Returns:
        transform callable (PIL Image -> Tensor)
    """
    if aug_type == "weak":
        return WeakAugmentation(image_size=image_size, config=config)
    elif aug_type == "strong":
        return StrongAugmentation(image_size=image_size, **kwargs)
    elif aug_type == "test":
        return EvalTransforms(image_size=image_size)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")
