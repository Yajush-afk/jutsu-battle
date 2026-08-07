"""Image cropping and transforms shared by training and inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms

from jutsu_battle.number_identifier.hand_detector import BoundingBox

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImagePreprocessingError(ValueError):
    """Raised when an image or crop cannot be processed safely."""


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Stable preprocessing settings stored with trained checkpoints."""

    image_size: int = 160
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


def read_bgr_image(path: Path) -> np.ndarray[Any, Any]:
    """Read an image and raise a descriptive error on failure."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImagePreprocessingError(f"Unable to read image: {path}")
    return image


def validate_box(box: BoundingBox, image: np.ndarray[Any, Any]) -> BoundingBox:
    """Clamp a box to the image and reject empty results."""
    height, width = image.shape[:2]
    clamped = BoundingBox(
        x1=max(0, min(width, box.x1)),
        y1=max(0, min(height, box.y1)),
        x2=max(0, min(width, box.x2)),
        y2=max(0, min(height, box.y2)),
    )
    if clamped.width <= 0 or clamped.height <= 0:
        raise ImagePreprocessingError(f"Invalid crop box: {box}")
    return clamped


def crop_pad_resize_rgb(
    image: np.ndarray[Any, Any],
    box: BoundingBox,
    image_size: int,
) -> np.ndarray[Any, Any]:
    """Crop a hand region, pad it square, resize, and convert BGR to RGB."""
    valid_box = validate_box(box, image)
    crop = image[valid_box.y1 : valid_box.y2, valid_box.x1 : valid_box.x2]
    height, width = crop.shape[:2]
    square_size = max(height, width)
    top = (square_size - height) // 2
    bottom = square_size - height - top
    left = (square_size - width) // 2
    right = square_size - width - left
    square = cv2.copyMakeBorder(
        crop,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(24, 24, 24),
    )
    resized = cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def build_transform(
    config: PreprocessConfig,
    *,
    training: bool,
) -> Callable[[Image.Image], Tensor]:
    """Build train-time augmentation or deterministic evaluation transforms."""
    common: list[Any] = [transforms.Resize((config.image_size, config.image_size))]
    if training:
        common.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.08, 0.08),
                    scale=(0.9, 1.1),
                ),
                transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3)], p=0.15
                ),
            ]
        )
    common.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(config.mean, config.std),
        ]
    )
    if training:
        common.append(
            transforms.RandomErasing(
                p=0.15,
                scale=(0.01, 0.06),
                ratio=(0.5, 2.0),
                value="random",
            )
        )
    return cast(Callable[[Image.Image], Tensor], transforms.Compose(common))


def rgb_to_tensor(
    rgb_image: np.ndarray[Any, Any],
    transform: Callable[[Image.Image], Tensor],
) -> Tensor:
    """Apply a configured TorchVision transform to an RGB array."""
    return transform(Image.fromarray(rgb_image))
