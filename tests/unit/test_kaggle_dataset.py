"""Tests for public Kaggle dataset discovery and manifest preparation."""

import csv
import json
from pathlib import Path

from PIL import Image

from jutsu_battle.number_identifier.kaggle_dataset import (
    DIGIT_LABELS,
    PublicGestureDataset,
    prepare_kaggle_manifests,
    read_public_manifest,
)
from jutsu_battle.number_identifier.preprocessing import PreprocessConfig


def create_digit_folders(root: Path, images_per_class: int = 10) -> None:
    """Create a nested, valid ten-class image-folder fixture."""
    wrapped = root / "download-wrapper" / "dataset"
    for label in DIGIT_LABELS:
        directory = wrapped / label
        directory.mkdir(parents=True)
        for index in range(images_per_class):
            color = (int(label) * 20, index * 10, 80)
            Image.new("RGB", (32, 24), color).save(directory / f"{index}.jpg")


def test_prepare_finds_nested_classes_and_balances_splits(tmp_path: Path) -> None:
    """Kaggle wrapper folders do not affect deterministic digit manifests."""
    dataset_root = tmp_path / "external"
    create_digit_folders(dataset_root)
    output = tmp_path / "manifests"
    paths = prepare_kaggle_manifests(dataset_root, output)

    assert len(read_public_manifest(paths["train"])) == 80
    assert len(read_public_manifest(paths["validation"])) == 10
    assert len(read_public_manifest(paths["test"])) == 10
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert tuple(summary["class_labels"]) == DIGIT_LABELS


def test_public_dataset_returns_model_ready_tensor(tmp_path: Path) -> None:
    """A manifest image becomes a normalized RGB tensor and class index."""
    dataset_root = tmp_path / "external"
    create_digit_folders(dataset_root)
    paths = prepare_kaggle_manifests(dataset_root, tmp_path / "manifests")
    dataset = PublicGestureDataset(
        paths["train"],
        dataset_root,
        DIGIT_LABELS,
        training=False,
        preprocess_config=PreprocessConfig(image_size=64),
    )
    tensor, label = dataset[0]
    assert tensor.shape == (3, 64, 64)
    assert 0 <= label <= 9


def test_manifest_paths_are_relative(tmp_path: Path) -> None:
    """Generated manifests remain movable with their extracted dataset."""
    dataset_root = tmp_path / "external"
    create_digit_folders(dataset_root)
    paths = prepare_kaggle_manifests(dataset_root, tmp_path / "manifests")
    with paths["train"].open(newline="", encoding="utf-8") as handle:
        first = next(csv.DictReader(handle))
    assert not Path(first["relative_path"]).is_absolute()
