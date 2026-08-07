"""Tests for number manifests, splits, preprocessing, and Torch datasets."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from jutsu_battle.number_identifier.dataset import (
    CAPTURE_FIELDS,
    DatasetValidationError,
    NumberImageDataset,
    build_manifests,
    create_assignments,
    read_split_manifest,
)
from jutsu_battle.number_identifier.hand_detector import BoundingBox
from jutsu_battle.number_identifier.preprocessing import (
    ImagePreprocessingError,
    crop_pad_resize_rgb,
)


def write_capture_fixture(root: Path) -> Path:
    """Create three subject-exclusive image records for smoke testing."""
    rows: list[dict[str, str | int]] = []
    for subject_index, subject in enumerate(("s1", "s2", "s3")):
        relative = Path(subject) / "session_01" / "0" / f"clip_{subject}" / "a.jpg"
        path = root / relative
        path.parent.mkdir(parents=True)
        image = np.full((40, 60, 3), 30 + subject_index * 80, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        rows.append(
            {
                "sample_id": f"sample_{subject}",
                "relative_path": relative.as_posix(),
                "label": "0",
                "subject_id": subject,
                "session_id": "session_01",
                "clip_id": f"clip_{subject}",
                "hand_count": 1,
                "timestamp_ms": 1000,
                "crop_x1": 5,
                "crop_y1": 5,
                "crop_x2": 55,
                "crop_y2": 35,
            }
        )
    manifest = root / "capture_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_create_assignments_is_subject_exclusive_and_deterministic() -> None:
    """A fixed seed always locks each subject into exactly one split."""
    first = create_assignments((f"s{index}" for index in range(15)), seed=42)
    second = create_assignments((f"s{index}" for index in range(15)), seed=42)
    assert first == second
    assert not set(first.train) & set(first.validation)
    assert not set(first.train) & set(first.test)
    assert not set(first.validation) & set(first.test)
    assert len(first.train) == 10
    assert len(first.validation) == 2
    assert len(first.test) == 3


def test_small_assignment_requires_explicit_smoke_flag() -> None:
    """Production manifest creation rejects an undersized participant pool."""
    with pytest.raises(DatasetValidationError):
        create_assignments(("s1", "s2", "s3"))
    assignment = create_assignments(("s1", "s2", "s3"), allow_small=True)
    assert len(assignment.train) == 1


def test_crop_pad_resize_preserves_target_shape() -> None:
    """A rectangular hand region becomes a square RGB model input."""
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    output = crop_pad_resize_rgb(image, BoundingBox(10, 5, 70, 35), 160)
    assert output.shape == (160, 160, 3)


def test_crop_rejects_empty_box() -> None:
    """Invalid crop geometry fails before model input is created."""
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    with pytest.raises(ImagePreprocessingError):
        crop_pad_resize_rgb(image, BoundingBox(20, 20, 20, 30), 160)


def test_build_manifests_and_load_dataset(tmp_path: Path) -> None:
    """Captured metadata becomes three loadable subject-exclusive datasets."""
    capture_root = tmp_path / "capture"
    capture_root.mkdir()
    capture_manifest = write_capture_fixture(capture_root)
    output = tmp_path / "manifests"
    paths = build_manifests(
        capture_manifest,
        capture_root,
        output,
        output / "assignments.json",
        allow_small=True,
        duplicate_distance=0,
    )

    subjects = {
        split: {sample.subject_id for sample in read_split_manifest(path)}
        for split, path in paths.items()
    }
    assert not subjects["train"] & subjects["validation"]
    assert not subjects["train"] & subjects["test"]

    dataset = NumberImageDataset(paths["train"], capture_root, training=False)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 160, 160)
    assert label == 0
