"""Validated manifests and PyTorch datasets for hand-number images."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from jutsu_battle.number_identifier.hand_detector import BoundingBox
from jutsu_battle.number_identifier.preprocessing import (
    PreprocessConfig,
    build_transform,
    crop_pad_resize_rgb,
    read_bgr_image,
    rgb_to_tensor,
)
from jutsu_battle.number_identifier.spec import CLASS_LABELS, class_to_index

CAPTURE_FIELDS = (
    "sample_id",
    "relative_path",
    "label",
    "subject_id",
    "session_id",
    "clip_id",
    "hand_count",
    "timestamp_ms",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
)
MANIFEST_FIELDS = (*CAPTURE_FIELDS, "split")


class DatasetValidationError(ValueError):
    """Raised when captured data violates the locked dataset contract."""


@dataclass(frozen=True, slots=True)
class NumberSample:
    """One validated model sample."""

    sample_id: str
    relative_path: str
    label: str
    subject_id: str
    session_id: str
    clip_id: str
    hand_count: int
    timestamp_ms: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int
    split: str

    @property
    def box(self) -> BoundingBox:
        """Return the stored full-frame crop coordinates."""
        return BoundingBox(self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2)

    @classmethod
    def from_mapping(cls, row: Mapping[str, str], *, split: str = "") -> NumberSample:
        """Parse and validate one CSV row."""
        missing = [field for field in CAPTURE_FIELDS if field not in row]
        if missing:
            raise DatasetValidationError(f"Missing manifest fields: {missing}")
        label = row["label"]
        if label not in CLASS_LABELS:
            raise DatasetValidationError(f"Unsupported label: {label}")
        try:
            return cls(
                sample_id=row["sample_id"],
                relative_path=row["relative_path"],
                label=label,
                subject_id=row["subject_id"],
                session_id=row["session_id"],
                clip_id=row["clip_id"],
                hand_count=int(row["hand_count"]),
                timestamp_ms=int(row["timestamp_ms"]),
                crop_x1=int(row["crop_x1"]),
                crop_y1=int(row["crop_y1"]),
                crop_x2=int(row["crop_x2"]),
                crop_y2=int(row["crop_y2"]),
                split=row.get("split", split),
            )
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(f"Invalid numeric metadata: {row}") from error


@dataclass(frozen=True, slots=True)
class SplitAssignments:
    """Locked participant membership for every dataset split."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int

    def split_for(self, subject_id: str) -> str:
        """Return the unique split containing a subject."""
        memberships = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        matches = [
            name for name, subjects in memberships.items() if subject_id in subjects
        ]
        if len(matches) != 1:
            raise DatasetValidationError(
                f"Subject {subject_id!r} must appear in exactly one split"
            )
        return matches[0]

    @property
    def subjects(self) -> set[str]:
        """Return every assigned participant."""
        return {*self.train, *self.validation, *self.test}


def read_capture_manifest(path: Path) -> list[NumberSample]:
    """Read captured frame metadata before split assignment."""
    if not path.is_file():
        raise DatasetValidationError(f"Capture manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise DatasetValidationError("Capture manifest is empty")
    return [NumberSample.from_mapping(row) for row in rows]


def create_assignments(
    subjects: Iterable[str],
    *,
    seed: int = 20260808,
    validation_count: int = 2,
    test_count: int = 3,
    minimum_subjects: int = 15,
    allow_small: bool = False,
) -> SplitAssignments:
    """Create deterministic subject-exclusive splits."""
    unique_subjects = sorted(set(subjects))
    if len(unique_subjects) < minimum_subjects and not allow_small:
        raise DatasetValidationError(
            f"Need at least {minimum_subjects} participants; "
            f"found {len(unique_subjects)}"
        )
    if allow_small and len(unique_subjects) < 3:
        raise DatasetValidationError("Smoke datasets still require at least 3 subjects")
    random.Random(seed).shuffle(unique_subjects)
    if allow_small and len(unique_subjects) < minimum_subjects:
        validation_count = 1
        test_count = 1
    validation = tuple(sorted(unique_subjects[:validation_count]))
    test_end = validation_count + test_count
    test = tuple(sorted(unique_subjects[validation_count:test_end]))
    train = tuple(sorted(unique_subjects[validation_count + test_count :]))
    if not train:
        raise DatasetValidationError("Training split cannot be empty")
    return SplitAssignments(train=train, validation=validation, test=test, seed=seed)


def save_or_validate_assignments(
    assignments: SplitAssignments,
    path: Path,
) -> SplitAssignments:
    """Persist new assignments or ensure an existing lock matches exactly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        existing = SplitAssignments(
            train=tuple(data["train"]),
            validation=tuple(data["validation"]),
            test=tuple(data["test"]),
            seed=int(data["seed"]),
        )
        if existing != assignments:
            raise DatasetValidationError(
                f"Existing split lock differs from requested assignments: {path}"
            )
        return existing
    path.write_text(json.dumps(asdict(assignments), indent=2) + "\n", encoding="utf-8")
    return assignments


def difference_hash(sample: NumberSample, capture_root: Path) -> int:
    """Return a compact perceptual hash for near-duplicate rejection."""
    image = read_bgr_image(capture_root / sample.relative_path)
    crop = crop_pad_resize_rgb(image, sample.box, 16)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    return int("".join("1" if value else "0" for value in bits.flat), 2)


def hamming_distance(left: int, right: int) -> int:
    """Return bit distance between two perceptual hashes."""
    return (left ^ right).bit_count()


def select_clip_samples(
    samples: Sequence[NumberSample],
    capture_root: Path,
    *,
    maximum_frames: int = 5,
    duplicate_distance: int = 3,
) -> list[NumberSample]:
    """Select temporally spaced, non-near-duplicate frames from a clip."""
    ordered = sorted(samples, key=lambda sample: sample.timestamp_ms)
    if len(ordered) <= maximum_frames:
        candidates = ordered
    else:
        indices = np.linspace(0, len(ordered) - 1, num=maximum_frames * 2, dtype=int)
        candidates = [ordered[index] for index in dict.fromkeys(indices.tolist())]
    selected: list[NumberSample] = []
    hashes: list[int] = []
    for candidate in candidates:
        candidate_hash = difference_hash(candidate, capture_root)
        nearest_distance = min(
            (hamming_distance(candidate_hash, value) for value in hashes),
            default=64,
        )
        if nearest_distance < duplicate_distance:
            continue
        selected.append(candidate)
        hashes.append(candidate_hash)
        if len(selected) == maximum_frames:
            break
    if not selected and ordered:
        selected.append(ordered[0])
    return selected


def validate_sample_files(samples: Sequence[NumberSample], capture_root: Path) -> None:
    """Ensure all manifest paths stay inside the capture root and exist."""
    root = capture_root.resolve()
    for sample in samples:
        path = (capture_root / sample.relative_path).resolve()
        if not path.is_relative_to(root):
            raise DatasetValidationError(f"Sample escapes capture root: {path}")
        if not path.is_file():
            raise DatasetValidationError(f"Sample image not found: {path}")


def build_manifests(
    capture_manifest: Path,
    capture_root: Path,
    output_directory: Path,
    assignment_path: Path,
    *,
    seed: int = 20260808,
    maximum_frames_per_clip: int = 5,
    duplicate_distance: int = 3,
    allow_small: bool = False,
) -> dict[str, Path]:
    """Validate captured data and write locked train/validation/test CSVs."""
    samples = read_capture_manifest(capture_manifest)
    validate_sample_files(samples, capture_root)
    assignments = create_assignments(
        (sample.subject_id for sample in samples),
        seed=seed,
        allow_small=allow_small,
    )
    assignments = save_or_validate_assignments(assignments, assignment_path)

    clips: dict[str, list[NumberSample]] = defaultdict(list)
    for sample in samples:
        clips[sample.clip_id].append(sample)
    selected = [
        sample
        for clip_samples in clips.values()
        for sample in select_clip_samples(
            clip_samples,
            capture_root,
            maximum_frames=maximum_frames_per_clip,
            duplicate_distance=duplicate_distance,
        )
    ]
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        path = output_directory / f"numbers_{split}.csv"
        split_samples = [
            sample
            for sample in selected
            if assignments.split_for(sample.subject_id) == split
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for sample in split_samples:
                writer.writerow({**asdict(sample), "split": split})
        paths[split] = path
    return paths


def read_split_manifest(path: Path) -> list[NumberSample]:
    """Read one generated split manifest."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [NumberSample.from_mapping(row) for row in csv.DictReader(handle)]


class NumberImageDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset backed by a generated split manifest."""

    def __init__(
        self,
        manifest_path: Path,
        capture_root: Path,
        *,
        training: bool,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.samples = read_split_manifest(manifest_path)
        if not self.samples:
            raise DatasetValidationError(f"Split manifest is empty: {manifest_path}")
        self.capture_root = capture_root
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.transform = build_transform(self.preprocess_config, training=training)
        self.label_indices = class_to_index()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        image = read_bgr_image(self.capture_root / sample.relative_path)
        rgb = crop_pad_resize_rgb(
            image,
            sample.box,
            self.preprocess_config.image_size,
        )
        return rgb_to_tensor(rgb, self.transform), self.label_indices[sample.label]


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-manifest command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=Path("data/raw/numbers"))
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        default=Path("data/raw/numbers/capture_manifest.csv"),
    )
    parser.add_argument("--output-directory", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--assignment-path",
        type=Path,
        default=Path("data/manifests/number_subject_splits.json"),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--allow-small", action="store_true")
    return parser


def main() -> int:
    """Build dataset manifests from captured frame metadata."""
    args = build_parser().parse_args()
    try:
        paths = build_manifests(
            args.capture_manifest,
            args.capture_root,
            args.output_directory,
            args.assignment_path,
            seed=args.seed,
            allow_small=args.allow_small,
        )
    except DatasetValidationError as error:
        print(f"Dataset error: {error}")
        return 1
    for split, path in paths.items():
        print(f"{split}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
