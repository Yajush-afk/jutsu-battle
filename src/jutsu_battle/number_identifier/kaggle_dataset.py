"""Prepare reproducible manifests from the extracted Kaggle gesture dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset

from jutsu_battle.number_identifier.preprocessing import (
    PreprocessConfig,
    build_transform,
)

DIGIT_LABELS: tuple[str, ...] = tuple(str(value) for value in range(10))
ALPHABET_LABELS: tuple[str, ...] = tuple(chr(value) for value in range(65, 91))
ALL_LABELS: tuple[str, ...] = (*DIGIT_LABELS, *ALPHABET_LABELS)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_FIELDS = ("relative_path", "label", "split")


class KaggleDatasetError(ValueError):
    """Raised when the extracted public dataset violates its contract."""


@dataclass(frozen=True, slots=True)
class PublicImageSample:
    """One public gesture image and its stable class label."""

    relative_path: str
    label: str
    split: str


def labels_for_mode(mode: str) -> tuple[str, ...]:
    """Resolve a user-facing label mode into deterministic class order."""
    if mode == "digits":
        return DIGIT_LABELS
    if mode == "all":
        return ALL_LABELS
    raise KaggleDatasetError("labels must be either 'digits' or 'all'")


def find_class_directories(
    dataset_root: Path, labels: tuple[str, ...]
) -> dict[str, Path]:
    """Find one non-empty image directory for every requested label."""
    if not dataset_root.is_dir():
        raise KaggleDatasetError(
            f"Dataset directory not found: {dataset_root.resolve()}"
        )
    candidates: dict[str, list[Path]] = {label: [] for label in labels}
    for directory in (dataset_root, *dataset_root.rglob("*")):
        if not directory.is_dir() or directory.name not in candidates:
            continue
        if any(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            for child in directory.iterdir()
        ):
            candidates[directory.name].append(directory)
    missing = [label for label, paths in candidates.items() if not paths]
    ambiguous = {label: paths for label, paths in candidates.items() if len(paths) > 1}
    if missing:
        raise KaggleDatasetError(
            "Missing class folders: "
            + ", ".join(missing)
            + ". Point --dataset-root at the extracted dataset."
        )
    if ambiguous:
        details = "; ".join(
            f"{label}: {', '.join(path.as_posix() for path in paths)}"
            for label, paths in ambiguous.items()
        )
        raise KaggleDatasetError(f"Multiple image folders found per label: {details}")
    return {label: paths[0] for label, paths in candidates.items()}


def validate_image(path: Path) -> None:
    """Decode enough of an image to reject corrupt dataset files."""
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise KaggleDatasetError(f"Unreadable image: {path}") from error


def split_class_paths(
    paths: list[Path],
    *,
    seed: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, list[Path]]:
    """Deterministically split one class while preserving class balance."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise KaggleDatasetError(
            "Validation and test fractions must be between 0 and 1"
        )
    if validation_fraction + test_fraction >= 1:
        raise KaggleDatasetError(
            "Validation and test fractions must sum to less than 1"
        )
    if len(paths) < 10:
        raise KaggleDatasetError(
            f"Each class needs at least 10 images; found {len(paths)}"
        )
    shuffled = sorted(paths)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    return {
        "test": shuffled[:test_count],
        "validation": shuffled[test_count : test_count + validation_count],
        "train": shuffled[test_count + validation_count :],
    }


def prepare_kaggle_manifests(
    dataset_root: Path,
    output_directory: Path,
    *,
    label_mode: str = "digits",
    seed: int = 20260808,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
    verify_images: bool = True,
) -> dict[str, Path]:
    """Validate the extracted dataset and write balanced split manifests."""
    labels = labels_for_mode(label_mode)
    class_directories = find_class_directories(dataset_root, labels)
    split_rows: dict[str, list[PublicImageSample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    source_counts: dict[str, int] = {}
    for class_index, label in enumerate(labels):
        paths = sorted(
            path
            for path in class_directories[label].iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if verify_images:
            for path in paths:
                validate_image(path)
        source_counts[label] = len(paths)
        class_splits = split_class_paths(
            paths,
            seed=seed + class_index,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        for split, split_paths in class_splits.items():
            split_rows[split].extend(
                PublicImageSample(
                    relative_path=path.relative_to(dataset_root).as_posix(),
                    label=label,
                    split=split,
                )
                for path in split_paths
            )

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split, rows in split_rows.items():
        random.Random(seed).shuffle(rows)
        path = output_directory / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
        manifest_paths[split] = path
        split_counts[split] = dict(Counter(row.label for row in rows))

    summary = {
        "schema_version": 1,
        "source": "debabratakuiry/hand-sign-gesture-dataset-az-and-09-25k-images",
        "dataset_root": dataset_root.resolve().as_posix(),
        "label_mode": label_mode,
        "class_labels": labels,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "source_counts": source_counts,
        "split_counts": split_counts,
        "warning": (
            "The public metadata has no participant IDs; image-level splits may "
            "overestimate generalization when nearby frames share a source session."
        ),
    }
    summary_path = output_directory / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_paths["summary"] = summary_path
    return manifest_paths


def read_public_manifest(path: Path) -> list[PublicImageSample]:
    """Read and validate a generated public-dataset manifest."""
    if not path.is_file():
        raise KaggleDatasetError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise KaggleDatasetError(f"Manifest is empty: {path}")
    try:
        return [PublicImageSample(**row) for row in rows]
    except TypeError as error:
        raise KaggleDatasetError(f"Malformed manifest: {path}") from error


class PublicGestureDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset for folder-based Kaggle hand-sign images."""

    def __init__(
        self,
        manifest_path: Path,
        dataset_root: Path,
        class_labels: tuple[str, ...],
        *,
        training: bool,
        preprocess_config: PreprocessConfig,
    ) -> None:
        self.samples = read_public_manifest(manifest_path)
        self.dataset_root = dataset_root.resolve()
        self.class_to_index = {
            label: index for index, label in enumerate(class_labels)
        }
        sample_labels = {sample.label for sample in self.samples}
        unknown = sorted(sample_labels - self.class_to_index.keys())
        if unknown:
            raise KaggleDatasetError(f"Manifest contains unexpected labels: {unknown}")
        self.transform = build_transform(preprocess_config, training=training)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        path = (self.dataset_root / sample.relative_path).resolve()
        if not path.is_relative_to(self.dataset_root):
            raise KaggleDatasetError(f"Sample escapes dataset root: {path}")
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, UnidentifiedImageError) as error:
            raise KaggleDatasetError(f"Unable to decode image: {path}") from error
        return tensor, self.class_to_index[sample.label]


def build_parser() -> argparse.ArgumentParser:
    """Build the public dataset preparation CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "data/external/kaggle/hand-sign-gesture-dataset-az-09-25k-images"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("data/manifests/kaggle_digits"),
    )
    parser.add_argument("--labels", choices=("digits", "all"), default="digits")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--skip-image-verification", action="store_true")
    return parser


def main() -> int:
    """Prepare public-dataset manifests from the command line."""
    args = build_parser().parse_args()
    try:
        paths = prepare_kaggle_manifests(
            args.dataset_root,
            args.output_directory,
            label_mode=args.labels,
            seed=args.seed,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            verify_images=not args.skip_image_verification,
        )
    except KaggleDatasetError as error:
        print(f"Dataset preparation error: {error}")
        return 1
    for name, path in paths.items():
        print(f"{name:>10}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
