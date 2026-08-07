"""Generate non-production fixture images for end-to-end pipeline smoke tests."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from jutsu_battle.number_identifier.dataset import CAPTURE_FIELDS, build_manifests
from jutsu_battle.number_identifier.spec import CLASS_LABELS


def render_fixture(
    label: str, subject_index: int, variation: int
) -> np.ndarray[Any, Any]:
    """Render a deterministic colored pattern; it is not realistic hand data."""
    image = np.full((192, 256, 3), 25 + subject_index * 18, dtype=np.uint8)
    seed = subject_index * 10_000 + CLASS_LABELS.index(label) * 100 + variation
    generator = np.random.default_rng(seed)
    color = tuple(int(value) for value in generator.integers(80, 240, size=3))
    if label == "unknown":
        for offset in range(0, 180, 20):
            cv2.line(image, (30 + offset, 25), (10 + offset, 165), color, 5)
    else:
        count = int(label)
        cv2.rectangle(image, (60, 105), (196, 175), color, -1)
        for index in range(count):
            x = 62 + index * 13
            cv2.rectangle(image, (x, 35 + variation), (x + 8, 112), color, -1)
        cv2.putText(
            image,
            label,
            (205, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    noise = generator.normal(0, 3, image.shape).astype(np.int16)
    result = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cast(np.ndarray[Any, Any], result)


def generate_smoke_dataset(root: Path, *, overwrite: bool = False) -> dict[str, Path]:
    """Create a complete three-subject fixture dataset and split manifests."""
    if root.exists() and overwrite:
        shutil.rmtree(root)
    capture_root = root / "capture"
    manifest_directory = root / "manifests"
    capture_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []
    for subject_index in range(3):
        subject = f"smoke_subject_{subject_index + 1}"
        for label in CLASS_LABELS:
            for variation in range(2):
                clip_id = f"{subject}_{label}_{variation}"
                relative = Path(subject) / "session_01" / label / clip_id / "frame.jpg"
                path = capture_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                fixture = render_fixture(label, subject_index, variation)
                if not cv2.imwrite(str(path), fixture):
                    raise OSError(f"Unable to write smoke fixture: {path}")
                rows.append(
                    {
                        "sample_id": f"{clip_id}_frame",
                        "relative_path": relative.as_posix(),
                        "label": label,
                        "subject_id": subject,
                        "session_id": "session_01",
                        "clip_id": clip_id,
                        "hand_count": (
                            1 if label in (*CLASS_LABELS[:6], "unknown") else 2
                        ),
                        "timestamp_ms": 1000 + variation,
                        "crop_x1": 0,
                        "crop_y1": 0,
                        "crop_x2": 256,
                        "crop_y2": 192,
                    }
                )
    capture_manifest = capture_root / "capture_manifest.csv"
    with capture_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAPTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return build_manifests(
        capture_manifest,
        capture_root,
        manifest_directory,
        manifest_directory / "number_subject_splits.json",
        allow_small=True,
        duplicate_distance=0,
    )


def main() -> int:
    """Generate smoke fixtures from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/phase1_smoke"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = generate_smoke_dataset(args.root, overwrite=args.overwrite)
    for split, path in paths.items():
        print(f"{split}: {path}")
    print("WARNING: generated patterns are for smoke testing, not model evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
