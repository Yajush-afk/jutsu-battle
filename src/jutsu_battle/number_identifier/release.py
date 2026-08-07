"""Create and optionally publish a guarded number-identifier release bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from jutsu_battle.number_identifier.model_asset import sha256_file

DEFAULT_TAG = "v0.1.0-number-identifier"


class ReleaseValidationError(ValueError):
    """Raised when artifacts are not eligible for a production release."""


def validate_release_inputs(
    checkpoint_path: Path,
    evaluation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Refuse smoke, failed, or mismatched model/evaluation artifacts."""
    if not checkpoint_path.is_file():
        raise ReleaseValidationError(f"Checkpoint not found: {checkpoint_path}")
    if not evaluation_path.is_file():
        raise ReleaseValidationError(f"Evaluation not found: {evaluation_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if bool(checkpoint.get("smoke_only", True)):
        raise ReleaseValidationError("Smoke checkpoints cannot be released")
    if bool(evaluation.get("smoke_only", True)):
        raise ReleaseValidationError("Smoke evaluations cannot be released")
    acceptance = evaluation.get("acceptance", {})
    if not bool(acceptance.get("passed", False)):
        failed = [
            name
            for name, passed in acceptance.get("checks", {}).items()
            if not passed
        ]
        raise ReleaseValidationError(f"Evaluation gates failed: {failed}")
    if evaluation.get("model_name") != checkpoint.get("model_name"):
        raise ReleaseValidationError("Evaluation and checkpoint model names differ")
    return checkpoint, evaluation


def current_commit() -> str:
    """Return the source commit included in release metadata."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_release_bundle(
    checkpoint_path: Path,
    evaluation_directory: Path,
    output_root: Path,
    *,
    tag: str = DEFAULT_TAG,
) -> tuple[Path, Path]:
    """Copy eligible artifacts into a checksummed archive."""
    evaluation_path = evaluation_directory / "evaluation.json"
    checkpoint, evaluation = validate_release_inputs(
        checkpoint_path, evaluation_path
    )
    bundle_directory = output_root / tag
    if bundle_directory.exists():
        raise ReleaseValidationError(
            f"Release output already exists; choose a new tag: {bundle_directory}"
        )
    bundle_directory.mkdir(parents=True)
    bundled_checkpoint = bundle_directory / "number_identifier.pt"
    shutil.copy2(checkpoint_path, bundled_checkpoint)
    reports_directory = bundle_directory / "reports"
    shutil.copytree(evaluation_directory, reports_directory)
    for documentation in (
        Path("docs/number-gesture-spec.md"),
        Path("docs/number-dataset-card.md"),
        Path("docs/number-model-card.md"),
    ):
        if documentation.is_file():
            shutil.copy2(documentation, bundle_directory / documentation.name)
    manifest = {
        "schema_version": 1,
        "tag": tag,
        "source_commit": current_commit(),
        "model_name": checkpoint["model_name"],
        "class_labels": checkpoint["class_labels"],
        "model_sha256": sha256_file(bundled_checkpoint),
        "evaluation_acceptance": evaluation["acceptance"],
    }
    (bundle_directory / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    archive_base = output_root / tag
    archive = Path(
        shutil.make_archive(
            archive_base.as_posix(),
            "zip",
            root_dir=bundle_directory,
        )
    )
    return bundle_directory, archive


def publish_release(tag: str, archive: Path) -> None:
    """Create the GitHub release only after local artifact validation."""
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            archive.as_posix(),
            "--title",
            "Two-Hand Number Identifier",
            "--notes",
            "Guarded Phase 1 number-identifier release.",
        ],
        check=True,
    )


def main() -> int:
    """Validate, bundle, and optionally publish production artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/releases"))
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    try:
        directory, archive = create_release_bundle(
            args.checkpoint,
            args.evaluation_directory,
            args.output_root,
            tag=args.tag,
        )
        print(f"Release directory: {directory}")
        print(f"Release archive: {archive}")
        if args.publish:
            publish_release(args.tag, archive)
    except (OSError, ReleaseValidationError, subprocess.CalledProcessError) as error:
        print(f"Release refused: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
