"""Tests for production release guards and artifact bundling."""

import json
from pathlib import Path

import pytest
import torch

from jutsu_battle.number_identifier.release import (
    ReleaseValidationError,
    create_release_bundle,
    validate_release_inputs,
)
from jutsu_battle.number_identifier.spec import CLASS_LABELS


def write_release_fixtures(
    root: Path,
    *,
    smoke_only: bool,
    passed: bool,
) -> tuple[Path, Path]:
    """Write minimal trusted checkpoint and evaluation fixture files."""
    checkpoint = root / "model.pt"
    torch.save(
        {
            "model_name": "mini_cnn",
            "class_labels": CLASS_LABELS,
            "smoke_only": smoke_only,
        },
        checkpoint,
    )
    evaluation_directory = root / "evaluation"
    evaluation_directory.mkdir()
    evaluation = {
        "model_name": "mini_cnn",
        "smoke_only": smoke_only,
        "acceptance": {
            "passed": passed,
            "checks": {"macro_f1": passed},
        },
    }
    (evaluation_directory / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    return checkpoint, evaluation_directory


def test_release_rejects_smoke_checkpoint(tmp_path: Path) -> None:
    """Synthetic checkpoint metadata cannot enter a production bundle."""
    checkpoint, evaluation = write_release_fixtures(
        tmp_path, smoke_only=True, passed=True
    )
    with pytest.raises(ReleaseValidationError, match="Smoke checkpoints"):
        validate_release_inputs(checkpoint, evaluation / "evaluation.json")


def test_release_rejects_failed_acceptance(tmp_path: Path) -> None:
    """A real-data checkpoint still requires every evaluation gate."""
    checkpoint, evaluation = write_release_fixtures(
        tmp_path, smoke_only=False, passed=False
    )
    with pytest.raises(ReleaseValidationError, match="Evaluation gates"):
        validate_release_inputs(checkpoint, evaluation / "evaluation.json")


def test_release_bundle_contains_checksum_and_reports(tmp_path: Path) -> None:
    """Eligible artifacts produce a self-describing directory and ZIP archive."""
    checkpoint, evaluation = write_release_fixtures(
        tmp_path, smoke_only=False, passed=True
    )
    directory, archive = create_release_bundle(
        checkpoint,
        evaluation,
        tmp_path / "releases",
        tag="test-release",
    )
    assert archive.is_file()
    assert (directory / "number_identifier.pt").is_file()
    manifest = json.loads(
        (directory / "release_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["model_sha256"]) == 64
