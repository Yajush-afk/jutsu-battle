"""Tests for the number-data capture state and metadata writer."""

from pathlib import Path

import numpy as np
import pytest

from jutsu_battle.number_identifier.capture import (
    CaptureConfig,
    ClipWriter,
    delete_captured_clip,
    is_capture_ready,
    sanitize_identifier,
)
from jutsu_battle.number_identifier.hand_detector import (
    BoundingBox,
    DetectedHand,
    HandDetection,
    LandmarkPoint,
)


def detection_with_hands(count: int) -> HandDetection:
    """Build a small deterministic detection fixture."""
    hand = DetectedHand(
        landmarks=tuple(LandmarkPoint(0.5, 0.5, 0.0) for _ in range(21)),
        handedness="Left",
        score=0.9,
    )
    return HandDetection(hands=tuple(hand for _ in range(count)))


@pytest.mark.parametrize("value", ["subject_001", "session-02", "A12"])
def test_sanitize_identifier_accepts_safe_path_components(value: str) -> None:
    """Safe collection identifiers pass unchanged."""
    assert value == sanitize_identifier(value, "identifier")


@pytest.mark.parametrize("value", ["", "../escape", "two words", "a/b"])
def test_sanitize_identifier_rejects_unsafe_values(value: str) -> None:
    """Unsafe or empty collection identifiers cannot become directories."""
    with pytest.raises(ValueError):
        sanitize_identifier(value, "identifier")


def test_capture_readiness_enforces_canonical_hand_count() -> None:
    """One/two-hand labels and unknown enforce their documented counts."""
    one_hand = detection_with_hands(1)
    two_hands = detection_with_hands(2)
    assert is_capture_ready("5", one_hand)
    assert not is_capture_ready("5", two_hands)
    assert is_capture_ready("6", two_hands)
    assert not is_capture_ready("6", one_hand)
    assert is_capture_ready("unknown", one_hand)
    assert is_capture_ready("unknown", two_hands)


def test_clip_writer_persists_image_and_manifest(tmp_path: Path) -> None:
    """One sampled frame produces an image, clip JSON, and CSV record."""
    config = CaptureConfig(
        subject_id="subject_001",
        session_id="session_01",
        output_root=tmp_path,
        capture_fps=1000.0,
    )
    writer = ClipWriter(config, "3")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    wrote = writer.maybe_write(
        frame,
        detection_with_hands(1),
        BoundingBox(1, 2, 30, 31),
    )
    directory = writer.finalize()

    assert wrote
    assert len(list(directory.glob("*.jpg"))) == 1
    assert (directory / "clip.json").is_file()
    manifest = (tmp_path / "capture_manifest.csv").read_text(encoding="utf-8")
    assert "subject_001" in manifest
    assert ",3," in manifest

    delete_captured_clip(config, directory)
    assert not directory.exists()
    updated_manifest = (tmp_path / "capture_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert "subject_001" not in updated_manifest
