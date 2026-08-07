"""Tests for temporal number stability and duplicate-event suppression."""

import pytest

from jutsu_battle.number_identifier.predictor import (
    NumberPrediction,
    PredictionStabilizer,
)


def prediction(label: str, confidence: float = 0.9) -> NumberPrediction:
    """Create a minimal frame-level prediction fixture."""
    return NumberPrediction(
        label=label,
        confidence=confidence,
        probabilities=tuple(0.0 for _ in range(12)),
        hand_count=1,
        inference_ms=1.0,
    )


def test_stabilizer_emits_once_for_a_held_pose() -> None:
    """Stable votes confirm once and holding does not generate duplicates."""
    stabilizer = PredictionStabilizer(
        window_size=3,
        required_votes=2,
        stable_duration_ms=100,
        release_duration_ms=50,
    )
    assert stabilizer.update(prediction("3"), 0) is None
    assert stabilizer.update(prediction("3"), 20) is None
    event = stabilizer.update(prediction("3"), 130)
    assert event is not None
    assert event.label == "3"
    assert stabilizer.update(prediction("3"), 250) is None


def test_stabilizer_rearms_after_unknown_release() -> None:
    """A neutral interval allows the same number to be confirmed again."""
    stabilizer = PredictionStabilizer(
        window_size=2,
        required_votes=1,
        stable_duration_ms=0,
        release_duration_ms=50,
    )
    assert stabilizer.update(prediction("2"), 0) is None
    assert stabilizer.update(prediction("2"), 1) is not None
    assert stabilizer.update(prediction("unknown"), 10) is None
    assert stabilizer.update(prediction("unknown"), 70) is None
    assert stabilizer.update(prediction("2"), 80) is None
    assert stabilizer.update(prediction("2"), 81) is not None


def test_stabilizer_validates_vote_configuration() -> None:
    """Impossible vote requirements fail at construction time."""
    with pytest.raises(ValueError):
        PredictionStabilizer(window_size=3, required_votes=4)
