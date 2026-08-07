"""Tests for calibration, rejection, and release acceptance gates."""

import numpy as np

from jutsu_battle.number_identifier.evaluate import (
    AcceptanceThresholds,
    acceptance_results,
    expected_calibration_error,
    unknown_false_acceptance_rate,
)


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    """Perfect one-hot predictions have no calibration gap."""
    probabilities = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    error, bins = expected_calibration_error(
        probabilities, np.asarray([0, 1]), bins=2
    )
    assert error == 0.0
    assert sum(item["count"] for item in bins) == 2


def test_unknown_false_acceptance_counts_numeric_predictions() -> None:
    """Unknown samples predicted as numbers are explicitly measured."""
    targets = np.asarray([11, 11, 2])
    predictions = np.asarray([11, 4, 2])
    assert unknown_false_acceptance_rate(targets, predictions) == 0.5


def test_smoke_checkpoint_cannot_pass_acceptance() -> None:
    """Generated fixtures can verify code but can never authorize release."""
    metrics = {
        "macro_f1": 1.0,
        "minimum_per_class_recall": 1.0,
        "unknown_false_acceptance_rate": 0.0,
        "latency_ms": {"p95": 1.0},
    }
    result = acceptance_results(
        metrics, AcceptanceThresholds(), smoke_only=True
    )
    assert result["passed"] is False
    assert result["checks"]["not_smoke_checkpoint"] is False
