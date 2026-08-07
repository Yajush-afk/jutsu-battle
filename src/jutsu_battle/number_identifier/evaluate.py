"""Evaluate a number classifier and enforce production acceptance gates."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from jutsu_battle.number_identifier.dataset import NumberImageDataset
from jutsu_battle.number_identifier.models import load_model_checkpoint
from jutsu_battle.number_identifier.preprocessing import PreprocessConfig
from jutsu_battle.number_identifier.spec import CLASS_LABELS
from jutsu_battle.number_identifier.train import select_device


@dataclass(frozen=True, slots=True)
class AcceptanceThresholds:
    """Quality and latency required before a model can be released."""

    macro_f1_minimum: float = 0.90
    per_class_recall_minimum: float = 0.80
    unknown_false_acceptance_maximum: float = 0.10
    p95_inference_ms_maximum: float = 50.0


def expected_calibration_error(
    probabilities: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any],
    *,
    bins: int = 10,
) -> tuple[float, list[dict[str, float]]]:
    """Calculate top-label expected calibration error and bin statistics."""
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    statistics: list[dict[str, float]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        if count == 0:
            statistics.append(
                {
                    "lower": float(lower),
                    "upper": float(upper),
                    "count": 0.0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                }
            )
            continue
        bin_accuracy = float(correctness[mask].mean())
        bin_confidence = float(confidences[mask].mean())
        weight = count / len(targets)
        error += weight * abs(bin_accuracy - bin_confidence)
        statistics.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": float(count),
                "accuracy": bin_accuracy,
                "confidence": bin_confidence,
            }
        )
    return error, statistics


def unknown_false_acceptance_rate(
    targets: np.ndarray[Any, Any], predictions: np.ndarray[Any, Any]
) -> float:
    """Return the fraction of unknown samples incorrectly accepted as numbers."""
    unknown_index = CLASS_LABELS.index("unknown")
    unknown_mask = targets == unknown_index
    unknown_count = int(unknown_mask.sum())
    if unknown_count == 0:
        return 1.0
    return float((predictions[unknown_mask] != unknown_index).mean())


def acceptance_results(
    metrics: dict[str, Any],
    thresholds: AcceptanceThresholds,
    *,
    smoke_only: bool,
) -> dict[str, Any]:
    """Evaluate every release gate without concealing individual failures."""
    checks = {
        "not_smoke_checkpoint": not smoke_only,
        "macro_f1": metrics["macro_f1"] >= thresholds.macro_f1_minimum,
        "minimum_per_class_recall": (
            metrics["minimum_per_class_recall"]
            >= thresholds.per_class_recall_minimum
        ),
        "unknown_false_acceptance": (
            metrics["unknown_false_acceptance_rate"]
            <= thresholds.unknown_false_acceptance_maximum
        ),
        "p95_inference_latency": (
            metrics["latency_ms"]["p95"]
            <= thresholds.p95_inference_ms_maximum
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": asdict(thresholds),
    }


def synchronize(device: torch.device) -> None:
    """Synchronize accelerator kernels before timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[float]]:
    """Collect probabilities, targets, and per-sample inference latency."""
    probabilities: list[np.ndarray[Any, Any]] = []
    targets: list[np.ndarray[Any, Any]] = []
    latencies: list[float] = []
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            synchronize(device)
            started = time.perf_counter()
            logits = model(inputs)
            synchronize(device)
            elapsed_ms = (time.perf_counter() - started) * 1000
            batch_size = inputs.size(0)
            latencies.extend([elapsed_ms / batch_size] * batch_size)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(probabilities), np.concatenate(targets), latencies


def grouped_accuracy(
    groups: list[str],
    targets: np.ndarray[Any, Any],
    predictions: np.ndarray[Any, Any],
) -> dict[str, dict[str, float | int]]:
    """Return sample count and accuracy for each metadata group."""
    result: dict[str, dict[str, float | int]] = {}
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        mask = group_array == group
        result[group] = {
            "count": int(mask.sum()),
            "accuracy": float(accuracy_score(targets[mask], predictions[mask])),
        }
    return result


def write_confusion_artifacts(
    matrix: np.ndarray[Any, Any], output_directory: Path
) -> None:
    """Write a machine-readable CSV and human-readable confusion heatmap."""
    csv_path = output_directory / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("actual/predicted", *CLASS_LABELS))
        for label, row in zip(CLASS_LABELS, matrix.tolist(), strict=True):
            writer.writerow((label, *row))
    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.tight_layout()
    figure.savefig(output_directory / "confusion_matrix.png", dpi=160)
    plt.close(figure)


def write_calibration_plot(
    bins: list[dict[str, float]], output_directory: Path
) -> None:
    """Write a reliability diagram from calculated calibration bins."""
    populated = [item for item in bins if item["count"] > 0]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    if populated:
        axis.plot(
            [item["confidence"] for item in populated],
            [item["accuracy"] for item in populated],
            marker="o",
            label="Model",
        )
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "calibration.png", dpi=160)
    plt.close(figure)


def evaluate_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    capture_root: Path,
    output_directory: Path,
    *,
    device_name: str = "auto",
    batch_size: int = 64,
    num_workers: int = 2,
    thresholds: AcceptanceThresholds | None = None,
) -> Path:
    """Evaluate one image classifier and write all release artifacts."""
    thresholds = thresholds or AcceptanceThresholds()
    device = select_device(device_name)
    model, checkpoint = load_model_checkpoint(
        checkpoint_path.as_posix(), device=device
    )
    preprocess_data = checkpoint["preprocess"]
    preprocess = PreprocessConfig(
        image_size=int(preprocess_data["image_size"]),
        mean=tuple(preprocess_data["mean"]),
        std=tuple(preprocess_data["std"]),
    )
    dataset = NumberImageDataset(
        manifest_path,
        capture_root,
        training=False,
        preprocess_config=preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    probabilities, targets, latencies = collect_predictions(model, loader, device)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(CLASS_LABELS)),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    calibration_error, calibration_bins = expected_calibration_error(
        probabilities, targets
    )
    latency_array = np.asarray(latencies)
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": checkpoint_path.as_posix(),
        "manifest": manifest_path.as_posix(),
        "model_name": checkpoint["model_name"],
        "smoke_only": bool(checkpoint["smoke_only"]),
        "sample_count": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "minimum_per_class_recall": float(recall.min()),
        "unknown_false_acceptance_rate": unknown_false_acceptance_rate(
            targets, predictions
        ),
        "expected_calibration_error": calibration_error,
        "latency_ms": {
            "median": float(np.median(latency_array)),
            "p95": float(np.percentile(latency_array, 95)),
            "mean": float(latency_array.mean()),
        },
        "model_size_megabytes": checkpoint_path.stat().st_size / (1024 * 1024),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(CLASS_LABELS)
        },
        "breakdowns": {
            "subject": grouped_accuracy(
                [sample.subject_id for sample in dataset.samples], targets, predictions
            ),
            "hand_count": grouped_accuracy(
                [str(sample.hand_count) for sample in dataset.samples],
                targets,
                predictions,
            ),
        },
        "calibration_bins": calibration_bins,
    }
    metrics["acceptance"] = acceptance_results(
        metrics,
        thresholds,
        smoke_only=bool(checkpoint["smoke_only"]),
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        targets,
        predictions,
        labels=np.arange(len(CLASS_LABELS)),
        target_names=CLASS_LABELS,
        zero_division=0,
        output_dict=True,
    )
    (output_directory / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    matrix = confusion_matrix(
        targets, predictions, labels=np.arange(len(CLASS_LABELS))
    )
    write_confusion_artifacts(matrix, output_directory)
    write_calibration_plot(calibration_bins, output_directory)
    metrics_path = output_directory / "evaluation.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics_path


def main() -> int:
    """Evaluate a trusted project checkpoint from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    metrics_path = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        args.capture_root,
        args.output_directory,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print(f"Evaluation report: {metrics_path}")
    print(f"Acceptance passed: {metrics['acceptance']['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
