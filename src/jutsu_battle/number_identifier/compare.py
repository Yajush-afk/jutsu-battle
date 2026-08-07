"""Train image and landmark baselines under one comparison contract."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from jutsu_battle.number_identifier.landmarks import (
    LandmarkFeatureDataset,
    extract_landmark_cache,
)
from jutsu_battle.number_identifier.model_asset import ensure_model_asset
from jutsu_battle.number_identifier.models import create_model, parameter_count
from jutsu_battle.number_identifier.train import (
    TrainingConfig,
    run_epoch,
    save_checkpoint,
    seed_everything,
    select_device,
    train_model,
)


def train_landmark_model(
    train_cache: Path,
    validation_cache: Path,
    output_directory: Path,
    config: TrainingConfig,
    *,
    device_name: str = "auto",
    smoke_only: bool = False,
) -> Path:
    """Train the landmark MLP using the same optimizer and metric contract."""
    seed_everything(config.random_seed)
    device = select_device(device_name)
    train_dataset = LandmarkFeatureDataset(train_cache)
    validation_dataset = LandmarkFeatureDataset(validation_cache)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    model = create_model("landmark_mlp").to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    output_directory.mkdir(parents=True, exist_ok=True)
    best_path = output_directory / "best.pt"
    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, config.maximum_epochs + 1):
        training = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        with torch.inference_mode():
            validation = run_epoch(
                model, validation_loader, criterion, device, optimizer=None
            )
        print(
            f"model=landmark_mlp epoch={epoch} train_loss={training.loss:.4f} "
            f"val_f1={validation.macro_f1:.4f}"
        )
        if validation.macro_f1 > best_f1:
            best_f1 = validation.macro_f1
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                config=config,
                epoch=epoch,
                metrics=validation,
                smoke_only=smoke_only,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.early_stopping_patience:
            break
    return best_path


def checkpoint_summary(path: Path) -> dict[str, Any]:
    """Extract comparison-safe metadata from a trusted project checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = create_model(str(checkpoint["model_name"]))
    return {
        "model_name": checkpoint["model_name"],
        "checkpoint": path.as_posix(),
        "validation": checkpoint["metrics"],
        "parameter_count": parameter_count(model),
        "smoke_only": checkpoint["smoke_only"],
    }


def run_comparison(
    train_manifest: Path,
    validation_manifest: Path,
    capture_root: Path,
    output_directory: Path,
    *,
    config: TrainingConfig,
    device_name: str,
    smoke_only: bool,
    include_pretrained: bool,
) -> Path:
    """Train all configured baselines and write a machine-readable comparison."""
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints: list[Path] = []
    started_at = time.perf_counter()
    mini_config = TrainingConfig(**{**asdict(config), "model_name": "mini_cnn"})
    checkpoints.append(
        train_model(
            train_manifest,
            validation_manifest,
            capture_root,
            output_directory / "mini_cnn",
            mini_config,
            device_name=device_name,
            smoke_only=smoke_only,
        )
    )
    mobile_config = TrainingConfig(
        **{
            **asdict(config),
            "model_name": "mobilenet_v3_small",
            "pretrained": include_pretrained,
        }
    )
    checkpoints.append(
        train_model(
            train_manifest,
            validation_manifest,
            capture_root,
            output_directory / "mobilenet_v3_small",
            mobile_config,
            device_name=device_name,
            smoke_only=smoke_only,
        )
    )

    model_asset = ensure_model_asset()
    train_cache = extract_landmark_cache(
        train_manifest,
        capture_root,
        model_asset,
        output_directory / "landmarks_train.npz",
    )
    validation_cache = extract_landmark_cache(
        validation_manifest,
        capture_root,
        model_asset,
        output_directory / "landmarks_validation.npz",
    )
    landmark_config = TrainingConfig(
        **{**asdict(config), "model_name": "landmark_mlp", "pretrained": False}
    )
    checkpoints.append(
        train_landmark_model(
            train_cache,
            validation_cache,
            output_directory / "landmark_mlp",
            landmark_config,
            device_name=device_name,
            smoke_only=smoke_only,
        )
    )
    summaries = [checkpoint_summary(path) for path in checkpoints]
    best = max(summaries, key=lambda item: float(item["validation"]["macro_f1"]))
    report = {
        "schema_version": 1,
        "smoke_only": smoke_only,
        "elapsed_seconds": time.perf_counter() - started_at,
        "models": summaries,
        "selected_model": best["model_name"],
    }
    report_path = output_directory / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    """Run all comparison models from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    config = TrainingConfig(
        maximum_epochs=args.epochs,
        early_stopping_patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    report = run_comparison(
        args.train_manifest,
        args.validation_manifest,
        args.capture_root,
        args.output_directory,
        config=config,
        device_name=args.device,
        smoke_only=args.smoke_only,
        include_pretrained=not args.no_pretrained,
    )
    print(f"Comparison report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

