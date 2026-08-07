"""Train and checkpoint number-gesture image classifiers."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from jutsu_battle.number_identifier.dataset import NumberImageDataset
from jutsu_battle.number_identifier.models import create_model, parameter_count
from jutsu_battle.number_identifier.preprocessing import PreprocessConfig
from jutsu_battle.number_identifier.spec import CLASS_LABELS


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Reproducible baseline training configuration."""

    model_name: str = "mini_cnn"
    pretrained: bool = False
    image_size: int = 160
    batch_size: int = 32
    maximum_epochs: int = 50
    early_stopping_patience: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    random_seed: int = 20260808
    num_workers: int = 2


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Loss and classification quality from one full dataset pass."""

    loss: float
    accuracy: float
    macro_f1: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit device or select the best available accelerator."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None,
) -> EpochMetrics:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * inputs.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    sample_count = len(targets)
    if sample_count == 0:
        raise ValueError("Cannot run an epoch over an empty dataset")
    accuracy = sum(
        left == right for left, right in zip(predictions, targets, strict=True)
    )
    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=accuracy / sample_count,
        macro_f1=float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    config: TrainingConfig,
    epoch: int,
    metrics: EpochMetrics,
    smoke_only: bool,
) -> None:
    """Persist model state with all compatibility-critical metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "model_name": config.model_name,
            "class_labels": CLASS_LABELS,
            "preprocess": asdict(PreprocessConfig(config.image_size)),
            "training_config": asdict(config),
            "epoch": epoch,
            "metrics": asdict(metrics),
            "smoke_only": smoke_only,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def train_model(
    train_manifest: Path,
    validation_manifest: Path,
    capture_root: Path,
    output_directory: Path,
    config: TrainingConfig,
    *,
    device_name: str = "auto",
    smoke_only: bool = False,
) -> Path:
    """Train with early stopping and return the best checkpoint path."""
    seed_everything(config.random_seed)
    device = select_device(device_name)
    preprocess = PreprocessConfig(image_size=config.image_size)
    train_dataset = NumberImageDataset(
        train_manifest,
        capture_root,
        training=True,
        preprocess_config=preprocess,
    )
    validation_dataset = NumberImageDataset(
        validation_manifest,
        capture_root,
        training=False,
        preprocess_config=preprocess,
    )
    generator = torch.Generator().manual_seed(config.random_seed)
    loader_options: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **loader_options,
    )
    model = create_model(config.model_name, pretrained=config.pretrained).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_directory / "tensorboard"))
    best_path = output_directory / "best.pt"
    best_f1 = -1.0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    try:
        for epoch in range(1, config.maximum_epochs + 1):
            training = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
            )
            with torch.inference_mode():
                validation = run_epoch(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    optimizer=None,
                )
            for name, metrics in (("train", training), ("validation", validation)):
                writer.add_scalar(f"loss/{name}", metrics.loss, epoch)
                writer.add_scalar(f"accuracy/{name}", metrics.accuracy, epoch)
                writer.add_scalar(f"macro_f1/{name}", metrics.macro_f1, epoch)
            history.append(
                {
                    "epoch": epoch,
                    "train": asdict(training),
                    "validation": asdict(validation),
                }
            )
            print(
                f"epoch={epoch} train_loss={training.loss:.4f} "
                f"val_loss={validation.loss:.4f} val_f1={validation.macro_f1:.4f}"
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
    finally:
        writer.close()
    summary = {
        "schema_version": 1,
        "model_name": config.model_name,
        "parameter_count": parameter_count(model),
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started_at,
        "smoke_only": smoke_only,
        "history": history,
    }
    (output_directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return best_path


def build_parser() -> argparse.ArgumentParser:
    """Create the baseline training command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-name", default="mini_cnn")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    return parser


def main() -> int:
    """Train a number classifier from CLI arguments."""
    args = build_parser().parse_args()
    config = TrainingConfig(
        model_name=args.model_name,
        pretrained=args.pretrained,
        batch_size=args.batch_size,
        maximum_epochs=args.epochs,
        early_stopping_patience=args.patience,
        num_workers=args.workers,
    )
    checkpoint = train_model(
        args.train_manifest,
        args.validation_manifest,
        args.capture_root,
        args.output_directory,
        config,
        device_name=args.device,
        smoke_only=args.smoke_only,
    )
    print(f"Best checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
