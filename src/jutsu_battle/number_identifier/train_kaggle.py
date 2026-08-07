"""Train a hand-sign CNN from terminal using prepared Kaggle manifests."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from jutsu_battle.number_identifier.kaggle_dataset import (
    PublicGestureDataset,
)
from jutsu_battle.number_identifier.models import create_model, parameter_count
from jutsu_battle.number_identifier.preprocessing import PreprocessConfig


@dataclass(frozen=True, slots=True)
class KaggleTrainingConfig:
    """Hardware-conscious defaults for the RTX 3050 Ti 4 GB."""

    model_name: str = "mini_cnn"
    pretrained: bool = False
    image_size: int = 160
    batch_size: int = 32
    maximum_epochs: int = 35
    early_stopping_patience: int = 7
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    random_seed: int = 20260808
    num_workers: int = 4
    mixed_precision: bool = True


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Dataset and output locations loaded from the YAML configuration."""

    dataset_root: Path
    manifest_directory: Path
    output_directory: Path
    label_mode: str


@dataclass(frozen=True, slots=True)
class EpochResult:
    """Metrics and timing from one complete loader pass."""

    loss: float
    accuracy: float
    macro_f1: float
    seconds: float


def seed_everything(seed: int) -> None:
    """Seed all random generators and use stable cuDNN behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def select_device(requested: str) -> torch.device:
    """Select CUDA automatically while honoring explicit device requests."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    return device


def read_yaml_config(path: Path) -> tuple[TrainingPaths, KaggleTrainingConfig]:
    """Load and type-normalize the terminal training configuration."""
    if not path.is_file():
        raise ValueError(f"Configuration not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or not isinstance(raw.get("training"), Mapping):
        raise ValueError(f"Invalid training configuration: {path}")
    training_raw = cast(Mapping[str, Any], raw["training"])
    config = KaggleTrainingConfig(
        model_name=str(training_raw["model_name"]),
        pretrained=bool(training_raw.get("pretrained", False)),
        image_size=int(training_raw["image_size"]),
        batch_size=int(training_raw["batch_size"]),
        maximum_epochs=int(training_raw["maximum_epochs"]),
        early_stopping_patience=int(training_raw["early_stopping_patience"]),
        learning_rate=float(training_raw["learning_rate"]),
        minimum_learning_rate=float(training_raw["minimum_learning_rate"]),
        weight_decay=float(training_raw["weight_decay"]),
        label_smoothing=float(training_raw["label_smoothing"]),
        random_seed=int(training_raw["random_seed"]),
        num_workers=int(training_raw["num_workers"]),
        mixed_precision=bool(training_raw["mixed_precision"]),
    )
    paths = TrainingPaths(
        dataset_root=Path(str(raw["dataset_root"])),
        manifest_directory=Path(str(raw["manifest_directory"])),
        output_directory=Path(str(training_raw["output_directory"])),
        label_mode=str(raw["labels"]),
    )
    return paths, config


def load_class_labels(summary_path: Path) -> tuple[str, ...]:
    """Load the class order locked by dataset preparation."""
    if not summary_path.is_file():
        raise ValueError(
            f"Dataset summary not found: {summary_path}. Run "
            "jutsu-prepare-kaggle first."
        )
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    labels = tuple(str(label) for label in data["class_labels"])
    if not labels or len(set(labels)) != len(labels):
        raise ValueError(f"Invalid class labels in {summary_path}: {labels}")
    return labels


def make_loader(
    dataset: PublicGestureDataset,
    *,
    config: KaggleTrainingConfig,
    device: torch.device,
    training: bool,
) -> DataLoader[Any]:
    """Construct a deterministic, GPU-friendly data loader."""
    options: dict[str, Any] = {
        "batch_size": config.batch_size,
        "shuffle": training,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    if training:
        options["generator"] = torch.Generator().manual_seed(config.random_seed)
    return DataLoader(dataset, **options)


def run_loader(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    epoch: int,
    phase: str,
    show_progress: bool,
) -> EpochResult:
    """Run one train or validation pass with live terminal metrics."""
    training = optimizer is not None
    model.train(training)
    started = time.perf_counter()
    total_loss = 0.0
    correct = 0
    sample_count = 0
    predictions: list[int] = []
    targets: list[int] = []
    progress = tqdm(
        loader,
        desc=f"epoch {epoch:02d} {phase:<5}",
        unit="batch",
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for inputs, labels in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((batch_predictions == labels).sum().item())
        sample_count += batch_size
        predictions.extend(batch_predictions.detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        progress.set_postfix(
            loss=f"{total_loss / sample_count:.4f}",
            accuracy=f"{correct / sample_count:.3f}",
        )
    if not sample_count:
        raise ValueError(f"{phase} loader is empty")
    return EpochResult(
        loss=total_loss / sample_count,
        accuracy=correct / sample_count,
        macro_f1=float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        seconds=time.perf_counter() - started,
    )


def save_training_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    config: KaggleTrainingConfig,
    class_labels: tuple[str, ...],
    epoch: int,
    validation: EpochResult,
    best_f1: float,
) -> None:
    """Save model weights plus everything needed for inference or resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "dataset_source": (
                "debabratakuiry/hand-sign-gesture-dataset-az-and-09-25k-images"
            ),
            "model_name": config.model_name,
            "class_labels": class_labels,
            "preprocess": asdict(PreprocessConfig(image_size=config.image_size)),
            "training_config": asdict(config),
            "epoch": epoch,
            "metrics": asdict(validation),
            "best_f1": best_f1,
            "smoke_only": False,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Write spreadsheet-friendly epoch analytics."""
    fieldnames = (
        "epoch",
        "learning_rate",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
        "epoch_seconds",
        "gpu_peak_memory_mb",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def train_kaggle_model(
    paths: TrainingPaths,
    config: KaggleTrainingConfig,
    *,
    device_name: str = "auto",
    resume_path: Path | None = None,
    show_progress: bool = True,
) -> Path:
    """Train the configured CNN and return its best validation checkpoint."""
    seed_everything(config.random_seed)
    device = select_device(device_name)
    class_labels = load_class_labels(paths.manifest_directory / "dataset_summary.json")
    preprocess = PreprocessConfig(image_size=config.image_size)
    train_dataset = PublicGestureDataset(
        paths.manifest_directory / "train.csv",
        paths.dataset_root,
        class_labels,
        training=True,
        preprocess_config=preprocess,
    )
    validation_dataset = PublicGestureDataset(
        paths.manifest_directory / "validation.csv",
        paths.dataset_root,
        class_labels,
        training=False,
        preprocess_config=preprocess,
    )
    train_loader = make_loader(
        train_dataset, config=config, device=device, training=True
    )
    validation_loader = make_loader(
        validation_dataset, config=config, device=device, training=False
    )
    model = create_model(
        config.model_name,
        class_count=len(class_labels),
        pretrained=config.pretrained,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.maximum_epochs,
        eta_min=config.minimum_learning_rate,
    )
    amp_enabled = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    start_epoch = 1
    best_f1 = -1.0
    stale_epochs = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if tuple(checkpoint["class_labels"]) != class_labels:
            raise ValueError("Resume checkpoint class labels do not match the dataset")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint["best_f1"])

    paths.output_directory.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(paths.output_directory / "tensorboard"))
    best_path = paths.output_directory / "best.pt"
    last_path = paths.output_directory / "last.pt"
    history_path = paths.output_directory / "history.csv"
    history: list[dict[str, Any]] = []
    print("\nJutsu Battle — Kaggle hand-sign CNN training")
    print(f"device          : {device}")
    if device.type == "cuda":
        print(f"gpu             : {torch.cuda.get_device_name(device)}")
    print(f"classes         : {', '.join(class_labels)}")
    print(f"train images    : {len(train_dataset):,}")
    print(f"validation images: {len(validation_dataset):,}")
    print(f"model parameters: {parameter_count(model):,}")
    print(f"mixed precision : {amp_enabled}")
    print(f"output          : {paths.output_directory}\n")
    try:
        for epoch in range(start_epoch, config.maximum_epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            training = run_loader(
                model,
                train_loader,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                mixed_precision=amp_enabled,
                epoch=epoch,
                phase="train",
                show_progress=show_progress,
            )
            with torch.inference_mode():
                validation = run_loader(
                    model,
                    validation_loader,
                    criterion,
                    device,
                    optimizer=None,
                    scaler=scaler,
                    mixed_precision=amp_enabled,
                    epoch=epoch,
                    phase="valid",
                    show_progress=show_progress,
                )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0
            )
            improved = validation.macro_f1 > best_f1
            if improved:
                best_f1 = validation.macro_f1
                stale_epochs = 0
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": training.loss,
                "train_accuracy": training.accuracy,
                "train_macro_f1": training.macro_f1,
                "validation_loss": validation.loss,
                "validation_accuracy": validation.accuracy,
                "validation_macro_f1": validation.macro_f1,
                "epoch_seconds": training.seconds + validation.seconds,
                "gpu_peak_memory_mb": peak_memory,
            }
            history.append(row)
            write_history(history_path, history)
            for phase, result in (("train", training), ("validation", validation)):
                writer.add_scalar(f"loss/{phase}", result.loss, epoch)
                writer.add_scalar(f"accuracy/{phase}", result.accuracy, epoch)
                writer.add_scalar(f"macro_f1/{phase}", result.macro_f1, epoch)
            writer.add_scalar("learning_rate", learning_rate, epoch)
            status = (
                "BEST"
                if improved
                else f"patience {stale_epochs}/{config.early_stopping_patience}"
            )
            print(
                f"epoch {epoch:02d}/{config.maximum_epochs} | "
                f"train loss {training.loss:.4f} acc {training.accuracy:.3%} "
                f"f1 {training.macro_f1:.4f} | val loss {validation.loss:.4f} "
                f"acc {validation.accuracy:.3%} f1 {validation.macro_f1:.4f} | "
                f"lr {learning_rate:.2e} | {row['epoch_seconds']:.1f}s | {status}"
            )
            scheduler.step()
            save_training_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                class_labels=class_labels,
                epoch=epoch,
                validation=validation,
                best_f1=best_f1,
            )
            if improved:
                save_training_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    class_labels=class_labels,
                    epoch=epoch,
                    validation=validation,
                    best_f1=best_f1,
                )
            if stale_epochs >= config.early_stopping_patience:
                print(
                    "Early stopping: validation F1 did not improve for "
                    f"{stale_epochs} epochs."
                )
                break
    except KeyboardInterrupt:
        print(f"\nTraining interrupted safely. Resume from: {last_path}")
        raise
    finally:
        writer.close()
    print(f"\nBest validation macro-F1: {best_f1:.4f}")
    print(f"Best checkpoint          : {best_path}")
    print(f"Epoch analytics          : {history_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    """Build the terminal-first training CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/kaggle_digits.yaml")
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> int:
    """Train the public-dataset CNN from the command line."""
    args = build_parser().parse_args()
    try:
        paths, config = read_yaml_config(args.config)
        values = asdict(config)
        if args.epochs is not None:
            values["maximum_epochs"] = args.epochs
        if args.batch_size is not None:
            values["batch_size"] = args.batch_size
        if args.workers is not None:
            values["num_workers"] = args.workers
        config = KaggleTrainingConfig(**values)
        train_kaggle_model(
            paths,
            config,
            device_name=args.device,
            resume_path=args.resume,
            show_progress=not args.no_progress,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Training error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
