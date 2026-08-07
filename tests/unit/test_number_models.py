"""Tests for CNN construction and compatible checkpoints."""

from pathlib import Path

import torch
from torch.optim import AdamW

from jutsu_battle.number_identifier.models import (
    MiniNumberCNN,
    load_model_checkpoint,
    parameter_count,
)
from jutsu_battle.number_identifier.train import (
    EpochMetrics,
    TrainingConfig,
    save_checkpoint,
)


def test_mini_cnn_output_shape() -> None:
    """The baseline produces one logit for every stable class label."""
    model = MiniNumberCNN()
    logits = model(torch.zeros(2, 3, 160, 160))
    assert logits.shape == (2, 12)
    assert parameter_count(model) > 100_000


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """Checkpoint metadata reconstructs the correct inference model."""
    model = MiniNumberCNN()
    optimizer = AdamW(model.parameters())
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=TrainingConfig(maximum_epochs=1),
        epoch=1,
        metrics=EpochMetrics(loss=1.0, accuracy=0.2, macro_f1=0.1),
        smoke_only=True,
    )
    loaded, metadata = load_model_checkpoint(
        path.as_posix(), device=torch.device("cpu")
    )
    assert isinstance(loaded, MiniNumberCNN)
    assert metadata["smoke_only"] is True
    assert loaded(torch.zeros(1, 3, 160, 160)).shape == (1, 12)
