"""End-to-end smoke test for terminal-first Kaggle CNN training."""

import csv
import json
from pathlib import Path

import torch
from PIL import Image

from jutsu_battle.number_identifier.kaggle_dataset import MANIFEST_FIELDS
from jutsu_battle.number_identifier.models import load_model_checkpoint
from jutsu_battle.number_identifier.train_kaggle import (
    KaggleTrainingConfig,
    TrainingPaths,
    train_kaggle_model,
)


def create_training_fixture(root: Path) -> TrainingPaths:
    """Create two small visual classes and prepared manifests."""
    dataset_root = root / "data"
    manifests = root / "manifests"
    manifests.mkdir()
    labels = ("0", "1")
    for label in labels:
        directory = dataset_root / label
        directory.mkdir(parents=True)
        for index in range(12):
            color = (int(label) * 180, index * 10, 50)
            Image.new("RGB", (64, 64), color).save(directory / f"{index}.jpg")
    for split, indices in (("train", range(8)), ("validation", range(8, 12))):
        with (manifests / f"{split}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for label in labels:
                for index in indices:
                    writer.writerow(
                        {
                            "relative_path": f"{label}/{index}.jpg",
                            "label": label,
                            "split": split,
                        }
                    )
    (manifests / "dataset_summary.json").write_text(
        json.dumps({"class_labels": labels}), encoding="utf-8"
    )
    return TrainingPaths(dataset_root, manifests, root / "output", "digits")


def test_one_epoch_writes_loadable_checkpoint(tmp_path: Path) -> None:
    """A complete epoch produces analytics and a dynamic-class checkpoint."""
    paths = create_training_fixture(tmp_path)
    config = KaggleTrainingConfig(
        image_size=64,
        batch_size=4,
        maximum_epochs=1,
        early_stopping_patience=1,
        num_workers=0,
    )
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    best = train_kaggle_model(
        paths,
        config,
        device_name=device_name,
        show_progress=False,
    )

    model, metadata = load_model_checkpoint(
        best.as_posix(), device=torch.device(device_name)
    )
    assert tuple(metadata["class_labels"]) == ("0", "1")
    assert model(torch.zeros(1, 3, 64, 64, device=device_name)).shape == (1, 2)
    assert (paths.output_directory / "last.pt").is_file()
    assert (paths.output_directory / "history.csv").is_file()
