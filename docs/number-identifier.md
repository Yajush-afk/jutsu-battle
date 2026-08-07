# Kaggle Number Identifier Operator Guide

The Phase 1 learning milestone trains a CNN on the numeric `0`–`9` subset of
Debabrata Kuiry's **Hand Sign Gesture Dataset (A–Z & 0–9)**. The public dataset
is never committed to Git.

## 1. Download and extract

Download the dataset archive from:

<https://www.kaggle.com/datasets/debabratakuiry/hand-sign-gesture-dataset-az-and-09-25k-images>

Extract its contents into exactly:

```text
data/external/kaggle/hand-sign-gesture-dataset-az-09-25k-images/
```

An extra wrapper directory created by Kaggle is acceptable. The importer finds
the class folders recursively. It expects image-containing folders named `0`
through `9`; alphabet folders are ignored in the first experiment.

## 2. Prepare deterministic manifests

```bash
conda activate naruto
python -m pip install --editable . --no-deps
jutsu-prepare-kaggle
```

Preparation verifies every image and creates a balanced 80/10/10 image-level
train/validation/test split under `data/manifests/kaggle_digits/`. It does not
copy or alter the source images.

Inspect `data/manifests/kaggle_digits/dataset_summary.json` before training. The
public description does not provide participant or recording-session IDs, so a
random image split may contain nearby frames from the same source session in
different splits. Report this limitation with the final metrics.

## 3. Train the digit CNN from a terminal

```bash
jutsu-train-kaggle --config configs/kaggle_digits.yaml
```

The command prints live batch loss/accuracy progress and an epoch summary with
training and validation loss, accuracy, macro-F1, learning rate, time, early
stopping state, and whether a new best checkpoint was saved.

Artifacts are written to:

```text
outputs/kaggle_digits/mini_cnn/best.pt
outputs/kaggle_digits/mini_cnn/last.pt
outputs/kaggle_digits/mini_cnn/history.csv
outputs/kaggle_digits/mini_cnn/tensorboard/
```

`best.pt` is selected by validation macro-F1. `last.pt` supports recovery after
an interruption:

```bash
jutsu-train-kaggle \
  --config configs/kaggle_digits.yaml \
  --resume outputs/kaggle_digits/mini_cnn/last.pt
```

Temporary command-line overrides are available without editing YAML:

```bash
jutsu-train-kaggle --epochs 5 --batch-size 16 --workers 2
```

If CUDA runs out of memory, retry with `--batch-size 16`. Do not increase batch
size merely to maximize GPU memory; validation quality and stable execution are
more important.

## 4. Default hyperparameters

```text
Architecture:          four-block MiniNumberCNN, trained from scratch
Input:                 160 × 160 RGB
Classes:               0–9
Batch size:            32
Maximum epochs:        35
Optimizer:             AdamW
Learning rate:         3e-4
Minimum learning rate: 1e-6
Scheduler:             cosine annealing
Weight decay:          1e-4
Loss:                  cross-entropy
Label smoothing:       0.05
Early-stop patience:   7 epochs
Checkpoint metric:     validation macro-F1
Mixed precision:       enabled on CUDA
Workers:               4
Seed:                  20260808
```

These defaults target the installed RTX 3050 Ti Laptop GPU with 4 GB VRAM.

## 5. Optional visual analytics

No notebook is required. `history.csv` contains all epoch metrics. TensorBoard
is also available if a graph is useful:

```bash
tensorboard --logdir outputs/kaggle_digits/mini_cnn/tensorboard
```

The terminal prints the local address to open in a browser.

## 6. All 36 signs later

After completing and reviewing the digit milestone, the same importer supports
all letters and digits:

```bash
jutsu-prepare-kaggle \
  --labels all \
  --output-directory data/manifests/kaggle_all_signs

jutsu-train-kaggle --config configs/kaggle_all_signs.yaml
```

The 36-class configuration uses ImageNet-pretrained MobileNetV3-Small at
`192×192`, batch size 24. It is feasible on this machine, but it is a separate,
longer experiment rather than a requirement for the number identifier.

## 7. Webcam check

After training finishes, run the best checkpoint against the local camera:

```bash
jutsu-number-app \
  --checkpoint outputs/kaggle_digits/mini_cnn/best.pt
```

The public model uses the class order stored inside its checkpoint and does not
apply the older custom `6`–`10` two-hand rules. Press `Q` or Escape to exit and
`R` to clear temporal prediction state.

## 8. Personal fine-tuning

After the public-data checkpoint exists, inspect its label poses and test it
against the local webcam. Personal fine-tuning should then capture the same
`0`–`9` pose vocabulary, keep a personal holdout set, and use a smaller learning
rate. Do not fine-tune against webcam evaluation images; that would make the
reported result circular.
