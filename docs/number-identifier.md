# Two-Hand Number Identifier Operator Guide

The number identifier recognizes the canonical poses defined in
`docs/number-gesture-spec.md`. It is a self-contained vision milestone and a
source of reusable camera, preprocessing, training, evaluation, and inference
components for Jutsu Battle.

## 1. Environment

```bash
conda activate naruto
python -m pip install --editable . --no-deps
```

The first capture command automatically downloads the free MediaPipe
hand-landmarker asset to `models/mediapipe/`. No upload or account is required.

## 2. Camera check

```bash
jutsu-capture-numbers --probe-camera --camera-index 0
```

If camera index 0 is unavailable, try another index. The command reads one
frame and saves nothing.

## 3. Participant collection

Use anonymous participant and session identifiers:

```bash
jutsu-capture-numbers \
  --subject-id subject_001 \
  --session-id session_01 \
  --camera-index 0
```

Collector controls:

```text
A / D   previous or next label
U       jump to unknown
SPACE   record a three-second clip
X       discard the current or most recent clip
Q / ESC quit
```

Repeat for 15 participants and two sessions per participant. Capture four clips
per numeric class and at least ten unknown/transition clips per session. Raw
frames remain under `data/raw/numbers/` and are ignored by Git.

## 4. Locked manifests

```bash
jutsu-build-number-dataset
```

This refuses fewer than 15 participants, creates the locked 10/2/3 subject
split, rejects path escape and missing files, removes near-duplicates, and
writes:

```text
data/manifests/numbers_train.csv
data/manifests/numbers_validation.csv
data/manifests/numbers_test.csv
data/manifests/number_subject_splits.json
```

Never delete or regenerate the split lock after model selection begins.

## 5. Baseline training

```bash
jutsu-train-numbers \
  --train-manifest data/manifests/numbers_train.csv \
  --validation-manifest data/manifests/numbers_validation.csv \
  --capture-root data/raw/numbers \
  --output-directory outputs/number_cnn \
  --model-name mini_cnn
```

## 6. Model comparison

```bash
jutsu-compare-numbers \
  --train-manifest data/manifests/numbers_train.csv \
  --validation-manifest data/manifests/numbers_validation.csv \
  --capture-root data/raw/numbers \
  --output-directory outputs/number_comparison
```

This trains the custom CNN, ImageNet-pretrained MobileNetV3-Small, and the
normalized-landmark MLP. Model selection uses validation macro-F1 only.

## 7. Final test evaluation

After choosing one checkpoint, evaluate the locked test set once:

```bash
jutsu-evaluate-numbers \
  --checkpoint outputs/number_comparison/<selected-model>/best.pt \
  --manifest data/manifests/numbers_test.csv \
  --capture-root data/raw/numbers \
  --output-directory outputs/number_final_evaluation
```

The evaluator writes metrics, confusion artifacts, calibration, latency, and
release-gate results. Do not tune the model against this test report.

## 8. Live application

```bash
jutsu-number-app \
  --checkpoint outputs/number_comparison/<selected-model>/best.pt
```

Press `Q` or Escape to exit and `R` to reset temporal state.

## 9. Guarded release bundle

```bash
jutsu-release-number \
  --checkpoint outputs/number_comparison/<selected-model>/best.pt \
  --evaluation-directory outputs/number_final_evaluation \
  --output-root outputs/releases
```

The bundler refuses smoke checkpoints and any evaluation that does not pass
every acceptance gate. Add `--publish` only when the resulting production
bundle should create the GitHub release and tag.

## Smoke verification

Generated patterns can exercise the complete code path without pretending to
measure recognition quality:

```bash
python -m jutsu_battle.number_identifier.smoke_data --overwrite
```

Every smoke checkpoint contains `smoke_only: true`; the live app requires an
explicit diagnostic override and the release bundler always rejects it.

