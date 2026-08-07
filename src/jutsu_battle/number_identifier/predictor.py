"""Checkpoint-backed number prediction and temporal stabilization."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from jutsu_battle.number_identifier.hand_detector import HandDetection
from jutsu_battle.number_identifier.landmarks import normalized_landmark_features
from jutsu_battle.number_identifier.models import load_model_checkpoint
from jutsu_battle.number_identifier.preprocessing import (
    PreprocessConfig,
    build_transform,
    crop_pad_resize_rgb,
    rgb_to_tensor,
)
from jutsu_battle.number_identifier.spec import CLASS_LABELS, gesture_classes


@dataclass(frozen=True, slots=True)
class NumberPrediction:
    """One frame-level classifier result after rejection rules."""

    label: str
    confidence: float
    probabilities: tuple[float, ...]
    hand_count: int
    inference_ms: float


@dataclass(frozen=True, slots=True)
class StableNumberEvent:
    """A numeric pose confirmed once after temporal stabilization."""

    label: str
    confidence: float
    confirmed_at_ms: int


class NumberPredictor:
    """Run image or landmark checkpoints with shared rejection semantics."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: torch.device,
        confidence_threshold: float = 0.75,
        allow_smoke: bool = False,
    ) -> None:
        self.model, self.metadata = load_model_checkpoint(
            checkpoint_path.as_posix(), device=device
        )
        if bool(self.metadata["smoke_only"]) and not allow_smoke:
            raise ValueError(
                "Smoke-only checkpoint refused. Pass --allow-smoke only for "
                "diagnostics."
            )
        self.device = device
        self.model_name = str(self.metadata["model_name"])
        self.confidence_threshold = confidence_threshold
        preprocess = self.metadata["preprocess"]
        self.preprocess_config = PreprocessConfig(
            image_size=int(preprocess["image_size"]),
            mean=tuple(preprocess["mean"]),
            std=tuple(preprocess["std"]),
        )
        self.transform = build_transform(self.preprocess_config, training=False)
        self.required_hands = {
            item.label: item.required_hands for item in gesture_classes()
        }

    def predict(
        self,
        bgr_frame: np.ndarray[Any, Any],
        detection: HandDetection,
        *,
        padding_ratio: float = 0.2,
    ) -> NumberPrediction:
        """Return one rejected or accepted frame-level prediction."""
        if detection.hand_count == 0:
            return NumberPrediction(
                label="no_hands",
                confidence=1.0,
                probabilities=tuple(0.0 for _ in CLASS_LABELS),
                hand_count=0,
                inference_ms=0.0,
            )
        if self.model_name == "landmark_mlp":
            feature_array = normalized_landmark_features(detection)
            inputs = torch.from_numpy(feature_array).unsqueeze(0).to(self.device)
        else:
            box = detection.union_box(bgr_frame.shape, padding_ratio)
            if box is None:
                return NumberPrediction(
                    label="no_hands",
                    confidence=1.0,
                    probabilities=tuple(0.0 for _ in CLASS_LABELS),
                    hand_count=0,
                    inference_ms=0.0,
                )
            rgb = crop_pad_resize_rgb(
                bgr_frame,
                box,
                self.preprocess_config.image_size,
            )
            inputs = rgb_to_tensor(rgb, self.transform).unsqueeze(0).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(inputs)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000
        confidence, index = probabilities_tensor.max(dim=0)
        label = CLASS_LABELS[int(index.item())]
        confidence_value = float(confidence.item())
        required_hands = self.required_hands[label]
        confidence_rejected = confidence_value < self.confidence_threshold
        hand_count_rejected = (
            required_hands is not None and required_hands != detection.hand_count
        )
        if confidence_rejected or hand_count_rejected:
            label = "unknown"
        return NumberPrediction(
            label=label,
            confidence=confidence_value,
            probabilities=tuple(
                float(value) for value in probabilities_tensor.detach().cpu().tolist()
            ),
            hand_count=detection.hand_count,
            inference_ms=inference_ms,
        )


class PredictionStabilizer:
    """Convert noisy frame predictions into one event per held numeric pose."""

    def __init__(
        self,
        *,
        window_size: int = 10,
        required_votes: int = 8,
        stable_duration_ms: int = 300,
        release_duration_ms: int = 150,
    ) -> None:
        if not 0 < required_votes <= window_size:
            raise ValueError("required_votes must be between 1 and window_size")
        self.window: deque[NumberPrediction] = deque(maxlen=window_size)
        self.required_votes = required_votes
        self.stable_duration_ms = stable_duration_ms
        self.release_duration_ms = release_duration_ms
        self.armed = True
        self.candidate: str | None = None
        self.candidate_since_ms: int | None = None
        self.release_since_ms: int | None = None

    def update(
        self,
        prediction: NumberPrediction,
        timestamp_ms: int,
    ) -> StableNumberEvent | None:
        """Update temporal state and optionally emit a newly confirmed number."""
        if prediction.label in ("unknown", "no_hands"):
            self.window.clear()
            self.candidate = None
            self.candidate_since_ms = None
            if self.release_since_ms is None:
                self.release_since_ms = timestamp_ms
            if timestamp_ms - self.release_since_ms >= self.release_duration_ms:
                self.armed = True
            return None

        self.release_since_ms = None
        self.window.append(prediction)
        counts = Counter(item.label for item in self.window)
        majority, votes = counts.most_common(1)[0]
        if votes < self.required_votes:
            self.candidate = None
            self.candidate_since_ms = None
            return None
        if majority != self.candidate:
            self.candidate = majority
            self.candidate_since_ms = timestamp_ms
            return None
        if self.candidate_since_ms is None:
            self.candidate_since_ms = timestamp_ms
            return None
        if timestamp_ms - self.candidate_since_ms < self.stable_duration_ms:
            return None
        if not self.armed:
            return None
        matching = [item.confidence for item in self.window if item.label == majority]
        self.armed = False
        return StableNumberEvent(
            label=majority,
            confidence=float(np.mean(matching)),
            confirmed_at_ms=timestamp_ms,
        )
