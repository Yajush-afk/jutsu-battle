"""MediaPipe hand detection and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


@dataclass(frozen=True, slots=True)
class LandmarkPoint:
    """One normalized hand landmark."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Pixel-space rectangle with an exclusive lower-right edge."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return the box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Return the box height in pixels."""
        return self.y2 - self.y1


@dataclass(frozen=True, slots=True)
class DetectedHand:
    """Landmarks and handedness for one detected hand."""

    landmarks: tuple[LandmarkPoint, ...]
    handedness: str
    score: float


@dataclass(frozen=True, slots=True)
class HandDetection:
    """All hands detected in one frame."""

    hands: tuple[DetectedHand, ...]

    @property
    def hand_count(self) -> int:
        """Return the number of detected hands."""
        return len(self.hands)

    def union_box(
        self,
        frame_shape: tuple[int, ...],
        padding_ratio: float = 0.2,
    ) -> BoundingBox | None:
        """Return a padded box enclosing every detected landmark."""
        if not self.hands:
            return None
        height, width = frame_shape[:2]
        xs = [point.x * width for hand in self.hands for point in hand.landmarks]
        ys = [point.y * height for hand in self.hands for point in hand.landmarks]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)
        padding = max(right - left, bottom - top) * padding_ratio
        return BoundingBox(
            x1=max(0, int(left - padding)),
            y1=max(0, int(top - padding)),
            x2=min(width, int(right + padding) + 1),
            y2=min(height, int(bottom + padding) + 1),
        )


class HandDetector:
    """Synchronous MediaPipe Tasks hand-landmarker wrapper."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_hands: int = 2,
        minimum_detection_confidence: float = 0.5,
        minimum_presence_confidence: float = 0.5,
        minimum_tracking_confidence: float = 0.5,
    ) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=minimum_detection_confidence,
            min_hand_presence_confidence=minimum_presence_confidence,
            min_tracking_confidence=minimum_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, bgr_frame: np.ndarray[Any, Any]) -> HandDetection:
        """Detect hands in an OpenCV BGR image."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect(image)
        hands: list[DetectedHand] = []
        for index, raw_landmarks in enumerate(result.hand_landmarks):
            category = result.handedness[index][0]
            landmarks = tuple(
                LandmarkPoint(x=point.x, y=point.y, z=point.z)
                for point in raw_landmarks
            )
            hands.append(
                DetectedHand(
                    landmarks=landmarks,
                    handedness=category.category_name or "Unknown",
                    score=float(category.score),
                )
            )
        return HandDetection(hands=tuple(hands))

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def draw_detection(
    frame: np.ndarray[Any, Any], detection: HandDetection
) -> np.ndarray[Any, Any]:
    """Draw landmarks and handedness on a copy of the input frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    for hand in detection.hands:
        points = [
            (int(point.x * width), int(point.y * height))
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], (80, 220, 120), 2)
        for point in points:
            cv2.circle(output, point, 3, (40, 120, 255), -1)
        wrist = points[0]
        cv2.putText(
            output,
            f"{hand.handedness} {hand.score:.2f}",
            (wrist[0], max(20, wrist[1] - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output
