"""Real-time two-hand number-recognition desktop application."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jutsu_battle.number_identifier.capture import CaptureConfig, open_camera
from jutsu_battle.number_identifier.hand_detector import (
    HandDetection,
    HandDetector,
    draw_detection,
)
from jutsu_battle.number_identifier.model_asset import (
    DEFAULT_MODEL_PATH,
    ensure_model_asset,
)
from jutsu_battle.number_identifier.predictor import (
    NumberPrediction,
    NumberPredictor,
    PredictionStabilizer,
    StableNumberEvent,
)
from jutsu_battle.number_identifier.train import select_device

WINDOW_NAME = "Jutsu Battle - Two-Hand Number Identifier"


def draw_application_ui(
    frame: np.ndarray[Any, Any],
    prediction: NumberPrediction,
    last_event: StableNumberEvent | None,
    *,
    fps: float,
) -> np.ndarray[Any, Any]:
    """Draw readable live state without obscuring the user's hands."""
    output = frame.copy()
    panel = output.copy()
    cv2.rectangle(panel, (0, 0), (output.shape[1], 115), (10, 15, 24), -1)
    cv2.addWeighted(panel, 0.78, output, 0.22, 0, output)
    display_label = prediction.label.replace("_", " ").upper()
    cv2.putText(
        output,
        display_label,
        (25, 55),
        cv2.FONT_HERSHEY_DUPLEX,
        1.45,
        (80, 225, 135) if prediction.label.isdigit() else (80, 180, 255),
        3,
        cv2.LINE_AA,
    )
    details = (
        f"confidence {prediction.confidence:.2f} | hands {prediction.hand_count} | "
        f"model {prediction.inference_ms:.1f} ms | {fps:.1f} FPS"
    )
    cv2.putText(
        output,
        details,
        (28, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (225, 230, 240),
        2,
        cv2.LINE_AA,
    )
    if last_event is not None:
        text = f"CONFIRMED: {last_event.label}"
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
        x = max(20, (output.shape[1] - size[0]) // 2)
        y = output.shape[0] - 38
        cv2.putText(
            output,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (70, 235, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def run_application(
    checkpoint_path: Path,
    *,
    camera_index: int,
    model_asset_path: Path,
    device_name: str,
    confidence_threshold: float,
    allow_smoke: bool,
    headless_frames: int,
) -> None:
    """Run interactive rendering or a finite headless camera diagnostic."""
    camera_config = CaptureConfig(
        subject_id="runtime",
        session_id="runtime",
        output_root=Path("data/private/runtime"),
        camera_index=camera_index,
        frame_width=640,
        frame_height=480,
    )
    camera = open_camera(camera_config)
    device = select_device(device_name)
    predictor = NumberPredictor(
        checkpoint_path,
        device=device,
        confidence_threshold=confidence_threshold,
        allow_smoke=allow_smoke,
    )
    stabilizer = PredictionStabilizer()
    fps_history: deque[float] = deque(maxlen=30)
    last_event: StableNumberEvent | None = None
    processed = 0
    detection = HandDetection(hands=())
    prediction = NumberPrediction(
        label="no_hands",
        confidence=1.0,
        probabilities=tuple(0.0 for _ in range(12)),
        hand_count=0,
        inference_ms=0.0,
    )
    try:
        with HandDetector(
            model_asset_path,
            max_hands=2,
            processing_width=320,
        ) as detector:
            while True:
                started = time.perf_counter()
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed")
                frame = cv2.flip(frame, 1)
                if processed % 2 == 0:
                    detection = detector.detect(frame)
                    prediction = predictor.predict(frame, detection)
                    timestamp_ms = int(time.monotonic() * 1000)
                    event = stabilizer.update(prediction, timestamp_ms)
                    if event is not None:
                        last_event = event
                        print(
                            f"confirmed={event.label} "
                            f"confidence={event.confidence:.3f}"
                        )
                elapsed = time.perf_counter() - started
                fps_history.append(1.0 / max(elapsed, 1e-6))
                processed += 1
                if headless_frames:
                    if processed >= headless_frames:
                        mean_fps = float(np.mean(fps_history))
                        print(
                            f"Headless probe passed: {processed} frames, "
                            f"mean {mean_fps:.1f} FPS, state={prediction.label}"
                        )
                        return
                    continue
                display = draw_detection(frame, detection)
                display = draw_application_ui(
                    display,
                    prediction,
                    last_event,
                    fps=float(np.mean(fps_history)),
                )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return
                if key == ord("r"):
                    stabilizer = PredictionStabilizer()
                    last_event = None
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    """Run the live number-recognition application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-asset", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--headless-frames", type=int, default=0)
    args = parser.parse_args()
    try:
        model_asset = ensure_model_asset(args.model_asset)
        run_application(
            args.checkpoint,
            camera_index=args.camera_index,
            model_asset_path=model_asset,
            device_name=args.device,
            confidence_threshold=args.confidence_threshold,
            allow_smoke=args.allow_smoke,
            headless_frames=args.headless_frames,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Application error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
