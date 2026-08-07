"""Guided webcam collector for canonical two-hand number gestures."""

from __future__ import annotations

import argparse
import csv
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from jutsu_battle.number_identifier.hand_detector import (
    BoundingBox,
    HandDetection,
    HandDetector,
    draw_detection,
)
from jutsu_battle.number_identifier.model_asset import (
    DEFAULT_MODEL_PATH,
    ModelAssetError,
    ensure_model_asset,
)
from jutsu_battle.number_identifier.spec import CLASS_LABELS, gesture_classes

WINDOW_NAME = "Jutsu Battle - Number Data Capture"


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Runtime settings for one collection session."""

    subject_id: str
    session_id: str
    output_root: Path
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    clip_duration_seconds: float = 3.0
    capture_fps: float = 5.0
    padding_ratio: float = 0.2
    jpeg_quality: int = 95


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """Metadata written for one captured image."""

    sample_id: str
    relative_path: str
    label: str
    subject_id: str
    session_id: str
    clip_id: str
    hand_count: int
    timestamp_ms: int
    crop_x1: int
    crop_y1: int
    crop_x2: int
    crop_y2: int


def sanitize_identifier(value: str, field_name: str) -> str:
    """Validate identifiers used as path components."""
    normalized = value.strip()
    if not normalized or not all(char.isalnum() or char in "-_" for char in normalized):
        raise ValueError(
            f"{field_name} must contain only letters, numbers, '-' or '_'"
        )
    return normalized


def expected_hand_count(label: str) -> int | None:
    """Return the canonical hand count for a label."""
    return next(
        item.required_hands for item in gesture_classes() if item.label == label
    )


def is_capture_ready(label: str, detection: HandDetection) -> bool:
    """Return whether the detected hand count is valid for this label."""
    required = expected_hand_count(label)
    if label == "unknown":
        return detection.hand_count in (1, 2)
    return detection.hand_count == required


class ClipWriter:
    """Write sampled frames and append their metadata atomically by clip."""

    fieldnames = tuple(CapturedFrame.__dataclass_fields__)

    def __init__(self, config: CaptureConfig, label: str) -> None:
        self.config = config
        self.label = label
        self.clip_id = uuid.uuid4().hex
        self.clip_directory = (
            config.output_root
            / config.subject_id
            / config.session_id
            / label
            / self.clip_id
        )
        self.clip_directory.mkdir(parents=True, exist_ok=False)
        self.frames: list[CapturedFrame] = []
        self._last_capture_time = 0.0
        self._started_at = time.monotonic()

    @property
    def finished(self) -> bool:
        """Return whether the configured clip duration elapsed."""
        return time.monotonic() - self._started_at >= self.config.clip_duration_seconds

    def maybe_write(
        self,
        frame: np.ndarray[Any, Any],
        detection: HandDetection,
        crop_box: BoundingBox,
    ) -> bool:
        """Write a frame when the sampling interval elapsed."""
        now = time.monotonic()
        if now - self._last_capture_time < 1.0 / self.config.capture_fps:
            return False
        self._last_capture_time = now
        sample_id = uuid.uuid4().hex
        image_path = self.clip_directory / f"{sample_id}.jpg"
        success = cv2.imwrite(
            str(image_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
        )
        if not success:
            raise OSError(f"Unable to write captured frame: {image_path}")
        relative_path = image_path.relative_to(self.config.output_root)
        metadata = CapturedFrame(
            sample_id=sample_id,
            relative_path=relative_path.as_posix(),
            label=self.label,
            subject_id=self.config.subject_id,
            session_id=self.config.session_id,
            clip_id=self.clip_id,
            hand_count=detection.hand_count,
            timestamp_ms=int(time.time() * 1000),
            crop_x1=crop_box.x1,
            crop_y1=crop_box.y1,
            crop_x2=crop_box.x2,
            crop_y2=crop_box.y2,
        )
        self.frames.append(metadata)
        return True

    def finalize(self) -> Path:
        """Write clip metadata and append it to the private manifest."""
        clip_metadata_path = self.clip_directory / "clip.json"
        clip_metadata_path.write_text(
            json.dumps([asdict(frame) for frame in self.frames], indent=2),
            encoding="utf-8",
        )
        manifest_path = self.config.output_root / "capture_manifest.csv"
        manifest_exists = manifest_path.exists()
        with manifest_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not manifest_exists:
                writer.writeheader()
            writer.writerows(asdict(frame) for frame in self.frames)
        return self.clip_directory

    def discard(self) -> None:
        """Remove an unfinished or rejected clip."""
        for path in self.clip_directory.glob("*"):
            path.unlink()
        self.clip_directory.rmdir()


def delete_captured_clip(config: CaptureConfig, clip_directory: Path) -> None:
    """Delete a finalized clip and remove its rows from the private manifest."""
    clip_id = clip_directory.name
    manifest_path = config.output_root / "capture_manifest.csv"
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        retained = [row for row in rows if row["clip_id"] != clip_id]
        temporary_manifest = manifest_path.with_suffix(".csv.tmp")
        with temporary_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ClipWriter.fieldnames)
            writer.writeheader()
            writer.writerows(retained)
        temporary_manifest.replace(manifest_path)
    for path in clip_directory.glob("*"):
        path.unlink()
    clip_directory.rmdir()


def open_camera(config: CaptureConfig) -> cv2.VideoCapture:
    """Open and configure the selected camera."""
    camera = cv2.VideoCapture(config.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Unable to open camera index {config.camera_index}")
    return camera


def probe_camera(config: CaptureConfig) -> None:
    """Read one frame to verify camera access without saving data."""
    camera = open_camera(config)
    try:
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Camera opened but did not return a frame")
        print(f"Camera probe passed: {frame.shape[1]}x{frame.shape[0]}")
    finally:
        camera.release()


def overlay_capture_ui(
    frame: np.ndarray[Any, Any],
    *,
    label: str,
    detection: HandDetection,
    recording: bool,
    frame_count: int,
) -> np.ndarray[Any, Any]:
    """Draw collector instructions and state."""
    output = draw_detection(frame, detection)
    ready = is_capture_ready(label, detection)
    color = (80, 220, 120) if ready else (40, 80, 255)
    lines = (
        f"Label: {label} | Hands: {detection.hand_count}",
        "READY" if ready else "Adjust hands to match the specification",
        f"{'RECORDING' if recording else 'IDLE'} | Saved frames: {frame_count}",
        "SPACE record | A/D label | U unknown | X discard clip | Q quit",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (24, 36 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color if index == 1 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


def run_capture(config: CaptureConfig, model_path: Path) -> None:
    """Run the interactive guided collector."""
    camera = open_camera(config)
    label_index = 0
    writer: ClipWriter | None = None
    last_clip: Path | None = None
    try:
        with HandDetector(model_path=model_path, max_hands=2) as detector:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed")
                frame = cv2.flip(frame, 1)
                detection = detector.detect(frame)
                label = CLASS_LABELS[label_index]
                crop_box = detection.union_box(frame.shape, config.padding_ratio)
                if writer is not None:
                    if is_capture_ready(label, detection) and crop_box is not None:
                        writer.maybe_write(frame, detection, crop_box)
                    if writer.finished and writer.frames:
                        last_clip = writer.finalize()
                        writer = None
                    elif writer.finished:
                        writer.discard()
                        writer = None
                display = overlay_capture_ui(
                    frame,
                    label=label,
                    detection=detection,
                    recording=writer is not None,
                    frame_count=0 if writer is None else len(writer.frames),
                )
                if crop_box is not None:
                    cv2.rectangle(
                        display,
                        (crop_box.x1, crop_box.y1),
                        (crop_box.x2, crop_box.y2),
                        (255, 180, 40),
                        2,
                    )
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a") and writer is None:
                    label_index = (label_index - 1) % len(CLASS_LABELS)
                elif key == ord("d") and writer is None:
                    label_index = (label_index + 1) % len(CLASS_LABELS)
                elif key == ord("u") and writer is None:
                    label_index = CLASS_LABELS.index("unknown")
                elif key == ord(" ") and writer is None:
                    if is_capture_ready(label, detection):
                        writer = ClipWriter(config, label)
                elif key == ord("x"):
                    if writer is not None:
                        writer.discard()
                        writer = None
                    elif last_clip is not None and last_clip.exists():
                        delete_captured_clip(config, last_clip)
                        last_clip = None
    finally:
        if writer is not None:
            writer.discard()
        camera.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    """Create the collector command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-id", default="subject_001")
    parser.add_argument("--session-id", default="session_01")
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/numbers"))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--probe-camera", action="store_true")
    return parser


def main() -> int:
    """Run the collector CLI."""
    args = build_parser().parse_args()
    try:
        config = CaptureConfig(
            subject_id=sanitize_identifier(args.subject_id, "subject-id"),
            session_id=sanitize_identifier(args.session_id, "session-id"),
            output_root=args.output_root,
            camera_index=args.camera_index,
        )
        if args.probe_camera:
            probe_camera(config)
            return 0
        model_path = ensure_model_asset(
            args.model_path,
            allow_download=not args.no_download,
        )
        run_capture(config, model_path)
    except (ModelAssetError, OSError, RuntimeError, ValueError) as error:
        print(f"Capture error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
