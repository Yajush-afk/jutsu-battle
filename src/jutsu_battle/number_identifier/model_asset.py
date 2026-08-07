"""Download and validate the free MediaPipe hand-landmarker asset."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from pathlib import Path

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/mediapipe/hand_landmarker.task")
MINIMUM_MODEL_BYTES = 1_000_000


class ModelAssetError(RuntimeError):
    """Raised when the hand-landmarker asset is missing or invalid."""


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model_asset(path: Path) -> None:
    """Reject missing or obviously incomplete model assets."""
    if not path.is_file():
        raise ModelAssetError(f"Hand-landmarker model not found: {path}")
    if path.stat().st_size < MINIMUM_MODEL_BYTES:
        raise ModelAssetError(f"Hand-landmarker model is incomplete: {path}")

    checksum_path = path.with_suffix(f"{path.suffix}.sha256")
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip()
        actual = sha256_file(path)
        if expected != actual:
            raise ModelAssetError(f"Checksum mismatch for {path}")


def ensure_model_asset(
    path: Path = DEFAULT_MODEL_PATH,
    *,
    allow_download: bool = True,
) -> Path:
    """Return a valid model path, downloading it over HTTPS when permitted."""
    try:
        validate_model_asset(path)
    except ModelAssetError:
        if not allow_download:
            raise
    else:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.download")
    request = urllib.request.Request(
        HAND_LANDMARKER_URL,
        headers={"User-Agent": "jutsu-battle/0.1"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # noqa: S310
            temporary_path.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        validate_model_asset(temporary_path)
        temporary_path.replace(path)
        checksum = sha256_file(path)
        path.with_suffix(f"{path.suffix}.sha256").write_text(
            f"{checksum}\n", encoding="utf-8"
        )
    except (OSError, urllib.error.URLError) as error:
        temporary_path.unlink(missing_ok=True)
        raise ModelAssetError(
            "Unable to download the MediaPipe hand-landmarker model. "
            f"Download {HAND_LANDMARKER_URL} to {path}."
        ) from error

    validate_model_asset(path)
    return path
