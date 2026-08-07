"""Canonical labels and pose metadata for number gestures."""

from dataclasses import dataclass

NUMBER_LABELS: tuple[str, ...] = tuple(str(value) for value in range(11))
UNKNOWN_LABEL = "unknown"
CLASS_LABELS: tuple[str, ...] = (*NUMBER_LABELS, UNKNOWN_LABEL)
DETECTOR_ONLY_STATES: tuple[str, ...] = ("no_hands",)


@dataclass(frozen=True, slots=True)
class GestureClass:
    """Definition of one model output class."""

    label: str
    index: int
    required_hands: int | None
    extended_fingers: int | None


def gesture_classes() -> tuple[GestureClass, ...]:
    """Return model classes in their stable output-index order."""
    numeric_classes = tuple(
        GestureClass(
            label=label,
            index=index,
            required_hands=1 if index <= 5 else 2,
            extended_fingers=index,
        )
        for index, label in enumerate(NUMBER_LABELS)
    )
    return (
        *numeric_classes,
        GestureClass(
            label=UNKNOWN_LABEL,
            index=len(NUMBER_LABELS),
            required_hands=None,
            extended_fingers=None,
        ),
    )


def class_to_index() -> dict[str, int]:
    """Map stable class labels to model-output indices."""
    return {gesture.label: gesture.index for gesture in gesture_classes()}


def index_to_class() -> dict[int, str]:
    """Map model-output indices to stable class labels."""
    return {gesture.index: gesture.label for gesture in gesture_classes()}

