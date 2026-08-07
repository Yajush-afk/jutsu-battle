"""Tests for the canonical number-gesture vocabulary."""

from jutsu_battle.number_identifier.spec import (
    CLASS_LABELS,
    class_to_index,
    gesture_classes,
    index_to_class,
)


def test_class_order_is_stable() -> None:
    """Numeric labels precede the unknown class in numeric order."""
    expected = (*tuple(str(value) for value in range(11)), "unknown")
    assert expected == CLASS_LABELS
    assert tuple(item.label for item in gesture_classes()) == CLASS_LABELS


def test_hand_requirements_change_after_five() -> None:
    """Counts above five require two hands while unknown is unconstrained."""
    classes = gesture_classes()
    assert all(item.required_hands == 1 for item in classes[:6])
    assert all(item.required_hands == 2 for item in classes[6:11])
    assert classes[-1].required_hands is None


def test_class_index_mappings_round_trip() -> None:
    """Every configured class has a unique reversible output index."""
    forward = class_to_index()
    reverse = index_to_class()
    assert len(forward) == len(reverse) == 12
    assert all(reverse[index] == label for label, index in forward.items())
