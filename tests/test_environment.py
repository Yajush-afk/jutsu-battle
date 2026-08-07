"""Smoke tests for the initial project environment."""

from jutsu_battle import __version__


def test_package_version() -> None:
    """The package exposes the version declared for the bootstrap release."""
    assert __version__ == "0.0.1"

