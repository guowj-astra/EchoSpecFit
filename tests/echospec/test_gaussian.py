"""Smoke tests for future Gaussian model utilities."""

from echospec import emission


def test_emission_module_imports() -> None:
    assert emission.__all__ == []

