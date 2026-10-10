"""Tests for pipeline stellar-template resolution."""

import numpy as np
import pytest

from echospec.pipeline import _resolve_stellar_templates


def test_do_stellar_without_templates_raises_clear_file_not_found(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(
        FileNotFoundError,
        match=(
            "No stellar templates found. Please run: "
            "python scripts/download_ppxf_templates.py --library miles"
        ),
    ):
        _resolve_stellar_templates(None, None, template_library="miles")


def test_do_stellar_loads_templates_from_default_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    template_dir = tmp_path / ".echospec" / "templates" / "miles"
    template_dir.mkdir(parents=True)
    cache_path = template_dir / "ppxf_templates.npz"
    wave = np.linspace(3500.0, 7500.0, 20)
    templates = np.column_stack([np.ones_like(wave), 1.0 + 0.01 * wave])
    np.savez_compressed(
        cache_path,
        template_wave=wave,
        template_flux=templates,
    )

    loaded_wave, loaded_templates = _resolve_stellar_templates(
        None,
        None,
        template_library="miles",
        max_templates=1,
    )

    np.testing.assert_allclose(loaded_wave, wave)
    np.testing.assert_allclose(loaded_templates, templates[:, :1])


def test_partial_stellar_template_inputs_use_default_library(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    template_dir = tmp_path / ".echospec" / "templates" / "miles"
    template_dir.mkdir(parents=True)
    wave = np.linspace(3500.0, 7500.0, 20)
    templates = np.column_stack([np.ones_like(wave), 1.0 + 0.01 * wave])
    np.savez_compressed(
        template_dir / "ppxf_templates.npz",
        template_wave=wave,
        template_flux=templates,
    )

    loaded_wave, loaded_templates = _resolve_stellar_templates(wave, None)

    np.testing.assert_allclose(loaded_wave, wave)
    np.testing.assert_allclose(loaded_templates, templates)
