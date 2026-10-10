"""Tests for local stellar-template discovery and loading."""

import numpy as np
import pytest
from astropy.io import fits

from echospec.templates import (
    find_template_files,
    get_template_cache_dir,
    load_template_library,
)


def test_load_template_library_reads_fake_fits_templates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    template_dir = get_template_cache_dir() / "miles"
    template_dir.mkdir(parents=True)

    wave = 3500.0 + 2.0 * np.arange(5)
    _write_template_fits(template_dir / "template_1.fits", wave, np.arange(5.0))
    _write_template_fits(template_dir / "template_2.fits", wave, np.arange(5.0) + 10.0)
    (template_dir / "ignore.md").write_text("not a template\n")

    files = find_template_files("miles")
    assert len(files) == 2

    template_wave, template_flux = load_template_library("miles")

    np.testing.assert_allclose(template_wave, wave)
    assert template_flux.shape == (5, 2)
    np.testing.assert_allclose(template_flux[:, 0], np.arange(5.0))
    np.testing.assert_allclose(template_flux[:, 1], np.arange(5.0) + 10.0)


def test_load_template_library_honors_max_templates_and_wavelength_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    template_dir = get_template_cache_dir() / "miles"
    template_dir.mkdir(parents=True)

    wave = 3500.0 + 2.0 * np.arange(5)
    _write_template_fits(template_dir / "template_1.fits", wave, np.arange(5.0))
    _write_template_fits(template_dir / "template_2.fits", wave, np.arange(5.0) + 10.0)

    template_wave, template_flux = load_template_library(
        "miles",
        wavelength_range=(3502.0, 3506.0),
        max_templates=1,
    )

    np.testing.assert_allclose(template_wave, [3502.0, 3504.0, 3506.0])
    assert template_flux.shape == (3, 1)


def test_load_template_library_accepts_explicit_local_path(tmp_path) -> None:
    template_dir = tmp_path / "local_templates"
    template_dir.mkdir()

    wave = 3500.0 + 2.0 * np.arange(5)
    _write_template_fits(template_dir / "template_1.fits", wave, np.arange(5.0))

    template_wave, template_flux = load_template_library(local_path=template_dir)

    np.testing.assert_allclose(template_wave, wave)
    assert template_flux.shape == (5, 1)


def test_load_template_library_rejects_mismatched_fits_grids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    template_dir = get_template_cache_dir() / "miles"
    template_dir.mkdir(parents=True)

    wave = 3500.0 + 2.0 * np.arange(5)
    _write_template_fits(template_dir / "template_1.fits", wave, np.arange(5.0))
    _write_template_fits(template_dir / "template_2.fits", wave + 1.0, np.arange(5.0))

    with pytest.raises(ValueError, match="wavelength grids"):
        load_template_library("miles")


def test_load_template_library_missing_files_has_download_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="download_ppxf_templates.py"):
        load_template_library("miles")


def _write_template_fits(path, wave: np.ndarray, flux: np.ndarray) -> None:
    header = fits.Header()
    header["CRVAL1"] = float(wave[0])
    header["CDELT1"] = float(wave[1] - wave[0])
    fits.PrimaryHDU(data=np.asarray(flux, dtype=float), header=header).writeto(path)
