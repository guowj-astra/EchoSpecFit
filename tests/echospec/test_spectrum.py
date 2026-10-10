"""Tests for AGNSpectrum."""

import numpy as np

from echospec.spectrum import AGNSpectrum


def test_from_arrays_works_with_err() -> None:
    spec = AGNSpectrum.from_arrays(
        wave=[5002.0, 5000.0, 5001.0],
        flux=[3.0, 1.0, 2.0],
        err=[0.3, 0.1, 0.2],
        z=0.1,
        survey="DESI",
        object_id="target-1",
        meta={"epoch": 1},
    )

    np.testing.assert_allclose(spec.wave, [5000.0, 5001.0, 5002.0])
    np.testing.assert_allclose(spec.flux, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(spec.err, [0.1, 0.2, 0.3])
    assert spec.z == 0.1
    assert spec.survey == "DESI"
    assert spec.object_id == "target-1"
    assert spec.meta == {"epoch": 1}


def test_from_arrays_works_with_ivar() -> None:
    spec = AGNSpectrum.from_arrays(
        wave=[5000.0, 5001.0],
        flux=[1.0, 2.0],
        ivar=[100.0, 25.0],
    )

    np.testing.assert_allclose(spec.err, [0.1, 0.2])


def test_to_rest_frame_divides_wavelengths_correctly() -> None:
    spec = AGNSpectrum.from_arrays(
        wave=[5500.0, 6600.0],
        flux=[1.0, 2.0],
        err=[0.1, 0.2],
        z=0.1,
    )

    rest = spec.to_rest_frame()

    np.testing.assert_allclose(rest.wave, [5000.0, 6000.0])
    np.testing.assert_allclose(rest.flux, spec.flux)
    np.testing.assert_allclose(rest.err, spec.err)
    assert rest.z == 0.0


def test_select_window_returns_correct_wavelength_range() -> None:
    spec = AGNSpectrum.from_arrays(
        wave=[5500.0, 5610.0, 5720.0],
        flux=[1.0, 2.0, 3.0],
        err=[0.1, 0.2, 0.3],
        z=0.1,
    )

    selected = spec.select_window(5050.0, 5150.0)

    np.testing.assert_allclose(selected.wave, [5610.0])
    np.testing.assert_allclose(selected.flux, [2.0])
    np.testing.assert_allclose(selected.err, [0.2])
    assert selected.z == spec.z


def test_mask_bad_pixels_removes_nans_and_non_positive_errors() -> None:
    spec = AGNSpectrum.from_arrays(
        wave=[5000.0, np.nan, 5002.0, 5003.0, 5004.0],
        flux=[1.0, 2.0, np.nan, 4.0, 5.0],
        err=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    spec.err[3] = 0.0

    masked = spec.mask_bad_pixels()

    np.testing.assert_allclose(masked.wave, [5000.0, 5004.0])
    np.testing.assert_allclose(masked.flux, [1.0, 5.0])
    np.testing.assert_allclose(masked.err, [0.1, 0.5])
