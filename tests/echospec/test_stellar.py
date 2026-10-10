"""Tests for stellar-continuum fitting helpers."""

import numpy as np
import pytest

from echospec.spectrum import AGNSpectrum
from echospec.stellar import (
    _as_template_matrix,
    fit_stellar_ppxf,
    subtract_stellar_continuum,
)


def test_template_orientation_detection() -> None:
    template_wave = np.linspace(4000.0, 5000.0, 50)
    templates = np.column_stack(
        [
            np.ones_like(template_wave),
            1.0 + 0.1 * (template_wave - template_wave.mean()) / np.ptp(template_wave),
        ]
    )

    np.testing.assert_allclose(_as_template_matrix(templates, len(template_wave)), templates)
    np.testing.assert_allclose(
        _as_template_matrix(templates.T, len(template_wave)),
        templates,
    )


def test_fit_stellar_ppxf_smoke_with_synthetic_templates() -> None:
    pytest.importorskip("ppxf")
    template_wave = np.linspace(3800.0, 5700.0, 1267)
    template_1 = 1.0 - 0.2 * np.exp(-0.5 * ((template_wave - 4300.0) / 3.0) ** 2)
    template_2 = 1.0 - 0.15 * np.exp(-0.5 * ((template_wave - 5175.0) / 4.0) ** 2)
    templates = np.column_stack([template_1, template_2])
    galaxy_pixels = (template_wave >= 4000.0) & (template_wave <= 5500.0)
    wave = template_wave[galaxy_pixels]
    flux = (0.65 * template_1 + 0.35 * template_2)[galaxy_pixels]
    err = np.full_like(wave, 0.02)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=flux, err=err)

    result = fit_stellar_ppxf(
        spec,
        template_wave=template_wave,
        template_flux=templates,
        mdegree=4,
        mask_emission_lines=False,
        include_agn_powerlaw=False,
    )
    residual_spec = subtract_stellar_continuum(spec, result)

    assert np.isfinite(result.velocity)
    assert np.isfinite(result.sigma)
    assert result.weights.shape[0] == 2
    assert len(result.goodpixels) > 0
    assert np.isfinite(result.stellar_model).all()
    assert np.nanmedian(np.abs(residual_spec.flux)) < 0.1
