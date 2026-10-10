from types import SimpleNamespace

import numpy as np

from desipecfit.staged import (
    _feii_unit,
    downweight_spectrum_for_host,
    fit_fixed_host_feii,
)
from echospec import AGNSpectrum
from echospec.feii import load_feii_template


def test_ppxf_downweighting_preserves_full_wavelength_grid():
    wave = np.arange(4200.0, 5601.0, 2.0)
    spec = AGNSpectrum(
        wave=wave,
        flux=np.ones_like(wave),
        err=np.full_like(wave, 0.2),
        z=0.0,
        survey="DESI",
        object_id="mock",
    )

    weighted, ignored = downweight_spectrum_for_host(
        spec,
        windows=((4434.0, 4684.0),),
        error_scale=1.0e6,
    )

    assert np.array_equal(weighted.wave, spec.wave)
    assert np.array_equal(weighted.flux, spec.flux)
    assert np.all(weighted.err[~ignored] == spec.err[~ignored])
    assert np.all(weighted.err[ignored] > 1.0e5 * spec.err[ignored])


def test_fixed_host_feii_recovers_mock_with_real_errors(tmp_path):
    wave = np.arange(4200.0, 5601.0, 2.0)
    template_flux = (
        np.exp(-0.5 * ((wave - 4550.0) / 18.0) ** 2)
        + 0.8 * np.exp(-0.5 * ((wave - 4625.0) / 25.0) ** 2)
        + 0.6 * np.exp(-0.5 * ((wave - 5250.0) / 35.0) ** 2)
    )
    template_path = tmp_path / "feii.txt"
    np.savetxt(template_path, np.column_stack((wave, template_flux)))
    template = load_feii_template(feii_template_path=template_path)
    unit = _feii_unit(
        wave,
        template.wave,
        template.flux,
        sigma_kms=900.0,
        shift_kms=0.0,
    )

    host_shape = 1.0 + 0.08 * np.sin((wave - 4200.0) / 180.0)
    host_scale = 1.25
    powerlaw_amp = 1.8
    powerlaw_alpha = -1.0
    feii_flux = 35.0
    flux = (
        host_scale * host_shape
        + powerlaw_amp * (wave / 5100.0) ** powerlaw_alpha
        + feii_flux * unit
    )
    flux += 2.0 * np.exp(-0.5 * ((wave - 4861.0) / 22.0) ** 2)
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum(
        wave=wave,
        flux=flux,
        err=err,
        z=0.0,
        survey="DESI",
        object_id="mock",
    )
    stellar = SimpleNamespace(stellar_model=host_shape)

    result = fit_fixed_host_feii(
        spec,
        stellar,
        feii_template_path=template_path,
        powerlaw_grid=(-1.5, -1.0, -0.5),
        feii_sigma_grid=(600.0, 900.0, 1300.0),
        feii_shift_grid=(-300.0, 0.0, 300.0),
    )

    assert result["usable"]
    assert result["errors"] == "input_spectrum_errors"
    assert abs(result["host_scale"] - host_scale) < 0.15
    assert abs(result["powerlaw_alpha"] - powerlaw_alpha) < 0.2
    assert abs(result["feii_flux_4434_4684"] - feii_flux) < 4.0
    assert not np.any(result["fit_mask"] & result["emission_mask"])
