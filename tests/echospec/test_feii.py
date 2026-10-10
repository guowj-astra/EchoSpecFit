"""Tests for DASpec-style Fe II template handling."""

from pathlib import Path
import tempfile

import numpy as np

from echospec.feii import (
    C_KMS,
    broaden_shift_feii_template,
    compute_feii_effective_width,
    estimate_feii_template_intrinsic_width,
    inspect_feii_template,
)
from echospec.fitting import fit_hb_oiii, prefit_feii_grid
from echospec.spectrum import AGNSpectrum


def _synthetic_template() -> tuple[np.ndarray, np.ndarray]:
    wave = np.linspace(4200.0, 5700.0, 3001)
    flux = (
        np.exp(-0.5 * ((wave - 4550.0) / 22.0) ** 2)
        + 0.55 * np.exp(-0.5 * ((wave - 4650.0) / 28.0) ** 2)
        + 0.45 * np.exp(-0.5 * ((wave - 5015.0) / 18.0) ** 2)
        + 0.35 * np.exp(-0.5 * ((wave - 5250.0) / 60.0) ** 2)
    )
    return wave, flux


def test_feii_unit_model_integrates_to_one() -> None:
    template_wave, template_flux = _synthetic_template()
    output_wave = np.linspace(4200.0, 5700.0, 3001)

    unit, diagnostics = broaden_shift_feii_template(
        template_wave,
        template_flux,
        output_wave,
        width_kms=1600.0,
        shift_kms=300.0,
        return_diagnostics=True,
    )
    mask = (output_wave >= 4434.0) & (output_wave <= 4684.0)

    assert np.isclose(np.trapz(unit[mask], output_wave[mask]), 1.0, rtol=1.0e-5)
    assert diagnostics["dloglam"] > 0.0
    assert diagnostics["sigma_pix"] > 0.0
    assert diagnostics["width_kms"] == 1600.0
    assert diagnostics["shift_kms"] == 300.0
    assert diagnostics["norm_region_integral"] > 0.0
    assert diagnostics["finite_fraction"] == 1.0


def test_prefit_feii_grid_recovers_integrated_flux_width_and_shift() -> None:
    template_wave, template_flux = _synthetic_template()
    wave = np.linspace(4200.0, 5700.0, 3001)
    true_flux = 100.0
    true_width = 600.0
    true_shift = 300.0
    unit = broaden_shift_feii_template(
        template_wave,
        template_flux,
        wave,
        width_kms=true_width,
        shift_kms=true_shift,
    )
    continuum = np.full_like(wave, 2.0)
    err = np.full_like(wave, 0.2)
    flux = continuum + true_flux * unit

    prefit = prefit_feii_grid(
        wave,
        flux,
        err,
        (template_wave, template_flux),
        sigma_grid=[0.0, true_width, 1200.0],
        velocity_grid=[-300.0, true_shift, 900.0],
        feii_template_method_grid=["raw"],
        feii_smooth_sigma_grid=[0.0],
        continuum_flux=continuum,
    )

    assert np.isclose(prefit.best_feii_flux_4434_4684, true_flux, rtol=1.0e-4)
    assert prefit.best_feii_width_kms == true_width
    assert prefit.best_feii_shift_kms == true_shift
    assert np.isfinite(prefit.chi2_grid).any()


def test_feii_velocity_shift_moves_feature_by_v_over_c() -> None:
    template_wave = np.linspace(4400.0, 4700.0, 3001)
    template_flux = np.exp(-0.5 * ((template_wave - 4550.0) / 1.5) ** 2)
    output_wave = template_wave.copy()
    shift_kms = 600.0

    shifted = broaden_shift_feii_template(
        template_wave,
        template_flux,
        output_wave,
        width_kms=50.0,
        shift_kms=shift_kms,
    )
    peak_wave = output_wave[np.nanargmax(shifted)]

    assert np.isclose((peak_wave - 4550.0) / 4550.0, shift_kms / C_KMS, rtol=0.03)


def test_feii_model_is_zero_not_nan_outside_template_range() -> None:
    template_wave, template_flux = _synthetic_template()
    output_wave = np.linspace(3600.0, 5900.0, 4601)

    unit = broaden_shift_feii_template(
        template_wave,
        template_flux,
        output_wave,
        width_kms=1600.0,
        shift_kms=0.0,
    )

    assert np.isfinite(unit).all()
    assert np.all(unit[output_wave < template_wave.min()] == 0.0)
    assert np.all(unit[output_wave > template_wave.max()] == 0.0)


def test_inspect_feii_template_reports_grid_widths_and_plot() -> None:
    template_wave, template_flux = _synthetic_template()

    with tempfile.TemporaryDirectory() as tmp:
        diagnostics = inspect_feii_template(template_wave, template_flux, output_dir=tmp)
        plot_path = Path(tmp) / "feii_template_inspection.png"

        assert diagnostics["template_grid_type"] == "linear_wavelength"
        assert diagnostics["template_median_dlambda"] > 0.0
        assert diagnostics["template_median_dloglam"] > 0.0
        assert np.isfinite(diagnostics["estimated_template_fwhm_kms"])
        assert np.isfinite(diagnostics["estimated_template_sigma_kms"])
        assert np.isfinite(diagnostics["global_template_sigma_kms"])
        assert plot_path.exists()


def test_feii_template_intrinsic_and_effective_width_diagnostics() -> None:
    template_wave, template_flux = _synthetic_template()

    intrinsic = estimate_feii_template_intrinsic_width(template_wave, template_flux)
    assert np.isfinite(intrinsic["blue_template_fwhm_kms"])
    assert np.isfinite(intrinsic["blue_template_sigma_kms"])
    assert np.isfinite(intrinsic["central_template_fwhm_kms"])
    assert np.isfinite(intrinsic["central_template_sigma_kms"])
    assert np.isfinite(intrinsic["red_template_fwhm_kms"])
    assert np.isfinite(intrinsic["red_template_sigma_kms"])
    assert np.isfinite(intrinsic["global_template_sigma_kms"])

    effective = compute_feii_effective_width(
        intrinsic["global_template_sigma_kms"],
        smooth_sigma_kms=300.0,
        extra_sigma_kms=400.0,
    )
    expected_sigma = np.sqrt(intrinsic["global_template_sigma_kms"] ** 2 + 300.0**2 + 400.0**2)
    assert np.isclose(effective["feii_effective_sigma_kms"], expected_sigma)
    assert np.isclose(effective["feii_effective_fwhm_kms"], 2.3548 * expected_sigma)
    assert "approximate diagnostic" in effective["warning"]


def test_prefit_feii_grid_no_feii_does_not_run_to_max_width() -> None:
    template_wave, template_flux = _synthetic_template()
    wave = np.linspace(4200.0, 5700.0, 3001)
    continuum = np.full_like(wave, 3.0)
    err = np.full_like(wave, 0.2)

    prefit = prefit_feii_grid(
        wave,
        continuum,
        err,
        (template_wave, template_flux),
        sigma_grid=[0.0, 600.0, 1500.0],
        velocity_grid=[-400.0, 0.0, 400.0],
        feii_template_method_grid=["smoothed"],
        feii_smooth_sigma_grid=[500.0],
        continuum_flux=continuum,
    )

    assert prefit.best_feii_flux_4434_4684 <= 1.0e-8
    assert (
        prefit.flags["flag_feii_unconstrained"]
        or prefit.best_feii_extra_sigma_kms != 1500.0
    )


def test_hb_full_fit_fixed_grid_keeps_feii_width_shift_and_oiii_ratio() -> None:
    template_wave, template_flux = _synthetic_template()
    wave = np.linspace(4200.0, 5700.0, 3001)
    unit = broaden_shift_feii_template(
        template_wave,
        template_flux,
        wave,
        width_kms=1600.0,
        shift_kms=300.0,
    )
    continuum = np.full_like(wave, 2.0)
    err = np.full_like(wave, 0.2)
    flux = continuum + 80.0 * unit
    flux += _integrated_gaussian(wave, 4861.33, 20.0, 0.0, 160.0)
    flux += _integrated_gaussian(wave, 4958.91, 15.0, 0.0, 160.0)
    flux += _integrated_gaussian(wave, 5006.84, 44.7, 0.0, 160.0)
    spec = AGNSpectrum.from_arrays(wave, flux, err=err, z=0.0)

    result = fit_hb_oiii(
        spec,
        include_broad_hb=True,
        include_feii=True,
        feii_template=(template_wave, template_flux),
        include_oiii_wing=False,
        feii_width_strategy="fixed_grid",
        max_nfev=500,
    )
    info = result.fit_info

    assert info["feii_width_kms_final"] == info["feii_width_kms_prefit"]
    assert info["feii_shift_kms_final"] == info["feii_shift_kms_prefit"]
    assert np.isclose(
        result.line_flux["OIII_5007"] / result.line_flux["OIII_4959"],
        2.98,
        rtol=1.0e-8,
    )


def _integrated_gaussian(
    wave: np.ndarray,
    rest_wave: float,
    flux: float,
    velocity: float,
    sigma_velocity: float,
) -> np.ndarray:
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    return flux / (sigma_wave * np.sqrt(2.0 * np.pi)) * np.exp(
        -0.5 * ((wave - center) / sigma_wave) ** 2
    )
