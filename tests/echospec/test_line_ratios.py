"""Tests for constrained line ratios in emission-line fits."""

import numpy as np
import pytest

from echospec.diagnostics import (
    compute_bpt_coordinates,
    compute_o32,
    compute_sii_bpt_coordinates,
    safe_log10_ratio,
)
from echospec.feii import feii_unit_flux_template, broaden_shift_feii_template
from echospec.fitting import (
    compute_feii_region_diagnostics,
    fit_ha_nii_sii,
    fit_hb_oiii,
    fit_hb_oiii_bic,
    fit_oii,
    measure_combined_profile_properties,
    monte_carlo_fit,
)
from echospec.fitting import _fit_hb_oiii_hardcoded
from echospec.recipes import Recipe
from echospec.spectrum import AGNSpectrum


C_KMS = 299792.458


def integrated_gaussian(
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


def test_hb_oiii_fit_recovers_synthetic_line_fluxes() -> None:
    wave = np.linspace(4700.0, 5100.0, 1601)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    true_fluxes = {
        "Hb_narrow": 40.0,
        "OIII_4959": 30.0,
        "OIII_5007": 89.4,
    }
    line_flux = (
        integrated_gaussian(wave, 4861.33, true_fluxes["Hb_narrow"], 120.0, 180.0)
        + integrated_gaussian(wave, 4958.91, true_fluxes["OIII_4959"], 120.0, 180.0)
        + integrated_gaussian(wave, 5006.84, true_fluxes["OIII_5007"], 120.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_hb_oiii(spec, include_broad_hb=False, include_feii=False)

    assert result.success
    assert np.isclose(
        result.line_flux["OIII_5007"] / result.line_flux["OIII_4959"],
        2.98,
        rtol=1.0e-12,
    )
    for name, expected_flux in true_fluxes.items():
        np.testing.assert_allclose(
            result.line_flux[name],
            expected_flux,
            rtol=0.1,
        )


def test_hb_oiii_yaml_matches_legacy_hardcoded_fitter() -> None:
    recipe = Recipe.from_yaml("recipes/hb_oiii.yaml")
    assert recipe.name == "hb_oiii"

    wave = np.linspace(4700.0, 5100.0, 1601)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 120.0, 180.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 120.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 120.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    recipe_result = fit_hb_oiii(spec, include_broad_hb=False, include_feii=False)
    hardcoded_result = _fit_hb_oiii_hardcoded(spec, include_broad_hb=False, include_feii=False)

    assert recipe_result.success
    assert hardcoded_result.success
    for name, flux in hardcoded_result.line_flux.items():
        np.testing.assert_allclose(recipe_result.line_flux[name], flux, rtol=1.0e-5)


def test_hb_oiii_bic_rejects_unneeded_broad_component() -> None:
    wave = np.linspace(4700.0, 5100.0, 1601)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 0.0, 180.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 0.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 0.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_hb_oiii_bic(spec, include_feii=False, include_oiii_wing=False)

    assert result.success
    assert not result.fit_info["broad_balmer_selected"]
    assert "Hb_broad" not in result.line_flux
    assert result.fit_info["delta_bic_broad"] < 10.0


def test_hb_oiii_bic_selects_required_broad_component() -> None:
    wave = np.linspace(4700.0, 5100.0, 1601)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 0.0, 180.0)
        + integrated_gaussian(wave, 4861.33, 120.0, 0.0, 1800.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 0.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 0.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_hb_oiii_bic(spec, include_feii=False, include_oiii_wing=False)

    assert result.success
    assert result.fit_info["broad_balmer_selected"]
    assert "Hb_broad" in result.line_flux
    assert result.fit_info["delta_bic_broad"] > 10.0


def test_hb_oiii_double_broad_model_recovers_total_broad_flux() -> None:
    wave = np.linspace(4700.0, 5100.0, 1601)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    broad_1 = 80.0
    broad_2 = 100.0
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 0.0, 180.0)
        + integrated_gaussian(wave, 4861.33, broad_1, 0.0, 1500.0)
        + integrated_gaussian(wave, 4861.33, broad_2, 0.0, 4000.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 0.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 0.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_hb_oiii(
        spec,
        include_feii=False,
        include_oiii_wing=False,
        broad_hb_model="double",
    )

    assert result.success
    assert np.isclose(
        result.line_flux["OIII_5007"] / result.line_flux["OIII_4959"],
        2.98,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.line_flux["Hb_broad_flux_total"],
        broad_1 + broad_2,
        rtol=0.1,
    )
    assert result.fit_info["Hb_broad_fwhm_combined"] > 1000.0


def test_hb_oiii_missing_feii_template_does_not_crash(monkeypatch) -> None:
    wave = np.linspace(4700.0, 5100.0, 801)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 0.0, 180.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 0.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 0.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    import echospec.fitting as fitting

    monkeypatch.setattr(fitting, "_default_feii_template_path", lambda: None)
    result = fit_hb_oiii(spec, include_feii=True, include_broad_hb=False)

    assert result.success
    assert result.fit_info["flag_feii_missing_template"]
    assert not result.fit_info["include_feii"]


def test_hb_oiii_feii_diagnostics_use_blue_and_red_regions() -> None:
    wave = np.linspace(4400.0, 5600.0, 2401)
    continuum = 2.0 - 0.0005 * (wave - 4900.0)
    feii_template_flux = (
        0.7 * np.exp(-0.5 * ((wave - 4550.0) / 45.0) ** 2)
        + 0.5 * np.exp(-0.5 * ((wave - 5250.0) / 90.0) ** 2)
        + 0.25 * np.exp(-0.5 * ((wave - 5450.0) / 70.0) ** 2)
    )
    feii_template = (wave, feii_template_flux)
    feii_amp = 8.0
    broad_1 = 45.0
    broad_2 = 55.0
    line_flux = (
        feii_amp * feii_template_flux
        + integrated_gaussian(wave, 4861.33, 35.0, 0.0, 160.0)
        + integrated_gaussian(wave, 4861.33, broad_1, 0.0, 1500.0)
        + integrated_gaussian(wave, 4861.33, broad_2, 0.0, 4000.0)
        + integrated_gaussian(wave, 4958.91, 25.0, 0.0, 160.0)
        + integrated_gaussian(wave, 5006.84, 74.5, 0.0, 160.0)
    )
    err = np.full_like(wave, 0.08)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_hb_oiii(
        spec,
        include_feii=True,
        feii_template=feii_template,
        include_oiii_wing=False,
        broad_hb_model="double",
    )

    assert result.success
    assert result.fit_info["feii_flux_4434_4684"] > 0.0
    assert result.fit_info["feii_amp"] > 0.0
    assert result.fit_info["feii_sigma"] == result.fit_info["feii_extra_sigma_kms"]
    assert result.fit_info["feii_sigma_conv"] == (
        result.fit_info["feii_smooth_sigma_kms"] + result.fit_info["feii_extra_sigma_kms"]
    )
    assert result.fit_info["feii_sigma_pix"] > 0.0
    assert result.fit_info["feii_broadening_diagnostics"]["output_finite_fraction"] == 1.0
    assert result.fit_info["feii_flux_blue_region"] > 0.0
    assert result.fit_info["feii_flux_red_region"] > 0.0
    assert result.fit_info["broad_hb_fraction_red_feii"] < 0.3
    assert not result.fit_info["flag_broad_hb_eats_feii"]


def test_feii_region_diagnostics_report_model_balance() -> None:
    wave = np.linspace(4400.0, 5600.0, 1201)
    continuum = np.ones_like(wave)
    feii = (
        np.exp(-0.5 * ((wave - 4550.0) / 35.0) ** 2)
        + 0.6 * np.exp(-0.5 * ((wave - 5300.0) / 80.0) ** 2)
    )
    observed = continuum + feii
    diagnostics = compute_feii_region_diagnostics(
        wave,
        observed,
        {"continuum": continuum, "FeII": feii},
    )

    assert diagnostics["blue_feii_model_flux"] > 0.0
    assert diagnostics["red_feii_model_flux"] > 0.0
    assert diagnostics["feii_blue_red_model_ratio"] > 0.0
    assert diagnostics["blue_feii_residual_rms"] < 1.0e-10
    assert diagnostics["red_feii_residual_rms"] < 1.0e-10


def test_feii_broadening_uses_log_lambda_velocity_grid() -> None:
    template_wave = np.linspace(4400.0, 5600.0, 2401)
    template_flux = np.exp(-0.5 * ((template_wave - 5000.0) / 8.0) ** 2)
    output_wave = template_wave.copy()

    broadened, diagnostics = broaden_shift_feii_template(
        template_wave,
        template_flux,
        output_wave,
        width_kms=1200.0,
        shift_kms=300.0,
        return_diagnostics=True,
    )

    dloglam = np.median(np.diff(np.log(template_wave)))
    assert np.isfinite(broadened).all()
    assert diagnostics["output_finite_fraction"] == 1.0
    assert np.isclose(diagnostics["dloglam"], dloglam)
    assert np.isclose(diagnostics["sigma_pix"], (1200.0 / C_KMS) / dloglam)
    assert diagnostics["width_kms"] == 1200.0
    assert diagnostics["shift_kms"] == 300.0
    assert diagnostics["feii_template_sigma_kms"] == 0.0
    assert diagnostics["sigma_conv_kms"] == 1200.0
    assert np.isclose(diagnostics["norm_region_integral"], diagnostics["normalization_integral_4434_4684"])
    assert diagnostics["template_after_broadening"]["max"] < diagnostics[
        "template_before_broadening"
    ]["max"]


def test_feii_unit_template_integrates_to_one_in_daspec_window() -> None:
    template_wave = np.linspace(4400.0, 5600.0, 2401)
    template_flux = np.exp(-0.5 * ((template_wave - 4550.0) / 30.0) ** 2)

    unit, diagnostics = feii_unit_flux_template(
        template_wave,
        template_wave,
        template_flux,
        width_kms=200.0,
        shift_kms=0.0,
        return_diagnostics=True,
    )
    mask = (template_wave >= 4434.0) & (template_wave <= 4684.0)

    assert np.isclose(np.trapz(unit[mask], template_wave[mask]), 1.0)
    assert diagnostics["normalization_integral_4434_4684"] > 0.0
    assert np.isclose(diagnostics["unit_template_integral_4434_4684"], 1.0)


def test_feii_broadening_warns_for_unstable_sigma_pix() -> None:
    template_wave = np.linspace(4400.0, 5600.0, 2401)
    template_flux = np.exp(-0.5 * ((template_wave - 4550.0) / 8.0) ** 2)

    with pytest.warns(RuntimeWarning, match="sigma_pix"):
        broaden_shift_feii_template(
            template_wave,
            template_flux,
            template_wave,
            width_kms=1.0,
            shift_kms=0.0,
        )


def test_feii_broadening_subtracts_template_intrinsic_width() -> None:
    template_wave = np.linspace(4400.0, 5600.0, 2401)
    template_flux = np.exp(-0.5 * ((template_wave - 4550.0) / 8.0) ** 2)

    _, diagnostics = broaden_shift_feii_template(
        template_wave,
        template_flux,
        template_wave,
        width_kms=2000.0,
        shift_kms=0.0,
        feii_template_sigma_kms=1200.0,
        feii_broadening_mode="target",
        return_diagnostics=True,
    )

    expected_conv = np.sqrt(2000.0**2 - 1200.0**2)
    dloglam = np.median(np.diff(np.log(template_wave)))
    assert np.isclose(diagnostics["sigma_conv_kms"], expected_conv)
    assert np.isclose(diagnostics["sigma_pix"], (expected_conv / C_KMS) / dloglam)


def test_feii_broadening_skips_when_template_is_already_broader() -> None:
    template_wave = np.linspace(4400.0, 5600.0, 2401)
    template_flux = np.exp(-0.5 * ((template_wave - 4550.0) / 8.0) ** 2)

    with pytest.warns(RuntimeWarning, match="template intrinsic sigma"):
        _, diagnostics = broaden_shift_feii_template(
            template_wave,
            template_flux,
            template_wave,
            width_kms=1000.0,
            shift_kms=0.0,
            feii_template_sigma_kms=1200.0,
            feii_broadening_mode="target",
            return_diagnostics=True,
        )

    assert diagnostics["sigma_conv_kms"] == 0.0
    assert diagnostics["sigma_pix"] == 0.0


def test_hb_oiii_can_fix_feii_sigma_below_default_bounds() -> None:
    wave = np.linspace(4400.0, 5600.0, 2401)
    continuum = np.full_like(wave, 2.0)
    feii_template_flux = np.exp(-0.5 * ((wave - 4550.0) / 30.0) ** 2)
    feii_template = (wave, feii_template_flux)
    flux = (
        continuum
        + 4.0 * feii_template_flux
        + integrated_gaussian(wave, 4861.33, 20.0, 0.0, 160.0)
        + integrated_gaussian(wave, 4958.91, 15.0, 0.0, 160.0)
        + integrated_gaussian(wave, 5006.84, 44.7, 0.0, 160.0)
    )
    err = np.full_like(wave, 0.1)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=flux, err=err)

    result = fit_hb_oiii(
        spec,
        include_broad_hb=False,
        include_feii=True,
        feii_template=feii_template,
        include_oiii_wing=False,
        feii_fixed_sigma_kms=200.0,
    )

    assert result.success
    assert result.fit_info["feii_sigma_fixed"]
    assert result.fit_info["feii_width_kms"] == 200.0
    assert result.fit_info["feii_sigma"] == 200.0
    assert result.params["feii_width_kms"] == 200.0
    assert result.params["feii_sigma"] == 200.0
    assert result.fit_info["feii_flux_4434_4684"] > 0.0


def test_ha_nii_sii_fit_recovers_synthetic_line_fluxes() -> None:
    wave = np.linspace(6400.0, 6800.0, 1601)
    continuum = 3.0 - 0.001 * (wave - 6600.0)
    true_fluxes = {
        "Ha_narrow": 80.0,
        "NII_6548": 25.0,
        "NII_6584": 74.0,
        "SII_6716": 35.0,
        "SII_6731": 28.0,
    }
    line_flux = (
        integrated_gaussian(wave, 6562.80, true_fluxes["Ha_narrow"], -90.0, 160.0)
        + integrated_gaussian(wave, 6548.05, true_fluxes["NII_6548"], -90.0, 160.0)
        + integrated_gaussian(wave, 6583.45, true_fluxes["NII_6584"], -90.0, 160.0)
        + integrated_gaussian(wave, 6716.44, true_fluxes["SII_6716"], -90.0, 160.0)
        + integrated_gaussian(wave, 6730.82, true_fluxes["SII_6731"], -90.0, 160.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_ha_nii_sii(spec, include_broad_ha=False)

    assert result.success
    assert np.isclose(
        result.line_flux["NII_6584"] / result.line_flux["NII_6548"],
        2.96,
        rtol=1.0e-12,
    )
    for name, expected_flux in true_fluxes.items():
        np.testing.assert_allclose(
            result.line_flux[name],
            expected_flux,
            rtol=0.1,
        )


def test_ha_nii_sii_double_broad_model_recovers_total_broad_flux() -> None:
    wave = np.linspace(6400.0, 6800.0, 1601)
    continuum = 3.0 - 0.001 * (wave - 6600.0)
    broad_1 = 120.0
    broad_2 = 180.0
    line_flux = (
        integrated_gaussian(wave, 6562.80, 80.0, 0.0, 160.0)
        + integrated_gaussian(wave, 6562.80, broad_1, 0.0, 1500.0)
        + integrated_gaussian(wave, 6562.80, broad_2, 0.0, 3000.0)
        + integrated_gaussian(wave, 6548.05, 25.0, 0.0, 160.0)
        + integrated_gaussian(wave, 6583.45, 74.0, 0.0, 160.0)
        + integrated_gaussian(wave, 6716.44, 35.0, 0.0, 160.0)
        + integrated_gaussian(wave, 6730.82, 28.0, 0.0, 160.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_ha_nii_sii(spec, broad_ha_model="double")

    assert result.success
    assert np.isclose(
        result.line_flux["NII_6584"] / result.line_flux["NII_6548"],
        2.96,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(
        result.line_flux["Ha_broad_flux_total"],
        broad_1 + broad_2,
        rtol=0.1,
    )
    assert result.fit_info["Ha_broad_fwhm_combined"] > 1000.0


def test_measure_combined_profile_properties_two_gaussians() -> None:
    wave = np.linspace(4700.0, 5025.0, 1601)
    profile = (
        integrated_gaussian(wave, 4861.33, 80.0, -100.0, 1200.0)
        + integrated_gaussian(wave, 4861.33, 120.0, 200.0, 3000.0)
    )

    props = measure_combined_profile_properties(wave, profile, 4861.33)

    np.testing.assert_allclose(props["integrated_flux"], 200.0, rtol=0.05)
    assert props["FWHM_kms"] > 1000.0
    assert props["line_dispersion_kms"] > 0.0
    assert abs(props["centroid_velocity_kms"]) < 1000.0


def test_oii_blend_fit_recovers_synthetic_line_flux() -> None:
    wave = np.linspace(3650.0, 3800.0, 1201)
    continuum = 1.5 + 0.0008 * (wave - 3725.0)
    expected_flux = 55.0
    line_flux = integrated_gaussian(wave, 3727.50, expected_flux, 70.0, 170.0)
    err = np.full_like(wave, 0.03)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_oii(spec, mode="blend")

    assert result.success
    np.testing.assert_allclose(
        result.line_flux["OII_3727_blend"],
        expected_flux,
        rtol=0.1,
    )


def test_oii_doublet_fit_recovers_synthetic_line_fluxes() -> None:
    wave = np.linspace(3650.0, 3800.0, 1201)
    continuum = 1.5 + 0.0008 * (wave - 3725.0)
    true_fluxes = {
        "OII_3726": 35.0,
        "OII_3729": 50.0,
    }
    line_flux = (
        integrated_gaussian(wave, 3726.03, true_fluxes["OII_3726"], -60.0, 150.0)
        + integrated_gaussian(wave, 3728.82, true_fluxes["OII_3729"], -60.0, 150.0)
    )
    err = np.full_like(wave, 0.03)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = fit_oii(spec, mode="doublet")

    assert result.success
    for name, expected_flux in true_fluxes.items():
        np.testing.assert_allclose(
            result.line_flux[name],
            expected_flux,
            rtol=0.1,
        )


def test_safe_log10_ratio_known_values() -> None:
    assert safe_log10_ratio(100.0, 10.0) == 1.0
    np.testing.assert_allclose(
        safe_log10_ratio([100.0, 1.0], [10.0, 10.0]),
        [1.0, -1.0],
    )


def test_safe_log10_ratio_error_propagation() -> None:
    value, sigma = safe_log10_ratio(
        numerator=100.0,
        denominator=50.0,
        numerator_err=10.0,
        denominator_err=5.0,
    )

    expected_sigma = (1.0 / np.log(10.0)) * np.sqrt(0.1**2 + 0.1**2)
    np.testing.assert_allclose(value, np.log10(2.0))
    np.testing.assert_allclose(sigma, expected_sigma)


def test_bpt_coordinates_known_values() -> None:
    results = {
        "OIII_5007": 300.0,
        "Hb_narrow": 100.0,
        "NII_6584": 50.0,
        "Ha_narrow": 100.0,
    }

    x, y = compute_bpt_coordinates(results)

    np.testing.assert_allclose(x, np.log10(0.5))
    np.testing.assert_allclose(y, np.log10(3.0))


def test_sii_bpt_coordinates_known_values() -> None:
    results = {
        "OIII_5007": 300.0,
        "Hb_narrow": 100.0,
        "SII_6717": 40.0,
        "SII_6731": 60.0,
        "Ha_narrow": 200.0,
    }

    x, y = compute_sii_bpt_coordinates(results)

    np.testing.assert_allclose(x, np.log10(0.5))
    np.testing.assert_allclose(y, np.log10(3.0))


def test_o32_known_values() -> None:
    results = {
        "OIII_5007": 400.0,
        "OII_3727_blend": 100.0,
    }

    np.testing.assert_allclose(compute_o32(results), np.log10(4.0))


def test_monte_carlo_fit_estimates_hb_oiii_uncertainties() -> None:
    wave = np.linspace(4700.0, 5100.0, 801)
    continuum = 2.0 + 0.001 * (wave - 4900.0)
    line_flux = (
        integrated_gaussian(wave, 4861.33, 40.0, 120.0, 180.0)
        + integrated_gaussian(wave, 4958.91, 30.0, 120.0, 180.0)
        + integrated_gaussian(wave, 5006.84, 89.4, 120.0, 180.0)
    )
    err = np.full_like(wave, 0.05)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + line_flux, err=err)

    result = monte_carlo_fit(
        spec,
        fit_hb_oiii,
        n_mc=8,
        random_seed=123,
        include_broad_hb=False,
        include_feii=False,
        include_oiii_wing=False,
    )

    assert result.success
    assert result.mc_summary["failure_count"] == 0
    assert result.mc_summary["n_success"] == 8
    assert set(result.line_flux_err) == {"Hb_narrow", "OIII_4959", "OIII_5007"}
    assert result.line_flux_err["OIII_5007"] > 0.0
    assert "median" in result.mc_summary["line_flux"]["OIII_5007"]
