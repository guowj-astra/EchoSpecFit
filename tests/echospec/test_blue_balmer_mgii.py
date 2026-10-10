"""Tests for optional blue-Balmer and Mg II fitting modes."""

import numpy as np
from types import SimpleNamespace

from echospec.fitting import (
    fit_blue_balmer_extended,
    fit_blue_balmer_window,
    fit_mgii_uv,
    fit_mgii_uvfeii,
    should_enable_blue_balmer_mode,
)
from echospec.plotting import plot_blue_balmer_fit, plot_mgii_fit
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


def test_should_enable_blue_balmer_mode_reports_requested_diagnostics() -> None:
    wave = np.linspace(3850.0, 4450.0, 901)
    err = np.full_like(wave, 0.1)
    flux = np.ones_like(wave) + integrated_gaussian(wave, 4340.47, 10.0, 0.0, 220.0)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=flux, err=err)

    enable, diagnostics = should_enable_blue_balmer_mode(spec, None)

    assert enable
    assert diagnostics["enable_blue_balmer"]
    assert diagnostics["triggers"]["covers_3800_4700"]
    assert diagnostics["flag_balmer_continuum_region_skipped"]
    assert "residual_snr_hgamma" in diagnostics
    assert "residual_snr_hdelta" in diagnostics
    assert "powerlaw_fraction_4200" in diagnostics
    assert "host_fraction_4200" in diagnostics


def test_fit_blue_balmer_window_recovers_blue_lines() -> None:
    wave = np.linspace(3850.0, 4450.0, 1201)
    continuum = np.full_like(wave, 2.0)
    flux = (
        continuum
        + integrated_gaussian(wave, 4340.47, 25.0, 0.0, 220.0)
        + integrated_gaussian(wave, 4101.74, 16.0, 0.0, 220.0)
        + integrated_gaussian(wave, 3868.76, 8.0, 0.0, 220.0)
    )
    spec = AGNSpectrum.from_arrays(wave=wave, flux=flux, err=np.full_like(wave, 0.08))

    result = fit_blue_balmer_window(spec, None, force=True)

    assert result.success
    assert result.window_name == "blue_balmer"
    assert result.line_flux["Hgamma_narrow_flux"] > 10.0
    assert result.line_flux["Hdelta_narrow_flux"] > 5.0
    assert result.line_flux["NeIII_3869_flux"] > 1.0
    assert result.fit_info["flag_balmer_continuum_region_skipped"]


def test_blue_balmer_broad_lines_tie_to_hbeta_profile_and_bounds() -> None:
    wave = np.linspace(3850.0, 4450.0, 1201)
    continuum = np.full_like(wave, 1.0)
    hgamma_broad = integrated_gaussian(wave, 4340.47, 24.0, 300.0, 1800.0)
    hdelta_broad = integrated_gaussian(wave, 4101.74, 14.0, -200.0, 3500.0)
    spec = AGNSpectrum.from_arrays(
        wave=wave,
        flux=continuum + hgamma_broad + hdelta_broad,
        err=np.full_like(wave, 0.08),
    )
    initial = SimpleNamespace(
        model_flux=continuum,
        window_name="hb_oiii",
        params={
            "narrow_velocity": 10.0,
            "narrow_sigma": 180.0,
            "hb_broad_1_velocity": 300.0,
            "hb_broad_1_sigma": 1800.0,
            "hb_broad_1_flux": 40.0,
            "hb_broad_2_velocity": -200.0,
            "hb_broad_2_sigma": 3500.0,
            "hb_broad_2_flux": 30.0,
        },
        line_flux={"Hb_narrow": 20.0, "Hb_broad_1": 40.0, "Hb_broad_2": 30.0},
        fit_info={"Hb_broad_snr": 20.0},
    )

    result = fit_blue_balmer_window(
        spec,
        initial,
        force=True,
        include_neiii=False,
        force_broad_blue_balmer=True,
    )

    assert result.success
    assert result.params["hbeta_broad_1_velocity"] == 300.0
    assert result.params["hbeta_broad_1_sigma"] == 1800.0
    assert result.params["hbeta_broad_2_velocity"] == -200.0
    assert result.params["hbeta_broad_2_sigma"] == 3500.0
    assert result.line_flux["Hgamma_broad_flux_total"] <= 0.8 * 70.0 + 1.0e-6
    assert result.line_flux["Hdelta_broad_flux_total"] <= 0.6 * 70.0 + 1.0e-6
    assert "flag_hgamma_required" in result.fit_info
    assert "flag_hdelta_overfit" in result.fit_info


def test_fit_mgii_uv_recovers_broad_mgii_without_template() -> None:
    wave = np.linspace(2200.0, 3090.0, 1201)
    continuum = 1.5 + 0.0002 * (wave - 2800.0)
    mgii = (
        integrated_gaussian(wave, 2796.35, 30.0, 0.0, 1800.0)
        + integrated_gaussian(wave, 2803.53, 30.0, 0.0, 1800.0)
    )
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + mgii, err=np.full_like(wave, 0.05))

    result = fit_mgii_uv(spec, include_uv_feii=False, include_narrow_mgii=False, broad_mgii_model="single")

    assert result.success
    assert result.window_name == "mgii_uv"
    assert result.line_flux["MgII_broad"] > 20.0
    assert result.fit_info["fit_mode"] == "mgii_uvfeii"


def test_fit_mgii_uvfeii_model_selection_and_plots(tmp_path) -> None:
    wave = np.linspace(2200.0, 3090.0, 1201)
    continuum = 1.5 * (wave / 2800.0) ** -1.0
    template_flux = (
        np.exp(-0.5 * ((wave - 2450.0) / 80.0) ** 2)
        + 0.7 * np.exp(-0.5 * ((wave - 3020.0) / 55.0) ** 2)
    )
    uv_feii = 20.0 * template_flux / np.trapz(template_flux, wave)
    mgii = integrated_gaussian(wave, 2798.75, 80.0, 0.0, 2200.0)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=continuum + uv_feii + mgii, err=np.full_like(wave, 0.04))

    result = fit_mgii_uvfeii(
        spec,
        include_uv_feii=True,
        include_mgii_broad=True,
        include_mgii_narrow=True,
        uv_feii_template=(wave, template_flux),
    )

    assert result.success
    assert result.window_name == "mgii_uv"
    assert result.fit_info["selected_model"] in {"M1", "M2", "M3"}
    assert "delta_bic_mgii_broad" in result.fit_info
    assert "UV_FeII_2200_3090" in result.line_flux or result.fit_info["selected_model"] == "M0"
    fig = plot_mgii_fit(spec, result, tmp_path / "mgii.png")
    assert (tmp_path / "mgii.png").exists()
    fig.clf()


def test_plot_blue_balmer_fit(tmp_path) -> None:
    wave = np.linspace(3850.0, 4450.0, 1201)
    continuum = np.full_like(wave, 2.0)
    flux = continuum + integrated_gaussian(wave, 4340.47, 25.0, 0.0, 220.0)
    spec = AGNSpectrum.from_arrays(wave=wave, flux=flux, err=np.full_like(wave, 0.08))
    result = fit_blue_balmer_window(spec, None, force=True)

    fig = plot_blue_balmer_fit(spec, result, tmp_path / "blue.png")

    assert (tmp_path / "blue.png").exists()
    fig.clf()
