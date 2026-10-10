"""Plotting helpers for spectra, fits, and diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from echospec.feii import broaden_shift_feii_template, load_feii_template
from echospec.spectrum import AGNSpectrum

if TYPE_CHECKING:
    from echospec.compare import EpochComparisonResult
    from echospec.fitting import EmissionFitResult
else:
    EpochComparisonResult = Any
    EmissionFitResult = Any

FIT_WINDOWS = {
    "blue_balmer": (3850.0, 4450.0),
    "blue_balmer_extended": (3850.0, 4450.0),
    "hb_oiii": (4200.0, 5700.0),
    "ha_nii_sii": (6400.0, 6800.0),
    "mgii_uv": (2200.0, 3090.0),
    "oii_blend": (3650.0, 3800.0),
    "oii_doublet": (3650.0, 3800.0),
}
FEII_BLUE_REGION = (4434.0, 4684.0)
FEII_RED_REGION = (5100.0, 5600.0)
FEII_NORM_REGION = (4434.0, 4684.0)


def plot_emission_fit(
    spec: AGNSpectrum,
    result: EmissionFitResult,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot observed flux, best-fit model, continuum, and residual."""

    wave, flux = _matched_wave_flux(spec, result)
    fig, (ax_spec, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    ax_spec.plot(wave, flux, color="0.25", lw=1.0, label="Observed")
    ax_spec.plot(wave, result.model_flux, color="#d62728", lw=1.5, label="Model")
    ax_spec.plot(
        wave,
        result.continuum_flux,
        color="#1f77b4",
        lw=1.2,
        ls="--",
        label="Continuum",
    )
    _plot_fit_components(ax_spec, wave, result)
    ax_spec.set_ylabel("Flux")
    ax_spec.legend(loc="best", frameon=False)
    if title is not None:
        ax_spec.set_title(title)

    ax_resid.axhline(0.0, color="0.55", lw=1.0)
    ax_resid.plot(wave, result.residual_flux, color="0.15", lw=1.0)
    ax_resid.set_xlabel("Rest wavelength (Angstrom)")
    ax_resid.set_ylabel("Residual")

    _save_if_requested(fig, output_path)
    return fig


def plot_hb_oiii_feii_diagnostic(
    spec: AGNSpectrum,
    result: EmissionFitResult,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot the staged H-beta + [O III] + Fe II diagnostic panels."""

    wave, flux = _matched_wave_flux(spec, result)
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(10.0, 12.5),
        sharex=False,
        gridspec_kw={"hspace": 0.32},
    )
    info = getattr(result, "fit_info", {}) or {}
    if title is None:
        title = (
            "Hbeta + [O III] + Fe II | "
            f"chi2={_fmt_float(info.get('final_joint_chi2', info.get('chi2')))} "
            f"BIC={_fmt_float(info.get('final_joint_bic'))} "
            f"FeII4434-4684={_fmt_float(info.get('feii_flux_4434_4684', info.get('feii_amp')))} "
            f"extra_sigma={_fmt_float(info.get('feii_extra_sigma_kms', info.get('feii_width_kms', info.get('feii_sigma'))))} "
            f"method={info.get('feii_template_method_final', info.get('feii_template_method_prefit', 'raw'))} "
            f"smooth={_fmt_float(info.get('feii_smooth_sigma_kms_final', info.get('feii_smooth_sigma_kms_prefit')))} "
            f"mode={info.get('feii_broadening_mode', 'extra')} "
            f"fixed={bool(info.get('feii_sigma_fixed'))} "
            f"sigma_conv={_fmt_float(info.get('feii_sigma_conv'))} "
            f"sigma_pix={_fmt_float(info.get('feii_sigma_pix'))} "
            f"shift={_fmt_float(info.get('feii_shift_kms', info.get('feii_velocity')))} "
            f"strategy={info.get('feii_width_strategy', 'unknown')} "
            f"chi2_blue={_fmt_float(info.get('feii_chi2_blue', info.get('blue_feii_chi2')))} "
            f"chi2_ped={_fmt_float(info.get('feii_chi2_pedestal'))} "
            f"chi2_red={_fmt_float(info.get('feii_chi2_red', info.get('red_feii_chi2')))} "
            f"too_broad={bool(info.get('flag_feii_too_broad'))} "
            f"unconstrained={bool(info.get('flag_feii_unconstrained'))} "
            f"hf_overfit={bool(info.get('flag_feii_high_frequency_overfit'))} "
            f"shape_mismatch={bool(info.get('flag_single_feii_template_shape_mismatch'))} "
            f"feii_boundary={bool(info.get('flag_feii_boundary'))} "
            f"feii_underfit={bool(info.get('flag_feii_underfit'))} "
            f"cont_eats={bool(info.get('flag_continuum_eats_feii'))} "
            f"broad_eats={bool(info.get('flag_broad_hb_eats_feii'))} "
            f"underOIII FeII={_fmt_float(info.get('feii_fraction_under_oiii'))} "
            f"bHb={_fmt_float(info.get('broad_hb_fraction_under_oiii'))} "
            f"wing={_fmt_float(info.get('oiii_wing_fraction_under_oiii'))}"
        )

    _plot_hb_region_panel(
        axes[0],
        wave,
        flux,
        result,
        (4200.0, 5700.0),
        "full Hbeta + [O III] + Fe II window",
        show_all=True,
    )
    axes[0].set_title(title, fontsize=10)
    _plot_hb_region_panel(
        axes[1],
        wave,
        flux,
        result,
        FEII_NORM_REGION,
        "Fe II flux normalization 4434-4684",
        shade_region=FEII_NORM_REGION,
        annotation=(
            f"observed residual int={_fmt_float(info.get('blue_feii_observed_flux'))}  "
            f"FeII model int={_fmt_float(info.get('feii_flux_4434_4684', info.get('blue_feii_model_flux')))}  "
            f"width={_fmt_float(info.get('feii_width_kms'))} km/s  "
            f"shift={_fmt_float(info.get('feii_shift_kms'))} km/s"
        ),
    )
    _plot_hb_region_panel(
        axes[2],
        wave,
        flux,
        result,
        (4750.0, 5100.0),
        "Hbeta + [O III] core",
        show_all=True,
    )
    _plot_hb_region_panel(
        axes[3],
        wave,
        flux,
        result,
        (4920.0, 5050.0),
        "[O III] pedestal",
        show_all=True,
        annotation=(
            f"underOIII FeII={_fmt_float(info.get('feii_fraction_under_oiii'))}  "
            f"bHb={_fmt_float(info.get('broad_hb_fraction_under_oiii'))}  "
            f"wing={_fmt_float(info.get('oiii_wing_fraction_under_oiii'))}"
        ),
    )
    _plot_hb_region_panel(
        axes[4],
        wave,
        flux,
        result,
        FEII_RED_REGION,
        "red Fe II",
        shade_region=FEII_RED_REGION,
        annotation=(
            f"chi2={_fmt_float(info.get('feii_chi2_red', info.get('feii_red_chi2')))}  "
            f"FeII frac={_fmt_float(info.get('feii_fraction_red'))}  "
            f"broad frac={_fmt_float(info.get('broad_hb_fraction_red'))}  "
            f"cont frac={_fmt_float(info.get('continuum_fraction_red'))}"
        ),
    )
    axes[-1].set_xlabel("Rest wavelength (Angstrom)")
    for ax in axes:
        ax.set_ylabel("Flux")
        ax.legend(frameon=False, fontsize=7, ncol=2)

    _save_if_requested(fig, output_path)
    return fig


def plot_blue_balmer_fit(
    spec: AGNSpectrum,
    result: EmissionFitResult,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot blue-Balmer extended fit diagnostics."""

    wave, flux = _matched_wave_flux(spec, result)
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), gridspec_kw={"hspace": 0.35})
    panels = [
        ((3850.0, 4450.0), "blue-Balmer 3850-4450"),
        ((4070.0, 4140.0), "Hdelta region"),
        ((4305.0, 4375.0), "Hgamma region"),
        ((3850.0, 4005.0), "[Ne III] / Hepsilon blend"),
    ]
    for ax, (region, label) in zip(axes[:4], panels):
        _plot_region_model(ax, wave, flux, result, region, label)
    axes[4].axhline(0.0, color="0.55", lw=1.0)
    axes[4].plot(wave, result.residual_flux, color="0.15", lw=0.9)
    axes[4].set_xlim(3850.0, 4450.0)
    axes[4].set_title("residual")
    axes[4].set_xlabel("Rest wavelength (Angstrom)")
    axes[4].set_ylabel("Residual")
    info = getattr(result, "fit_info", {}) or {}
    fig.suptitle(
        title
        or (
            "blue-Balmer 3850-4450 "
            f"Hg_req={bool(info.get('flag_hgamma_required'))} "
            f"Hd_req={bool(info.get('flag_hdelta_required'))} "
            f"skip_BC={bool(info.get('flag_balmer_continuum_region_skipped'))} "
            f"Hg_over={bool(info.get('flag_hgamma_overfit'))} "
            f"Hd_over={bool(info.get('flag_hdelta_overfit'))}"
        ),
        y=0.995,
        fontsize=11,
    )
    _save_if_requested(fig, output_path)
    return fig


def plot_mgii_fit(
    spec: AGNSpectrum,
    result: EmissionFitResult,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Figure:
    """Plot Mg II + UV Fe II diagnostics."""

    wave, flux = _matched_wave_flux(spec, result)
    fig, axes = plt.subplots(5, 1, figsize=(9, 12), gridspec_kw={"hspace": 0.35})
    panels = [
        ((2200.0, 3090.0), "full Mg II window"),
        ((2200.0, 2700.0), "UV Fe II anchor 2200-2700"),
        ((2700.0, 2900.0), "Mg II 2700-2900"),
        ((2900.0, 3090.0), "red UV Fe II 2900-3090"),
    ]
    for ax, (region, label) in zip(axes[:4], panels):
        _plot_region_model(ax, wave, flux, result, region, label)
    axes[4].axhline(0.0, color="0.55", lw=1.0)
    axes[4].plot(wave, result.residual_flux, color="0.15", lw=0.9)
    axes[4].set_xlim(2200.0, 3090.0)
    axes[4].set_title("residual")
    axes[4].set_xlabel("Rest wavelength (Angstrom)")
    axes[4].set_ylabel("Residual")
    info = getattr(result, "fit_info", {}) or {}
    fig.suptitle(
        title
        or (
            f"MgII {info.get('selected_model', '')} "
            f"dBIC_uv={_fmt_float(info.get('delta_bic_uv_feii'))} "
            f"dBIC_broad={_fmt_float(info.get('delta_bic_mgii_broad'))} "
            f"SNR_broad={_fmt_float(info.get('mgii_broad_snr'))} "
            f"secure={bool(info.get('secure_mgii_broad'))}"
        ),
        y=0.995,
        fontsize=11,
    )
    _save_if_requested(fig, output_path)
    return fig


def _plot_region_model(
    ax: plt.Axes,
    wave: np.ndarray,
    flux: np.ndarray,
    result: EmissionFitResult,
    region: tuple[float, float],
    label: str,
) -> None:
    mask = (wave >= region[0]) & (wave <= region[1])
    ax.plot(wave[mask], flux[mask], color="0.25", lw=0.9, label="Observed")
    ax.plot(wave[mask], result.model_flux[mask], color="#d62728", lw=1.2, label="Model")
    ax.plot(wave[mask], result.continuum_flux[mask], color="#1f77b4", lw=0.9, ls="--", label="Continuum")
    for name, values in (getattr(result, "component_flux", {}) or {}).items():
        values = np.asarray(values, dtype=float)
        if values.shape != wave.shape:
            continue
        ax.plot(wave[mask], values[mask], lw=0.75, ls=":", label=name)
    ax.set_xlim(region)
    ax.set_title(label, fontsize=9)
    ax.set_ylabel("Flux")
    ax.legend(frameon=False, fontsize=7, ncol=3)


def plot_feii_broadening_grid(
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | str | Path,
    output_path: str | Path | None = None,
    sigma_grid: tuple[float, ...] | list[float] = (800.0, 1200.0, 2000.0, 3000.0, 4000.0),
    velocity: float = 0.0,
) -> Figure:
    """Plot original and log-lambda broadened Fe II templates across sigma values."""

    if isinstance(feii_template, (str, Path)):
        loaded = load_feii_template(feii_template_path=feii_template)
        if loaded is None:
            raise ValueError("Fe II template could not be loaded.")
        template_wave, template_flux = loaded.wave, loaded.flux
    elif isinstance(feii_template, tuple):
        template_wave, template_flux = feii_template
    else:
        data = np.asarray(feii_template, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Fe II template array must have wavelength and flux columns.")
        template_wave, template_flux = data[:, 0], data[:, 1]

    finite = np.isfinite(template_wave) & np.isfinite(template_flux)
    template_wave = np.asarray(template_wave[finite], dtype=float)
    template_flux = np.asarray(template_flux[finite], dtype=float)
    order = np.argsort(template_wave)
    template_wave = template_wave[order]
    template_flux = template_flux[order]
    output_wave = np.linspace(
        max(4200.0, float(np.nanmin(template_wave))),
        min(5700.0, float(np.nanmax(template_wave))),
        3000,
    )
    original_on_output = np.interp(output_wave, template_wave, template_flux, left=0.0, right=0.0)
    norm_mask = (output_wave >= FEII_NORM_REGION[0]) & (output_wave <= FEII_NORM_REGION[1])
    original_norm = float(np.trapz(original_on_output[norm_mask], output_wave[norm_mask]))
    if np.isfinite(original_norm) and original_norm > 0.0:
        original_on_output = original_on_output / original_norm

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.0, 10.0),
        sharex=False,
        gridspec_kw={"hspace": 0.3},
    )
    panels = (
        (axes[0], (output_wave[0], output_wave[-1]), "full template"),
        (axes[1], FEII_NORM_REGION, "4434-4684 normalization"),
        (axes[2], (4920.0, 5050.0), "[O III] pedestal"),
        (axes[3], FEII_RED_REGION, "5100-5600 red Fe II"),
    )
    for ax, region, label in panels:
        mask_original = (output_wave >= region[0]) & (output_wave <= region[1])
        ax.plot(
            output_wave[mask_original],
            original_on_output[mask_original],
            color="0.2",
            lw=0.9,
            label="original unit",
        )
        mask_output = (output_wave >= region[0]) & (output_wave <= region[1])
        for sigma in sigma_grid:
            unit = broaden_shift_feii_template(
                template_wave,
                template_flux,
                output_wave,
                width_kms=float(sigma),
                shift_kms=float(velocity),
            )
            ax.plot(
                output_wave[mask_output],
                unit[mask_output],
                lw=0.9,
                label=f"{sigma:.0f} km/s",
            )
        ax.set_xlim(region)
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("Unit Fe II")
        ax.legend(frameon=False, fontsize=7, ncol=3)
    axes[-1].set_xlabel("Rest wavelength (Angstrom)")
    fig.suptitle(f"Fe II broadening grid | velocity={velocity:.0f} km/s", fontsize=11)
    _save_if_requested(fig, output_path)
    return fig


def _plot_hb_region_panel(
    ax: plt.Axes,
    wave: np.ndarray,
    flux: np.ndarray,
    result: EmissionFitResult,
    region: tuple[float, float],
    label: str,
    show_all: bool = False,
    shade_region: tuple[float, float] | None = None,
    annotation: str | None = None,
) -> None:
    mask = (wave >= region[0]) & (wave <= region[1])
    components = getattr(result, "component_flux", {}) or {}
    continuum = np.asarray(components.get("continuum", result.continuum_flux), dtype=float)
    if shade_region is not None:
        ax.axvspan(shade_region[0], shade_region[1], color="#efe6ff", alpha=0.45, label=label)
    ax.plot(wave[mask], flux[mask], color="0.25", lw=0.85, label="observed residual")
    ax.plot(wave[mask], result.model_flux[mask], color="#d62728", lw=1.2, label="total model")
    ax.plot(wave[mask], continuum[mask], color="#1f77b4", ls="--", lw=0.9, label="local continuum")

    styles = {
        "FeII": ("#9467bd", "-", "Fe II"),
        "Hb_narrow": ("#2ca02c", ":", "narrow Hbeta"),
        "Hb_broad_1": ("#ff7f0e", "--", "broad Hbeta 1"),
        "Hb_broad_2": ("#bcbd22", "--", "broad Hbeta 2"),
        "Hb_broad_combined": ("#8c564b", "-.", "combined broad Hbeta"),
        "OIII_core": ("#17becf", ":", "[O III] core"),
        "OIII_wing": ("#e377c2", ":", "[O III] wing"),
    }
    names = (
        tuple(styles)
        if show_all
        else ("FeII", "Hb_broad_combined", "OIII_core", "OIII_wing")
    )
    for name in names:
        values = components.get(name)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        if len(values) != len(wave):
            continue
        color, ls, component_label = styles[name]
        ax.plot(
            wave[mask],
            continuum[mask] + values[mask],
            color=color,
            ls=ls,
            lw=0.85,
            alpha=0.85,
            label=component_label,
        )
    ax.set_xlim(region)
    ax.set_title(label, fontsize=9)
    if annotation:
        ax.text(
            0.01,
            0.93,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
        )


def plot_epoch_overlay(
    spec1: AGNSpectrum,
    result1: EmissionFitResult,
    spec2: AGNSpectrum,
    result2: EmissionFitResult,
    output_path: str | Path | None = None,
) -> Figure:
    """Overlay two spectra and their fitted models in the same fitting window."""

    wave1, flux1 = _matched_wave_flux(spec1, result1)
    wave2, flux2 = _matched_wave_flux(spec2, result2)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    label1 = _epoch_label(spec1, "Epoch 1")
    label2 = _epoch_label(spec2, "Epoch 2")
    ax.plot(wave1, flux1, color="0.35", lw=1.0, alpha=0.85, label=f"{label1} observed")
    ax.plot(wave1, result1.model_flux, color="#1f77b4", lw=1.5, label=f"{label1} model")
    ax.plot(wave2, flux2, color="0.65", lw=1.0, alpha=0.85, label=f"{label2} observed")
    ax.plot(wave2, result2.model_flux, color="#d62728", lw=1.5, label=f"{label2} model")
    ax.set_xlabel("Rest wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.legend(loc="best", frameon=False, ncol=2)

    _save_if_requested(fig, output_path)
    return fig


def plot_bpt_migration(
    comparison: EpochComparisonResult,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot N II BPT coordinates for two epochs with a migration arrow."""

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    x1, y1 = comparison.bpt_sdss
    x2, y2 = comparison.bpt_desi

    _plot_bpt_demarcation(ax)
    ax.scatter([x1], [y1], color="#1f77b4", s=55, label="Epoch 1", zorder=3)
    ax.scatter([x2], [y2], color="#d62728", s=55, label="Epoch 2", zorder=3)
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "0.2"},
    )
    ax.set_xlabel(r"log([N II]/H$\alpha$)")
    ax.set_ylabel(r"log([O III]/H$\beta$)")
    ax.legend(loc="best", frameon=False)

    _save_if_requested(fig, output_path)
    return fig


def plot_line_flux_change_summary(
    comparison: EpochComparisonResult,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot flux ratios for key emission-line groups."""

    labels, ratios = _key_line_ratios(comparison.flux_ratio)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(labels))
    ax.axhline(1.0, color="0.5", lw=1.0, ls="--")
    ax.bar(x, ratios, color="#4c78a8", width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flux ratio (epoch 2 / epoch 1)")
    ax.set_ylim(bottom=0.0)

    _save_if_requested(fig, output_path)
    return fig


def plot_iterative_joint_fit(
    result: Any,
    output_path: str | Path | None = None,
) -> Figure:
    """Plot the three-pass iterative joint fit diagnostics."""

    pass1 = result.pass1_stellar_result
    pass3 = result.pass3_joint_result
    pass4 = getattr(result, "pass4_stellar_refit_result", None)
    joint = result.final_result
    if joint is None:
        raise ValueError("Iterative joint result does not contain a final fit.")
    wave = np.arange(len(joint.total_model), dtype=float)
    # Recover wavelength from the component arrays through the pass-1 model length.
    # The pipeline stores components on the original spectrum grid in diagnostics.
    wave = np.asarray(joint.diagnostics.get("wave", wave), dtype=float)
    if len(wave) != len(joint.total_model):
        wave = np.arange(len(joint.total_model), dtype=float)
    observed = joint.total_model + joint.residual
    components = joint.components
    fig, axes = plt.subplots(8, 1, figsize=(11, 18), gridspec_kw={"hspace": 0.34})

    _plot_joint_panel(
        axes[0],
        wave,
        observed,
        joint,
        (float(np.nanmin(wave)), float(np.nanmax(wave))),
        "full observed spectrum",
        ("stellar", "powerlaw", "feii"),
    )
    axes[0].plot(wave, pass1.stellar_model, color="#8c564b", lw=0.8, ls=":", label="Pass 1 stellar initial")
    if pass4 is not None:
        axes[0].plot(
            wave,
            pass4.stellar_model,
            color="#2ca02c",
            lw=0.8,
            ls=":",
            label="Pass 4 stellar updated",
        )
    title = (
        "fit_strategy=iterative_joint "
        f"chi2={_fmt_float(joint.chi2)} BIC={_fmt_float(joint.bic)} "
        f"A_star={_fmt_float(joint.params.get('stellar_scale'))} "
        f"alpha={_fmt_float(joint.params.get('pl_alpha'))} "
        f"PL5100={_fmt_float(joint.diagnostics.get('powerlaw_fraction_5100'))} "
        f"FeII4434-4684={_fmt_float(joint.params.get('feii_flux_4434_4684', joint.params.get('feii_amp')))} "
        f"smooth={_fmt_float(joint.params.get('feii_smooth_sigma_kms'))} "
        f"extra={_fmt_float(joint.params.get('feii_extra_sigma_kms', joint.params.get('feii_sigma')))} "
        f"shift={_fmt_float(joint.params.get('feii_shift_kms', joint.params.get('feii_velocity')))} "
        f"quality={joint.diagnostics.get('fit_quality')}"
    )
    axes[0].set_title(title, fontsize=10)
    axes[0].legend(frameon=False, fontsize=7, ncol=3)

    pass3_nonstellar = _nonstellar_from_components(pass3.components, wave)
    pass5_nonstellar = _nonstellar_from_components(joint.components, wave)
    axes[1].plot(wave, observed - pass1.stellar_model, color="0.25", lw=0.7, label="observed - Pass 1 stellar")
    axes[1].plot(wave, pass3_nonstellar, color="#ff7f0e", lw=1.0, label="Pass 3 nonstellar")
    axes[1].plot(wave, pass5_nonstellar, color="#d62728", lw=1.0, label="Pass 5 nonstellar")
    axes[1].set_xlim(4200, 5700)
    axes[1].set_title("residual after Pass 1 stellar subtraction")
    axes[1].legend(frameon=False, fontsize=7)

    _plot_joint_panel(axes[2], wave, observed, joint, (4200, 5700), "Hbeta full window", ("stellar", "powerlaw", "feii", "hbeta_broad_total", "hbeta_narrow", "oiii_core", "oiii_wing"))
    _plot_joint_panel(axes[3], wave, observed, joint, (4920, 5050), "[O III] pedestal", ("stellar", "powerlaw", "feii", "hbeta_broad_total", "oiii_core", "oiii_wing"))
    _plot_joint_panel(axes[4], wave, observed, joint, (5100, 5600), "red Fe II", ("powerlaw", "feii", "hbeta_broad_total"))
    _plot_joint_panel(axes[5], wave, observed, joint, (6400, 6800), "Halpha", ("stellar", "powerlaw", "halpha_narrow", "halpha_broad_total", "nii", "sii"))
    _plot_joint_panel(axes[6], wave, observed, joint, (3650, 3800), "[O II]", ("stellar", "powerlaw", "oii"))
    axes[7].axhline(0.0, color="0.6", lw=0.8)
    axes[7].plot(wave, joint.residual, color="0.2", lw=0.7)
    axes[7].set_title("final residual")
    axes[7].set_xlabel("Rest wavelength (Angstrom)")
    for ax in axes:
        ax.set_ylabel("Flux")
    _save_if_requested(fig, output_path)
    return fig


def _nonstellar_from_components(
    components: dict[str, np.ndarray],
    wave: np.ndarray,
) -> np.ndarray:
    total = np.zeros_like(wave)
    for name, values in components.items():
        if name == "stellar":
            continue
        if len(values) == len(wave):
            total = total + values
    return total


def _plot_joint_panel(
    ax: plt.Axes,
    wave: np.ndarray,
    observed: np.ndarray,
    joint: Any,
    region: tuple[float, float],
    title: str,
    component_names: tuple[str, ...],
) -> None:
    mask = (wave >= region[0]) & (wave <= region[1])
    components = joint.components
    ax.plot(wave[mask], observed[mask], color="0.25", lw=0.75, label="observed")
    ax.plot(wave[mask], joint.total_model[mask], color="#d62728", lw=1.1, label="total")
    styles = {
        "stellar": ("#8c564b", "--", "stellar"),
        "powerlaw": ("#1f77b4", "--", "power-law"),
        "feii": ("#9467bd", "-", "Fe II"),
        "hbeta_narrow": ("#2ca02c", ":", "narrow Hbeta"),
        "hbeta_broad_total": ("#ff7f0e", "-.", "broad Hbeta"),
        "oiii_core": ("#17becf", ":", "[O III] core"),
        "oiii_wing": ("#e377c2", ":", "[O III] wing"),
        "halpha_narrow": ("#2ca02c", ":", "narrow Halpha"),
        "halpha_broad_total": ("#ff7f0e", "-.", "broad Halpha"),
        "nii": ("#17becf", ":", "[N II]"),
        "sii": ("#e377c2", ":", "[S II]"),
        "oii": ("#17becf", ":", "[O II]"),
    }
    for name in component_names:
        values = components.get(name)
        if values is None or len(values) != len(wave):
            continue
        color, ls, label = styles.get(name, ("0.5", ":", name))
        ax.plot(wave[mask], values[mask], color=color, ls=ls, lw=0.85, label=label)
    ax.set_xlim(region)
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=7, ncol=3)


def _matched_wave_flux(
    spec: AGNSpectrum,
    result: EmissionFitResult,
) -> tuple[np.ndarray, np.ndarray]:
    n_model = len(result.model_flux)
    if len(spec.wave) == n_model:
        return spec.wave, spec.flux

    if result.window_name in FIT_WINDOWS:
        rest_min, rest_max = FIT_WINDOWS[result.window_name]
        selected = spec.select_window(rest_min, rest_max).mask_bad_pixels()
        if len(selected.wave) == n_model:
            return selected.wave, selected.flux

    raise ValueError("Could not match spectrum wavelengths to fit result arrays.")


def _plot_fit_components(
    ax: plt.Axes,
    wave: np.ndarray,
    result: EmissionFitResult,
) -> None:
    components = getattr(result, "component_flux", {}) or {}
    styles = {
        "FeII": {"color": "#9467bd", "ls": "-", "label": "Fe II"},
        "Hb_narrow": {"color": "#2ca02c", "ls": ":", "label": r"H$\beta$ narrow"},
        "Hb_broad_1": {"color": "#ff7f0e", "ls": "--", "label": r"H$\beta$ broad 1"},
        "Hb_broad_2": {"color": "#bcbd22", "ls": "--", "label": r"H$\beta$ broad 2"},
        "Hb_broad_combined": {
            "color": "#8c564b",
            "ls": "-.",
            "label": r"H$\beta$ broad total",
        },
        "OIII_core": {"color": "#17becf", "ls": ":", "label": "[O III] core"},
        "OIII_wing": {"color": "#e377c2", "ls": ":", "label": "[O III] wing"},
        "Ha_narrow": {"color": "#2ca02c", "ls": ":", "label": r"H$\alpha$ narrow"},
        "Ha_broad_1": {"color": "#ff7f0e", "ls": "--", "label": r"H$\alpha$ broad 1"},
        "Ha_broad_2": {"color": "#bcbd22", "ls": "--", "label": r"H$\alpha$ broad 2"},
        "Ha_broad_combined": {
            "color": "#8c564b",
            "ls": "-.",
            "label": r"H$\alpha$ broad total",
        },
        "NII": {"color": "#17becf", "ls": ":", "label": "[N II]"},
        "SII": {"color": "#e377c2", "ls": ":", "label": "[S II]"},
    }
    continuum = components.get("continuum", result.continuum_flux)
    for name, values in components.items():
        if name == "continuum" or name not in styles:
            continue
        values = np.asarray(values, dtype=float)
        if len(values) != len(wave):
            continue
        style = styles[name]
        ax.plot(
            wave,
            continuum + values,
            color=style["color"],
            lw=0.9,
            ls=style["ls"],
            alpha=0.8,
            label=style["label"],
        )


def _epoch_label(spec: AGNSpectrum, fallback: str) -> str:
    if spec.survey:
        return spec.survey
    return fallback


def _plot_bpt_demarcation(ax: plt.Axes) -> None:
    x_kauffmann = np.linspace(-1.5, 0.04, 200)
    y_kauffmann = 0.61 / (x_kauffmann - 0.05) + 1.3
    ax.plot(x_kauffmann, y_kauffmann, color="0.55", lw=1.0, ls=":", label="Kauffmann 2003")

    x_kewley = np.linspace(-1.5, 0.45, 200)
    y_kewley = 0.61 / (x_kewley - 0.47) + 1.19
    ax.plot(x_kewley, y_kewley, color="0.35", lw=1.0, ls="-.", label="Kewley 2001")


def _key_line_ratios(flux_ratio: dict[str, float]) -> tuple[list[str], list[float]]:
    candidates = [
        ("[O II]", ("OII_3727_blend", "OII_3727")),
        (r"H$\beta$", ("Hb_narrow",)),
        ("[O III]", ("OIII_5007",)),
        (r"H$\alpha$", ("Ha_narrow",)),
        ("[N II]", ("NII_6584",)),
    ]
    labels: list[str] = []
    ratios: list[float] = []
    for label, names in candidates:
        value = _first_available(flux_ratio, names)
        if value is not None:
            labels.append(label)
            ratios.append(value)

    sii_values = [
        flux_ratio[name]
        for name in ("SII_6717", "SII_6716", "SII_6731")
        if name in flux_ratio
    ]
    if sii_values:
        labels.append("[S II]")
        ratios.append(float(np.nanmean(sii_values)))
    return labels, ratios


def _first_available(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    if "OII_3726" in values and "OII_3729" in values:
        return float(np.nanmean([values["OII_3726"], values["OII_3729"]]))
    return None


def _fmt_float(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.1f}"


def _save_if_requested(fig: Figure, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


__all__ = [
    "plot_emission_fit",
    "plot_blue_balmer_fit",
    "plot_feii_broadening_grid",
    "plot_hb_oiii_feii_diagnostic",
    "plot_mgii_fit",
    "plot_iterative_joint_fit",
    "plot_epoch_overlay",
    "plot_bpt_migration",
    "plot_line_flux_change_summary",
]
