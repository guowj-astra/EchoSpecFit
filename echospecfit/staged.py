"""Two-stage host decomposition and optical Fe II continuum fitting."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares, lsq_linear


C_KMS = 299792.458
GAUSSIAN_FWHM = 2.3548200450309493
FEII_TEMPLATE_FWHM_KMS = 900.0
FEII_TEMPLATE_SIGMA_KMS = FEII_TEMPLATE_FWHM_KMS / GAUSSIAN_FWHM
FEII_NORM_WINDOW = (4434.0, 4684.0)
DEFAULT_STAGE2_WINDOW = (4200.0, 5600.0)
DEFAULT_STAGE2_EMISSION_MASKS = (
    (4240.0, 4430.0),
    (4450.0, 4495.0),
    (4650.0, 4725.0),
    (4740.0, 5055.0),
    (5140.0, 5225.0),
    (5255.0, 5320.0),
    (5385.0, 5435.0),
)
DEFAULT_PPXF_FEII_WINDOWS = (
    (2200.0, 3090.0),
    (4434.0, 4684.0),
    (4920.0, 5050.0),
    (5100.0, 5600.0),
)
DEFAULT_STAGE2_POWERLAW_GRID = (-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0)
DEFAULT_FEII_SIGMA_GRID = (400.0, 600.0, 900.0, 1300.0, 1800.0, 2500.0, 3500.0, 4800.0)
DEFAULT_FEII_SHIFT_GRID = (-1200.0, -900.0, -600.0, -300.0, 0.0, 300.0, 600.0, 900.0, 1200.0)
DEFAULT_HOST_POWERLAW_SLOPES = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0)
DEFAULT_TEMPLATE_AGES_GYR = (0.1, 0.5, 1.0, 3.0, 6.0, 10.0, 14.0)


def interval_mask(
    wave: np.ndarray,
    intervals: Iterable[tuple[float, float]],
) -> np.ndarray:
    """Return a Boolean mask that is true inside any interval."""

    mask = np.zeros_like(np.asarray(wave), dtype=bool)
    for low, high in intervals:
        mask |= (wave >= low) & (wave <= high)
    return mask


def downweight_spectrum_for_host(
    spec: Any,
    windows: tuple[tuple[float, float], ...] = DEFAULT_PPXF_FEII_WINDOWS,
    error_scale: float = 1.0e6,
) -> tuple[Any, np.ndarray]:
    """Keep the full grid while removing Fe II windows from pPXF leverage."""

    from echospec import AGNSpectrum

    wave = np.asarray(spec.wave, dtype=float)
    err = np.asarray(spec.err, dtype=float).copy()
    ignored = interval_mask(wave, windows)
    finite = err[np.isfinite(err) & (err > 0.0)]
    if finite.size == 0:
        raise ValueError("No positive finite uncertainties are available for pPXF.")
    high_noise = max(float(np.nanmax(finite)), float(np.nanmedian(finite))) * float(
        error_scale
    )
    err[ignored] = high_noise
    meta = dict(getattr(spec, "meta", {}) or {})
    meta.update(
        {
            "desipecfit_ppxf_feii_windows_downweighted": [list(item) for item in windows],
            "desipecfit_ppxf_feii_error_scale": float(error_scale),
        }
    )
    return (
        AGNSpectrum(
            wave=wave.copy(),
            flux=np.asarray(spec.flux, dtype=float).copy(),
            err=err,
            z=float(getattr(spec, "z", 0.0)),
            survey=str(getattr(spec, "survey", "DESI")),
            object_id=str(getattr(spec, "object_id", "")),
            meta=meta,
        ),
        ignored,
    )


def fit_fixed_host_feii(
    spec: Any,
    stellar: Any,
    *,
    feii_template_path: str | Path,
    fit_window: tuple[float, float] = DEFAULT_STAGE2_WINDOW,
    emission_masks: tuple[tuple[float, float], ...] = DEFAULT_STAGE2_EMISSION_MASKS,
    powerlaw_grid: tuple[float, ...] = DEFAULT_STAGE2_POWERLAW_GRID,
    feii_sigma_grid: tuple[float, ...] = DEFAULT_FEII_SIGMA_GRID,
    feii_shift_grid: tuple[float, ...] = DEFAULT_FEII_SHIFT_GRID,
    host_scale_max: float = 5.0,
) -> dict[str, Any]:
    """Fit host amplitude, one power law, and broadened optical Fe II.

    The pPXF stellar mixture and kinematics are represented by ``stellar_model``
    and remain fixed. Only its non-negative amplitude is allowed to change.
    The likelihood uses the input spectrum uncertainties without an error floor.
    """

    wave = np.asarray(spec.wave, dtype=float)
    flux = np.asarray(spec.flux, dtype=float)
    err = np.asarray(spec.err, dtype=float)
    host_shape = np.asarray(stellar.stellar_model, dtype=float)
    if not (len(wave) == len(flux) == len(err) == len(host_shape)):
        raise ValueError("Spectrum and pPXF host arrays must share one wavelength grid.")

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(err)
        & (err > 0.0)
        & np.isfinite(host_shape)
        & (wave >= fit_window[0])
        & (wave <= fit_window[1])
    )
    emission_mask = interval_mask(wave, emission_masks) & valid
    fit_mask = valid & ~emission_mask
    if np.count_nonzero(fit_mask) < 80:
        raise ValueError("Too few unmasked pixels for the staged continuum fit.")

    template_wave, template_flux = _load_feii_template(Path(feii_template_path))
    sigma_grid = np.asarray(feii_sigma_grid, dtype=float)
    shift_grid = np.asarray(feii_shift_grid, dtype=float)
    slope_grid = np.asarray(powerlaw_grid, dtype=float)
    if sigma_grid.size == 0 or shift_grid.size == 0 or slope_grid.size == 0:
        raise ValueError("Fe II width, shift, and power-law grids must be non-empty.")

    best: dict[str, float] | None = None
    for sigma in sigma_grid:
        for shift in shift_grid:
            unit = _feii_unit(
                wave,
                template_wave,
                template_flux,
                sigma_kms=float(sigma),
                shift_kms=float(shift),
            )
            for alpha in slope_grid:
                powerlaw_shape = np.power(wave / 5100.0, float(alpha))
                matrix = np.column_stack((host_shape, powerlaw_shape, unit))
                coeff, chi2 = _solve_nonnegative_linear(matrix, flux, err, fit_mask)
                candidate = {
                    "chi2": float(chi2),
                    "host_scale": float(coeff[0]),
                    "powerlaw_amp": float(coeff[1]),
                    "feii_flux_4434_4684": float(coeff[2]),
                    "powerlaw_alpha": float(alpha),
                    "feii_sigma_kms": float(sigma),
                    "feii_shift_kms": float(shift),
                }
                if best is None or candidate["chi2"] < best["chi2"]:
                    best = candidate
    if best is None:
        raise RuntimeError("No staged Fe II grid candidate completed.")

    level = float(np.nanmedian(np.abs(flux[fit_mask])))
    if not np.isfinite(level) or level <= 0.0:
        level = 1.0
    sigma_half_width = max(300.0, 0.25 * best["feii_sigma_kms"])
    alpha_bounds = (
        max(-4.0, best["powerlaw_alpha"] - 0.75),
        min(2.0, best["powerlaw_alpha"] + 0.75),
    )
    sigma_bounds = (
        max(float(sigma_grid.min()), best["feii_sigma_kms"] - sigma_half_width),
        min(float(sigma_grid.max()), best["feii_sigma_kms"] + sigma_half_width),
    )
    shift_bounds = (
        max(-1500.0, best["feii_shift_kms"] - 300.0),
        min(1500.0, best["feii_shift_kms"] + 300.0),
    )
    x0 = np.array(
        [
            best["host_scale"],
            best["powerlaw_amp"],
            best["feii_flux_4434_4684"],
            best["powerlaw_alpha"],
            best["feii_sigma_kms"],
            best["feii_shift_kms"],
        ],
        dtype=float,
    )
    lower = np.array(
        [0.0, 0.0, 0.0, alpha_bounds[0], sigma_bounds[0], shift_bounds[0]],
        dtype=float,
    )
    upper = np.array(
        [
            float(host_scale_max),
            20.0 * level,
            5000.0 * level,
            alpha_bounds[1],
            sigma_bounds[1],
            shift_bounds[1],
        ],
        dtype=float,
    )
    x0 = np.clip(x0, lower + 1.0e-10, upper - 1.0e-10)

    def components(
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        host = params[0] * host_shape
        powerlaw = params[1] * np.power(wave / 5100.0, params[3])
        feii = params[2] * _feii_unit(
            wave,
            template_wave,
            template_flux,
            sigma_kms=params[4],
            shift_kms=params[5],
        )
        return host, powerlaw, feii, host + powerlaw + feii

    def residual(params: np.ndarray) -> np.ndarray:
        model = components(params)[3]
        return (flux[fit_mask] - model[fit_mask]) / err[fit_mask]

    nonlinear = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="linear",
        x_scale="jac",
        max_nfev=800,
    )
    params = np.asarray(nonlinear.x, dtype=float)
    host_model, powerlaw_model, feii_model, total_model = components(params)
    pull = np.full_like(flux, np.nan, dtype=float)
    pull[valid] = (flux[valid] - total_model[valid]) / err[valid]
    chi2 = float(np.sum(np.square(pull[fit_mask])))
    ndata = int(np.count_nonzero(fit_mask))
    nparams = 6

    nofe_best_chi2 = float("inf")
    for alpha in slope_grid:
        matrix = np.column_stack(
            (host_shape, np.power(wave / 5100.0, float(alpha)))
        )
        _, nofe_chi2 = _solve_nonnegative_linear(matrix, flux, err, fit_mask)
        nofe_best_chi2 = min(nofe_best_chi2, nofe_chi2)
    bic = chi2 + nparams * math.log(max(ndata, 1))
    nofe_bic = nofe_best_chi2 + 3 * math.log(max(ndata, 1))

    continuum_without_feii = host_model + powerlaw_model
    blue = (
        (wave >= FEII_NORM_WINDOW[0])
        & (wave <= FEII_NORM_WINDOW[1])
        & np.isfinite(continuum_without_feii)
    )
    continuum_level = float(np.nanmedian(continuum_without_feii[blue]))
    feii_flux = float(params[2])
    feii_ew = feii_flux / continuum_level if continuum_level > 0.0 else float("nan")

    sample_5100 = int(np.nanargmin(np.abs(wave - 5100.0)))
    denominator_5100 = host_model[sample_5100] + powerlaw_model[sample_5100]
    host_fraction_5100 = (
        float(host_model[sample_5100] / denominator_5100)
        if denominator_5100 > 0.0
        else float("nan")
    )
    sigma_at_bound = bool(
        abs(params[4] - sigma_bounds[0]) < 1.0
        or abs(params[4] - sigma_bounds[1]) < 1.0
    )
    shift_at_bound = bool(
        abs(params[5] - shift_bounds[0]) < 1.0
        or abs(params[5] - shift_bounds[1]) < 1.0
    )
    delta_bic_feii = float(nofe_bic - bic)
    stellar_velocity = float(getattr(stellar, "velocity", np.nan))
    stellar_sigma = float(getattr(stellar, "sigma", np.nan))
    host_kinematics_reliable = bool(
        np.isfinite(stellar_velocity)
        and np.isfinite(stellar_sigma)
        and abs(stellar_velocity) < 500.0
        and 0.0 < stellar_sigma < 500.0
    )
    host_decomposition_reliable = bool(
        host_fraction_5100 < 0.10 or host_kinematics_reliable
    )
    feii_detected = bool(delta_bic_feii > 10.0 and feii_flux > 0.0)
    feii_well_constrained = bool(
        feii_detected and not sigma_at_bound and not shift_at_bound
    )
    feii_secure = bool(
        feii_well_constrained
        and host_decomposition_reliable
        and chi2 / max(ndata - nparams, 1) < 3.0
    )
    if feii_secure:
        quality = "secure"
    elif not feii_detected:
        quality = "weak_or_undetected"
    elif sigma_at_bound or shift_at_bound:
        quality = "parameter_boundary"
    elif not host_decomposition_reliable:
        quality = "host_decomposition_unreliable"
    else:
        quality = "continuum_mismatch"
    return {
        "enabled": True,
        "success": bool(nonlinear.success),
        "usable": bool(np.isfinite(chi2)),
        "reason": str(nonlinear.message),
        "message": str(nonlinear.message),
        "errors": "input_spectrum_errors",
        "host_kinematics": "fixed_from_ppxf",
        "host_scale": float(params[0]),
        "host_fraction_hb": host_fraction_5100,
        "host_fraction_5100": host_fraction_5100,
        "powerlaw_amp_5100": float(params[1]),
        "powerlaw_alpha": float(params[3]),
        "feii_amp": feii_flux,
        "feii_flux_4434_4684": feii_flux,
        "feii_ew_4434_4684": float(feii_ew),
        "feii_velocity": float(params[5]),
        "feii_shift_kms": float(params[5]),
        "feii_sigma": float(params[4]),
        "feii_sigma_kms": float(params[4]),
        "feii_fwhm_kms": float(GAUSSIAN_FWHM * params[4]),
        "feii_sigma_at_bound": sigma_at_bound,
        "feii_shift_at_bound": shift_at_bound,
        "feii_detected": feii_detected,
        "feii_well_constrained": feii_well_constrained,
        "feii_secure": feii_secure,
        "quality": quality,
        "host_kinematics_reliable": host_kinematics_reliable,
        "host_decomposition_reliable": host_decomposition_reliable,
        "chi2": chi2,
        "redchi": chi2 / max(ndata - nparams, 1),
        "bic": float(bic),
        "no_feii_bic": float(nofe_bic),
        "delta_bic_feii": delta_bic_feii,
        "ndata": ndata,
        "nparams": nparams,
        "fit_window": tuple(float(value) for value in fit_window),
        "emission_masks": tuple(tuple(float(value) for value in item) for item in emission_masks),
        "valid_mask": valid,
        "fit_mask": fit_mask,
        "emission_mask": emission_mask,
        "host_model": host_model,
        "powerlaw_model": powerlaw_model,
        "feii_model": feii_model,
        "total_model": total_model,
        "residual": flux - total_model,
        "pull": pull,
        "coarse_grid_best": best,
        "parameter_bounds": {
            "powerlaw_alpha": alpha_bounds,
            "feii_sigma_kms": sigma_bounds,
            "feii_shift_kms": shift_bounds,
        },
    }


def continuum_subtracted_spectrum(
    spec: Any,
    staged: dict[str, Any],
    *,
    subtract_feii: bool = True,
) -> Any:
    """Subtract the staged host and continuum before emission-line fitting."""

    from echospec import AGNSpectrum

    if not staged.get("usable"):
        raise ValueError("The staged continuum result is not usable.")
    continuum = np.asarray(staged["host_model"], dtype=float) + np.asarray(
        staged["powerlaw_model"], dtype=float
    )
    if subtract_feii:
        continuum = continuum + np.asarray(staged["feii_model"], dtype=float)
    meta = dict(getattr(spec, "meta", {}) or {})
    meta["desipecfit_staged_continuum"] = {
        key: value
        for key, value in staged.items()
        if key
        not in {
            "valid_mask",
            "fit_mask",
            "emission_mask",
            "host_model",
            "powerlaw_model",
            "feii_model",
            "total_model",
            "residual",
            "pull",
        }
    }
    meta["desipecfit_staged_feii_subtracted"] = bool(subtract_feii)
    return AGNSpectrum(
        wave=np.asarray(spec.wave, dtype=float).copy(),
        flux=np.asarray(spec.flux, dtype=float) - continuum,
        err=np.asarray(spec.err, dtype=float).copy(),
        z=float(getattr(spec, "z", 0.0)),
        survey=str(getattr(spec, "survey", "DESI")),
        object_id=str(getattr(spec, "object_id", "")),
        meta=meta,
    ).mask_bad_pixels()


def write_staged_fits(
    path: str | Path,
    *,
    spectrum: Any,
    rest_spec: Any,
    stellar: Any,
    staged: dict[str, Any],
) -> Path:
    """Write measured errors, masks, and all continuum components to FITS."""

    from astropy.io import fits

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wave = np.asarray(rest_spec.wave, dtype=float)
    flux = np.asarray(rest_spec.flux, dtype=float)
    err = np.asarray(rest_spec.err, dtype=float)
    primary = fits.PrimaryHDU()
    header = primary.header
    header["OBJECT"] = str(getattr(spectrum, "object_id", ""))[:68]
    header["SURVEY"] = str(getattr(spectrum, "survey", "DESI"))[:20]
    header["REDSHIFT"] = float(getattr(spectrum, "z", 0.0))
    header["PPXFVEL"] = float(stellar.velocity)
    header["PPXFSIG"] = float(stellar.sigma)
    header["HSCALE"] = float(staged["host_scale"])
    header["PLALPHA"] = float(staged["powerlaw_alpha"])
    header["FEFLUX"] = float(staged["feii_flux_4434_4684"])
    header["FESIG"] = float(staged["feii_sigma_kms"])
    header["FEFWHM"] = float(staged["feii_fwhm_kms"])
    header["FESHIFT"] = float(staged["feii_shift_kms"])
    header["FEEW"] = float(staged["feii_ew_4434_4684"])
    header["DELBIC"] = float(staged["delta_bic_feii"])
    header["FEDETECT"] = bool(staged.get("feii_detected", False))
    header["FESECURE"] = bool(staged.get("feii_secure", False))
    header["FEQUAL"] = str(staged.get("quality", ""))[:20]
    header["ERRTYPE"] = "INPUT"
    header["HOSTKIN"] = "FIXED"

    ignored = np.asarray(
        staged.get("ppxf_feii_ignored_mask", np.zeros_like(wave, dtype=bool)),
        dtype=bool,
    )
    if ignored.shape != wave.shape:
        ignored = np.zeros_like(wave, dtype=bool)
    columns = [
        fits.Column(name="REST_WAVE", format="D", array=wave),
        fits.Column(name="FLUX", format="D", array=flux),
        fits.Column(name="ERR", format="D", array=err),
        fits.Column(name="PPXF_HOST_SHAPE", format="D", array=stellar.stellar_model),
        fits.Column(
            name="PPXF_POWERLAW",
            format="D",
            array=(
                np.zeros_like(wave)
                if stellar.agn_model is None
                else np.asarray(stellar.agn_model, dtype=float)
            ),
        ),
        fits.Column(name="HOST_MODEL", format="D", array=staged["host_model"]),
        fits.Column(name="POWERLAW_MODEL", format="D", array=staged["powerlaw_model"]),
        fits.Column(name="FEII_MODEL", format="D", array=staged["feii_model"]),
        fits.Column(name="TOTAL_MODEL", format="D", array=staged["total_model"]),
        fits.Column(name="RESIDUAL", format="D", array=staged["residual"]),
        fits.Column(name="PULL", format="D", array=staged["pull"]),
        fits.Column(name="FIT_MASK", format="L", array=staged["fit_mask"]),
        fits.Column(name="EMISSION_MASK", format="L", array=staged["emission_mask"]),
        fits.Column(name="PPXF_FEII_MASK", format="L", array=ignored),
    ]
    model_hdu = fits.BinTableHDU.from_columns(columns, name="MODEL_COMPONENTS")
    summary = {
        key: value
        for key, value in staged.items()
        if key
        not in {
            "valid_mask",
            "fit_mask",
            "emission_mask",
            "ppxf_feii_ignored_mask",
            "host_model",
            "powerlaw_model",
            "feii_model",
            "total_model",
            "residual",
            "pull",
        }
    }
    encoded = json.dumps(_json_safe(summary), sort_keys=True)
    summary_hdu = fits.BinTableHDU.from_columns(
        [fits.Column(name="JSON", format=f"{max(len(encoded), 1)}A", array=[encoded])],
        name="FIT_SUMMARY",
    )
    fits.HDUList([primary, model_hdu, summary_hdu]).writeto(output, overwrite=True)
    return output


def make_staged_diagnostic_figure(
    *,
    spectrum: Any,
    rest_spec: Any,
    stellar: Any,
    staged: dict[str, Any],
) -> Any:
    """Create the default four-panel staged continuum diagnostic."""

    import matplotlib.pyplot as plt

    wave = np.asarray(rest_spec.wave, dtype=float)
    flux = np.asarray(rest_spec.flux, dtype=float)
    err = np.asarray(rest_spec.err, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0.0)
    window = valid & (wave >= DEFAULT_STAGE2_WINDOW[0]) & (wave <= DEFAULT_STAGE2_WINDOW[1])
    colors = {
        "data": "#333333",
        "ppxf": "#D55E00",
        "host": "#0072B2",
        "powerlaw": "#CC79A7",
        "feii": "#009E73",
        "model": "#E69F00",
        "residual": "#6B7280",
    }
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(8.2, 7.4),
        sharex=False,
        gridspec_kw={"height_ratios": (1.35, 1.2, 1.15, 0.8)},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)

    axes[0].plot(wave[valid], flux[valid], color=colors["data"], lw=0.55, alpha=0.72, label="Observed")
    axes[0].plot(wave[valid], np.asarray(stellar.total_model)[valid], color=colors["ppxf"], lw=1.05, label="Stage 1 pPXF total")
    axes[0].plot(wave[valid], np.asarray(stellar.stellar_model)[valid], color=colors["host"], lw=0.9, label="pPXF host shape")
    if stellar.agn_model is not None:
        axes[0].plot(wave[valid], np.asarray(stellar.agn_model)[valid], color=colors["powerlaw"], lw=0.8, ls="--", label="pPXF power law")
    axes[0].set_title("Stage 1: host decomposition; Fe II windows downweighted", fontsize=10)
    axes[0].set_ylabel("Flux", fontsize=9)
    axes[0].legend(loc="upper right", ncol=4, fontsize=7, frameon=False)
    _robust_ylim(axes[0], flux[valid])

    axes[1].plot(wave[window], flux[window], color=colors["data"], lw=0.6, alpha=0.72, label="Observed")
    axes[1].plot(wave[window], staged["total_model"][window], color=colors["model"], lw=1.35, label="Host + power law + Fe II")
    axes[1].plot(wave[window], staged["host_model"][window], color=colors["host"], lw=0.9, label="Fixed-shape host")
    axes[1].plot(wave[window], staged["powerlaw_model"][window], color=colors["powerlaw"], lw=0.9, ls="--", label="Power law")
    axes[1].plot(wave[window], staged["feii_model"][window], color=colors["feii"], lw=1.05, label="Optical Fe II")
    _shade_masks(axes[1], staged["emission_masks"])
    axes[1].set_title("Stage 2: fixed host kinematics; input errors used", fontsize=10)
    axes[1].set_ylabel("Flux", fontsize=9)
    axes[1].legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
    _robust_ylim(axes[1], flux[window])

    host_subtracted = flux - np.asarray(staged["host_model"])
    axes[2].plot(wave[window], host_subtracted[window], color=colors["data"], lw=0.6, alpha=0.72, label="Observed - scaled host")
    axes[2].plot(wave[window], (staged["powerlaw_model"] + staged["feii_model"])[window], color=colors["model"], lw=1.25, label="Power law + Fe II")
    axes[2].fill_between(wave[window], staged["powerlaw_model"][window], (staged["powerlaw_model"] + staged["feii_model"])[window], color=colors["feii"], alpha=0.25, linewidth=0.0, label="Fe II contribution")
    axes[2].plot(wave[window], staged["powerlaw_model"][window], color=colors["powerlaw"], lw=0.9, ls="--")
    _shade_masks(axes[2], staged["emission_masks"])
    axes[2].set_ylabel("Host-subtracted flux", fontsize=9)
    axes[2].legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
    _robust_ylim(axes[2], host_subtracted[window])

    fit_mask = np.asarray(staged["fit_mask"], dtype=bool)
    axes[3].axhline(0.0, color="#111111", lw=0.65)
    pull_display = np.where(fit_mask, np.asarray(staged["pull"]), np.nan)
    axes[3].plot(wave, pull_display, color=colors["residual"], lw=0.55)
    _shade_masks(axes[3], staged["emission_masks"])
    axes[3].set_xlim(DEFAULT_STAGE2_WINDOW)
    axes[3].set_ylim(-6.0, 6.0)
    axes[3].set_ylabel("Residual / error", fontsize=9)
    axes[3].set_xlabel(r"Rest wavelength ($\AA$)", fontsize=9)

    object_id = str(getattr(spectrum, "object_id", ""))
    z = float(getattr(spectrum, "z", 0.0))
    fig.suptitle(
        f"{object_id}   z={z:.4f}   Fe II EW={staged['feii_ew_4434_4684']:.1f} A   "
        f"FWHM={staged['feii_fwhm_kms']:.0f} km s$^{{-1}}$   "
        f"host fraction$_{{5100}}$={staged['host_fraction_5100']:.2f}",
        fontsize=11,
    )
    return fig


def load_representative_stellar_templates(
    *,
    library: str,
    local_path: str | Path | None = None,
    n_metallicities: int = 6,
    target_ages_gyr: tuple[float, ...] = DEFAULT_TEMPLATE_AGES_GYR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load an age/metallicity-balanced E-MILES subset for pPXF."""

    from echospec.templates import _load_fits_template, get_template_cache_dir

    directory = (
        Path(local_path).expanduser()
        if local_path is not None
        else get_template_cache_dir() / library
    )
    grouped: dict[float, list[tuple[float, Path]]] = {}
    for path in sorted(directory.glob("*.fits")):
        parsed = _parse_emiles_name(path)
        if parsed is None:
            continue
        metallicity, age = parsed
        grouped.setdefault(metallicity, []).append((age, path))
    if not grouped:
        raise FileNotFoundError(
            f"No parseable E-MILES FITS templates were found under {directory}."
        )

    metallicities = sorted(grouped)
    n_keep = min(max(int(n_metallicities), 1), len(metallicities))
    indices = np.unique(
        np.rint(np.linspace(0, len(metallicities) - 1, n_keep)).astype(int)
    )
    files: list[Path] = []
    for index in indices:
        candidates = grouped[metallicities[int(index)]]
        for target_age in target_ages_gyr:
            _, selected = min(candidates, key=lambda item: abs(item[0] - target_age))
            files.append(selected)
    files = list(dict.fromkeys(files))

    template_wave: np.ndarray | None = None
    columns: list[np.ndarray] = []
    for path in files:
        wave, flux = _load_fits_template(path)
        wave = np.asarray(wave, dtype=float)
        if template_wave is None:
            template_wave = wave
        elif not np.allclose(template_wave, wave, rtol=0.0, atol=1.0e-8):
            raise ValueError("Selected E-MILES templates do not share one grid.")
        columns.append(np.asarray(flux, dtype=float))
    if template_wave is None or not columns:
        raise RuntimeError("No representative stellar templates were loaded.")
    return template_wave, np.column_stack(columns), [path.name for path in files]


def _parse_emiles_name(path: Path) -> tuple[float, float] | None:
    match = re.search(
        r"Z(?P<sign>[mp])(?P<metal>[0-9]+\.[0-9]+)T(?P<age>[0-9]+\.[0-9]+)_i[PT]p0\.00_baseFe\.fits$",
        path.name,
    )
    if match is None:
        return None
    metallicity = float(match.group("metal"))
    if match.group("sign") == "m":
        metallicity *= -1.0
    return metallicity, float(match.group("age"))


@lru_cache(maxsize=8)
def _load_feii_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from echospec.feii import load_feii_template

    loaded = load_feii_template(feii_template_path=path)
    if loaded is None:
        raise FileNotFoundError(f"Could not load Fe II template: {path}")
    return np.asarray(loaded.wave, dtype=float), np.asarray(loaded.flux, dtype=float)


def _feii_unit(
    wave: np.ndarray,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    *,
    sigma_kms: float,
    shift_kms: float,
) -> np.ndarray:
    from echospec.feii import feii_unit_flux_template

    return np.asarray(
        feii_unit_flux_template(
            wave,
            template_wave,
            template_flux,
            width_kms=float(sigma_kms),
            shift_kms=float(shift_kms),
            feii_template_sigma_kms=FEII_TEMPLATE_SIGMA_KMS,
            feii_broadening_mode="target",
            norm_window=FEII_NORM_WINDOW,
        ),
        dtype=float,
    )


def _solve_nonnegative_linear(
    matrix: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    weighted_matrix = matrix[mask] / err[mask, None]
    weighted_flux = flux[mask] / err[mask]
    fit = lsq_linear(
        weighted_matrix,
        weighted_flux,
        bounds=(0.0, np.inf),
        lsmr_tol="auto",
        max_iter=1000,
    )
    coeff = np.asarray(fit.x, dtype=float)
    pull = (flux[mask] - matrix[mask] @ coeff) / err[mask]
    return coeff, float(np.sum(pull * pull))


def _shade_masks(ax: Any, intervals: Iterable[tuple[float, float]]) -> None:
    for low, high in intervals:
        ax.axvspan(low, high, color="#D1D5DB", alpha=0.30, linewidth=0.0)


def _robust_ylim(ax: Any, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 10:
        return
    low, high = np.nanpercentile(finite, (1.0, 99.0))
    span = high - low
    if np.isfinite(span) and span > 0.0:
        ax.set_ylim(low - 0.08 * span, high + 0.25 * span)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "DEFAULT_FEII_SHIFT_GRID",
    "DEFAULT_FEII_SIGMA_GRID",
    "DEFAULT_HOST_POWERLAW_SLOPES",
    "DEFAULT_PPXF_FEII_WINDOWS",
    "DEFAULT_STAGE2_EMISSION_MASKS",
    "DEFAULT_STAGE2_POWERLAW_GRID",
    "DEFAULT_STAGE2_WINDOW",
    "continuum_subtracted_spectrum",
    "downweight_spectrum_for_host",
    "fit_fixed_host_feii",
    "interval_mask",
    "load_representative_stellar_templates",
    "make_staged_diagnostic_figure",
    "write_staged_fits",
]
