"""Shared fitting infrastructure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from lmfit import Parameters, minimize

from echospec.feii import FEII_NORM_WINDOW, feii_unit_flux_template, load_feii_template
from echospec.recipes import Recipe, RecipeEmissionFitter
from echospec.spectrum import AGNSpectrum

C_KMS = 299792.458
GAUSSIAN_FWHM_FACTOR = 2.3548
BROAD_BALMER_VELOCITY_BOUNDS = (-10000.0, 10000.0)
BROAD_BALMER_SIGMA_BOUNDS = (1.0e-6, 30000.0)
HB_BROAD_1_VELOCITY_BOUNDS = (-3000.0, 3000.0)
HB_BROAD_1_SIGMA_BOUNDS = (800.0, 6000.0)
HB_BROAD_2_VELOCITY_BOUNDS = (-5000.0, 5000.0)
HB_BROAD_2_SIGMA_BOUNDS = (1500.0, 12000.0)
HA_BROAD_1_VELOCITY_BOUNDS = (-2000.0, 2000.0)
HA_BROAD_1_SIGMA_BOUNDS = (800.0, 3000.0)
HA_BROAD_2_VELOCITY_BOUNDS = (-2000.0, 2000.0)
HA_BROAD_2_SIGMA_BOUNDS = (1500.0, 3500.0)
FEII_VELOCITY_BOUNDS = (-1500.0, 1500.0)
FEII_SIGMA_BOUNDS = (0.0, 1500.0)
FEII_DEFAULT_WIDTH_BOUNDS = (0.0, 1500.0)
FEII_DEFAULT_SHIFT_BOUNDS = (-1500.0, 1500.0)
FEII_DEFAULT_SIGMA_GRID = (0.0, 300.0, 600.0, 1000.0, 1500.0)
FEII_DEFAULT_VELOCITY_GRID = (-800.0, -400.0, 0.0, 400.0, 800.0)
HBETA = 4861.33
OIII_4959 = 4958.91
OIII_5007 = 5006.84
OIII_RATIO = 2.98
DEFAULT_HB_NARROW_MIN_OIII_RATIO = 1.0 / 20.0
HB_OIII_WINDOW = (4200.0, 5700.0)
FEII_BLUE_REGION = (4434.0, 4684.0)
FEII_PEDESTAL_REGION = (4920.0, 5050.0)
FEII_RED_REGION = (5100.0, 5600.0)
OIII_UNDERLYING_FEII_REGION = (4920.0, 5050.0)
FEII_DIAGNOSTIC_MASKS = (
    (4670.0, 4705.0),
    (4810.0, 4910.0),
    (4953.0, 4965.0),
    (5001.0, 5013.0),
)
FEII_PREFIT_LINE_MASKS = FEII_DIAGNOSTIC_MASKS
HB_CONTINUUM_ANCHOR_WINDOWS = (
    (4200.0, 4230.0),
    (5650.0, 5700.0),
)
HB_CONTINUUM_FALLBACK_ANCHOR_WINDOWS = (
    (4400.0, 4425.0),
    (5550.0, 5600.0),
)
HALPHA = 6562.80
OI_6300 = 6300.30
OI_6363 = 6363.78
OI_RATIO = 3.0
NII_6548 = 6548.05
NII_6584 = 6583.45
SII_6716 = 6716.44
SII_6731 = 6730.82
NII_RATIO = 2.96
HA_NII_SII_WINDOW = (6250.0, 6800.0)
OII_3726 = 3726.03
OII_BLEND = 3727.50
OII_3729 = 3728.82
OII_WINDOW = (3650.0, 3800.0)
CONTINUUM_PIVOT = 4900.0
_LEGACY_RECIPE_DIR = Path(__file__).resolve().parents[1] / "recipes"
RECIPE_DIR = (
    _LEGACY_RECIPE_DIR
    if _LEGACY_RECIPE_DIR.is_dir()
    else Path(__file__).resolve().parent / "data" / "recipes"
)


@dataclass
class EmissionFitResult:
    """Container for emission-line fit outputs."""

    success: bool
    message: str
    params: dict[str, float]
    line_flux: dict[str, float]
    line_flux_err: dict[str, float]
    model_flux: np.ndarray
    continuum_flux: np.ndarray
    residual_flux: np.ndarray
    chi2: float
    redchi: float
    ndata: int
    nvarys: int
    window_name: str
    param_err: dict[str, float] = field(default_factory=dict)
    mc_summary: dict[str, Any] = field(default_factory=dict)
    fit_info: dict[str, Any] = field(default_factory=dict)
    component_flux: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def bic(self) -> float:
        """Bayesian information criterion for this fit."""

        return fit_bic(self)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Alias for fit diagnostics."""

        return self.fit_info

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly summary of the fit."""

        summary: dict[str, Any] = {
            "success": self.success,
            "message": self.message,
            "window_name": self.window_name,
            "chi2": self.chi2,
            "redchi": self.redchi,
            "bic": self.bic,
            "ndata": self.ndata,
            "nvarys": self.nvarys,
            "params": self.params,
            "param_err": self.param_err,
            "line_flux": self.line_flux,
            "line_flux_err": self.line_flux_err,
            "mc_summary": self.mc_summary,
            "fit_info": self.fit_info,
            "warnings": self.warnings,
        }
        if include_arrays:
            summary.update(
                {
                    "model_flux": self.model_flux.tolist(),
                    "continuum_flux": self.continuum_flux.tolist(),
                    "residual_flux": self.residual_flux.tolist(),
                    "component_flux": {
                        name: values.tolist()
                        for name, values in self.component_flux.items()
                    },
                }
            )
        return summary


@dataclass(frozen=True)
class FeIIPrefitResult:
    """DASpec-style Fe II grid prefit summary."""

    best_feii_flux_4434_4684: float
    best_feii_width_kms: float
    best_feii_extra_sigma_kms: float
    best_feii_shift_kms: float
    best_feii_template_method: str
    best_feii_smooth_sigma_kms: float
    chi2_blue: float
    chi2_pedestal: float
    chi2_red: float
    chi2_total: float
    second_best_chi2: float
    chi2_grid: np.ndarray
    flux_grid: np.ndarray
    method_grid: tuple[str, ...]
    smooth_sigma_grid: np.ndarray
    width_grid: np.ndarray
    shift_grid: np.ndarray
    continuum_anchor_params: dict[str, Any]
    flags: dict[str, bool]
    fit_mask_npix: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_feii_flux_4434_4684": float(self.best_feii_flux_4434_4684),
            "best_feii_width_kms": float(self.best_feii_width_kms),
            "best_feii_extra_sigma_kms": float(self.best_feii_extra_sigma_kms),
            "best_feii_shift_kms": float(self.best_feii_shift_kms),
            "best_feii_template_method": self.best_feii_template_method,
            "best_feii_smooth_sigma_kms": float(self.best_feii_smooth_sigma_kms),
            "chi2_blue": float(self.chi2_blue),
            "chi2_pedestal": float(self.chi2_pedestal),
            "chi2_red": float(self.chi2_red),
            "chi2_total": float(self.chi2_total),
            "second_best_chi2": float(self.second_best_chi2),
            "chi2_grid": self.chi2_grid.tolist(),
            "flux_grid": self.flux_grid.tolist(),
            "method_grid": list(self.method_grid),
            "smooth_sigma_grid": self.smooth_sigma_grid.tolist(),
            "width_grid": self.width_grid.tolist(),
            "shift_grid": self.shift_grid.tolist(),
            "continuum_anchor_params": self.continuum_anchor_params,
            "flags": self.flags,
            "fit_mask_npix": int(self.fit_mask_npix),
        }


@dataclass
class FullSpectrumFitResult:
    """Full-spectrum refinement of continuum and emission components."""

    success: bool
    message: str
    params: dict[str, float]
    line_flux: dict[str, float]
    model_flux: np.ndarray
    stellar_flux: np.ndarray
    agn_flux: np.ndarray
    emission_flux: np.ndarray
    residual_flux: np.ndarray
    chi2: float
    redchi: float
    ndata: int
    nvarys: int
    fit_info: dict[str, Any] = field(default_factory=dict)

    @property
    def bic(self) -> float:
        if self.ndata <= 0:
            return float("nan")
        return float(self.chi2 + self.nvarys * np.log(self.ndata))

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "success": self.success,
            "message": self.message,
            "chi2": self.chi2,
            "redchi": self.redchi,
            "bic": self.bic,
            "ndata": self.ndata,
            "nvarys": self.nvarys,
            "params": self.params,
            "line_flux": self.line_flux,
            "fit_info": self.fit_info,
        }
        if include_arrays:
            summary.update(
                {
                    "model_flux": self.model_flux.tolist(),
                    "stellar_flux": self.stellar_flux.tolist(),
                    "agn_flux": self.agn_flux.tolist(),
                    "emission_flux": self.emission_flux.tolist(),
                    "residual_flux": self.residual_flux.tolist(),
                }
            )
        return summary


def fit_bic(result: EmissionFitResult) -> float:
    """Return BIC = chi2 + k ln(n) for an emission-line fit."""

    if result.ndata <= 0:
        return float("nan")
    return float(result.chi2 + result.nvarys * np.log(result.ndata))


def _gaussian_integrated(
    wave: np.ndarray,
    rest_wave: float,
    flux: float,
    velocity: float,
    sigma_velocity: float,
) -> np.ndarray:
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    norm = flux / (sigma_wave * np.sqrt(2.0 * np.pi))
    return norm * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)


def measure_combined_profile_properties(
    wave: np.ndarray,
    profile: np.ndarray,
    restwave: float,
) -> dict[str, float]:
    """Measure integrated flux, FWHM, dispersion, and centroid of a line profile."""

    wave = np.asarray(wave, dtype=float)
    profile = np.asarray(profile, dtype=float)
    good = np.isfinite(wave) & np.isfinite(profile) & (profile > 0.0)
    if np.count_nonzero(good) < 2:
        return {
            "integrated_flux": 0.0,
            "FWHM_kms": float("nan"),
            "line_dispersion_kms": float("nan"),
            "centroid_velocity_kms": float("nan"),
        }

    w = wave[good]
    p = profile[good]
    integrated_flux = float(np.trapz(p, w))
    if integrated_flux <= 0.0:
        centroid = float("nan")
        dispersion = float("nan")
    else:
        centroid = float(np.trapz(w * p, w) / integrated_flux)
        variance = float(np.trapz(((w - centroid) ** 2) * p, w) / integrated_flux)
        dispersion = np.sqrt(max(variance, 0.0)) * C_KMS / restwave

    peak = float(np.nanmax(p))
    half_peak = 0.5 * peak
    above = p >= half_peak
    if np.count_nonzero(above) < 2:
        fwhm_kms = float("nan")
    else:
        indices = np.flatnonzero(above)
        left_index = indices[0]
        right_index = indices[-1]
        left = _half_max_crossing(w, p, left_index, half_peak, side="left")
        right = _half_max_crossing(w, p, right_index, half_peak, side="right")
        fwhm_kms = (right - left) * C_KMS / restwave if right > left else float("nan")

    centroid_velocity = (centroid / restwave - 1.0) * C_KMS if np.isfinite(centroid) else float("nan")
    return {
        "integrated_flux": integrated_flux,
        "FWHM_kms": float(fwhm_kms),
        "line_dispersion_kms": float(dispersion),
        "centroid_velocity_kms": float(centroid_velocity),
    }


def _half_max_crossing(
    wave: np.ndarray,
    profile: np.ndarray,
    index: int,
    half_peak: float,
    side: str,
) -> float:
    if side == "left":
        if index == 0:
            return float(wave[index])
        x0, x1 = wave[index - 1], wave[index]
        y0, y1 = profile[index - 1], profile[index]
    else:
        if index == len(wave) - 1:
            return float(wave[index])
        x0, x1 = wave[index], wave[index + 1]
        y0, y1 = profile[index], profile[index + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 + (half_peak - y0) * (x1 - x0) / (y1 - y0))


def _continuum(wave: np.ndarray, params: Parameters) -> np.ndarray:
    return params["cont_intercept"].value + params["cont_slope"].value * (
        wave - CONTINUUM_PIVOT
    )


def _model_hb_oiii_flux(
    wave: np.ndarray,
    params: Parameters,
    include_broad_hb: bool,
    broad_hb_model: str = "double",
    include_feii: bool = False,
    feii_template: tuple[np.ndarray, np.ndarray] | None = None,
    include_oiii_wing: bool = True,
    feii_template_sigma_kms: float = 0.0,
) -> np.ndarray:
    narrow_velocity = params["narrow_velocity"].value
    narrow_sigma = params["narrow_sigma"].value
    model = _continuum(wave, params)
    if include_feii and feii_template is not None and _has_feii_parameters(params):
        model += _feii_component(
            wave,
            params,
            feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
    model += _gaussian_integrated(
        wave, HBETA, params["hb_narrow_flux"].value, narrow_velocity, narrow_sigma
    )
    model += _gaussian_integrated(
        wave, OIII_4959, params["oiii_4959_flux"].value, narrow_velocity, narrow_sigma
    )
    model += _gaussian_integrated(
        wave, OIII_5007, params["oiii_5007_flux"].value, narrow_velocity, narrow_sigma
    )
    if include_oiii_wing and "oiii_4959_wing_flux" in params:
        wing_velocity = params["oiii_wing_velocity"].value
        wing_sigma = params["oiii_wing_sigma"].value
        model += _gaussian_integrated(
            wave,
            OIII_4959,
            params["oiii_4959_wing_flux"].value,
            wing_velocity,
            wing_sigma,
        )
        model += _gaussian_integrated(
            wave,
            OIII_5007,
            params["oiii_5007_wing_flux"].value,
            wing_velocity,
            wing_sigma,
        )
    if include_broad_hb:
        model += _broad_balmer_profile(wave, params, "hb", broad_hb_model)
    return model


def _residual_hb_oiii(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    include_broad_hb: bool,
    broad_hb_model: str = "double",
    include_feii: bool = False,
    feii_template: tuple[np.ndarray, np.ndarray] | None = None,
    include_oiii_wing: bool = True,
    feii_template_sigma_kms: float = 0.0,
) -> np.ndarray:
    return (
        flux
        - _model_hb_oiii_flux(
            wave,
            params,
            include_broad_hb,
            broad_hb_model=broad_hb_model,
            include_feii=include_feii,
            feii_template=feii_template,
            include_oiii_wing=include_oiii_wing,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
    ) / err


def _estimate_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    line_centers: tuple[float, ...],
    half_width: float = 25.0,
) -> tuple[float, float]:
    line_mask = np.zeros_like(wave, dtype=bool)
    for line_center in line_centers:
        line_mask |= np.abs(wave - line_center) < half_width
    continuum_mask = ~line_mask
    if np.count_nonzero(continuum_mask) >= 2:
        slope, intercept_at_zero = np.polyfit(
            wave[continuum_mask], flux[continuum_mask], 1
        )
        return float(intercept_at_zero + slope * CONTINUUM_PIVOT), float(slope)
    return float(np.nanmedian(flux)), 0.0


def _estimate_hb_continuum_from_anchors(
    wave: np.ndarray,
    flux: np.ndarray,
) -> tuple[float, float]:
    params = _estimate_feii_anchor_continuum(wave, flux)
    return float(params["intercept"]), float(params["slope"])


def _estimate_feii_anchor_continuum(
    wave: np.ndarray,
    flux: np.ndarray,
    anchor_windows: tuple[tuple[float, float], ...] = HB_CONTINUUM_ANCHOR_WINDOWS,
    fallback_windows: tuple[tuple[float, float], ...] = HB_CONTINUUM_FALLBACK_ANCHOR_WINDOWS,
) -> dict[str, Any]:
    """Estimate a linear continuum from Fe II anchor windows only."""

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    anchor_mask = np.zeros_like(wave, dtype=bool)
    used_windows = anchor_windows
    for start, stop in used_windows:
        anchor_mask |= (wave >= start) & (wave <= stop)
    anchor_mask &= np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(anchor_mask) < 2:
        anchor_mask = np.zeros_like(wave, dtype=bool)
        used_windows = fallback_windows
        for start, stop in used_windows:
            anchor_mask |= (wave >= start) & (wave <= stop)
        anchor_mask &= np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(anchor_mask) >= 2:
        slope, intercept_at_zero = np.polyfit(wave[anchor_mask], flux[anchor_mask], 1)
        intercept = float(intercept_at_zero + slope * CONTINUUM_PIVOT)
        return {
            "intercept": intercept,
            "slope": float(slope),
            "pivot": float(CONTINUUM_PIVOT),
            "windows": [[float(a), float(b)] for a, b in used_windows],
            "n_anchor_pixels": int(np.count_nonzero(anchor_mask)),
            "fallback_used": bool(used_windows == fallback_windows),
        }
    intercept, slope = _estimate_continuum(wave, flux, (HBETA, OIII_4959, OIII_5007), half_width=50.0)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "pivot": float(CONTINUUM_PIVOT),
        "windows": [],
        "n_anchor_pixels": int(np.count_nonzero(anchor_mask)),
        "fallback_used": True,
    }


def _continuum_slope_bounds(
    wave: np.ndarray,
    flux: np.ndarray,
    max_fractional_change: float = 0.70,
) -> tuple[float, float]:
    finite = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(finite) < 2:
        return (-np.inf, np.inf)
    width = float(np.nanmax(wave[finite]) - np.nanmin(wave[finite]))
    if not np.isfinite(width) or width <= 0.0:
        return (-np.inf, np.inf)
    level = float(np.nanmedian(np.abs(flux[finite])))
    if not np.isfinite(level) or level <= 0.0:
        level = float(np.nanpercentile(np.abs(flux[finite]), 75))
    if not np.isfinite(level) or level <= 0.0:
        return (-np.inf, np.inf)
    max_abs_slope = max_fractional_change * level / width
    return (-max_abs_slope, max_abs_slope)


def _apply_hb_continuum_mode(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    continuum_mode: str,
) -> None:
    if continuum_mode not in {"constant", "linear_constrained", "linear_free"}:
        raise ValueError("continuum_mode must be constant, linear_constrained, or linear_free.")
    if "cont_slope" not in params:
        return
    if continuum_mode == "constant":
        params["cont_slope"].set(value=0.0, vary=False)
    elif continuum_mode == "linear_constrained":
        lower, upper = _continuum_slope_bounds(wave, flux)
        params["cont_slope"].set(min=lower, max=upper)


def _outside_intervals_mask(
    wave: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
) -> np.ndarray:
    mask = np.ones_like(wave, dtype=bool)
    for start, stop in intervals:
        mask &= ~((wave >= start) & (wave <= stop))
    return mask


def _region_mask(
    wave: np.ndarray,
    region: tuple[float, float],
    masks: tuple[tuple[float, float], ...] = (),
) -> np.ndarray:
    mask = (wave >= region[0]) & (wave <= region[1])
    if masks:
        mask &= _outside_intervals_mask(wave, masks)
    return mask


def _integrate_component(
    wave: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> float:
    good = mask & np.isfinite(wave) & np.isfinite(values)
    if np.count_nonzero(good) < 2:
        return 0.0
    return float(np.trapz(np.clip(values[good], 0.0, None), wave[good]))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    return float(numerator / denominator)


def compute_feii_region_diagnostics(
    wave: np.ndarray,
    residual: np.ndarray,
    model_components: dict[str, np.ndarray],
) -> dict[str, float]:
    """Measure Fe II blue/red diagnostic region residuals and model balance."""

    wave = np.asarray(wave, dtype=float)
    residual = np.asarray(residual, dtype=float)
    continuum = np.asarray(
        model_components.get("continuum", np.zeros_like(wave)),
        dtype=float,
    )
    feii = np.asarray(model_components.get("FeII", np.zeros_like(wave)), dtype=float)
    broad_hb = np.asarray(
        model_components.get("Hb_broad_combined", np.zeros_like(wave)),
        dtype=float,
    )
    total_model = continuum.copy()
    for name, values in model_components.items():
        if name == "continuum":
            continue
        values = np.asarray(values, dtype=float)
        if values.shape == total_model.shape:
            total_model += values
    observed_feii_like = residual - continuum

    diagnostics: dict[str, float] = {}
    residual_fluxes: dict[str, float] = {}
    model_fluxes: dict[str, float] = {}
    for label, region in (("blue", FEII_BLUE_REGION), ("red", FEII_RED_REGION)):
        mask = _region_mask(wave, region, FEII_DIAGNOSTIC_MASKS)
        good = mask & np.isfinite(wave) & np.isfinite(observed_feii_like) & np.isfinite(feii)
        observed_flux = _integrate_component(wave, observed_feii_like, good)
        model_flux = _integrate_component(wave, feii, good)
        broad_flux = _integrate_component(wave, broad_hb, good)
        continuum_flux = _integrate_component(wave, continuum, good)
        total_flux = _integrate_component(wave, total_model, good)
        diff = residual[good] - total_model[good]
        if diff.size:
            rms = float(np.sqrt(np.nanmean(diff**2)))
            chi2 = float(np.nansum(diff**2))
        else:
            rms = float("nan")
            chi2 = float("nan")
        diagnostics[f"{label}_feii_observed_flux"] = observed_flux
        diagnostics[f"{label}_feii_model_flux"] = model_flux
        diagnostics[f"{label}_feii_broad_hb_flux"] = broad_flux
        diagnostics[f"{label}_feii_continuum_flux"] = continuum_flux
        diagnostics[f"{label}_feii_total_model_flux"] = total_flux
        diagnostics[f"{label}_feii_residual_rms"] = rms
        diagnostics[f"{label}_feii_chi2"] = chi2
        diagnostics[f"{label}_feii_fraction"] = _safe_ratio(model_flux, total_flux)
        diagnostics[f"{label}_broad_hb_fraction"] = _safe_ratio(broad_flux, total_flux)
        diagnostics[f"{label}_continuum_fraction"] = _safe_ratio(continuum_flux, total_flux)
        residual_fluxes[label] = (
            float(np.trapz(np.abs(diff), wave[good])) if diff.size >= 2 else 0.0
        )
        model_fluxes[label] = model_flux

    diagnostics["feii_blue_red_model_ratio"] = _safe_ratio(
        model_fluxes.get("blue", float("nan")),
        model_fluxes.get("red", float("nan")),
    )
    diagnostics["feii_blue_red_residual_ratio"] = _safe_ratio(
        residual_fluxes.get("blue", float("nan")),
        residual_fluxes.get("red", float("nan")),
    )
    return diagnostics


def _estimate_line_flux(
    wave: np.ndarray,
    residual_flux: np.ndarray,
    rest_wave: float,
    half_width: float = 8.0,
) -> float:
    mask = np.abs(wave - rest_wave) <= half_width
    if np.count_nonzero(mask) < 2:
        return 0.0
    positive_residual = np.clip(residual_flux[mask], 0.0, None)
    return float(np.trapz(positive_residual, wave[mask]))


def _estimate_sii_local_flux_floor(
    wave: np.ndarray,
    flux: np.ndarray,
    rest_wave: float,
    floor_fraction: float = 0.15,
) -> tuple[float, float]:
    """Estimate a conservative [S II] flux floor from nearby sidebands."""

    continuum_mask = (
        ((wave >= 6675.0) & (wave <= 6695.0))
        | ((wave >= 6748.0) & (wave <= 6775.0))
    )
    line_mask = np.abs(wave - rest_wave) <= 9.0
    good_cont = continuum_mask & np.isfinite(flux)
    good_line = line_mask & np.isfinite(flux)
    if np.count_nonzero(good_line) < 2 or np.count_nonzero(good_cont) < 3:
        return 0.0, 0.0
    continuum_level = float(np.nanmedian(flux[good_cont]))
    positive = np.clip(flux[good_line] - continuum_level, 0.0, None)
    direct_flux = float(np.trapz(positive, wave[good_line]))
    if not np.isfinite(direct_flux) or direct_flux <= 0.0:
        return 0.0, 0.0
    return direct_flux, float(max(floor_fraction * direct_flux, 0.0))


def _initial_parameters(
    wave: np.ndarray,
    flux: np.ndarray,
    include_broad_hb: bool,
    broad_hb_model: str = "double",
    include_feii: bool = False,
    include_oiii_wing: bool = True,
    feii_scale_mode: str = "single",
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    hb_narrow_min_oiii_ratio: float = DEFAULT_HB_NARROW_MIN_OIII_RATIO,
    fix_hb_narrow_to_oiii_floor: bool = False,
) -> Parameters:
    cont_intercept, cont_slope = _estimate_continuum(
        wave,
        flux,
        (HBETA, OIII_4959, OIII_5007),
    )
    continuum_flux = cont_intercept + cont_slope * (wave - CONTINUUM_PIVOT)
    residual_flux = flux - continuum_flux
    oiii_5007_flux = max(_estimate_line_flux(wave, residual_flux, OIII_5007), 1.0e-12)
    oiii_4959_flux = max(oiii_5007_flux / OIII_RATIO, 1.0e-12)
    hb_narrow_flux = max(_estimate_line_flux(wave, residual_flux, HBETA), 1.0e-12)
    hb_broad_flux = max(0.2 * hb_narrow_flux, 1.0e-12)
    oiii_wing_flux = max(0.1 * oiii_5007_flux / OIII_RATIO, 1.0e-12)

    params = Parameters()
    params.add("cont_intercept", value=cont_intercept)
    params.add("cont_slope", value=cont_slope)
    params.add("narrow_velocity", value=0.0, min=-800.0, max=800.0)
    narrow_sigma_min, narrow_sigma_max = narrow_sigma_bounds
    params.add(
        "narrow_sigma",
        value=float(np.clip(200.0, narrow_sigma_min, narrow_sigma_max)),
        min=float(narrow_sigma_min),
        max=float(narrow_sigma_max),
    )
    params.add("oiii_4959_flux", value=oiii_4959_flux, min=0.0)
    params.add("oiii_5007_flux", expr=f"{OIII_RATIO} * oiii_4959_flux")

    if include_oiii_wing:
        params.add("oiii_wing_velocity", value=-200.0, min=-1500.0, max=1000.0)
        params.add("oiii_wing_sigma", value=250.0, min=50.0, max=2000.0)
        params.add("oiii_4959_wing_flux", value=oiii_wing_flux, min=0.0)
        params.add("oiii_5007_wing_flux", expr=f"{OIII_RATIO} * oiii_4959_wing_flux")

    _add_hb_narrow_flux_parameter(
        params,
        initial_flux=hb_narrow_flux,
        hb_narrow_min_oiii_ratio=hb_narrow_min_oiii_ratio,
        fix_to_floor=fix_hb_narrow_to_oiii_floor,
    )

    if include_feii:
        _add_feii_parameters(
            params,
            flux_initial=max(
                _integrate_feii_norm_window(wave, residual_flux),
                1.0e-12,
            ),
            feii_scale_mode=feii_scale_mode,
        )

    if include_broad_hb:
        _add_broad_balmer_parameters(
            params,
            prefix="hb",
            model=broad_hb_model,
            initial_flux=hb_broad_flux,
            broad1_sigma=1500.0,
            broad2_sigma=4000.0,
        )

    return params


def _add_hb_narrow_flux_parameter(
    params: Parameters,
    initial_flux: float,
    hb_narrow_min_oiii_ratio: float,
    fix_to_floor: bool = False,
) -> None:
    """Add narrow H-beta flux, optionally constrained by [O III] 5007 flux."""

    ratio = _as_float(hb_narrow_min_oiii_ratio)
    initial = max(float(initial_flux), 1.0e-12)
    if not np.isfinite(ratio) or ratio <= 0.0:
        params.add("hb_narrow_flux", value=initial, min=0.0)
        return

    oiii_expr = "oiii_5007_flux"
    if "oiii_5007_wing_flux" in params:
        oiii_expr = "(oiii_5007_flux + oiii_5007_wing_flux)"
    params.add("hb_narrow_min_oiii_ratio", value=float(ratio), vary=False)
    if fix_to_floor:
        params.add(
            "hb_narrow_flux",
            expr=f"hb_narrow_min_oiii_ratio * {oiii_expr}",
        )
        return
    floor = float(ratio) * _hb_narrow_oiii_reference_flux(params)
    params.add("hb_narrow_flux_excess", value=max(initial - floor, 1.0e-12), min=0.0)
    params.add(
        "hb_narrow_flux",
        expr=f"hb_narrow_min_oiii_ratio * {oiii_expr} + hb_narrow_flux_excess",
    )


def _hb_narrow_oiii_reference_flux(params: Parameters) -> float:
    """Return the [O III] 5007 flux used for the narrow H-beta floor."""

    value = _as_float(params["oiii_5007_flux"].value if "oiii_5007_flux" in params else np.nan)
    if "oiii_5007_wing_flux" in params:
        wing = _as_float(params["oiii_5007_wing_flux"].value)
        if np.isfinite(wing) and wing > 0.0:
            value = (value if np.isfinite(value) else 0.0) + wing
    return float(value) if np.isfinite(value) and value > 0.0 else 0.0


def _line_flux_hb_oiii_dict(
    params: Parameters,
    include_broad_hb: bool,
    broad_hb_model: str = "double",
    include_oiii_wing: bool = True,
) -> dict[str, float]:
    line_flux = {
        "Hb_narrow": float(params["hb_narrow_flux"].value),
        "OIII_4959": float(params["oiii_4959_flux"].value),
        "OIII_5007": float(params["oiii_5007_flux"].value),
    }
    if "feii_flux_4434_4684" in params:
        line_flux["FeII_4434_4684"] = float(params["feii_flux_4434_4684"].value)
    if _has_feii_multiscale_parameters(params):
        line_flux["FeII_blue_4434_4684"] = float(params["feii_flux_blue_4434_4684"].value)
        line_flux["FeII_central_4920_5050"] = float(params["feii_flux_central_4920_5050"].value)
        line_flux["FeII_red_5100_5600"] = float(params["feii_flux_red_5100_5600"].value)
        line_flux["FeII_4434_4684"] = line_flux["FeII_blue_4434_4684"]
    if include_oiii_wing and "oiii_4959_wing_flux" in params:
        line_flux["OIII_4959_wing"] = float(params["oiii_4959_wing_flux"].value)
        line_flux["OIII_5007_wing"] = float(params["oiii_5007_wing_flux"].value)
    if include_broad_hb:
        broad_flux = _broad_balmer_flux_dict(params, "hb", broad_hb_model)
        line_flux.update(broad_flux)
        line_flux["Hb_broad"] = float(
            broad_flux.get("Hb_broad_flux_total", broad_flux.get("Hb_broad_1", 0.0))
        )
        line_flux["Hb_broad_total"] = line_flux["Hb_broad"]
    return line_flux


def _add_broad_balmer_parameters(
    params: Parameters,
    prefix: str,
    model: str,
    initial_flux: float,
    broad1_sigma: float,
    broad2_sigma: float,
) -> None:
    if model not in {"single", "double"}:
        raise ValueError("broad Balmer model must be 'single' or 'double'.")
    base = f"{prefix}_broad_1"
    vmin, vmax, smin, smax = _broad_balmer_component_bounds(prefix, 1)
    params.add(
        f"{base}_velocity",
        value=0.0,
        min=vmin,
        max=vmax,
    )
    params.add(
        f"{base}_sigma",
        value=broad1_sigma,
        min=smin,
        max=smax,
    )
    params.add(f"{base}_flux", value=max(float(initial_flux), 1.0e-12), min=0.0)
    if model == "double":
        base = f"{prefix}_broad_2"
        vmin, vmax, smin, smax = _broad_balmer_component_bounds(prefix, 2)
        params.add(
            f"{base}_velocity",
            value=0.0,
            min=vmin,
            max=vmax,
        )
        params.add(
            f"{base}_sigma",
            value=broad2_sigma,
            min=smin,
            max=smax,
        )
        params.add(f"{base}_flux", value=max(float(initial_flux), 1.0e-12), min=0.0)


def _broad_balmer_component_bounds(
    prefix: str,
    component_index: int,
) -> tuple[float, float, float, float]:
    if prefix == "hb" and component_index == 1:
        return (
            HB_BROAD_1_VELOCITY_BOUNDS[0],
            HB_BROAD_1_VELOCITY_BOUNDS[1],
            HB_BROAD_1_SIGMA_BOUNDS[0],
            HB_BROAD_1_SIGMA_BOUNDS[1],
        )
    if prefix == "hb" and component_index == 2:
        return (
            HB_BROAD_2_VELOCITY_BOUNDS[0],
            HB_BROAD_2_VELOCITY_BOUNDS[1],
            HB_BROAD_2_SIGMA_BOUNDS[0],
            HB_BROAD_2_SIGMA_BOUNDS[1],
        )
    if prefix == "ha" and component_index == 1:
        return (
            HA_BROAD_1_VELOCITY_BOUNDS[0],
            HA_BROAD_1_VELOCITY_BOUNDS[1],
            HA_BROAD_1_SIGMA_BOUNDS[0],
            HA_BROAD_1_SIGMA_BOUNDS[1],
        )
    if prefix == "ha" and component_index == 2:
        return (
            HA_BROAD_2_VELOCITY_BOUNDS[0],
            HA_BROAD_2_VELOCITY_BOUNDS[1],
            HA_BROAD_2_SIGMA_BOUNDS[0],
            HA_BROAD_2_SIGMA_BOUNDS[1],
        )
    return (
        BROAD_BALMER_VELOCITY_BOUNDS[0],
        BROAD_BALMER_VELOCITY_BOUNDS[1],
        BROAD_BALMER_SIGMA_BOUNDS[0],
        BROAD_BALMER_SIGMA_BOUNDS[1],
    )


def _broad_balmer_profile(
    wave: np.ndarray,
    params: Parameters,
    prefix: str,
    model: str,
) -> np.ndarray:
    restwave = HBETA if prefix == "hb" else HALPHA
    profile = _gaussian_integrated(
        wave,
        restwave,
        params[f"{prefix}_broad_1_flux"].value,
        params[f"{prefix}_broad_1_velocity"].value,
        params[f"{prefix}_broad_1_sigma"].value,
    )
    if model == "double" and f"{prefix}_broad_2_flux" in params:
        profile += _gaussian_integrated(
            wave,
            restwave,
            params[f"{prefix}_broad_2_flux"].value,
            params[f"{prefix}_broad_2_velocity"].value,
            params[f"{prefix}_broad_2_sigma"].value,
        )
    return profile


def _broad_balmer_flux_dict(
    params: Parameters,
    prefix: str,
    model: str,
) -> dict[str, float]:
    label = "Hb" if prefix == "hb" else "Ha"
    values = {f"{label}_broad_1": float(params[f"{prefix}_broad_1_flux"].value)}
    if model == "double" and f"{prefix}_broad_2_flux" in params:
        values[f"{label}_broad_2"] = float(params[f"{prefix}_broad_2_flux"].value)
    values[f"{label}_broad_flux_total"] = float(sum(values.values()))
    return values


def _load_feii_template(
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None,
    feii_template_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if feii_template is None and feii_template_path is None:
        default_path = _default_feii_template_path()
        if default_path is None:
            return None
        feii_template_path = default_path
    template = load_feii_template(feii_template, feii_template_path)
    if template is None:
        return None
    return template.wave, template.flux


def _default_feii_template_path() -> Path | None:
    candidates = (
        Path(__file__).resolve().parents[1] / "templates" / "fetemplate_no3",
        Path.cwd() / "templates" / "fetemplate_no3",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _feii_component(
    wave: np.ndarray,
    params: Parameters,
    template: tuple[np.ndarray, np.ndarray],
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
) -> np.ndarray:
    if _has_feii_multiscale_parameters(params):
        return _feii_multiscale_component(
            wave,
            params,
            template,
            feii_template_sigma_kms=feii_template_sigma_kms,
            feii_broadening_mode=feii_broadening_mode,
        )
    template_wave, template_flux = template
    unit = feii_unit_flux_template(
        wave,
        template_wave,
        template_flux,
        width_kms=_feii_model_sigma(params),
        shift_kms=_feii_shift(params),
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
    )
    return _feii_flux(params) * unit


def _feii_multiscale_component(
    wave: np.ndarray,
    params: Parameters,
    template: tuple[np.ndarray, np.ndarray],
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
) -> np.ndarray:
    pieces = (
        ("feii_flux_blue_4434_4684", FEII_BLUE_REGION, 20.0),
        ("feii_flux_central_4920_5050", FEII_PEDESTAL_REGION, 12.5),
        ("feii_flux_red_5100_5600", FEII_RED_REGION, 30.0),
    )
    model = np.zeros_like(wave, dtype=float)
    for param_name, region, edge_width in pieces:
        if param_name not in params:
            continue
        unit = _feii_tapered_unit_component(
            wave,
            params,
            template,
            region,
            edge_width,
            feii_template_sigma_kms=feii_template_sigma_kms,
            feii_broadening_mode=feii_broadening_mode,
        )
        model += float(params[param_name].value) * unit
    return model


def _feii_tapered_unit_component(
    wave: np.ndarray,
    params: Parameters,
    template: tuple[np.ndarray, np.ndarray],
    region: tuple[float, float],
    edge_width: float,
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
) -> np.ndarray:
    template_wave, template_flux = template
    unit = feii_unit_flux_template(
        wave,
        template_wave,
        template_flux,
        width_kms=_feii_model_sigma(params),
        shift_kms=_feii_shift(params),
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
        norm_window=region,
    )
    tapered = unit * _cosine_taper(wave, region, edge_width)
    mask = (wave >= region[0]) & (wave <= region[1]) & np.isfinite(tapered)
    if np.count_nonzero(mask) < 2:
        return np.zeros_like(wave, dtype=float)
    integral = float(np.trapz(tapered[mask], wave[mask]))
    if not np.isfinite(integral) or integral <= 0.0:
        return np.zeros_like(wave, dtype=float)
    return tapered / integral


def _cosine_taper(
    wave: np.ndarray,
    region: tuple[float, float],
    edge_width: float,
) -> np.ndarray:
    start, stop = region
    weights = np.zeros_like(wave, dtype=float)
    inside = (wave >= start) & (wave <= stop)
    weights[inside] = 1.0
    left = inside & (wave < start + edge_width)
    right = inside & (wave > stop - edge_width)
    if edge_width > 0.0:
        weights[left] = 0.5 * (1.0 - np.cos(np.pi * (wave[left] - start) / edge_width))
        weights[right] = 0.5 * (1.0 - np.cos(np.pi * (stop - wave[right]) / edge_width))
    return weights


def _feii_component_diagnostics(
    wave: np.ndarray,
    params: Parameters,
    template: tuple[np.ndarray, np.ndarray],
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
) -> dict[str, Any]:
    template_wave, template_flux = template
    _, diagnostics = feii_unit_flux_template(
        wave,
        template_wave,
        template_flux,
        width_kms=_feii_model_sigma(params),
        shift_kms=_feii_shift(params),
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
        return_diagnostics=True,
    )
    return diagnostics


def _estimate_feii_flux_4434_4684(
    wave: np.ndarray,
    flux: np.ndarray,
    continuum_flux: np.ndarray,
    feii_template: tuple[np.ndarray, np.ndarray],
    feii_width_kms: float = 0.0,
    feii_shift_kms: float = 0.0,
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
) -> float:
    params = Parameters()
    _add_feii_parameters(
        params,
        flux_initial=1.0,
        width_initial=feii_width_kms,
        shift_initial=feii_shift_kms,
    )
    template_shape = _feii_component(
        wave,
        params,
        feii_template,
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
    )
    mask = (
        _region_mask(wave, FEII_BLUE_REGION, FEII_DIAGNOSTIC_MASKS)
        | _region_mask(wave, FEII_RED_REGION, FEII_DIAGNOSTIC_MASKS)
    )
    good = (
        mask
        & np.isfinite(flux)
        & np.isfinite(continuum_flux)
        & np.isfinite(template_shape)
        & (template_shape > 0.0)
    )
    if np.count_nonzero(good) < 3:
        return 1.0e-12
    residual = flux[good] - continuum_flux[good]
    template_values = template_shape[good]
    numerator = float(np.nansum(residual * template_values))
    denominator = float(np.nansum(template_values * template_values))
    if np.isfinite(denominator) and denominator > 0.0:
        return max(0.0, numerator / denominator, 1.0e-12)
    return 1.0e-12


def prefit_feii_grid(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    feii_template: tuple[np.ndarray, np.ndarray],
    sigma_grid: np.ndarray | list[float] | None = None,
    velocity_grid: np.ndarray | list[float] | None = None,
    feii_template_method_grid: list[str] | tuple[str, ...] | None = None,
    feii_smooth_sigma_grid: np.ndarray | list[float] | None = None,
    norm_region: tuple[float, float] = FEII_NORM_WINDOW,
    debug: bool = True,
    continuum_flux: np.ndarray | None = None,
    width_grid: np.ndarray | list[float] | None = None,
    shift_grid: np.ndarray | list[float] | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
    feii_template_sigma_kms: float = 0.0,
    fit_regions: tuple[tuple[float, float], ...] = (
        FEII_BLUE_REGION,
        FEII_PEDESTAL_REGION,
        FEII_RED_REGION,
    ),
    line_masks: tuple[tuple[float, float], ...] = FEII_DIAGNOSTIC_MASKS,
) -> FeIIPrefitResult:
    """Grid-prefit Fe II width/shift and solve DASpec-style integrated flux."""

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    err = np.asarray(err, dtype=float)
    if sigma_grid is None:
        sigma_grid = width_grid
    if velocity_grid is None:
        velocity_grid = shift_grid
    if sigma_grid is None:
        sigma_grid = (0.0,)
    if velocity_grid is None:
        velocity_grid = (-400.0, 0.0, 400.0)
    if feii_template_method_grid is None:
        feii_template_method_grid = ("smoothed", "smoothed", "smoothed", "smoothed")
    if feii_smooth_sigma_grid is None:
        feii_smooth_sigma_grid = (300.0, 500.0, 800.0, 1000.0)
    method_values = tuple(str(value) for value in feii_template_method_grid)
    smooth_values = np.asarray(feii_smooth_sigma_grid, dtype=float)
    if len(method_values) != len(smooth_values):
        raise ValueError("feii_template_method_grid and feii_smooth_sigma_grid must have the same length.")
    width_values = np.asarray(sigma_grid, dtype=float)
    shift_values = np.asarray(velocity_grid, dtype=float)
    width_values = width_values[
        (width_values >= feii_width_bounds[0]) & (width_values <= feii_width_bounds[1])
    ]
    shift_values = shift_values[
        (shift_values >= feii_shift_bounds[0]) & (shift_values <= feii_shift_bounds[1])
    ]
    if width_values.size == 0:
        width_values = np.asarray([float(np.clip(0.0, *feii_width_bounds))])
    if shift_values.size == 0:
        shift_values = np.asarray([0.0])

    anchor_params = _estimate_feii_anchor_continuum(wave, flux)
    if continuum_flux is None:
        continuum_flux = anchor_params["intercept"] + anchor_params["slope"] * (
            wave - anchor_params["pivot"]
        )
    else:
        continuum_flux = np.asarray(continuum_flux, dtype=float)

    residual = flux - continuum_flux
    best = {
        "flux": 1.0e-12,
        "width": float(width_values[0]),
        "shift": float(shift_values[0]),
        "method": method_values[0],
        "smooth": float(smooth_values[0]),
        "chi2": float("inf"),
        "chi2_blue": float("nan"),
        "chi2_pedestal": float("nan"),
        "chi2_red": float("nan"),
    }
    best_by_method: dict[str, dict[str, Any]] = {}
    chi2_grid = np.full((len(method_values), len(width_values), len(shift_values)), np.nan)
    flux_grid = np.full_like(chi2_grid, np.nan)
    template_wave, template_flux = feii_template
    region_defs = (
        ("blue", FEII_BLUE_REGION),
        ("pedestal", FEII_PEDESTAL_REGION),
        ("red", FEII_RED_REGION),
    )
    region_masks = {
        name: (
            _region_mask(wave, region, line_masks)
            & np.isfinite(wave)
            & np.isfinite(flux)
            & np.isfinite(err)
            & np.isfinite(continuum_flux)
            & (err > 0.0)
        )
        for name, region in region_defs
        if region in fit_regions
    }
    fit_mask_npix = int(sum(np.count_nonzero(mask) for mask in region_masks.values()))
    if fit_mask_npix < 3:
        return FeIIPrefitResult(
            best_feii_flux_4434_4684=1.0e-12,
            best_feii_width_kms=float(np.clip(0.0, *feii_width_bounds)),
            best_feii_extra_sigma_kms=float(np.clip(0.0, *feii_width_bounds)),
            best_feii_shift_kms=0.0,
            best_feii_template_method=method_values[0],
            best_feii_smooth_sigma_kms=float(smooth_values[0]),
            chi2_blue=float("nan"),
            chi2_pedestal=float("nan"),
            chi2_red=float("nan"),
            chi2_total=float("nan"),
            second_best_chi2=float("nan"),
            chi2_grid=chi2_grid,
            flux_grid=flux_grid,
            method_grid=method_values,
            smooth_sigma_grid=smooth_values,
            width_grid=width_values,
            shift_grid=shift_values,
            continuum_anchor_params=anchor_params,
            flags={"flag_feii_unconstrained": True},
            fit_mask_npix=fit_mask_npix,
        )
    for k, (method, smooth_sigma_kms) in enumerate(zip(method_values, smooth_values)):
        if method not in {"raw", "smoothed"}:
            raise ValueError("Fe II template method must be 'raw' or 'smoothed'.")
        method_smooth = 0.0 if method == "raw" else float(smooth_sigma_kms)
        for i, width_kms in enumerate(width_values):
            for j, shift_kms in enumerate(shift_values):
                try:
                    unit = feii_unit_flux_template(
                        wave,
                        template_wave,
                        template_flux,
                        width_kms=float(method_smooth + width_kms),
                        shift_kms=float(shift_kms),
                        feii_template_sigma_kms=feii_template_sigma_kms,
                        feii_broadening_mode="extra",
                        norm_window=norm_region,
                    )
                except ValueError:
                    continue
                fit_mask = np.zeros_like(wave, dtype=bool)
                for mask in region_masks.values():
                    fit_mask |= mask
                good = fit_mask & np.isfinite(unit)
                if np.count_nonzero(good) < 3:
                    continue
                t = unit[good]
                r = residual[good]
                wgt = 1.0 / np.square(err[good])
                denom = float(np.sum(t * t * wgt))
                if not np.isfinite(denom) or denom <= 0.0:
                    continue
                feii_flux = max(0.0, float(np.sum(r * t * wgt) / denom))
                region_chi2: dict[str, float] = {}
                balanced_terms = []
                for name, mask in region_masks.items():
                    region_good = mask & np.isfinite(unit)
                    n_region = int(np.count_nonzero(region_good))
                    if n_region < 2:
                        region_chi2[name] = float("nan")
                        continue
                    raw = float(
                        np.sum(
                            np.square(
                                (residual[region_good] - feii_flux * unit[region_good])
                                / err[region_good]
                            )
                        )
                    )
                    region_chi2[name] = raw
                    balanced_terms.append(raw / n_region)
                chi2 = float(np.sum(balanced_terms)) if balanced_terms else float("nan")
                chi2_grid[k, i, j] = chi2
                flux_grid[k, i, j] = feii_flux
                if np.isfinite(chi2) and chi2 < best["chi2"]:
                    best = {
                        "flux": feii_flux,
                        "width": float(width_kms),
                        "shift": float(shift_kms),
                        "method": method,
                        "smooth": float(method_smooth),
                        "chi2": chi2,
                        "chi2_blue": region_chi2.get("blue", float("nan")),
                        "chi2_pedestal": region_chi2.get("pedestal", float("nan")),
                        "chi2_red": region_chi2.get("red", float("nan")),
                    }
                method_best = best_by_method.get(method)
                if method_best is None or (np.isfinite(chi2) and chi2 < method_best["chi2"]):
                    best_by_method[method] = {
                        "flux": feii_flux,
                        "width": float(width_kms),
                        "shift": float(shift_kms),
                        "method": method,
                        "smooth": float(method_smooth),
                        "chi2": chi2,
                        "chi2_blue": region_chi2.get("blue", float("nan")),
                        "chi2_pedestal": region_chi2.get("pedestal", float("nan")),
                        "chi2_red": region_chi2.get("red", float("nan")),
                    }

    raw_best = best_by_method.get("raw")
    smoothed_best = best_by_method.get("smoothed")
    raw_high_frequency_overfit = False
    if raw_best is not None and smoothed_best is not None:
        raw_chi2 = float(raw_best["chi2"])
        smoothed_chi2 = float(smoothed_best["chi2"])
        if np.isfinite(raw_chi2) and np.isfinite(smoothed_chi2) and raw_chi2 > 0.0:
            raw_unit = feii_unit_flux_template(
                wave,
                template_wave,
                template_flux,
                width_kms=float(raw_best["smooth"] + raw_best["width"]),
                shift_kms=float(raw_best["shift"]),
                feii_template_sigma_kms=feii_template_sigma_kms,
                feii_broadening_mode="extra",
                norm_window=norm_region,
            )
            raw_model = float(raw_best["flux"]) * raw_unit
            raw_high_frequency_overfit = _feii_high_frequency_overfit(
                wave,
                residual,
                raw_model,
                region=FEII_RED_REGION,
            )
            if (smoothed_chi2 - raw_chi2) / raw_chi2 < 0.05:
                best = smoothed_best

    finite_chi2 = np.sort(chi2_grid[np.isfinite(chi2_grid)])
    second_best = float(finite_chi2[1]) if finite_chi2.size > 1 else float("nan")
    best_chi2 = float(best["chi2"])
    unconstrained = bool(
        not np.isfinite(best_chi2)
        or (
            np.isfinite(second_best)
            and best_chi2 > 0.0
            and (second_best - best_chi2) / best_chi2 < 0.02
        )
    )
    boundary = bool(
        np.isclose(best["width"], feii_width_bounds[0])
        or np.isclose(best["width"], feii_width_bounds[1])
        or np.isclose(best["shift"], feii_shift_bounds[0])
        or np.isclose(best["shift"], feii_shift_bounds[1])
    )
    shape_mismatch = bool(_feii_template_shape_mismatch(best))
    return FeIIPrefitResult(
        best_feii_flux_4434_4684=max(float(best["flux"]), 1.0e-12),
        best_feii_width_kms=float(best["width"]),
        best_feii_extra_sigma_kms=float(best["width"]),
        best_feii_shift_kms=float(best["shift"]),
        best_feii_template_method=str(best["method"]),
        best_feii_smooth_sigma_kms=float(best["smooth"]),
        chi2_blue=float(best["chi2_blue"]),
        chi2_pedestal=float(best["chi2_pedestal"]),
        chi2_red=float(best["chi2_red"]),
        chi2_total=best_chi2,
        second_best_chi2=second_best,
        chi2_grid=chi2_grid,
        flux_grid=flux_grid,
        method_grid=method_values,
        smooth_sigma_grid=smooth_values,
        width_grid=width_values,
        shift_grid=shift_values,
        continuum_anchor_params=anchor_params,
        flags={
            "flag_feii_unconstrained": unconstrained,
            "flag_feii_boundary": boundary,
            "flag_feii_continuum_degenerate": False,
            "flag_feii_high_frequency_overfit": raw_high_frequency_overfit,
            "flag_single_feii_template_shape_mismatch": shape_mismatch,
        },
        fit_mask_npix=fit_mask_npix,
    )


def _estimate_feii_amp(*args: Any, **kwargs: Any) -> float:
    """Backward-compatible alias; returns Fe II 4434-4684 integrated flux."""

    if "feii_sigma" in kwargs:
        kwargs["feii_width_kms"] = kwargs.pop("feii_sigma")
    if "feii_velocity" in kwargs:
        kwargs["feii_shift_kms"] = kwargs.pop("feii_velocity")
    return _estimate_feii_flux_4434_4684(*args, **kwargs)


def _feii_high_frequency_overfit(
    wave: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    region: tuple[float, float],
    smooth_width_angstrom: float = 50.0,
) -> bool:
    mask = (
        (wave >= region[0])
        & (wave <= region[1])
        & np.isfinite(wave)
        & np.isfinite(data)
        & np.isfinite(model)
    )
    if np.count_nonzero(mask) < 5:
        return False
    region_wave = wave[mask]
    data_values = data[mask]
    model_values = model[mask]
    low_data = _smooth_linear_wavelength(region_wave, data_values, smooth_width_angstrom)
    low_model = _smooth_linear_wavelength(region_wave, model_values, smooth_width_angstrom)
    data_hf = data_values - low_data
    model_hf = model_values - low_model
    data_rms = float(np.sqrt(np.nanmean(data_hf**2)))
    model_rms = float(np.sqrt(np.nanmean(model_hf**2)))
    return bool(np.isfinite(data_rms) and data_rms > 0.0 and model_rms > 0.7 * data_rms)


def _smooth_linear_wavelength(
    wave: np.ndarray,
    values: np.ndarray,
    smooth_width_angstrom: float,
) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    step = float(np.nanmedian(np.diff(wave)))
    if not np.isfinite(step) or step <= 0.0:
        return values.copy()
    sigma_pix = max(smooth_width_angstrom / step, 1.0)
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(values, sigma_pix, mode="nearest")
    except Exception:
        width = max(int(round(sigma_pix)), 1)
        kernel = np.ones(2 * width + 1, dtype=float)
        kernel /= np.sum(kernel)
        return np.convolve(values, kernel, mode="same")


def _feii_template_shape_mismatch(best: dict[str, Any]) -> bool:
    chis = np.asarray(
        [
            best.get("chi2_blue", np.nan),
            best.get("chi2_pedestal", np.nan),
            best.get("chi2_red", np.nan),
        ],
        dtype=float,
    )
    finite = chis[np.isfinite(chis) & (chis > 0.0)]
    if finite.size < 2:
        return False
    return bool(np.nanmax(finite) / np.nanmedian(finite) > 1.5)


def _add_feii_parameters(
    params: Parameters,
    flux_initial: float,
    width_initial: float = 0.0,
    shift_initial: float = 0.0,
    fixed_width: float | None = None,
    fixed_shift: float | None = None,
    width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
    feii_scale_mode: str = "single",
) -> None:
    params.add("feii_flux_4434_4684", value=max(float(flux_initial), 1.0e-12), min=0.0)
    if fixed_shift is None:
        params.add(
            "feii_shift_kms",
            value=float(np.clip(shift_initial, shift_bounds[0], shift_bounds[1])),
            min=shift_bounds[0],
            max=shift_bounds[1],
        )
    else:
        params.add(
            "feii_shift_kms",
            value=float(fixed_shift),
            min=shift_bounds[0],
            max=shift_bounds[1],
            vary=False,
        )
    if fixed_width is None:
        params.add(
            "feii_extra_sigma_kms",
            value=float(np.clip(width_initial, width_bounds[0], width_bounds[1])),
            min=width_bounds[0],
            max=width_bounds[1],
        )
    else:
        params.add(
            "feii_extra_sigma_kms",
            value=float(fixed_width),
            min=0.0 if fixed_width < width_bounds[0] else width_bounds[0],
            max=max(width_bounds[1], float(fixed_width)),
            vary=False,
        )
    params.add("feii_amp", expr="feii_flux_4434_4684")
    params.add("feii_velocity", expr="feii_shift_kms")
    params.add("feii_smooth_sigma_kms", value=0.0, min=0.0, max=1500.0, vary=False)
    params.add("feii_width_kms", expr="feii_extra_sigma_kms")
    params.add("feii_sigma", expr="feii_extra_sigma_kms")
    if feii_scale_mode == "blue_central_red":
        params.add("feii_flux_blue_4434_4684", value=max(float(flux_initial), 1.0e-12), min=0.0)
        params.add("feii_flux_central_4920_5050", value=max(0.5 * float(flux_initial), 1.0e-12), min=0.0)
        params.add("feii_flux_red_5100_5600", value=max(float(flux_initial), 1.0e-12), min=0.0)
    elif feii_scale_mode != "single":
        raise ValueError("feii_scale_mode must be 'single' or 'blue_central_red'.")


def _has_feii_parameters(params: Parameters) -> bool:
    return (
        "feii_flux_4434_4684" in params
        or "feii_amp" in params
        or _has_feii_multiscale_parameters(params)
    )


def _has_feii_multiscale_parameters(params: Parameters) -> bool:
    return (
        "feii_flux_blue_4434_4684" in params
        and "feii_flux_central_4920_5050" in params
        and "feii_flux_red_5100_5600" in params
    )


def _feii_flux(params: Parameters) -> float:
    if _has_feii_multiscale_parameters(params):
        return float(
            params["feii_flux_blue_4434_4684"].value
            + params["feii_flux_central_4920_5050"].value
            + params["feii_flux_red_5100_5600"].value
        )
    if "feii_flux_4434_4684" in params:
        return float(params["feii_flux_4434_4684"].value)
    return float(params["feii_amp"].value)


def _feii_width(params: Parameters) -> float:
    if "feii_extra_sigma_kms" in params:
        return float(params["feii_extra_sigma_kms"].value)
    if "feii_width_kms" in params:
        return float(params["feii_width_kms"].value)
    return float(params["feii_sigma"].value)


def _feii_smooth_sigma(params: Parameters) -> float:
    if "feii_smooth_sigma_kms" in params:
        return float(params["feii_smooth_sigma_kms"].value)
    return 0.0


def _feii_model_sigma(params: Parameters) -> float:
    return _feii_smooth_sigma(params) + _feii_width(params)


def _feii_shift(params: Parameters) -> float:
    if "feii_shift_kms" in params:
        return float(params["feii_shift_kms"].value)
    return float(params["feii_velocity"].value)


def _integrate_feii_norm_window(wave: np.ndarray, values: np.ndarray) -> float:
    mask = (
        np.isfinite(wave)
        & np.isfinite(values)
        & (wave >= FEII_NORM_WINDOW[0])
        & (wave <= FEII_NORM_WINDOW[1])
    )
    if np.count_nonzero(mask) < 2:
        return 1.0e-12
    return max(float(np.trapz(np.clip(values[mask], 0.0, None), wave[mask])), 1.0e-12)


def _initial_feii_prefit_parameters(
    wave: np.ndarray,
    flux: np.ndarray,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    feii_template_sigma_kms: float = 0.0,
    feii_fixed_sigma_kms: float | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
) -> Parameters:
    continuum_mask = _outside_intervals_mask(wave, FEII_PREFIT_LINE_MASKS)
    continuum_mask &= np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(continuum_mask) >= 2:
        slope, intercept_at_zero = np.polyfit(wave[continuum_mask], flux[continuum_mask], 1)
        cont_intercept = float(intercept_at_zero + slope * CONTINUUM_PIVOT)
        cont_slope = float(slope)
    else:
        cont_intercept = float(np.nanmedian(flux))
        cont_slope = 0.0

    params = Parameters()
    params.add("cont_intercept", value=cont_intercept)
    params.add("cont_slope", value=cont_slope)
    if include_feii and feii_template is not None:
        continuum_flux = cont_intercept + cont_slope * (wave - CONTINUUM_PIVOT)
        flux_initial = _estimate_feii_flux_4434_4684(
                wave,
                flux,
                continuum_flux,
                feii_template,
                feii_template_sigma_kms=feii_template_sigma_kms,
        )
        _add_feii_parameters(
            params,
            flux_initial=flux_initial,
            fixed_width=feii_fixed_sigma_kms,
            width_bounds=feii_width_bounds,
            shift_bounds=feii_shift_bounds,
        )
    return params


def _feii_prefit_constraints(
    params: Parameters,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    feii_fixed_sigma_kms: float | None = None,
    feii_fixed_shift_kms: float | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
) -> dict[str, float]:
    if not include_feii or feii_template is None or not _has_feii_parameters(params):
        return {
            "sigma": float("nan"),
            "sigma_min": feii_width_bounds[0],
            "sigma_max": feii_width_bounds[1],
            "velocity": float("nan"),
            "velocity_min": feii_shift_bounds[0],
            "velocity_max": feii_shift_bounds[1],
        }
    sigma = _feii_width(params)
    velocity = _feii_shift(params)
    if not np.isfinite(sigma):
        sigma = 2000.0
    if not np.isfinite(velocity):
        velocity = 0.0
    if feii_fixed_sigma_kms is not None:
        fixed_sigma = float(feii_fixed_sigma_kms)
        return {
            "sigma": fixed_sigma,
            "sigma_min": fixed_sigma,
            "sigma_max": fixed_sigma,
            "velocity": float(feii_fixed_shift_kms) if feii_fixed_shift_kms is not None else velocity,
            "velocity_min": (
                feii_shift_bounds[0]
                if feii_fixed_shift_kms is not None
                else max(feii_shift_bounds[0], velocity - 500.0)
            ),
            "velocity_max": (
                feii_shift_bounds[1]
                if feii_fixed_shift_kms is not None
                else min(feii_shift_bounds[1], velocity + 500.0)
            ),
            "sigma_fixed": True,
            "velocity_fixed": bool(feii_fixed_shift_kms is not None),
        }
    if feii_fixed_shift_kms is not None:
        velocity = float(feii_fixed_shift_kms)
    sigma_half_width = max(700.0, 0.35 * sigma)
    velocity_half_width = 500.0
    return {
        "sigma": sigma,
        "sigma_min": max(feii_width_bounds[0], sigma - sigma_half_width),
        "sigma_max": min(feii_width_bounds[1], sigma + sigma_half_width),
        "velocity": velocity,
        "velocity_min": (
            feii_shift_bounds[0]
            if feii_fixed_shift_kms is not None
            else max(feii_shift_bounds[0], velocity - velocity_half_width)
        ),
        "velocity_max": (
            feii_shift_bounds[1]
            if feii_fixed_shift_kms is not None
            else min(feii_shift_bounds[1], velocity + velocity_half_width)
        ),
        "sigma_fixed": False,
        "velocity_fixed": bool(feii_fixed_shift_kms is not None),
    }


def _apply_feii_prefit_constraints(
    params: Parameters,
    constraints: dict[str, float],
) -> None:
    if "feii_extra_sigma_kms" in params:
        sigma = float(constraints["sigma"])
        if bool(constraints.get("sigma_fixed", False)):
            params["feii_extra_sigma_kms"].set(value=sigma, min=0.0, max=FEII_SIGMA_BOUNDS[1], vary=False)
        else:
            sigma = float(np.clip(sigma, constraints["sigma_min"], constraints["sigma_max"]))
            params["feii_extra_sigma_kms"].set(
                value=sigma,
                min=constraints["sigma_min"],
                max=constraints["sigma_max"],
            )
    if "feii_shift_kms" in params:
        velocity = float(
            np.clip(
                constraints["velocity"],
                constraints["velocity_min"],
                constraints["velocity_max"],
            )
        )
        params["feii_shift_kms"].set(
            value=velocity,
            min=constraints["velocity_min"],
            max=constraints["velocity_max"],
            vary=not bool(constraints.get("velocity_fixed", False)),
        )


def _feii_prefit_grid(constraints: dict[str, float], name: str) -> list[float]:
    center = float(constraints[name])
    lower = float(constraints[f"{name}_min"])
    upper = float(constraints[f"{name}_max"])
    if not np.isfinite(center):
        center = 0.0
    if name == "sigma" and bool(constraints.get("sigma_fixed", False)):
        return [center]
    values = [center, lower, upper]
    return [float(np.clip(value, lower, upper)) for value in values]


def _model_feii_prefit_flux(
    wave: np.ndarray,
    params: Parameters,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    feii_template_sigma_kms: float = 0.0,
) -> np.ndarray:
    model = _continuum(wave, params)
    if include_feii and feii_template is not None and _has_feii_parameters(params):
        model += _feii_component(
            wave,
            params,
            feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
    return model


def _residual_feii_prefit(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    fit_mask: np.ndarray,
    feii_template_sigma_kms: float = 0.0,
) -> np.ndarray:
    resid = (
        flux
        - _model_feii_prefit_flux(
            wave,
            params,
            include_feii=include_feii,
            feii_template=feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
    ) / err
    return resid[fit_mask]


def _seed_params_from_fit(
    target: Parameters,
    source: Parameters,
    names: tuple[str, ...],
) -> None:
    for name in names:
        if name in target and name in source:
            target[name].set(value=float(source[name].value))


def _add_hb_broad_parameters_if_needed(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    broad_hb_model: str,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    feii_template_sigma_kms: float = 0.0,
) -> None:
    if "hb_broad_1_flux" in params:
        return
    continuum = _continuum(wave, params)
    line_model = _model_hb_oiii_flux(
        wave,
        params,
        include_broad_hb=False,
        broad_hb_model=broad_hb_model,
        include_feii=_has_feii_parameters(params),
        feii_template=feii_template,
        include_oiii_wing="oiii_4959_wing_flux" in params,
        feii_template_sigma_kms=feii_template_sigma_kms,
    ) - continuum
    residual = flux - continuum - line_model
    narrow_hb_flux = max(float(params["hb_narrow_flux"].value), 1.0e-12)
    broad_guess = max(
        0.2 * narrow_hb_flux,
        _estimate_line_flux(wave, residual, HBETA, half_width=60.0),
        1.0e-12,
    )
    _add_broad_balmer_parameters(
        params,
        prefix="hb",
        model=broad_hb_model,
        initial_flux=broad_guess,
        broad1_sigma=1500.0,
        broad2_sigma=4000.0,
    )


def _model_ha_nii_sii_flux(
    wave: np.ndarray,
    params: Parameters,
    include_broad_ha: bool,
    broad_ha_model: str = "double",
) -> np.ndarray:
    narrow_velocity = params["narrow_velocity"].value
    narrow_sigma = params["narrow_sigma"].value
    model = _continuum(wave, params)
    model += _gaussian_integrated(
        wave, HALPHA, params["ha_narrow_flux"].value, narrow_velocity, narrow_sigma
    )
    if "oi_6300_flux" in params:
        model += _gaussian_integrated(
            wave, OI_6300, params["oi_6300_flux"].value, narrow_velocity, narrow_sigma
        )
        model += _gaussian_integrated(
            wave, OI_6363, params["oi_6363_flux"].value, narrow_velocity, narrow_sigma
        )
    model += _gaussian_integrated(
        wave, NII_6548, params["nii_6548_flux"].value, narrow_velocity, narrow_sigma
    )
    model += _gaussian_integrated(
        wave, NII_6584, params["nii_6584_flux"].value, narrow_velocity, narrow_sigma
    )
    model += _gaussian_integrated(
        wave, SII_6716, params["sii_6716_flux"].value, narrow_velocity, narrow_sigma
    )
    model += _gaussian_integrated(
        wave, SII_6731, params["sii_6731_flux"].value, narrow_velocity, narrow_sigma
    )
    if include_broad_ha:
        model += _broad_balmer_profile(wave, params, "ha", broad_ha_model)
    return model


def _residual_ha_nii_sii(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    include_broad_ha: bool,
    broad_ha_model: str = "double",
) -> np.ndarray:
    return (
        flux
        - _model_ha_nii_sii_flux(
            wave,
            params,
            include_broad_ha,
            broad_ha_model=broad_ha_model,
        )
    ) / err


def _initial_ha_nii_sii_parameters(
    wave: np.ndarray,
    flux: np.ndarray,
    include_broad_ha: bool,
    broad_ha_model: str = "double",
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    narrow_kinematics: dict[str, float] | None = None,
    freeze_narrow_kinematics: bool = False,
) -> Parameters:
    cont_intercept, cont_slope = _estimate_continuum(
        wave,
        flux,
        (OI_6300, OI_6363, HALPHA, NII_6548, NII_6584, SII_6716, SII_6731),
    )
    continuum_flux = cont_intercept + cont_slope * (wave - CONTINUUM_PIVOT)
    residual_flux = flux - continuum_flux
    nii_6584_flux = max(_estimate_line_flux(wave, residual_flux, NII_6584), 1.0e-12)
    nii_6548_flux = max(nii_6584_flux / NII_RATIO, 1.0e-12)

    params = Parameters()
    params.add("cont_intercept", value=cont_intercept)
    params.add("cont_slope", value=cont_slope)
    narrow_velocity_init = _as_float((narrow_kinematics or {}).get("velocity", 0.0))
    if not np.isfinite(narrow_velocity_init):
        narrow_velocity_init = 0.0
    if narrow_kinematics and not freeze_narrow_kinematics:
        velocity_min = max(-800.0, narrow_velocity_init - 150.0)
        velocity_max = min(800.0, narrow_velocity_init + 150.0)
    else:
        velocity_min = -800.0
        velocity_max = 800.0
    params.add(
        "narrow_velocity",
        value=float(np.clip(narrow_velocity_init, velocity_min, velocity_max)),
        min=velocity_min,
        max=velocity_max,
        vary=not freeze_narrow_kinematics,
    )
    narrow_sigma_min, narrow_sigma_max = narrow_sigma_bounds
    narrow_sigma_init = _as_float((narrow_kinematics or {}).get("sigma", 200.0))
    if not np.isfinite(narrow_sigma_init):
        narrow_sigma_init = 200.0
    params.add(
        "narrow_sigma",
        value=float(np.clip(narrow_sigma_init, narrow_sigma_min, narrow_sigma_max)),
        min=float(narrow_sigma_min),
        max=float(narrow_sigma_max),
        vary=not freeze_narrow_kinematics,
    )
    ha_narrow_flux = max(_estimate_line_flux(wave, residual_flux, HALPHA), 1.0e-12)
    oi_6300_flux = max(_estimate_line_flux(wave, residual_flux, OI_6300), 1.0e-12)
    sii_6716_direct, sii_6716_floor = _estimate_sii_local_flux_floor(
        wave,
        flux,
        SII_6716,
    )
    sii_6731_direct, sii_6731_floor = _estimate_sii_local_flux_floor(
        wave,
        flux,
        SII_6731,
    )
    params.add(
        "ha_narrow_flux",
        value=ha_narrow_flux,
        min=0.0,
    )
    params.add("oi_6300_flux", value=oi_6300_flux, min=0.0)
    params.add("oi_6363_flux", expr=f"oi_6300_flux / {OI_RATIO}")
    params.add("nii_6548_flux", value=nii_6548_flux, min=0.0)
    params.add("nii_6584_flux", expr=f"{NII_RATIO} * nii_6548_flux")
    params.add(
        "sii_6716_flux",
        value=max(_estimate_line_flux(wave, residual_flux, SII_6716), sii_6716_direct, 1.0e-12),
        min=sii_6716_floor,
    )
    params.add(
        "sii_6731_flux",
        value=max(_estimate_line_flux(wave, residual_flux, SII_6731), sii_6731_direct, 1.0e-12),
        min=sii_6731_floor,
    )

    if include_broad_ha:
        _add_broad_balmer_parameters(
            params,
            prefix="ha",
            model=broad_ha_model,
            initial_flux=max(0.3 * ha_narrow_flux, 1.0e-12),
            broad1_sigma=1500.0,
            broad2_sigma=4000.0,
        )

    return params


def _initial_ha_nii_sii_parameters_from_narrow(
    wave: np.ndarray,
    flux: np.ndarray,
    narrow_result: EmissionFitResult,
    broad_ha_model: str,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    freeze_narrow_kinematics: bool = True,
) -> Parameters:
    """Initialize an H-alpha broad fit from an already converged narrow fit."""

    narrow_params = narrow_result.params or {}
    narrow_flux = narrow_result.line_flux or {}
    narrow_sigma_min, narrow_sigma_max = narrow_sigma_bounds

    def _get_param(name: str, default: float) -> float:
        value = _as_float(narrow_params.get(name, default))
        return value if np.isfinite(value) else float(default)

    def _get_flux(line_name: str, param_name: str, default: float = 1.0) -> float:
        value = _as_float(narrow_flux.get(line_name, narrow_params.get(param_name, default)))
        return float(value) if np.isfinite(value) and value > 0.0 else float(default)

    params = Parameters()
    params.add("cont_intercept", value=_get_param("cont_intercept", 0.0))
    params.add("cont_slope", value=_get_param("cont_slope", 0.0))
    params.add(
        "narrow_velocity",
        value=float(np.clip(_get_param("narrow_velocity", 0.0), -800.0, 800.0)),
        min=-800.0,
        max=800.0,
        vary=not freeze_narrow_kinematics,
    )
    params.add(
        "narrow_sigma",
        value=float(np.clip(_get_param("narrow_sigma", 200.0), narrow_sigma_min, narrow_sigma_max)),
        min=float(narrow_sigma_min),
        max=float(narrow_sigma_max),
        vary=not freeze_narrow_kinematics,
    )
    params.add(
        "ha_narrow_flux",
        value=max(_get_flux("Ha_narrow", "ha_narrow_flux"), 1.0e-12),
        min=0.0,
    )
    params.add(
        "oi_6300_flux",
        value=max(_get_flux("OI_6300", "oi_6300_flux"), 1.0e-12),
        min=0.0,
    )
    params.add("oi_6363_flux", expr=f"oi_6300_flux / {OI_RATIO}")
    nii_6548 = max(_get_flux("NII_6548", "nii_6548_flux"), 1.0e-12)
    params.add("nii_6548_flux", value=nii_6548, min=0.0)
    params.add("nii_6584_flux", expr=f"{NII_RATIO} * nii_6548_flux")
    sii_6716_direct, sii_6716_floor = _estimate_sii_local_flux_floor(
        wave,
        flux,
        SII_6716,
    )
    sii_6731_direct, sii_6731_floor = _estimate_sii_local_flux_floor(
        wave,
        flux,
        SII_6731,
    )
    params.add(
        "sii_6716_flux",
        value=max(_get_flux("SII_6716", "sii_6716_flux"), sii_6716_direct, 1.0e-12),
        min=sii_6716_floor,
    )
    params.add(
        "sii_6731_flux",
        value=max(_get_flux("SII_6731", "sii_6731_flux"), sii_6731_direct, 1.0e-12),
        min=sii_6731_floor,
    )

    residual_after_narrow = flux - narrow_result.model_flux if len(narrow_result.model_flux) == len(wave) else flux
    broad_guess = max(
        _estimate_line_flux(wave, residual_after_narrow, HALPHA, half_width=120.0),
        0.2 * params["ha_narrow_flux"].value,
        1.0e-12,
    )
    _add_broad_balmer_parameters(
        params,
        prefix="ha",
        model=broad_ha_model,
        initial_flux=broad_guess,
        broad1_sigma=1500.0,
        broad2_sigma=4000.0,
    )
    return params


def _line_flux_ha_nii_sii_dict(
    params: Parameters,
    include_broad_ha: bool,
    broad_ha_model: str = "double",
) -> dict[str, float]:
    line_flux = {
        "Ha_narrow": float(params["ha_narrow_flux"].value),
        "OI_6300": float(params["oi_6300_flux"].value),
        "OI_6363": float(params["oi_6363_flux"].value),
        "NII_6548": float(params["nii_6548_flux"].value),
        "NII_6584": float(params["nii_6584_flux"].value),
        "SII_6716": float(params["sii_6716_flux"].value),
        "SII_6731": float(params["sii_6731_flux"].value),
    }
    if include_broad_ha:
        broad_flux = _broad_balmer_flux_dict(params, "ha", broad_ha_model)
        line_flux.update(broad_flux)
        line_flux["Ha_broad"] = float(
            broad_flux.get("Ha_broad_flux_total", broad_flux.get("Ha_broad_1", 0.0))
        )
        line_flux["Ha_broad_total"] = line_flux["Ha_broad"]
    return line_flux


def _param_dict(params: Parameters) -> dict[str, float]:
    return {name: float(param.value) for name, param in params.items()}


def _param_err_dict(params: Parameters) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in params.items():
        stderr = getattr(param, "stderr", None)
        try:
            value = float(stderr)
        except (TypeError, ValueError):
            value = float("nan")
        out[name] = value if np.isfinite(value) else float("nan")
    return out


def _model_oii_flux(wave: np.ndarray, params: Parameters, mode: str) -> np.ndarray:
    velocity = params["oii_velocity"].value
    sigma = params["oii_sigma"].value
    model = _continuum(wave, params)
    if mode == "blend":
        model += _gaussian_integrated(
            wave, OII_BLEND, params["oii_3727_blend_flux"].value, velocity, sigma
        )
    elif mode == "doublet":
        model += _gaussian_integrated(
            wave, OII_3726, params["oii_3726_flux"].value, velocity, sigma
        )
        model += _gaussian_integrated(
            wave, OII_3729, params["oii_3729_flux"].value, velocity, sigma
        )
    else:
        raise ValueError("mode must be 'blend' or 'doublet'.")
    return model


def _residual_oii(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    mode: str,
) -> np.ndarray:
    return (flux - _model_oii_flux(wave, params, mode)) / err


def _initial_oii_parameters(wave: np.ndarray, flux: np.ndarray, mode: str) -> Parameters:
    if mode == "blend":
        line_centers = (OII_BLEND,)
    elif mode == "doublet":
        line_centers = (OII_3726, OII_3729)
    else:
        raise ValueError("mode must be 'blend' or 'doublet'.")

    cont_intercept, cont_slope = _estimate_continuum(wave, flux, line_centers)
    continuum_flux = cont_intercept + cont_slope * (wave - CONTINUUM_PIVOT)
    residual_flux = flux - continuum_flux

    params = Parameters()
    params.add("cont_intercept", value=cont_intercept)
    params.add("cont_slope", value=cont_slope)
    params.add("oii_velocity", value=0.0, min=-800.0, max=800.0)
    params.add("oii_sigma", value=180.0, min=50.0, max=1000.0)
    if mode == "blend":
        params.add(
            "oii_3727_blend_flux",
            value=max(_estimate_line_flux(wave, residual_flux, OII_BLEND, 10.0), 1.0e-12),
            min=0.0,
        )
    else:
        total_flux = max(
            _estimate_line_flux(wave, residual_flux, OII_BLEND, 12.0),
            1.0e-12,
        )
        params.add("oii_3726_flux", value=0.5 * total_flux, min=0.0)
        params.add("oii_3729_flux", value=0.5 * total_flux, min=0.0)
    return params


def _line_flux_oii_dict(params: Parameters, mode: str) -> dict[str, float]:
    if mode == "blend":
        return {"OII_3727_blend": float(params["oii_3727_blend_flux"].value)}
    if mode == "doublet":
        return {
            "OII_3726": float(params["oii_3726_flux"].value),
            "OII_3729": float(params["oii_3729_flux"].value),
        }
    raise ValueError("mode must be 'blend' or 'doublet'.")


def _hb_oiii_fit_info(
    wave: np.ndarray,
    err: np.ndarray,
    params: Parameters,
    include_broad_hb: bool,
    broad_hb_model: str,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    flag_feii_missing_template: bool,
    feii_template_path: str | Path | None = None,
    feii_template_sigma_kms: float = 0.0,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
) -> dict[str, Any]:
    narrow_sigma = _as_float(params.get("narrow_sigma").value if "narrow_sigma" in params else np.nan)
    narrow_sigma_min, narrow_sigma_max = narrow_sigma_bounds
    hb_narrow_flux = _as_float(params["hb_narrow_flux"].value if "hb_narrow_flux" in params else np.nan)
    hb_floor_ratio = _as_float(
        params["hb_narrow_min_oiii_ratio"].value
        if "hb_narrow_min_oiii_ratio" in params
        else 0.0
    )
    hb_floor_reference = _hb_narrow_oiii_reference_flux(params)
    hb_floor = float(hb_floor_ratio * hb_floor_reference) if np.isfinite(hb_floor_ratio) else 0.0
    hb_fixed_to_floor = bool(
        "hb_narrow_flux" in params
        and "hb_narrow_min_oiii_ratio" in params
        and "hb_narrow_flux_excess" not in params
    )
    hb_to_oiii = (
        float(hb_narrow_flux / hb_floor_reference)
        if np.isfinite(hb_narrow_flux) and hb_floor_reference > 0.0
        else float("nan")
    )
    info: dict[str, Any] = {
        "include_broad_hb": bool(include_broad_hb),
        "broad_hb_model": broad_hb_model if include_broad_hb else "none",
        "include_feii": bool(include_feii),
        "flag_feii_missing_template": bool(flag_feii_missing_template),
        "include_oiii_wing": "oiii_4959_wing_flux" in params,
        "hb_narrow_min_oiii_ratio": float(hb_floor_ratio) if np.isfinite(hb_floor_ratio) else 0.0,
        "hb_narrow_oiii5007_reference_flux": float(hb_floor_reference),
        "hb_narrow_flux_floor": float(hb_floor),
        "hb_narrow_fixed_to_oiii_floor": hb_fixed_to_floor,
        "hb_narrow_to_oiii5007_reference_ratio": hb_to_oiii,
        "flag_hb_narrow_at_oiii_floor": bool(
            hb_floor > 0.0
            and np.isfinite(hb_narrow_flux)
            and hb_narrow_flux <= hb_floor * 1.02
        ),
        "feii_template_path": str(feii_template_path) if feii_template_path is not None else "",
        "hb_oiii_narrow_sigma_bounds": [float(narrow_sigma_min), float(narrow_sigma_max)],
        "hb_oiii_narrow_sigma_kms": narrow_sigma,
        "hb_oiii_narrow_fwhm_kms": float(GAUSSIAN_FWHM_FACTOR * narrow_sigma)
        if np.isfinite(narrow_sigma)
        else float("nan"),
        "flag_hb_oiii_narrow_sigma_boundary": bool(
            np.isfinite(narrow_sigma)
            and (
                narrow_sigma <= float(narrow_sigma_min) + 1.0e-3
                or narrow_sigma >= float(narrow_sigma_max) - 1.0e-3
            )
        ),
    }
    if include_broad_hb:
        broad_profile = _broad_balmer_profile(wave, params, "hb", broad_hb_model)
        props = measure_combined_profile_properties(wave, broad_profile, HBETA)
        snr = _component_snr(wave, err, broad_profile)
        narrow_hb_flux = float(params["hb_narrow_flux"].value)
        fwhm_unphysical = bool(
            np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] < 1000.0
        )
        centroid_unphysical = bool(
            np.isfinite(props["centroid_velocity_kms"])
            and abs(props["centroid_velocity_kms"]) > 3000.0
        )
        flux_suspicious = bool(
            narrow_hb_flux > 0.0 and props["integrated_flux"] > 20.0 * narrow_hb_flux
        )
        continuum_degeneracy = _broad_hb_continuum_degeneracy(wave, broad_profile)
        broad_suspicious = flux_suspicious or continuum_degeneracy
        info.update(
            {
                "Hb_broad_flux_total": props["integrated_flux"],
                "Hb_broad_fwhm_combined": props["FWHM_kms"],
                "Hb_broad_line_dispersion": props["line_dispersion_kms"],
                "Hb_broad_centroid_velocity": props["centroid_velocity_kms"],
                "Hb_broad_snr": snr,
                "broad_hb_total_snr": snr,
                "broad_hb_combined_fwhm": props["FWHM_kms"],
                "broad_hb_centroid_velocity": props["centroid_velocity_kms"],
                "broad_hb_flux_total": props["integrated_flux"],
                "flag_broad_hb_weak": bool(snr < 3.0),
                "flag_broad_hb_boundary": _broad_balmer_boundary(params, "hb", broad_hb_model),
                "flag_broad_hb_unphysical": fwhm_unphysical or centroid_unphysical,
                "flag_broad_hb_suspicious": broad_suspicious,
                "flag_broad_hb_flux_suspicious": flux_suspicious,
                "flag_broad_hb_continuum_degeneracy": continuum_degeneracy,
                "flag_broad_balmer_weak": bool(snr < 3.0),
                "flag_broad_balmer_secure": bool(
                    snr > 5.0 and np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] > 1200.0
                ),
                "flag_broad_balmer_tentative": bool(
                    snr > 3.0 and np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] > 1000.0
                ),
                "flag_broad_balmer_unphysical": bool(
                    fwhm_unphysical or centroid_unphysical
                ),
                "flag_broad_balmer_boundary": _broad_balmer_boundary(params, "hb", broad_hb_model),
                "line_flux_aliases": {"Hb_broad_flux_total": props["integrated_flux"]},
            }
        )
    else:
        info["line_flux_aliases"] = {}

    if "oiii_4959_wing_flux" in params:
        wing_profile = _gaussian_integrated(
            wave,
            OIII_4959,
            params["oiii_4959_wing_flux"].value,
            params["oiii_wing_velocity"].value,
            params["oiii_wing_sigma"].value,
        ) + _gaussian_integrated(
            wave,
            OIII_5007,
            params["oiii_5007_wing_flux"].value,
            params["oiii_wing_velocity"].value,
            params["oiii_wing_sigma"].value,
        )
        wing_snr = _component_snr(wave, err, wing_profile)
        wing_flux_total = float(
            params["oiii_4959_wing_flux"].value + params["oiii_5007_wing_flux"].value
        )
        core_flux_total = float(params["oiii_4959_flux"].value + params["oiii_5007_flux"].value)
        info.update(
            {
                "OIII_wing_flux_total": wing_flux_total,
                "OIII_wing_snr": wing_snr,
                "OIII_wing_to_core_flux_ratio": (
                    wing_flux_total / core_flux_total if core_flux_total > 0 else float("nan")
                ),
                "flag_oiii_wing_weak": bool(wing_snr < 3.0),
                "flag_oiii_wing_boundary": bool(
                    _param_near_bounds(params["oiii_wing_velocity"], fraction=0.005)
                    or _param_near_bounds(params["oiii_wing_sigma"], fraction=0.005)
                ),
            }
        )
    else:
        info.update(
            {
                "OIII_wing_flux_total": 0.0,
                "OIII_wing_snr": 0.0,
                "OIII_wing_to_core_flux_ratio": 0.0,
                "flag_oiii_wing_weak": True,
                "flag_oiii_wing_boundary": False,
            }
        )

    if include_feii and feii_template is not None and _has_feii_parameters(params):
        feii = _feii_component(
            wave,
            params,
            feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
        broadening_diagnostics = _feii_component_diagnostics(
            wave,
            params,
            feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
        feii_flux = float(np.trapz(feii, wave))
        feii_snr = _component_snr(wave, err, feii)
        info.update(
            {
                "feii_flux_4434_4684": _feii_flux(params),
                "feii_scale_mode": "blue_central_red"
                if _has_feii_multiscale_parameters(params)
                else "single",
                "feii_extra_sigma_kms": _feii_width(params),
                "feii_smooth_sigma_kms": _feii_smooth_sigma(params),
                "feii_width_kms": _feii_width(params),
                "feii_shift_kms": _feii_shift(params),
                "feii_amp": _feii_flux(params),
                "feii_velocity": _feii_shift(params),
                "feii_sigma": _feii_width(params),
                "feii_template_sigma_kms": float(feii_template_sigma_kms),
                "feii_broadening_mode": broadening_diagnostics["feii_broadening_mode"],
                "feii_sigma_conv": broadening_diagnostics["sigma_conv_kms"],
                "feii_flux_integrated": feii_flux,
                "feii_snr": feii_snr,
                "feii_broadening_diagnostics": broadening_diagnostics,
                "feii_dloglam": broadening_diagnostics["dloglam"],
                "feii_sigma_pix": broadening_diagnostics["sigma_pix"],
                "feii_output_finite_fraction": broadening_diagnostics[
                    "output_finite_fraction"
                ],
                "flag_feii_boundary": _param_near_bounds(params["feii_shift_kms"])
                or _param_near_bounds(params["feii_extra_sigma_kms"]),
                "flag_feii_weak": bool(feii_snr < 3.0),
                "flag_feii_affects_narrow_hb": False,
            }
        )
        if _has_feii_multiscale_parameters(params):
            info.update(
                {
                    "feii_flux_blue_4434_4684": float(params["feii_flux_blue_4434_4684"].value),
                    "feii_flux_central_4920_5050": float(params["feii_flux_central_4920_5050"].value),
                    "feii_flux_red_5100_5600": float(params["feii_flux_red_5100_5600"].value),
                }
            )
    else:
        info.update(
            {
                "feii_scale_mode": "single",
                "feii_amp": 0.0,
                "feii_extra_sigma_kms": float("nan"),
                "feii_smooth_sigma_kms": float("nan"),
                "feii_flux_4434_4684": 0.0,
                "feii_velocity": float("nan"),
                "feii_sigma": float("nan"),
                "feii_width_kms": float("nan"),
                "feii_shift_kms": float("nan"),
                "feii_flux_integrated": 0.0,
                "feii_snr": 0.0,
                "flag_feii_boundary": False,
                "flag_feii_weak": True,
                "flag_feii_affects_narrow_hb": False,
            }
        )
    return info


def _hb_oiii_components(
    wave: np.ndarray,
    params: Parameters,
    include_broad_hb: bool,
    broad_hb_model: str,
    include_feii: bool,
    feii_template: tuple[np.ndarray, np.ndarray] | None,
    include_oiii_wing: bool,
    feii_template_sigma_kms: float = 0.0,
) -> dict[str, np.ndarray]:
    narrow_velocity = params["narrow_velocity"].value
    narrow_sigma = params["narrow_sigma"].value
    oiii_4959_core = _gaussian_integrated(
        wave, OIII_4959, params["oiii_4959_flux"].value, narrow_velocity, narrow_sigma
    )
    oiii_5007_core = _gaussian_integrated(
        wave, OIII_5007, params["oiii_5007_flux"].value, narrow_velocity, narrow_sigma
    )
    components = {
        "continuum": _continuum(wave, params),
        "Hb_narrow": _gaussian_integrated(
            wave, HBETA, params["hb_narrow_flux"].value, narrow_velocity, narrow_sigma
        ),
        "OIII_4959_core": oiii_4959_core,
        "OIII_5007_core": oiii_5007_core,
        "OIII_core": oiii_4959_core + oiii_5007_core,
    }
    if include_broad_hb:
        components["Hb_broad_1"] = _gaussian_integrated(
            wave,
            HBETA,
            params["hb_broad_1_flux"].value,
            params["hb_broad_1_velocity"].value,
            params["hb_broad_1_sigma"].value,
        )
        if broad_hb_model == "double" and "hb_broad_2_flux" in params:
            components["Hb_broad_2"] = _gaussian_integrated(
                wave,
                HBETA,
                params["hb_broad_2_flux"].value,
                params["hb_broad_2_velocity"].value,
                params["hb_broad_2_sigma"].value,
            )
        components["Hb_broad_combined"] = _broad_balmer_profile(
            wave, params, "hb", broad_hb_model
        )
    if include_oiii_wing and "oiii_4959_wing_flux" in params:
        oiii_4959_wing = _gaussian_integrated(
            wave,
            OIII_4959,
            params["oiii_4959_wing_flux"].value,
            params["oiii_wing_velocity"].value,
            params["oiii_wing_sigma"].value,
        )
        oiii_5007_wing = _gaussian_integrated(
            wave,
            OIII_5007,
            params["oiii_5007_wing_flux"].value,
            params["oiii_wing_velocity"].value,
            params["oiii_wing_sigma"].value,
        )
        components["OIII_4959_wing"] = oiii_4959_wing
        components["OIII_5007_wing"] = oiii_5007_wing
        components["OIII_wing"] = oiii_4959_wing + oiii_5007_wing
    if include_feii and feii_template is not None and _has_feii_parameters(params):
        components["FeII"] = _feii_component(
            wave,
            params,
            feii_template,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
    return components


def _augment_hb_feii_region_info(
    info: dict[str, Any],
    wave: np.ndarray,
    flux: np.ndarray,
    model_flux: np.ndarray,
    components: dict[str, np.ndarray],
) -> None:
    diagnostics = compute_feii_region_diagnostics(wave, flux, components)
    info.update(diagnostics)
    info["feii_flux_blue_region"] = diagnostics["blue_feii_model_flux"]
    info["feii_flux_red_region"] = diagnostics["red_feii_model_flux"]
    info["feii_flux_blue"] = diagnostics["blue_feii_model_flux"]
    info["feii_flux_red"] = diagnostics["red_feii_model_flux"]
    info["feii_blue_chi2"] = diagnostics["blue_feii_chi2"]
    info["feii_red_chi2"] = diagnostics["red_feii_chi2"]
    info["feii_chi2_blue"] = diagnostics["blue_feii_chi2"]
    info["feii_chi2_red"] = diagnostics["red_feii_chi2"]
    info["feii_blue_red_model_ratio"] = diagnostics["feii_blue_red_model_ratio"]

    broad = np.asarray(components.get("Hb_broad_combined", np.zeros_like(wave)), dtype=float)
    continuum = np.asarray(components.get("continuum", np.zeros_like(wave)), dtype=float)
    feii = np.asarray(components.get("FeII", np.zeros_like(wave)), dtype=float)
    total = np.asarray(model_flux, dtype=float)
    blue_mask = _region_mask(wave, FEII_BLUE_REGION, FEII_DIAGNOSTIC_MASKS)
    red_mask = _region_mask(wave, FEII_RED_REGION, FEII_DIAGNOSTIC_MASKS)
    broad_blue = _integrate_component(wave, broad, blue_mask)
    broad_red = _integrate_component(wave, broad, red_mask)
    continuum_blue = _integrate_component(wave, continuum, blue_mask)
    continuum_red = _integrate_component(wave, continuum, red_mask)
    feii_blue = _integrate_component(wave, feii, blue_mask)
    feii_red = _integrate_component(wave, feii, red_mask)
    total_blue = _integrate_component(wave, total, blue_mask)
    total_red = _integrate_component(wave, total, red_mask)
    frac_blue = _safe_ratio(broad_blue, total_blue)
    frac_red = _safe_ratio(broad_red, total_red)
    continuum_frac_blue = _safe_ratio(continuum_blue, total_blue)
    continuum_frac_red = _safe_ratio(continuum_red, total_red)
    feii_frac_blue = _safe_ratio(feii_blue, total_blue)
    feii_frac_red = _safe_ratio(feii_red, total_red)
    eats_feii = bool(
        (np.isfinite(frac_blue) and frac_blue > 0.5)
        or (np.isfinite(frac_red) and frac_red > 0.3)
    )
    model_residual = flux - total
    blue_diff = model_residual[blue_mask & np.isfinite(model_residual)]
    red_diff = model_residual[red_mask & np.isfinite(model_residual)]
    blue_underfit = bool(blue_diff.size and np.nanmedian(blue_diff) > 0.25 * np.nanstd(blue_diff))
    red_underfit = bool(red_diff.size and np.nanmedian(red_diff) > 0.25 * np.nanstd(red_diff))
    feii_underfit = bool(blue_underfit or red_underfit)
    continuum_eats_feii = bool(
        (
            (np.isfinite(continuum_frac_blue) and continuum_frac_blue > 0.85)
            or (np.isfinite(continuum_frac_red) and continuum_frac_red > 0.85)
        )
        and (
            diagnostics["blue_feii_chi2"] > 0.0
            or diagnostics["red_feii_chi2"] > 0.0
        )
        and (
            (np.isfinite(feii_frac_blue) and feii_frac_blue < 0.05)
            or (np.isfinite(feii_frac_red) and feii_frac_red < 0.05)
        )
    )
    info.update(
        {
            "broad_hb_fraction_blue_feii": frac_blue,
            "broad_hb_fraction_red_feii": frac_red,
            "broad_hb_fraction_blue": frac_blue,
            "broad_hb_fraction_red": frac_red,
            "continuum_fraction_blue": continuum_frac_blue,
            "continuum_fraction_red": continuum_frac_red,
            "feii_fraction_blue": feii_frac_blue,
            "feii_fraction_red": feii_frac_red,
            "flag_broad_hb_eats_feii": eats_feii,
            "flag_continuum_eats_feii": continuum_eats_feii,
            "flag_feii_underfit": feii_underfit,
        }
    )
    _augment_underlying_oiii_feii_info(info, wave, flux, model_flux, components)


def _augment_underlying_oiii_feii_info(
    info: dict[str, Any],
    wave: np.ndarray,
    flux: np.ndarray,
    model_flux: np.ndarray,
    components: dict[str, np.ndarray],
) -> None:
    mask = _region_mask(wave, OIII_UNDERLYING_FEII_REGION)
    feii = np.asarray(components.get("FeII", np.zeros_like(wave)), dtype=float)
    broad_hb = np.asarray(components.get("Hb_broad_combined", np.zeros_like(wave)), dtype=float)
    oiii_core = np.asarray(components.get("OIII_core", np.zeros_like(wave)), dtype=float)
    oiii_wing = np.asarray(components.get("OIII_wing", np.zeros_like(wave)), dtype=float)
    continuum = np.asarray(components.get("continuum", np.zeros_like(wave)), dtype=float)
    observed = np.asarray(flux, dtype=float)
    total = np.asarray(model_flux, dtype=float)

    feii_flux = _integrate_component(wave, feii, mask)
    broad_flux = _integrate_component(wave, broad_hb, mask)
    core_flux = _integrate_component(wave, oiii_core, mask)
    wing_flux = _integrate_component(wave, oiii_wing, mask)
    continuum_flux = _integrate_component(wave, continuum, mask)
    total_flux = _integrate_component(wave, total, mask)
    observed_flux = _integrate_component(wave, observed, mask)
    good = mask & np.isfinite(observed) & np.isfinite(total)
    residual_chi2 = float(np.nansum((observed[good] - total[good]) ** 2)) if np.any(good) else float("nan")

    info.update(
        {
            "feii_flux_4920_5050": feii_flux,
            "broad_hb_flux_4920_5050": broad_flux,
            "oiii_core_flux_4920_5050": core_flux,
            "oiii_wing_flux_4920_5050": wing_flux,
            "continuum_flux_4920_5050": continuum_flux,
            "total_model_flux_4920_5050": total_flux,
            "observed_flux_4920_5050": observed_flux,
            "residual_chi2_4920_5050": residual_chi2,
            "feii_fraction_under_oiii": _safe_ratio(feii_flux, total_flux),
            "broad_hb_fraction_under_oiii": _safe_ratio(broad_flux, total_flux),
            "oiii_wing_fraction_under_oiii": _safe_ratio(wing_flux, total_flux),
            "continuum_fraction_under_oiii": _safe_ratio(continuum_flux, total_flux),
        }
    )


def _ha_nii_sii_fit_info(
    wave: np.ndarray,
    err: np.ndarray,
    params: Parameters,
    include_broad_ha: bool,
    broad_ha_model: str,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
) -> dict[str, Any]:
    narrow_sigma = _as_float(params.get("narrow_sigma").value if "narrow_sigma" in params else np.nan)
    narrow_sigma_min, narrow_sigma_max = narrow_sigma_bounds
    sii_6716_floor = _as_float(params["sii_6716_flux"].min) if "sii_6716_flux" in params else 0.0
    sii_6731_floor = _as_float(params["sii_6731_flux"].min) if "sii_6731_flux" in params else 0.0
    info: dict[str, Any] = {
        "include_broad_ha": bool(include_broad_ha),
        "broad_ha_model": broad_ha_model if include_broad_ha else "none",
        "ha_narrow_sigma_bounds": [float(narrow_sigma_min), float(narrow_sigma_max)],
        "ha_narrow_sigma_kms": narrow_sigma,
        "sii_6716_flux_floor": sii_6716_floor,
        "sii_6731_flux_floor": sii_6731_floor,
        "flag_sii_local_flux_floor_applied": bool(
            max(sii_6716_floor, sii_6731_floor) > 0.0
        ),
        "ha_narrow_fwhm_kms": float(GAUSSIAN_FWHM_FACTOR * narrow_sigma)
        if np.isfinite(narrow_sigma)
        else float("nan"),
        "flag_ha_narrow_sigma_boundary": bool(
            np.isfinite(narrow_sigma)
            and (
                narrow_sigma <= float(narrow_sigma_min) + 1.0e-3
                or narrow_sigma >= float(narrow_sigma_max) - 1.0e-3
            )
        ),
    }
    if include_broad_ha:
        broad_profile = _broad_balmer_profile(wave, params, "ha", broad_ha_model)
        props = measure_combined_profile_properties(wave, broad_profile, HALPHA)
        snr = _component_snr(wave, err, broad_profile)
        sii_mask = _region_mask(wave, (6695.0, 6750.0))
        narrow_velocity = params["narrow_velocity"].value
        narrow_sigma = params["narrow_sigma"].value
        sii_profile = _gaussian_integrated(
            wave,
            SII_6716,
            params["sii_6716_flux"].value,
            narrow_velocity,
            narrow_sigma,
        ) + _gaussian_integrated(
            wave,
            SII_6731,
            params["sii_6731_flux"].value,
            narrow_velocity,
            narrow_sigma,
        )
        broad_sii_region_flux = _integrate_component(wave, broad_profile, sii_mask)
        sii_region_flux = _integrate_component(wave, sii_profile, sii_mask)
        broad_sii_fraction = _safe_ratio(
            broad_sii_region_flux,
            broad_sii_region_flux + sii_region_flux,
        )
        broad_boundary = _broad_balmer_boundary(params, "ha", broad_ha_model)
        broad_sii_degenerate = bool(
            np.isfinite(broad_sii_fraction)
            and broad_sii_fraction > 0.75
            and broad_sii_region_flux > max(3.0 * sii_region_flux, 1.0e-6)
            and (
                broad_boundary
                or params["sii_6716_flux"].value <= 1.0e-6
                or params["sii_6731_flux"].value <= 1.0e-6
            )
        )
        info.update(
            {
                "Ha_broad_flux_total": props["integrated_flux"],
                "Ha_broad_fwhm_combined": props["FWHM_kms"],
                "Ha_broad_line_dispersion": props["line_dispersion_kms"],
                "Ha_broad_centroid_velocity": props["centroid_velocity_kms"],
                "Ha_broad_snr": snr,
                "flag_broad_balmer_weak": bool(snr < 3.0),
                "flag_broad_balmer_secure": bool(
                    snr > 5.0 and np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] > 1200.0
                ),
                "flag_broad_balmer_tentative": bool(
                    snr > 3.0 and np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] > 1000.0
                ),
                "flag_broad_balmer_unphysical": bool(
                    np.isfinite(props["FWHM_kms"]) and props["FWHM_kms"] < 1000.0
                ),
                "flag_broad_balmer_boundary": broad_boundary,
                "Ha_broad_flux_in_sii_region": broad_sii_region_flux,
                "SII_model_flux_in_sii_region": sii_region_flux,
                "Ha_broad_sii_region_fraction": broad_sii_fraction,
                "flag_ha_broad_eats_sii": broad_sii_degenerate,
            }
        )
    return info


def _ha_nii_sii_components(
    wave: np.ndarray,
    params: Parameters,
    include_broad_ha: bool,
    broad_ha_model: str,
) -> dict[str, np.ndarray]:
    narrow_velocity = params["narrow_velocity"].value
    narrow_sigma = params["narrow_sigma"].value
    components = {
        "continuum": _continuum(wave, params),
        "Ha_narrow": _gaussian_integrated(
            wave, HALPHA, params["ha_narrow_flux"].value, narrow_velocity, narrow_sigma
        ),
        "OI_6300": _gaussian_integrated(
            wave, OI_6300, params["oi_6300_flux"].value, narrow_velocity, narrow_sigma
        ),
        "OI_6363": _gaussian_integrated(
            wave, OI_6363, params["oi_6363_flux"].value, narrow_velocity, narrow_sigma
        ),
        "NII_6548": _gaussian_integrated(
            wave, NII_6548, params["nii_6548_flux"].value, narrow_velocity, narrow_sigma
        ),
        "NII_6584": _gaussian_integrated(
            wave, NII_6584, params["nii_6584_flux"].value, narrow_velocity, narrow_sigma
        ),
        "SII_6716": _gaussian_integrated(
            wave, SII_6716, params["sii_6716_flux"].value, narrow_velocity, narrow_sigma
        ),
        "SII_6731": _gaussian_integrated(
            wave, SII_6731, params["sii_6731_flux"].value, narrow_velocity, narrow_sigma
        ),
    }
    if include_broad_ha:
        components["Ha_broad_1"] = _gaussian_integrated(
            wave,
            HALPHA,
            params["ha_broad_1_flux"].value,
            params["ha_broad_1_velocity"].value,
            params["ha_broad_1_sigma"].value,
        )
        if broad_ha_model == "double" and "ha_broad_2_flux" in params:
            components["Ha_broad_2"] = _gaussian_integrated(
                wave,
                HALPHA,
                params["ha_broad_2_flux"].value,
                params["ha_broad_2_velocity"].value,
                params["ha_broad_2_sigma"].value,
            )
        components["Ha_broad_combined"] = _broad_balmer_profile(
            wave, params, "ha", broad_ha_model
        )
    return components


def _component_snr(wave: np.ndarray, err: np.ndarray, component: np.ndarray) -> float:
    good = np.isfinite(wave) & np.isfinite(err) & (err > 0.0) & np.isfinite(component)
    if np.count_nonzero(good) < 2:
        return 0.0
    signal = float(np.trapz(np.clip(component[good], 0.0, None), wave[good]))
    dw = float(np.nanmedian(np.diff(wave[good])))
    noise = float(np.sqrt(np.sum((err[good] * dw) ** 2)))
    if not np.isfinite(noise) or noise <= 0.0:
        return 0.0
    return signal / noise


def _broad_hb_continuum_degeneracy(
    wave: np.ndarray,
    broad_profile: np.ndarray,
) -> bool:
    good = np.isfinite(wave) & np.isfinite(broad_profile) & (broad_profile > 0.0)
    if np.count_nonzero(good) < 3:
        return False
    peak = float(np.nanmax(broad_profile[good]))
    total = float(np.trapz(np.clip(broad_profile[good], 0.0, None), wave[good]))
    if not np.isfinite(peak) or peak <= 0.0 or not np.isfinite(total) or total <= 0.0:
        return False

    edge_mask = ((wave >= 4700.0) & (wave <= 4750.0)) | (
        (wave >= 5050.0) & (wave <= 5100.0)
    )
    edge_mask &= good
    if np.count_nonzero(edge_mask) < 3:
        return False
    edge_level = float(np.nanmedian(broad_profile[edge_mask]))
    edge_flux = float(np.trapz(np.clip(broad_profile[edge_mask], 0.0, None), wave[edge_mask]))
    return bool(edge_level > 0.08 * peak or edge_flux > 0.20 * total)


def _broad_balmer_boundary(params: Parameters, prefix: str, model: str) -> bool:
    names = [f"{prefix}_broad_1_velocity", f"{prefix}_broad_1_sigma"]
    if model == "double" and f"{prefix}_broad_2_sigma" in params:
        names.extend([f"{prefix}_broad_2_velocity", f"{prefix}_broad_2_sigma"])
    return any(_param_near_bounds(params[name]) for name in names if name in params)


def _param_near_bounds(param: Any, fraction: float = 0.05) -> bool:
    value = float(param.value)
    lower = float(param.min)
    upper = float(param.max)
    if not (np.isfinite(value) and np.isfinite(lower) and np.isfinite(upper)):
        return False
    width = upper - lower
    if width <= 0.0:
        return False
    return value <= lower + fraction * width or value >= upper - fraction * width


def _fit_hb_oiii_hardcoded(
    spec: AGNSpectrum,
    include_broad_hb: bool = True,
    include_feii: bool = True,
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    feii_template_path: str | Path | None = None,
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    hb_window: tuple[float, float] = HB_OIII_WINDOW,
    continuum_mode: str = "linear_constrained",
    max_nfev: int | None = 3000,
    feii_template_sigma_kms: float = 0.0,
    feii_fixed_sigma_kms: float | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
    feii_width_strategy: str = "fixed_grid",
    feii_sigma_strategy: str | None = None,
    feii_scale_mode: str = "single",
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    hb_narrow_min_oiii_ratio: float = DEFAULT_HB_NARROW_MIN_OIII_RATIO,
    fix_hb_narrow_to_oiii_floor: bool = False,
) -> EmissionFitResult:
    """Fit a minimal hard-coded H-beta + [O III] model in the rest-frame spectrum."""

    if feii_sigma_strategy is not None:
        feii_width_strategy = feii_sigma_strategy
    if feii_width_strategy not in {"fixed_grid", "refine_grid", "free"}:
        raise ValueError("feii_width_strategy must be 'fixed_grid', 'refine_grid', or 'free'.")
    if feii_scale_mode not in {"single", "blue_central_red"}:
        raise ValueError("feii_scale_mode must be 'single' or 'blue_central_red'.")
    window_spec = spec.select_window(*hb_window).mask_bad_pixels()
    if len(window_spec.wave) == 0:
        raise ValueError(
            f"No valid pixels available in the {hb_window[0]:.0f}-{hb_window[1]:.0f} Å fitting window."
        )

    feii_missing_template = False
    resolved_feii_path: str | Path | None = feii_template_path
    if include_feii and feii_template is None and feii_template_path is None:
        default_feii_path = _default_feii_template_path()
        if default_feii_path is not None:
            resolved_feii_path = default_feii_path
    loaded_feii = _load_feii_template(feii_template, resolved_feii_path)
    if include_feii and loaded_feii is None:
        warnings.warn(
            "Fe II template missing; Hβ broad fit may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )
        include_feii = False
        feii_missing_template = True
    if include_feii and loaded_feii is not None:
        overlap = (loaded_feii[0] >= hb_window[0]) & (loaded_feii[0] <= hb_window[1])
        if np.count_nonzero(overlap) < 2:
            raise ValueError("Fe II template does not overlap the H-beta + [O III] window.")

    cont_intercept, cont_slope = _estimate_hb_continuum_from_anchors(
        window_spec.wave,
        window_spec.flux,
    )
    stage1_params = Parameters()
    stage1_params.add("cont_intercept", value=cont_intercept)
    stage1_params.add("cont_slope", value=cont_slope)
    _apply_hb_continuum_mode(stage1_params, window_spec.wave, window_spec.flux, continuum_mode)

    stage2_params = _initial_feii_prefit_parameters(
        window_spec.wave,
        window_spec.flux,
        include_feii=include_feii,
        feii_template=loaded_feii,
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_fixed_sigma_kms=feii_fixed_sigma_kms,
        feii_width_bounds=feii_width_bounds,
        feii_shift_bounds=feii_shift_bounds,
    )
    _seed_params_from_fit(stage2_params, stage1_params, ("cont_intercept", "cont_slope"))
    _apply_hb_continuum_mode(stage2_params, window_spec.wave, window_spec.flux, continuum_mode)
    stage2_mask = (
        _region_mask(window_spec.wave, FEII_BLUE_REGION, FEII_DIAGNOSTIC_MASKS)
        | _region_mask(window_spec.wave, FEII_PEDESTAL_REGION, FEII_DIAGNOSTIC_MASKS)
        | _region_mask(window_spec.wave, FEII_RED_REGION, FEII_DIAGNOSTIC_MASKS)
    )
    stage2_mask &= np.isfinite(window_spec.flux) & np.isfinite(window_spec.err) & (window_spec.err > 0.0)
    feii_grid_prefit: dict[str, Any] | None = None
    if (
        include_feii
        and loaded_feii is not None
        and feii_width_strategy in {"fixed_grid", "refine_grid"}
        and np.count_nonzero(stage2_mask) >= 3
    ):
        if feii_fixed_sigma_kms is None:
            sigma_grid: np.ndarray | list[float] | None = None
        else:
            sigma_grid = [float(feii_fixed_sigma_kms)]
        feii_grid_prefit = prefit_feii_grid(
            window_spec.wave,
            window_spec.flux,
            window_spec.err,
            loaded_feii,
            sigma_grid=sigma_grid,
            feii_width_bounds=feii_width_bounds,
            feii_shift_bounds=feii_shift_bounds,
            feii_template_sigma_kms=feii_template_sigma_kms,
        )
        stage2_params["feii_flux_4434_4684"].set(
            value=float(feii_grid_prefit.best_feii_flux_4434_4684),
            min=0.0,
        )
        stage2_params["feii_extra_sigma_kms"].set(
            value=float(feii_grid_prefit.best_feii_width_kms),
            vary=feii_width_strategy == "refine_grid",
        )
        stage2_params["feii_smooth_sigma_kms"].set(
            value=float(feii_grid_prefit.best_feii_smooth_sigma_kms),
            vary=False,
        )
        stage2_params["feii_shift_kms"].set(
            value=float(feii_grid_prefit.best_feii_shift_kms),
            vary=feii_width_strategy == "refine_grid",
        )
        stage2 = None
    elif include_feii and loaded_feii is not None and np.count_nonzero(stage2_mask) >= 3:
        stage2 = minimize(
            _residual_feii_prefit,
            stage2_params,
            args=(
                window_spec.wave,
                window_spec.flux,
                window_spec.err,
                include_feii,
                loaded_feii,
                stage2_mask,
                feii_template_sigma_kms,
            ),
            max_nfev=max_nfev,
        )
        stage2_params = stage2.params
    else:
        stage2 = None
    feii_prefit_constraints = _feii_prefit_constraints(
        stage2_params,
        include_feii,
        loaded_feii,
        feii_fixed_sigma_kms=feii_fixed_sigma_kms,
        feii_fixed_shift_kms=(
            float(stage2_params["feii_shift_kms"].value)
            if include_feii and loaded_feii is not None and feii_width_strategy == "fixed_grid"
            else None
        ),
        feii_width_bounds=feii_width_bounds,
        feii_shift_bounds=feii_shift_bounds,
    )
    if feii_grid_prefit is not None and feii_width_strategy == "refine_grid":
        best_width = float(feii_grid_prefit.best_feii_width_kms)
        best_shift = float(feii_grid_prefit.best_feii_shift_kms)
        feii_prefit_constraints.update(
            {
                "sigma": best_width,
                "sigma_min": max(feii_width_bounds[0], best_width - 500.0),
                "sigma_max": min(feii_width_bounds[1], best_width + 500.0),
                "velocity": best_shift,
                "velocity_min": max(feii_shift_bounds[0], best_shift - 300.0),
                "velocity_max": min(feii_shift_bounds[1], best_shift + 300.0),
                "sigma_fixed": False,
                "velocity_fixed": False,
            }
        )
    elif feii_grid_prefit is not None and feii_width_strategy == "free":
        feii_prefit_constraints.update(
            {
                "sigma_min": feii_width_bounds[0],
                "sigma_max": feii_width_bounds[1],
                "velocity_min": feii_shift_bounds[0],
                "velocity_max": feii_shift_bounds[1],
                "sigma_fixed": False,
                "velocity_fixed": False,
            }
        )

    stage3_params = _initial_parameters(
        window_spec.wave,
        window_spec.flux,
        include_broad_hb=False,
        broad_hb_model=broad_hb_model,
        include_feii=include_feii,
        include_oiii_wing=include_oiii_wing,
        feii_scale_mode=feii_scale_mode,
        narrow_sigma_bounds=narrow_sigma_bounds,
        hb_narrow_min_oiii_ratio=hb_narrow_min_oiii_ratio,
        fix_hb_narrow_to_oiii_floor=fix_hb_narrow_to_oiii_floor,
    )
    _seed_params_from_fit(
        stage3_params,
        stage2_params if stage2 is not None else stage2_params,
        (
            "cont_intercept",
            "cont_slope",
            "feii_flux_4434_4684",
            "feii_shift_kms",
            "feii_extra_sigma_kms",
            "feii_smooth_sigma_kms",
            "feii_width_kms",
        ),
    )
    if feii_grid_prefit is not None and include_feii:
        if "feii_smooth_sigma_kms" in stage3_params:
            stage3_params["feii_smooth_sigma_kms"].set(
                value=float(feii_grid_prefit.best_feii_smooth_sigma_kms),
                vary=False,
            )
        if feii_scale_mode == "blue_central_red":
            initial_flux = float(feii_grid_prefit.best_feii_flux_4434_4684)
            for name, scale in (
                ("feii_flux_blue_4434_4684", 1.0),
                ("feii_flux_central_4920_5050", 0.5),
                ("feii_flux_red_5100_5600", 1.0),
            ):
                if name in stage3_params:
                    stage3_params[name].set(value=max(scale * initial_flux, 1.0e-12), min=0.0)
    _apply_feii_prefit_constraints(stage3_params, feii_prefit_constraints)
    _apply_hb_continuum_mode(stage3_params, window_spec.wave, window_spec.flux, continuum_mode)
    stage3 = minimize(
        _residual_hb_oiii,
        stage3_params,
        args=(
            window_spec.wave,
            window_spec.flux,
            window_spec.err,
            False,
            broad_hb_model,
            include_feii,
            loaded_feii,
            include_oiii_wing,
            feii_template_sigma_kms,
        ),
        max_nfev=max_nfev,
    )

    feii_sigma_grid = (
        _feii_prefit_grid(feii_prefit_constraints, "sigma")
        if include_feii
        else [float("nan")]
    )
    feii_velocity_grid = (
        _feii_prefit_grid(feii_prefit_constraints, "velocity")
        if include_feii
        else [float("nan")]
    )
    broad1_grid = [1200.0, 1800.0] if include_broad_hb else [float("nan")]
    broad2_grid = [3500.0, 6000.0] if include_broad_hb and broad_hb_model == "double" else [float("nan")]
    best_fit: Any | None = None
    best_params: Parameters | None = None
    best_chi2 = float("inf")
    best_bic = float("inf")
    best_trial_index = -1
    best_trial_settings: dict[str, float] = {}
    failed_trial_count = 0
    trial_index = 0
    for feii_sigma_init in feii_sigma_grid:
        for broad1_sigma_init in broad1_grid:
            for broad2_sigma_init in broad2_grid:
                for feii_velocity_init in feii_velocity_grid:
                    if trial_index >= 24:
                        break
                    params = stage3.params.copy()
                    _apply_hb_continuum_mode(params, window_spec.wave, window_spec.flux, continuum_mode)
                    if include_feii and loaded_feii is not None:
                        continuum_flux_init = _continuum(window_spec.wave, params)
                        if feii_width_strategy in {"fixed_grid", "refine_grid"} and feii_grid_prefit is not None:
                            feii_flux_init = float(
                                feii_grid_prefit.best_feii_flux_4434_4684
                            )
                        else:
                            feii_flux_init = _estimate_feii_flux_4434_4684(
                                window_spec.wave,
                                window_spec.flux,
                                continuum_flux_init,
                                loaded_feii,
                                feii_width_kms=feii_sigma_init,
                                feii_shift_kms=feii_velocity_init,
                                feii_template_sigma_kms=feii_template_sigma_kms,
                            )
                        params["feii_flux_4434_4684"].set(value=feii_flux_init, min=0.0)
                        params["feii_shift_kms"].set(
                            value=feii_velocity_init,
                            min=feii_prefit_constraints["velocity_min"],
                            max=feii_prefit_constraints["velocity_max"],
                            vary=feii_width_strategy != "fixed_grid",
                        )
                        params["feii_extra_sigma_kms"].set(
                            value=feii_sigma_init,
                            min=0.0 if feii_fixed_sigma_kms is not None else feii_prefit_constraints["sigma_min"],
                            max=max(feii_width_bounds[1], float(feii_sigma_init))
                            if feii_fixed_sigma_kms is not None
                            else feii_prefit_constraints["sigma_max"],
                            vary=feii_width_strategy != "fixed_grid"
                            and feii_fixed_sigma_kms is None,
                        )
                    if include_broad_hb:
                        _add_hb_broad_parameters_if_needed(
                            params,
                            window_spec.wave,
                            window_spec.flux,
                            broad_hb_model,
                            loaded_feii,
                            feii_template_sigma_kms,
                        )
                        params["hb_broad_1_sigma"].set(value=broad1_sigma_init)
                        if broad_hb_model == "double" and "hb_broad_2_sigma" in params:
                            params["hb_broad_2_sigma"].set(value=broad2_sigma_init)
                    try:
                        candidate = minimize(
                            _residual_hb_oiii,
                            params,
                            args=(
                                window_spec.wave,
                                window_spec.flux,
                                window_spec.err,
                                include_broad_hb,
                                broad_hb_model,
                                include_feii,
                                loaded_feii,
                                include_oiii_wing,
                                feii_template_sigma_kms,
                            ),
                            max_nfev=max_nfev,
                        )
                        candidate_model = _model_hb_oiii_flux(
                            window_spec.wave,
                            candidate.params,
                            include_broad_hb,
                            broad_hb_model=broad_hb_model,
                            include_feii=include_feii,
                            feii_template=loaded_feii,
                            include_oiii_wing=include_oiii_wing,
                            feii_template_sigma_kms=feii_template_sigma_kms,
                        )
                        candidate_residual = window_spec.flux - candidate_model
                        candidate_chi2 = float(np.sum((candidate_residual / window_spec.err) ** 2))
                        candidate_bic = float(candidate_chi2 + int(candidate.nvarys) * np.log(int(candidate.ndata)))
                        if np.isfinite(candidate_bic) and candidate_bic < best_bic:
                            best_fit = candidate
                            best_params = candidate.params.copy()
                            best_chi2 = candidate_chi2
                            best_bic = candidate_bic
                            best_trial_index = trial_index
                            best_trial_settings = {
                                "feii_sigma_init": float(feii_sigma_init),
                                "feii_velocity_init": float(feii_velocity_init),
                                "broad_hb_1_sigma_init": float(broad1_sigma_init),
                                "broad_hb_2_sigma_init": float(broad2_sigma_init),
                            }
                    except Exception:
                        failed_trial_count += 1
                    trial_index += 1
                if trial_index >= 24:
                    break
            if trial_index >= 24:
                break
        if trial_index >= 24:
            break

    if best_fit is None or best_params is None:
        raise RuntimeError("All H-beta + [O III] final joint fit trials failed.")
    fit = best_fit

    model_flux = _model_hb_oiii_flux(
        window_spec.wave,
        fit.params,
        include_broad_hb,
        broad_hb_model=broad_hb_model,
        include_feii=include_feii,
        feii_template=loaded_feii,
        include_oiii_wing=include_oiii_wing,
        feii_template_sigma_kms=feii_template_sigma_kms,
    )
    continuum_flux = _continuum(window_spec.wave, fit.params)
    residual_flux = window_spec.flux - model_flux
    chi2 = float(np.sum((residual_flux / window_spec.err) ** 2))
    line_flux = _line_flux_hb_oiii_dict(
        fit.params,
        include_broad_hb,
        broad_hb_model=broad_hb_model,
        include_oiii_wing=include_oiii_wing,
    )
    components = _hb_oiii_components(
        window_spec.wave,
        fit.params,
        include_broad_hb=include_broad_hb,
        broad_hb_model=broad_hb_model,
        include_feii=include_feii,
        feii_template=loaded_feii,
        include_oiii_wing=include_oiii_wing,
        feii_template_sigma_kms=feii_template_sigma_kms,
    )
    fit_info = _hb_oiii_fit_info(
        window_spec.wave,
        window_spec.err,
        fit.params,
        include_broad_hb=include_broad_hb,
        broad_hb_model=broad_hb_model,
        include_feii=include_feii,
        feii_template=loaded_feii,
        flag_feii_missing_template=feii_missing_template,
        feii_template_path=resolved_feii_path,
        feii_template_sigma_kms=feii_template_sigma_kms,
        narrow_sigma_bounds=narrow_sigma_bounds,
    )
    _augment_hb_feii_region_info(
        fit_info,
        window_spec.wave,
        window_spec.flux,
        model_flux,
        components,
    )
    fit_info["hb_window_min"] = float(hb_window[0])
    fit_info["hb_window_max"] = float(hb_window[1])
    fit_info["continuum_mode"] = continuum_mode
    fit_info["stage1_continuum_intercept"] = float(stage1_params["cont_intercept"].value)
    fit_info["stage1_continuum_slope"] = float(stage1_params["cont_slope"].value)
    fit_info["stage2_feii_success"] = (
        bool(getattr(stage2, "success", False))
        if stage2 is not None
        else bool(feii_grid_prefit is not None and np.isfinite(feii_grid_prefit.chi2_total))
    )
    fit_info["stage3_narrow_success"] = bool(stage3.success)
    fit_info["stage2_feii_chi2"] = (
        float(np.sum(getattr(stage2, "residual", np.array([])) ** 2))
        if stage2 is not None
        else (
            float(feii_grid_prefit.chi2_total)
            if feii_grid_prefit is not None
            else float("nan")
        )
    )
    fit_info["stage2_feii_sigma"] = float(feii_prefit_constraints["sigma"])
    fit_info["stage2_feii_sigma_min"] = float(feii_prefit_constraints["sigma_min"])
    fit_info["stage2_feii_sigma_max"] = float(feii_prefit_constraints["sigma_max"])
    fit_info["feii_sigma_fixed"] = bool(
        "feii_extra_sigma_kms" in fit.params and not fit.params["feii_extra_sigma_kms"].vary
    )
    fit_info["feii_shift_fixed"] = bool(
        "feii_shift_kms" in fit.params and not fit.params["feii_shift_kms"].vary
    )
    fit_info["feii_width_strategy"] = feii_width_strategy
    fit_info["flag_feii_too_broad"] = bool(
        np.isfinite(fit_info.get("feii_width_kms", np.nan))
        and fit_info.get("feii_width_kms", np.nan) > 3500.0
        and (
            not np.isfinite(fit_info.get("delta_bic_feii", np.nan))
            or fit_info.get("delta_bic_feii", np.nan) < 10.0
        )
    )
    finite_grid = np.isfinite(feii_grid_prefit.chi2_grid).any() if feii_grid_prefit is not None else True
    fit_info["flag_feii_unconstrained"] = bool(
        include_feii
        and loaded_feii is not None
        and (
            not np.isfinite(fit_info.get("feii_snr", np.nan))
            or not finite_grid
            or (feii_grid_prefit is not None and feii_grid_prefit.flags.get("flag_feii_unconstrained", False))
        )
    )
    fit_info["flag_feii_continuum_degenerate"] = bool(
        feii_grid_prefit.flags.get("flag_feii_continuum_degenerate", False)
        if feii_grid_prefit is not None
        else False
    )
    fit_info["flag_feii_high_frequency_overfit"] = bool(
        feii_grid_prefit.flags.get("flag_feii_high_frequency_overfit", False)
        if feii_grid_prefit is not None
        else False
    )
    fit_info["flag_single_feii_template_shape_mismatch"] = bool(
        feii_grid_prefit.flags.get("flag_single_feii_template_shape_mismatch", False)
        if feii_grid_prefit is not None
        else False
    )
    fit_info["feii_fixed_sigma_kms"] = (
        float(feii_fixed_sigma_kms) if feii_fixed_sigma_kms is not None else float("nan")
    )
    fit_info["stage2_feii_velocity"] = float(feii_prefit_constraints["velocity"])
    fit_info["stage2_feii_velocity_min"] = float(feii_prefit_constraints["velocity_min"])
    fit_info["stage2_feii_velocity_max"] = float(feii_prefit_constraints["velocity_max"])
    fit_info["stage3_narrow_chi2"] = float(np.sum(stage3.residual**2))
    fit_info["final_joint_chi2"] = float(best_chi2)
    fit_info["final_joint_bic"] = float(best_bic)
    fit_info["n_joint_fit_trials"] = int(trial_index)
    fit_info["best_trial_index"] = int(best_trial_index)
    fit_info["failed_trial_count"] = int(failed_trial_count)
    fit_info["feii_sigma_init_grid"] = feii_sigma_grid if include_feii else []
    fit_info["feii_velocity_init_grid"] = feii_velocity_grid if include_feii else []
    if feii_grid_prefit is not None:
        fit_info["feii_prefit_grid"] = feii_grid_prefit.to_dict()
        fit_info["feii_flux_4434_4684_prefit"] = float(
            feii_grid_prefit.best_feii_flux_4434_4684
        )
        fit_info["feii_extra_sigma_kms_prefit"] = float(
            feii_grid_prefit.best_feii_extra_sigma_kms
        )
        fit_info["feii_template_method_prefit"] = feii_grid_prefit.best_feii_template_method
        fit_info["feii_smooth_sigma_kms_prefit"] = float(
            feii_grid_prefit.best_feii_smooth_sigma_kms
        )
        fit_info["feii_width_kms_prefit"] = float(feii_grid_prefit.best_feii_width_kms)
        fit_info["feii_shift_kms_prefit"] = float(feii_grid_prefit.best_feii_shift_kms)
        fit_info["feii_chi2_blue"] = float(feii_grid_prefit.chi2_blue)
        fit_info["feii_chi2_pedestal"] = float(feii_grid_prefit.chi2_pedestal)
        fit_info["feii_chi2_red"] = float(feii_grid_prefit.chi2_red)
    fit_info["feii_flux_4434_4684_final"] = float(fit_info.get("feii_flux_4434_4684", np.nan))
    fit_info["feii_extra_sigma_kms_final"] = float(fit_info.get("feii_extra_sigma_kms", np.nan))
    fit_info["feii_smooth_sigma_kms_final"] = float(fit_info.get("feii_smooth_sigma_kms", np.nan))
    fit_info["feii_template_method_final"] = fit_info.get(
        "feii_template_method_prefit",
        "raw" if fit_info.get("feii_smooth_sigma_kms", 0.0) == 0.0 else "smoothed",
    )
    fit_info["feii_width_kms_final"] = float(fit_info.get("feii_width_kms", np.nan))
    fit_info["feii_shift_kms_final"] = float(fit_info.get("feii_shift_kms", np.nan))
    fit_info["best_feii_sigma_init"] = best_trial_settings.get("feii_sigma_init", float("nan"))
    fit_info["best_final_chi2"] = float(best_chi2)
    fit_info["best_final_bic"] = float(best_bic)
    fit_info["best_trial_settings"] = best_trial_settings
    if "cont_slope" in fit.params:
        fit_info["flag_continuum_slope_boundary"] = _param_near_bounds(fit.params["cont_slope"])
        fit_info["flag_continuum_may_absorb_feii"] = bool(
            fit_info["flag_continuum_slope_boundary"]
            and (
                fit_info.get("flag_continuum_eats_feii", False)
                or fit_info.get("flag_feii_underfit", False)
            )
        )
    if include_feii and loaded_feii is not None:
        try:
            no_feii = _fit_hb_oiii_hardcoded(
                spec,
                include_broad_hb=include_broad_hb,
                include_feii=False,
                include_oiii_wing=include_oiii_wing,
                broad_hb_model=broad_hb_model,
                hb_window=hb_window,
                continuum_mode=continuum_mode,
                max_nfev=max_nfev,
                feii_template_sigma_kms=feii_template_sigma_kms,
                feii_fixed_sigma_kms=feii_fixed_sigma_kms,
                feii_width_bounds=feii_width_bounds,
                feii_shift_bounds=feii_shift_bounds,
                feii_width_strategy=feii_width_strategy,
                feii_scale_mode=feii_scale_mode,
                narrow_sigma_bounds=narrow_sigma_bounds,
                fix_hb_narrow_to_oiii_floor=fix_hb_narrow_to_oiii_floor,
            )
            fit_info["bic_with_feii"] = fit_bic(
                EmissionFitResult(
                    success=bool(fit.success),
                    message=str(fit.message),
                    params=_param_dict(fit.params),
                    param_err=_param_err_dict(fit.params),
                    line_flux=line_flux,
                    line_flux_err={},
                    model_flux=model_flux,
                    continuum_flux=continuum_flux,
                    residual_flux=residual_flux,
                    chi2=chi2,
                    redchi=float(fit.redchi),
                    ndata=int(fit.ndata),
                    nvarys=int(fit.nvarys),
                    window_name="hb_oiii",
                )
            )
            fit_info["bic_without_feii"] = no_feii.bic
            fit_info["delta_bic_feii"] = no_feii.bic - fit_info["bic_with_feii"]
            reference = no_feii.line_flux.get("Hb_narrow", np.nan)
            current = line_flux.get("Hb_narrow", np.nan)
            fit_info["narrow_hb_flux_with_feii"] = float(current)
            fit_info["narrow_hb_flux_without_feii"] = float(reference)
            if np.isfinite(reference) and reference > 0.0 and np.isfinite(current):
                fit_info["narrow_hb_flux_change_percent_feii"] = float(
                    100.0 * (current - reference) / reference
                )
                fit_info["narrow_hb_change_due_to_feii"] = float(
                    (current - reference) / reference
                )
            else:
                fit_info["narrow_hb_flux_change_percent_feii"] = float("nan")
                fit_info["narrow_hb_change_due_to_feii"] = float("nan")
            fit_info["flag_feii_affects_narrow_hb"] = bool(
                np.isfinite(reference)
                and reference > 0.0
                and np.isfinite(current)
                and abs(current - reference) / reference > 0.2
            )
        except Exception:
            fit_info["bic_with_feii"] = float("nan")
            fit_info["bic_without_feii"] = float("nan")
            fit_info["delta_bic_feii"] = float("nan")
            fit_info["narrow_hb_flux_change_percent_feii"] = float("nan")
            fit_info["narrow_hb_change_due_to_feii"] = float("nan")
            fit_info["narrow_hb_flux_with_feii"] = float(line_flux.get("Hb_narrow", np.nan))
            fit_info["narrow_hb_flux_without_feii"] = float("nan")
            fit_info["flag_feii_affects_narrow_hb"] = False
    else:
        fit_info["bic_with_feii"] = float("nan")
        fit_info["bic_without_feii"] = float("nan")
        fit_info["delta_bic_feii"] = float("nan")
        fit_info["narrow_hb_flux_change_percent_feii"] = float("nan")
        fit_info["narrow_hb_change_due_to_feii"] = float("nan")
        fit_info["narrow_hb_flux_with_feii"] = float(line_flux.get("Hb_narrow", np.nan))
        fit_info["narrow_hb_flux_without_feii"] = float("nan")
    result_warnings: list[str] = []
    if feii_missing_template:
        result_warnings.append("Fe II template missing; Hβ broad fit may be unreliable.")
    if fit_info.get("flag_broad_hb_suspicious") or fit_info.get(
        "flag_broad_hb_unphysical"
    ):
        result_warnings.append(
            "Broad Hβ may be degenerate with Fe II/continuum; inspect manually."
        )
    if fit_info.get("flag_broad_hb_eats_feii"):
        result_warnings.append("Broad Hβ may be absorbing Fe II pseudo-continuum.")
    line_flux.update(fit_info.get("line_flux_aliases", {}))
    fit_info.pop("line_flux_aliases", None)

    return EmissionFitResult(
        success=bool(fit.success),
        message=str(fit.message),
        params=_param_dict(fit.params),
        param_err=_param_err_dict(fit.params),
        line_flux=line_flux,
        line_flux_err={},
        model_flux=model_flux,
        continuum_flux=continuum_flux,
        residual_flux=residual_flux,
        chi2=chi2,
        redchi=float(fit.redchi),
        ndata=int(fit.ndata),
        nvarys=int(fit.nvarys),
        window_name="hb_oiii",
        fit_info=fit_info,
        component_flux=components,
        warnings=result_warnings,
    )


def _fit_oii_hardcoded(spec: AGNSpectrum, mode: str = "doublet") -> EmissionFitResult:
    """Fit a minimal hard-coded [O II] blend or doublet model."""

    if mode not in {"blend", "doublet"}:
        raise ValueError("mode must be 'blend' or 'doublet'.")

    window_spec = spec.select_window(*OII_WINDOW).mask_bad_pixels()
    if len(window_spec.wave) == 0:
        raise ValueError("No valid pixels available in the 3650-3800 Å fitting window.")

    params = _initial_oii_parameters(window_spec.wave, window_spec.flux, mode)
    fit = minimize(
        _residual_oii,
        params,
        args=(window_spec.wave, window_spec.flux, window_spec.err, mode),
    )

    model_flux = _model_oii_flux(window_spec.wave, fit.params, mode)
    continuum_flux = _continuum(window_spec.wave, fit.params)
    residual_flux = window_spec.flux - model_flux
    chi2 = float(np.sum((residual_flux / window_spec.err) ** 2))

    return EmissionFitResult(
        success=bool(fit.success),
        message=str(fit.message),
        params=_param_dict(fit.params),
        param_err=_param_err_dict(fit.params),
        line_flux=_line_flux_oii_dict(fit.params, mode),
        line_flux_err={},
        model_flux=model_flux,
        continuum_flux=continuum_flux,
        residual_flux=residual_flux,
        chi2=chi2,
        redchi=float(fit.redchi),
        ndata=int(fit.ndata),
        nvarys=int(fit.nvarys),
        window_name=f"oii_{mode}",
    )


def _fit_ha_nii_sii_hardcoded(
    spec: AGNSpectrum,
    include_broad_ha: bool = True,
    broad_ha_model: str = "double",
    max_nfev: int | None = 3000,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    narrow_kinematics: dict[str, float] | None = None,
    freeze_narrow_kinematics: bool = False,
    initial_params: Parameters | None = None,
    fit_info_extra: dict[str, Any] | None = None,
) -> EmissionFitResult:
    """Fit a minimal hard-coded H-alpha + [N II] + [S II] model."""

    window_spec = spec.select_window(*HA_NII_SII_WINDOW).mask_bad_pixels()
    if len(window_spec.wave) == 0:
        raise ValueError("No valid pixels available in the 6400-6800 Å fitting window.")

    params = (
        initial_params.copy()
        if initial_params is not None
        else _initial_ha_nii_sii_parameters(
            window_spec.wave,
            window_spec.flux,
            include_broad_ha,
            broad_ha_model=broad_ha_model,
            narrow_sigma_bounds=narrow_sigma_bounds,
            narrow_kinematics=narrow_kinematics,
            freeze_narrow_kinematics=freeze_narrow_kinematics,
        )
    )
    fit = minimize(
        _residual_ha_nii_sii,
        params,
        args=(window_spec.wave, window_spec.flux, window_spec.err, include_broad_ha, broad_ha_model),
        max_nfev=max_nfev,
    )

    model_flux = _model_ha_nii_sii_flux(
        window_spec.wave,
        fit.params,
        include_broad_ha,
        broad_ha_model=broad_ha_model,
    )
    continuum_flux = _continuum(window_spec.wave, fit.params)
    residual_flux = window_spec.flux - model_flux
    chi2 = float(np.sum((residual_flux / window_spec.err) ** 2))
    optimizer_success = bool(fit.success)
    fit_message = str(fit.message)
    usable_fit = bool(
        optimizer_success
        or (
            np.isfinite(chi2)
            and np.isfinite(float(fit.redchi))
            and (
                "Could not estimate error-bars" in fit_message
                or "Tolerance seems to be too small" in fit_message
            )
        )
    )

    return EmissionFitResult(
        success=usable_fit,
        message=fit_message,
        params=_param_dict(fit.params),
        param_err=_param_err_dict(fit.params),
        line_flux=_line_flux_ha_nii_sii_dict(
            fit.params,
            include_broad_ha,
            broad_ha_model=broad_ha_model,
        ),
        line_flux_err={},
        model_flux=model_flux,
        continuum_flux=continuum_flux,
        residual_flux=residual_flux,
        chi2=chi2,
        redchi=float(fit.redchi),
        ndata=int(fit.ndata),
        nvarys=int(fit.nvarys),
        window_name="ha_nii_sii",
        fit_info={
            **_ha_nii_sii_fit_info(
                window_spec.wave,
                window_spec.err,
                fit.params,
                include_broad_ha=include_broad_ha,
                broad_ha_model=broad_ha_model,
                narrow_sigma_bounds=narrow_sigma_bounds,
            ),
            "optimizer_success": optimizer_success,
            "optimizer_message": fit_message,
            **(fit_info_extra or {}),
        },
        component_flux=_ha_nii_sii_components(
            window_spec.wave,
            fit.params,
            include_broad_ha=include_broad_ha,
            broad_ha_model=broad_ha_model,
        ),
    )


def fit_hb_oiii(
    spec: AGNSpectrum,
    include_broad_hb: bool = True,
    include_feii: bool = True,
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    feii_template_path: str | Path | None = None,
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    hb_window: tuple[float, float] = HB_OIII_WINDOW,
    continuum_mode: str = "linear_constrained",
    max_nfev: int | None = 3000,
    feii_template_sigma_kms: float = 0.0,
    feii_fixed_sigma_kms: float | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
    feii_width_strategy: str = "fixed_grid",
    feii_sigma_strategy: str | None = None,
    feii_scale_mode: str = "single",
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    hb_narrow_min_oiii_ratio: float = DEFAULT_HB_NARROW_MIN_OIII_RATIO,
    fix_hb_narrow_to_oiii_floor: bool = False,
) -> EmissionFitResult:
    """Fit H-beta + [O III] with broad Balmer, optional Fe II, and [O III] wing."""

    return _fit_hb_oiii_hardcoded(
        spec,
        include_broad_hb=include_broad_hb,
        include_feii=include_feii,
        feii_template=feii_template,
        feii_template_path=feii_template_path,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        hb_window=hb_window,
        continuum_mode=continuum_mode,
        max_nfev=max_nfev,
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_fixed_sigma_kms=feii_fixed_sigma_kms,
        feii_width_bounds=feii_width_bounds,
        feii_shift_bounds=feii_shift_bounds,
        feii_width_strategy=feii_width_strategy,
        feii_sigma_strategy=feii_sigma_strategy,
        feii_scale_mode=feii_scale_mode,
        narrow_sigma_bounds=narrow_sigma_bounds,
        hb_narrow_min_oiii_ratio=hb_narrow_min_oiii_ratio,
        fix_hb_narrow_to_oiii_floor=fix_hb_narrow_to_oiii_floor,
    )


def fit_hb_oiii_bic(
    spec: AGNSpectrum,
    bic_threshold: float = 10.0,
    include_feii: bool = True,
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    feii_template_path: str | Path | None = None,
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    hb_window: tuple[float, float] = HB_OIII_WINDOW,
    continuum_mode: str = "linear_constrained",
    max_nfev: int | None = 3000,
    feii_template_sigma_kms: float = 0.0,
    feii_fixed_sigma_kms: float | None = None,
    feii_width_bounds: tuple[float, float] = FEII_DEFAULT_WIDTH_BOUNDS,
    feii_shift_bounds: tuple[float, float] = FEII_DEFAULT_SHIFT_BOUNDS,
    feii_width_strategy: str = "fixed_grid",
    feii_sigma_strategy: str | None = None,
    feii_scale_mode: str = "single",
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    hb_narrow_min_oiii_ratio: float = DEFAULT_HB_NARROW_MIN_OIII_RATIO,
    fix_hb_narrow_to_oiii_floor: bool = False,
) -> EmissionFitResult:
    """Fit H-beta + [O III] and include broad H-beta only if BIC supports it."""

    common = {
        "include_feii": include_feii,
        "feii_template": feii_template,
        "feii_template_path": feii_template_path,
        "broad_hb_model": broad_hb_model,
        "hb_window": hb_window,
        "continuum_mode": continuum_mode,
        "max_nfev": max_nfev,
        "feii_template_sigma_kms": feii_template_sigma_kms,
        "feii_fixed_sigma_kms": feii_fixed_sigma_kms,
        "feii_width_bounds": feii_width_bounds,
        "feii_shift_bounds": feii_shift_bounds,
        "feii_width_strategy": feii_width_strategy,
        "feii_sigma_strategy": feii_sigma_strategy,
        "feii_scale_mode": feii_scale_mode,
        "narrow_sigma_bounds": narrow_sigma_bounds,
        "hb_narrow_min_oiii_ratio": hb_narrow_min_oiii_ratio,
        "fix_hb_narrow_to_oiii_floor": fix_hb_narrow_to_oiii_floor,
    }
    narrow_core = fit_hb_oiii(
        spec,
        include_broad_hb=False,
        include_oiii_wing=False,
        **common,
    )
    broad_core = fit_hb_oiii(
        spec,
        include_broad_hb=True,
        include_oiii_wing=False,
        **common,
    )
    if not include_oiii_wing:
        return select_broad_balmer_by_bic(
            narrow_result=narrow_core,
            broad_result=broad_core,
            bic_threshold=bic_threshold,
            broad_line_name="Hb_broad",
        )
    narrow_wing = fit_hb_oiii(
        spec,
        include_broad_hb=False,
        include_oiii_wing=True,
        **common,
    )
    broad_wing = fit_hb_oiii(
        spec,
        include_broad_hb=True,
        include_oiii_wing=True,
        **common,
    )
    return _select_hb_oiii_bic_with_oiii_wing(
        narrow_core=narrow_core,
        broad_core=broad_core,
        narrow_wing=narrow_wing,
        broad_wing=broad_wing,
        bic_threshold=bic_threshold,
    )


def _select_hb_oiii_bic_with_oiii_wing(
    narrow_core: EmissionFitResult,
    broad_core: EmissionFitResult,
    narrow_wing: EmissionFitResult,
    broad_wing: EmissionFitResult,
    bic_threshold: float,
) -> EmissionFitResult:
    """Select H-beta/[O III] model while requiring evidence for [O III] wing."""

    candidates = {
        "narrow_core": narrow_core,
        "broad_core": broad_core,
        "narrow_wing": narrow_wing,
        "broad_wing": broad_wing,
    }
    bic = {name: fit_bic(result) for name, result in candidates.items()}
    usable: list[tuple[str, EmissionFitResult, float]] = [("narrow_core", narrow_core, bic["narrow_core"])]
    rejected: list[str] = []

    if _bic_improves(bic["narrow_core"], bic["broad_core"], bic_threshold) and _hb_broad_result_usable(broad_core):
        usable.append(("broad_core", broad_core, bic["broad_core"]))
    else:
        rejected.append("broad_core:no_bic_or_pathological")

    if _bic_improves(bic["narrow_core"], bic["narrow_wing"], bic_threshold) and _oiii_wing_result_usable(narrow_wing):
        usable.append(("narrow_wing", narrow_wing, bic["narrow_wing"]))
    else:
        rejected.append("narrow_wing:no_bic_or_pathological")

    if (
        _bic_improves(bic["narrow_wing"], bic["broad_wing"], bic_threshold)
        and _bic_improves(bic["broad_core"], bic["broad_wing"], bic_threshold)
        and _hb_broad_result_usable(broad_wing)
        and _oiii_wing_result_usable(broad_wing)
    ):
        usable.append(("broad_wing", broad_wing, bic["broad_wing"]))
    else:
        rejected.append("broad_wing:no_bic_or_pathological")

    selected_label, selected, selected_bic = min(usable, key=lambda item: item[2])
    selected.fit_info.update(
        {
            "hb_oiii_model_selection": "bic_with_oiii_wing_gate",
            "selected_model": selected_label,
            "bic_threshold": float(bic_threshold),
            "bic_narrow_core": bic["narrow_core"],
            "bic_broad_core": bic["broad_core"],
            "bic_narrow_wing": bic["narrow_wing"],
            "bic_broad_wing": bic["broad_wing"],
            "selected_bic": selected_bic,
            "oiii_wing_selected": bool("wing" in selected_label),
            "broad_balmer_selected": bool("broad" in selected_label),
            "rejected_hb_oiii_models": rejected,
        }
    )
    selected.params.update(
        {
            "bic_narrow_core": bic["narrow_core"],
            "bic_broad_core": bic["broad_core"],
            "bic_narrow_wing": bic["narrow_wing"],
            "bic_broad_wing": bic["broad_wing"],
            "oiii_wing_selected": float("wing" in selected_label),
            "broad_balmer_selected": float("broad" in selected_label),
        }
    )
    return selected


def _bic_improves(base_bic: float, candidate_bic: float, threshold: float) -> bool:
    return bool(
        np.isfinite(base_bic)
        and np.isfinite(candidate_bic)
        and (base_bic - candidate_bic) > threshold
    )


def _hb_broad_result_usable(result: EmissionFitResult) -> bool:
    info = result.fit_info or {}
    if not bool(result.success):
        return False
    if bool(info.get("flag_broad_hb_unphysical", False)):
        return False
    snr = _as_float(info.get("Hb_broad_snr", info.get("broad_hb_total_snr")))
    fwhm = _as_float(info.get("Hb_broad_fwhm_combined", info.get("broad_hb_combined_fwhm")))
    centroid = _as_float(info.get("Hb_broad_centroid_velocity", info.get("broad_hb_centroid_velocity")))
    if not (np.isfinite(snr) and snr >= 3.0):
        return False
    if np.isfinite(fwhm) and fwhm > 10000.0:
        return False
    if np.isfinite(centroid) and abs(centroid) > 3000.0:
        return False
    return True


def _oiii_wing_result_usable(result: EmissionFitResult) -> bool:
    info = result.fit_info or {}
    params = result.params or {}
    if not bool(result.success):
        return False
    if not bool(info.get("include_oiii_wing", False)):
        return False
    if bool(info.get("flag_oiii_wing_boundary", False)):
        return False
    snr = _as_float(info.get("OIII_wing_snr"))
    if not (np.isfinite(snr) and snr >= 3.0):
        return False
    sigma = _as_float(params.get("oiii_wing_sigma"))
    velocity = _as_float(params.get("oiii_wing_velocity"))
    if np.isfinite(sigma) and (sigma <= 55.0 or sigma >= 1950.0):
        return False
    if np.isfinite(velocity) and (velocity <= -1490.0 or velocity >= 990.0):
        return False
    return True


def fit_ha_nii_sii(
    spec: AGNSpectrum,
    include_broad_ha: bool = True,
    broad_ha_model: str = "double",
    max_nfev: int | None = 3000,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    narrow_kinematics: dict[str, float] | None = None,
    freeze_narrow_kinematics: bool = False,
) -> EmissionFitResult:
    """Fit H-alpha + [N II] + [S II] with a double broad H-alpha default."""

    return _fit_ha_nii_sii_hardcoded(
        spec,
        include_broad_ha=include_broad_ha,
        broad_ha_model=broad_ha_model,
        max_nfev=max_nfev,
        narrow_sigma_bounds=narrow_sigma_bounds,
        narrow_kinematics=narrow_kinematics,
        freeze_narrow_kinematics=freeze_narrow_kinematics,
        fit_info_extra={
            "ha_narrow_kinematics_source": str((narrow_kinematics or {}).get("source", "")),
            "ha_narrow_kinematics_fixed": bool(freeze_narrow_kinematics),
        },
    )


def fit_ha_nii_sii_bic(
    spec: AGNSpectrum,
    bic_threshold: float = 10.0,
    broad_ha_model: str = "double",
    max_nfev: int | None = 3000,
    narrow_sigma_bounds: tuple[float, float] = (50.0, 500.0),
    narrow_kinematics: dict[str, float] | None = None,
    freeze_narrow_kinematics: bool = False,
) -> EmissionFitResult:
    """Fit H-alpha + [N II] + [S II] and include broad H-alpha only by BIC."""

    narrow = fit_ha_nii_sii(
        spec,
        include_broad_ha=False,
        broad_ha_model=broad_ha_model,
        max_nfev=max_nfev,
        narrow_sigma_bounds=narrow_sigma_bounds,
        narrow_kinematics=narrow_kinematics,
        freeze_narrow_kinematics=freeze_narrow_kinematics,
    )
    narrow.fit_info.update(
        {
            "ha_fit_strategy": "narrow_then_broad",
            "ha_fit_stage": "narrow_first_pass",
            "ha_narrow_kinematics_source": str((narrow_kinematics or {}).get("source", "")),
            "ha_narrow_kinematics_fixed": bool(freeze_narrow_kinematics),
        }
    )
    window_spec = spec.select_window(*HA_NII_SII_WINDOW).mask_bad_pixels()

    def _staged_broad_candidate(model: str) -> EmissionFitResult:
        initial = _initial_ha_nii_sii_parameters_from_narrow(
            window_spec.wave,
            window_spec.flux,
            narrow,
            model,
            narrow_sigma_bounds=narrow_sigma_bounds,
            freeze_narrow_kinematics=True,
        )
        return _fit_ha_nii_sii_hardcoded(
            spec,
            include_broad_ha=True,
            broad_ha_model=model,
            max_nfev=max_nfev,
            narrow_sigma_bounds=narrow_sigma_bounds,
            initial_params=initial,
            fit_info_extra={
                "ha_fit_strategy": "narrow_then_broad",
                "ha_fit_stage": "broad_after_narrow",
                "ha_broad_initialized_from_narrow": True,
                "ha_narrow_kinematics_fixed_in_broad": True,
                "ha_narrow_kinematics_source": str((narrow_kinematics or {}).get("source", "")),
                "ha_narrow_kinematics_fixed": bool(freeze_narrow_kinematics),
            },
        )

    broad = _staged_broad_candidate(broad_ha_model)
    if broad_ha_model == "double":
        single = _staged_broad_candidate("single")
        return _select_ha_bic_with_single_fallback(
            narrow_result=narrow,
            single_result=single,
            double_result=broad,
            bic_threshold=bic_threshold,
        )
    return select_broad_balmer_by_bic(
        narrow_result=narrow,
        broad_result=broad,
        bic_threshold=bic_threshold,
        broad_line_name="Ha_broad",
    )


def _select_ha_bic_with_single_fallback(
    narrow_result: EmissionFitResult,
    single_result: EmissionFitResult,
    double_result: EmissionFitResult,
    bic_threshold: float,
) -> EmissionFitResult:
    """Select H-alpha model while rejecting pathological broad-boundary fits."""

    bic_narrow = fit_bic(narrow_result)
    candidates = [("single", single_result), ("double", double_result)]
    usable_broad: list[tuple[str, EmissionFitResult, float]] = []
    rejected: list[str] = []
    for label, result in candidates:
        bic = fit_bic(result)
        if not np.isfinite(bic) or not np.isfinite(result.chi2) or not np.isfinite(result.redchi):
            rejected.append(f"{label}:failed")
            continue
        if _ha_broad_result_pathological(result):
            rejected.append(f"{label}:boundary_or_extreme")
            continue
        usable_broad.append((label, result, bic))

    selected = narrow_result
    selected_label = "narrow"
    broad_selected = False
    best_broad_label = ""
    best_broad_bic = float("nan")
    if usable_broad:
        best_broad_label, best_broad, best_broad_bic = min(usable_broad, key=lambda item: item[2])
        delta = bic_narrow - best_broad_bic
        if (np.isfinite(delta) and delta > bic_threshold) or (
            not bool(narrow_result.success) and np.isfinite(best_broad_bic)
        ):
            selected = best_broad
            selected_label = best_broad_label
            broad_selected = True

    selected.fit_info.update(
        {
            "broad_balmer_mode": "bic_single_fallback",
            "broad_line_name": "Ha_broad",
            "broad_balmer_selected": broad_selected,
            "selected_model": selected_label,
            "bic_threshold": float(bic_threshold),
            "bic_narrow": bic_narrow,
            "bic_single": fit_bic(single_result),
            "bic_double": fit_bic(double_result),
            "bic_broad": best_broad_bic,
            "best_usable_broad_model": best_broad_label,
            "rejected_broad_models": rejected,
            "delta_bic_broad": float(bic_narrow - best_broad_bic)
            if np.isfinite(best_broad_bic)
            else float("nan"),
            "narrow_success": bool(narrow_result.success),
            "single_success": bool(single_result.success),
            "double_success": bool(double_result.success),
            "single_optimizer_message": str(single_result.message),
            "double_optimizer_message": str(double_result.message),
            "broad_success": bool(selected.success) if broad_selected else False,
            "broad_optimizer_success": bool(selected.success) if broad_selected else False,
            "narrow_chi2": float(narrow_result.chi2),
            "single_chi2": float(single_result.chi2),
            "double_chi2": float(double_result.chi2),
            "narrow_nvarys": int(narrow_result.nvarys),
            "single_nvarys": int(single_result.nvarys),
            "double_nvarys": int(double_result.nvarys),
        }
    )
    selected.params.update(
        {
            "bic_narrow": bic_narrow,
            "bic_single": fit_bic(single_result),
            "bic_double": fit_bic(double_result),
            "bic_broad": best_broad_bic,
            "delta_bic_broad": selected.fit_info["delta_bic_broad"],
            "broad_balmer_selected": float(broad_selected),
        }
    )
    return selected


def _ha_broad_result_pathological(result: EmissionFitResult) -> bool:
    info = result.fit_info or {}
    params = result.params or {}
    fwhm = _as_float(info.get("Ha_broad_fwhm_combined"))
    if bool(info.get("flag_ha_broad_eats_sii", False)):
        return True
    # Boundary hits are recorded as diagnostics, but broad H-alpha should not be
    # discarded solely for touching a broad-component bound. In high-S/N changing
    # objects this can leave the H-alpha wing unfitted while BIC strongly favors
    # a broad component.
    if np.isfinite(fwhm) and fwhm > 10000.0:
        return True
    for idx in (1, 2):
        velocity = _as_float(params.get(f"ha_broad_{idx}_velocity"))
        sigma = _as_float(params.get(f"ha_broad_{idx}_sigma"))
        if np.isfinite(velocity) and abs(velocity) >= 4900.0:
            return True
        if np.isfinite(sigma) and sigma > 10000.0:
            return True
    return False


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def select_broad_balmer_by_bic(
    narrow_result: EmissionFitResult,
    broad_result: EmissionFitResult,
    bic_threshold: float = 10.0,
    broad_line_name: str = "broad_balmer",
) -> EmissionFitResult:
    """Choose narrow-only or broad-Balmer model from a BIC improvement test.

    Positive delta means the broad model has lower BIC. Broad Balmer is selected
    only when delta_bic_broad > bic_threshold.
    """

    bic_narrow = fit_bic(narrow_result)
    bic_broad = fit_bic(broad_result)
    delta_bic_broad = bic_narrow - bic_broad
    broad_selected = (
        bool(broad_result.success)
        and np.isfinite(delta_bic_broad)
        and delta_bic_broad > bic_threshold
    )
    if not narrow_result.success and broad_result.success:
        broad_selected = True

    selected = broad_result if broad_selected else narrow_result
    selected_label = "broad" if broad_selected else "narrow"
    selected.fit_info.update(
        {
            "broad_balmer_mode": "bic",
            "broad_line_name": broad_line_name,
            "broad_balmer_selected": broad_selected,
            "selected_model": selected_label,
            "bic_threshold": float(bic_threshold),
            "bic_narrow": bic_narrow,
            "bic_broad": bic_broad,
            "delta_bic_broad": float(delta_bic_broad),
            "narrow_success": bool(narrow_result.success),
            "broad_success": bool(broad_result.success),
            "narrow_chi2": float(narrow_result.chi2),
            "broad_chi2": float(broad_result.chi2),
            "narrow_nvarys": int(narrow_result.nvarys),
            "broad_nvarys": int(broad_result.nvarys),
        }
    )
    selected.params.update(
        {
            "bic_narrow": bic_narrow,
            "bic_broad": bic_broad,
            "delta_bic_broad": float(delta_bic_broad),
            "broad_balmer_selected": float(broad_selected),
        }
    )
    return selected


def fit_oii(spec: AGNSpectrum, mode: str = "doublet") -> EmissionFitResult:
    """Fit [O II] using the bundled blend or doublet YAML recipe."""

    if mode not in {"blend", "doublet"}:
        raise ValueError("mode must be 'blend' or 'doublet'.")
    recipe = Recipe.from_yaml(RECIPE_DIR / "oii.yaml", name=mode)
    return RecipeEmissionFitter(recipe).fit(spec)


def monte_carlo_fit(
    spec: AGNSpectrum,
    fit_func: Callable[..., EmissionFitResult],
    n_mc: int = 100,
    random_seed: int | None = None,
    **fit_kwargs: Any,
) -> EmissionFitResult:
    """Estimate line-flux uncertainties by perturbing fluxes within their errors."""

    if n_mc < 0:
        raise ValueError("n_mc must be non-negative.")

    original = fit_func(spec, **fit_kwargs)
    rng = np.random.default_rng(random_seed)
    line_samples: dict[str, list[float]] = {}
    param_samples: dict[str, list[float]] = {}
    fit_info_samples: dict[str, list[float]] = {}
    failure_count = 0

    for _ in range(n_mc):
        perturbed_flux = spec.flux + rng.normal(0.0, spec.err)
        perturbed_spec = AGNSpectrum(
            wave=spec.wave.copy(),
            flux=perturbed_flux,
            err=spec.err.copy(),
            z=spec.z,
            survey=spec.survey,
            object_id=spec.object_id,
            meta=spec.meta.copy(),
        )
        try:
            mc_result = fit_func(perturbed_spec, **fit_kwargs)
        except Exception:
            failure_count += 1
            continue
        if not mc_result.success:
            failure_count += 1
            continue

        for name, value in mc_result.line_flux.items():
            if np.isfinite(value):
                line_samples.setdefault(name, []).append(float(value))
        for name, value in mc_result.params.items():
            if np.isfinite(value):
                param_samples.setdefault(name, []).append(float(value))
        for name, value in (mc_result.fit_info or {}).items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric_value):
                fit_info_samples.setdefault(name, []).append(numeric_value)

    line_summary = _percentile_summary(line_samples)
    param_summary = _percentile_summary(param_samples)
    fit_info_summary = _percentile_summary(fit_info_samples)
    original.line_flux_err = {
        name: 0.5 * (summary["p84"] - summary["p16"])
        for name, summary in line_summary.items()
    }
    original.mc_summary = {
        "line_flux": line_summary,
        "params": param_summary,
        "fit_info": fit_info_summary,
        "n_mc": n_mc,
        "n_success": n_mc - failure_count,
        "failure_count": failure_count,
    }
    return original


def joint_refit_full_spectrum(
    spec: AGNSpectrum,
    stellar_result: Any,
    oii: EmissionFitResult | None = None,
    hb_oiii: EmissionFitResult | None = None,
    ha_nii_sii: EmissionFitResult | None = None,
    agn_powerlaw_slopes: tuple[float, ...] | None = None,
    continuum_mask_width_kms: float = 8000.0,
    line_flux_factor_bounds: tuple[float, float] = (0.2, 5.0),
    line_prior_sigma_frac: float | None = 0.30,
    fit_mask_mode: str = "line_windows",
    line_window_half_width_kms: float = 12000.0,
    max_nfev: int | None = 5000,
) -> FullSpectrumFitResult:
    """Refit stellar scale, power-law continuum, and all emission lines together.

    The stellar template mixture shape is fixed from pPXF and only its scale is
    varied. Power-law coefficients and all selected emission-line parameters are
    optimized simultaneously using the previous window fits as initial guesses.
    """

    clean_spec = spec.mask_bad_pixels()
    stellar_model = np.asarray(stellar_result.stellar_model, dtype=float)
    if len(stellar_model) != len(spec.wave):
        raise ValueError("stellar_result.stellar_model must match spec length.")

    slopes = agn_powerlaw_slopes
    if slopes is None:
        slopes = tuple(
            float(slope)
            for slope in getattr(stellar_result, "diagnostics", {}).get(
                "agn_powerlaw_slopes",
                (-10.0, -8.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
            )
        )

    valid_mask = (
        np.isfinite(spec.wave)
        & np.isfinite(spec.flux)
        & np.isfinite(spec.err)
        & (spec.err > 0)
        & np.isfinite(stellar_model)
    )

    params = Parameters()
    params.add("stellar_scale", value=1.0, min=0.0, max=5.0)
    line_specs: list[dict[str, Any]] = []
    _add_oii_joint_parameters(params, line_specs, oii, line_flux_factor_bounds)
    _add_hb_oiii_joint_parameters(params, line_specs, hb_oiii, line_flux_factor_bounds)
    _add_ha_nii_sii_joint_parameters(
        params,
        line_specs,
        ha_nii_sii,
        line_flux_factor_bounds,
    )
    if len(line_specs) == 0:
        raise ValueError("No emission-line fit results were provided for joint refit.")

    line_fit_mask = _joint_fit_pixel_mask(
        spec.wave,
        line_specs,
        mode=fit_mask_mode,
        half_width_kms=line_window_half_width_kms,
    )
    fit_mask = valid_mask & line_fit_mask
    if np.count_nonzero(fit_mask) < 10:
        raise ValueError("Insufficient valid pixels for joint full-spectrum fit.")

    wave = spec.wave[fit_mask]
    flux = spec.flux[fit_mask]
    err = spec.err[fit_mask]
    stellar_fit = stellar_model[fit_mask]
    basis = _powerlaw_basis(wave, slopes)

    agn_initial = _initial_powerlaw_coefficients(
        wave=wave,
        flux=flux,
        stellar_flux=stellar_fit,
        basis=basis,
        stellar_result=stellar_result,
        fit_mask=fit_mask,
        mask_width_kms=continuum_mask_width_kms,
    )
    for index, coeff in enumerate(agn_initial):
        params.add(f"agn_pl_{index}", value=float(max(coeff, 0.0)), min=0.0)

    fit = minimize(
        _joint_full_spectrum_residual,
        params,
        args=(
            wave,
            flux,
            err,
            stellar_fit,
            basis,
            line_specs,
            line_prior_sigma_frac,
        ),
        max_nfev=max_nfev,
    )
    model_fit = _joint_full_spectrum_model(wave, fit.params, stellar_fit, basis, line_specs)
    stellar_component = fit.params["stellar_scale"].value * stellar_fit
    agn_component = _joint_powerlaw_component(fit.params, basis)
    emission_component = model_fit - stellar_component - agn_component
    residual_fit = flux - model_fit
    chi2 = float(np.sum((residual_fit / err) ** 2))

    model_full = np.full_like(spec.flux, np.nan, dtype=float)
    stellar_full = np.full_like(spec.flux, np.nan, dtype=float)
    agn_full = np.full_like(spec.flux, np.nan, dtype=float)
    emission_full = np.full_like(spec.flux, np.nan, dtype=float)
    residual_full = np.full_like(spec.flux, np.nan, dtype=float)
    model_full[fit_mask] = model_fit
    stellar_full[fit_mask] = stellar_component
    agn_full[fit_mask] = agn_component
    emission_full[fit_mask] = emission_component
    residual_full[fit_mask] = residual_fit

    return FullSpectrumFitResult(
        success=bool(fit.success),
        message=str(fit.message),
        params=_param_dict(fit.params),
        line_flux=_joint_line_flux_dict(fit.params, line_specs),
        model_flux=model_full,
        stellar_flux=stellar_full,
        agn_flux=agn_full,
        emission_flux=emission_full,
        residual_flux=residual_full,
        chi2=chi2,
        redchi=float(fit.redchi),
        ndata=int(fit.ndata),
        nvarys=int(fit.nvarys),
        fit_info={
            "agn_powerlaw_slopes": tuple(float(slope) for slope in slopes),
            "n_lines": len(line_specs),
            "continuum_mask_width_kms": float(continuum_mask_width_kms),
            "line_flux_factor_bounds": tuple(float(v) for v in line_flux_factor_bounds),
            "line_prior_sigma_frac": (
                None if line_prior_sigma_frac is None else float(line_prior_sigma_frac)
            ),
            "fit_mask_mode": fit_mask_mode,
            "line_window_half_width_kms": float(line_window_half_width_kms),
            "n_valid_pixels": int(np.count_nonzero(valid_mask)),
            "n_fit_pixels": int(np.count_nonzero(fit_mask)),
            "line_flux_boundary_hits": _joint_line_boundary_hits(fit.params, line_specs),
            "n_line_flux_boundary_hits": len(
                _joint_line_boundary_hits(fit.params, line_specs)
            ),
            "fit_type": "joint_stellar_powerlaw_emission",
        },
    )


def _percentile_summary(samples: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        if len(values) == 0:
            continue
        p16, median, p84 = np.percentile(values, [16.0, 50.0, 84.0])
        summary[name] = {
            "median": float(median),
            "p16": float(p16),
            "p84": float(p84),
        }
    return summary


def _powerlaw_basis(wave: np.ndarray, slopes: tuple[float, ...]) -> np.ndarray:
    if len(slopes) == 0:
        return np.empty((len(wave), 0), dtype=float)
    pivot = float(np.nanmedian(wave))
    if not np.isfinite(pivot) or pivot <= 0:
        pivot = 5100.0
    columns = []
    for slope in slopes:
        column = (wave / pivot) ** float(slope)
        norm = float(np.nanmedian(column[np.isfinite(column)]))
        if np.isfinite(norm) and norm != 0:
            column = column / norm
        columns.append(column)
    return np.column_stack(columns)


def _initial_powerlaw_coefficients(
    wave: np.ndarray,
    flux: np.ndarray,
    stellar_flux: np.ndarray,
    basis: np.ndarray,
    stellar_result: Any,
    fit_mask: np.ndarray,
    mask_width_kms: float,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros(0, dtype=float)
    target_full = getattr(stellar_result, "agn_model", None)
    if target_full is not None and len(target_full) == len(fit_mask):
        target = np.asarray(target_full, dtype=float)[fit_mask]
    else:
        target = flux - stellar_flux
    good = (
        np.isfinite(wave)
        & np.all(np.isfinite(basis), axis=1)
        & np.isfinite(target)
        & _continuum_mask(wave, mask_width_kms)
    )
    if np.count_nonzero(good) <= basis.shape[1]:
        good = np.isfinite(wave) & np.all(np.isfinite(basis), axis=1) & np.isfinite(target)
    if np.count_nonzero(good) <= basis.shape[1]:
        return np.zeros(basis.shape[1], dtype=float)
    try:
        from scipy.optimize import nnls

        coeff, _ = nnls(basis[good], np.clip(target[good], 0.0, None))
        return coeff
    except Exception:
        coeff, *_ = np.linalg.lstsq(basis[good], np.clip(target[good], 0.0, None), rcond=None)
        return np.clip(coeff, 0.0, None)


def _continuum_mask(wave: np.ndarray, mask_width_kms: float) -> np.ndarray:
    centers = (
        OII_3726,
        OII_BLEND,
        OII_3729,
        HBETA,
        OIII_4959,
        OIII_5007,
        NII_6548,
        HALPHA,
        NII_6584,
        SII_6716,
        SII_6731,
    )
    good = np.ones_like(wave, dtype=bool)
    for center in centers:
        half_width = center * mask_width_kms / C_KMS
        good &= np.abs(wave - center) > half_width
    return good


def _joint_full_spectrum_residual(
    params: Parameters,
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    stellar_flux: np.ndarray,
    basis: np.ndarray,
    line_specs: list[dict[str, Any]],
    line_prior_sigma_frac: float | None,
) -> np.ndarray:
    data_residual = (
        flux
        - _joint_full_spectrum_model(wave, params, stellar_flux, basis, line_specs)
    ) / err
    prior_residual = _joint_line_prior_residual(
        params,
        line_specs,
        line_prior_sigma_frac,
    )
    if prior_residual.size == 0:
        return data_residual
    return np.concatenate([data_residual, prior_residual])


def _joint_full_spectrum_model(
    wave: np.ndarray,
    params: Parameters,
    stellar_flux: np.ndarray,
    basis: np.ndarray,
    line_specs: list[dict[str, Any]],
) -> np.ndarray:
    model = params["stellar_scale"].value * stellar_flux
    model = model + _joint_powerlaw_component(params, basis)
    for line in line_specs:
        flux = params[line["flux_param"]].value
        model += _gaussian_integrated(
            wave,
            line["restwave"],
            flux,
            params[f"{line['group']}_velocity"].value,
            params[f"{line['group']}_sigma"].value,
        )
    return model


def _joint_powerlaw_component(params: Parameters, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros(basis.shape[0], dtype=float)
    coeffs = np.array(
        [params[f"agn_pl_{index}"].value for index in range(basis.shape[1])],
        dtype=float,
    )
    return basis @ coeffs


def _joint_fit_pixel_mask(
    wave: np.ndarray,
    line_specs: list[dict[str, Any]],
    mode: str,
    half_width_kms: float,
) -> np.ndarray:
    if mode == "full":
        return np.ones_like(wave, dtype=bool)
    if mode != "line_windows":
        raise ValueError("fit_mask_mode must be 'line_windows' or 'full'.")
    mask = np.zeros_like(wave, dtype=bool)
    for line in line_specs:
        restwave = float(line["restwave"])
        half_width = restwave * half_width_kms / C_KMS
        mask |= np.abs(wave - restwave) <= half_width
    return mask


def _joint_line_prior_residual(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    line_prior_sigma_frac: float | None,
) -> np.ndarray:
    if line_prior_sigma_frac is None or line_prior_sigma_frac <= 0:
        return np.zeros(0, dtype=float)
    residuals: list[float] = []
    for line in line_specs:
        if line.get("tied", False):
            continue
        initial_flux = float(line.get("initial_flux", np.nan))
        if not np.isfinite(initial_flux) or initial_flux <= 0:
            continue
        sigma = max(abs(initial_flux) * line_prior_sigma_frac, 1.0e-12)
        residuals.append((params[line["flux_param"]].value - initial_flux) / sigma)
    return np.asarray(residuals, dtype=float)


def _joint_line_boundary_hits(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    tolerance_fraction: float = 0.02,
) -> dict[str, str]:
    hits: dict[str, str] = {}
    for line in line_specs:
        if line.get("tied", False):
            continue
        param = params[line["flux_param"]]
        value = float(param.value)
        lower = float(param.min)
        upper = float(param.max)
        if not np.isfinite(value) or not np.isfinite(lower) or not np.isfinite(upper):
            continue
        width = upper - lower
        if width <= 0:
            continue
        if value <= lower + tolerance_fraction * width:
            hits[line["name"]] = "lower"
        elif value >= upper - tolerance_fraction * width:
            hits[line["name"]] = "upper"
    return hits


def _add_group_params(
    params: Parameters,
    group: str,
    source: dict[str, float],
    source_prefix: str = "",
    broad: bool = False,
) -> None:
    velocity_key = f"{source_prefix}velocity" if source_prefix else "narrow_velocity"
    sigma_key = f"{source_prefix}sigma" if source_prefix else "narrow_sigma"
    if broad:
        params.add(
            f"{group}_velocity",
            value=float(source.get(velocity_key, 0.0)),
            min=BROAD_BALMER_VELOCITY_BOUNDS[0],
            max=BROAD_BALMER_VELOCITY_BOUNDS[1],
        )
        params.add(
            f"{group}_sigma",
            value=float(source.get(sigma_key, 1500.0)),
            min=BROAD_BALMER_SIGMA_BOUNDS[0],
            max=BROAD_BALMER_SIGMA_BOUNDS[1],
        )
    else:
        params.add(
            f"{group}_velocity",
            value=float(source.get(velocity_key, 0.0)),
            min=-800.0,
            max=800.0,
        )
        params.add(
            f"{group}_sigma",
            value=float(source.get(sigma_key, 200.0)),
            min=50.0,
            max=1000.0,
        )


def _add_line_param(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    name: str,
    restwave: float,
    group: str,
    initial_flux: float,
    factor_bounds: tuple[float, float],
    expr: str | None = None,
) -> None:
    flux_param = f"{name}_flux"
    if expr is None:
        value = max(float(initial_flux), 1.0e-12)
        if np.isfinite(initial_flux) and initial_flux > 1.0e-8:
            params.add(
                flux_param,
                value=value,
                min=max(1.0e-12, float(initial_flux) * factor_bounds[0]),
                max=max(float(initial_flux) * factor_bounds[1], value * 1.01),
            )
        else:
            params.add(flux_param, value=value, min=0.0)
    else:
        params.add(flux_param, expr=expr)
    line_specs.append(
        {
            "name": name,
            "restwave": float(restwave),
            "group": group,
            "flux_param": flux_param,
            "initial_flux": float(initial_flux),
            "tied": expr is not None,
        }
    )


def _add_oii_joint_parameters(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    result: EmissionFitResult | None,
    factor_bounds: tuple[float, float],
) -> None:
    if result is None or not result.success:
        return
    p = result.params
    _add_group_params(params, "oii", p, broad=False)
    if "OII_3727_blend" in result.line_flux:
        _add_line_param(
            params,
            line_specs,
            "OII_3727_blend",
            OII_BLEND,
            "oii",
            result.line_flux["OII_3727_blend"],
            factor_bounds,
        )
    elif {"OII_3726", "OII_3729"} <= result.line_flux.keys():
        _add_line_param(
            params,
            line_specs,
            "OII_3726",
            OII_3726,
            "oii",
            result.line_flux["OII_3726"],
            factor_bounds,
        )
        _add_line_param(
            params,
            line_specs,
            "OII_3729",
            OII_3729,
            "oii",
            result.line_flux["OII_3729"],
            factor_bounds,
        )


def _add_hb_oiii_joint_parameters(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    result: EmissionFitResult | None,
    factor_bounds: tuple[float, float],
) -> None:
    if result is None or not result.success:
        return
    p = result.params
    _add_group_params(params, "hb_narrow", p, broad=False)
    _add_line_param(
        params,
        line_specs,
        "Hb_narrow",
        HBETA,
        "hb_narrow",
        result.line_flux.get("Hb_narrow", 0.0),
        factor_bounds,
    )
    oiii_4959 = result.line_flux.get(
        "OIII_4959",
        result.line_flux.get("OIII_5007", 0.0) / OIII_RATIO,
    )
    _add_line_param(
        params,
        line_specs,
        "OIII_4959",
        OIII_4959,
        "hb_narrow",
        oiii_4959,
        factor_bounds,
    )
    _add_line_param(
        params,
        line_specs,
        "OIII_5007",
        OIII_5007,
        "hb_narrow",
        result.line_flux.get("OIII_5007", OIII_RATIO * oiii_4959),
        factor_bounds,
        expr=f"{OIII_RATIO} * OIII_4959_flux",
    )
    for index in (1, 2):
        name = f"Hb_broad_{index}"
        if name not in result.line_flux:
            continue
        group = f"hb_broad_{index}"
        params.add(
            f"{group}_velocity",
            value=float(p.get(f"hb_broad_{index}_velocity", 0.0)),
            min=_broad_balmer_component_bounds("hb", index)[0],
            max=_broad_balmer_component_bounds("hb", index)[1],
        )
        params.add(
            f"{group}_sigma",
            value=float(p.get(f"hb_broad_{index}_sigma", 1500.0 if index == 1 else 4000.0)),
            min=_broad_balmer_component_bounds("hb", index)[2],
            max=_broad_balmer_component_bounds("hb", index)[3],
        )
        _add_line_param(
            params,
            line_specs,
            name,
            HBETA,
            group,
            result.line_flux[name],
            factor_bounds,
        )
    if "Hb_broad" in result.line_flux and "Hb_broad_1" not in result.line_flux:
        _add_group_params(params, "hb_broad", p, source_prefix="broad_", broad=True)
        _add_line_param(
            params,
            line_specs,
            "Hb_broad",
            HBETA,
            "hb_broad",
            result.line_flux["Hb_broad"],
            factor_bounds,
        )


def _add_ha_nii_sii_joint_parameters(
    params: Parameters,
    line_specs: list[dict[str, Any]],
    result: EmissionFitResult | None,
    factor_bounds: tuple[float, float],
) -> None:
    if result is None or not result.success:
        return
    p = result.params
    _add_group_params(params, "ha_narrow", p, broad=False)
    _add_line_param(
        params,
        line_specs,
        "Ha_narrow",
        HALPHA,
        "ha_narrow",
        result.line_flux.get("Ha_narrow", 0.0),
        factor_bounds,
    )
    nii_6548 = result.line_flux.get(
        "NII_6548",
        result.line_flux.get("NII_6584", 0.0) / NII_RATIO,
    )
    _add_line_param(
        params,
        line_specs,
        "NII_6548",
        NII_6548,
        "ha_narrow",
        nii_6548,
        factor_bounds,
    )
    _add_line_param(
        params,
        line_specs,
        "NII_6584",
        NII_6584,
        "ha_narrow",
        result.line_flux.get("NII_6584", NII_RATIO * nii_6548),
        factor_bounds,
        expr=f"{NII_RATIO} * NII_6548_flux",
    )
    _add_line_param(
        params,
        line_specs,
        "SII_6716",
        SII_6716,
        "ha_narrow",
        result.line_flux.get("SII_6716", 0.0),
        factor_bounds,
    )
    _add_line_param(
        params,
        line_specs,
        "SII_6731",
        SII_6731,
        "ha_narrow",
        result.line_flux.get("SII_6731", 0.0),
        factor_bounds,
    )
    for index in (1, 2):
        name = f"Ha_broad_{index}"
        if name not in result.line_flux:
            continue
        group = f"ha_broad_{index}"
        vmin, vmax, smin, smax = _broad_balmer_component_bounds("ha", index)
        params.add(
            f"{group}_velocity",
            value=float(p.get(f"ha_broad_{index}_velocity", 0.0)),
            min=vmin,
            max=vmax,
        )
        params.add(
            f"{group}_sigma",
            value=float(p.get(f"ha_broad_{index}_sigma", 1500.0 if index == 1 else 4000.0)),
            min=smin,
            max=smax,
        )
        _add_line_param(
            params,
            line_specs,
            name,
            HALPHA,
            group,
            result.line_flux[name],
            factor_bounds,
        )
    if "Ha_broad" in result.line_flux and "Ha_broad_1" not in result.line_flux:
        _add_group_params(params, "ha_broad", p, source_prefix="broad_", broad=True)
        _add_line_param(
            params,
            line_specs,
            "Ha_broad",
            HALPHA,
            "ha_broad",
            result.line_flux["Ha_broad"],
            factor_bounds,
        )


def _joint_line_flux_dict(
    params: Parameters,
    line_specs: list[dict[str, Any]],
) -> dict[str, float]:
    flux = {line["name"]: float(params[line["flux_param"]].value) for line in line_specs}
    if "Hb_broad_1" in flux:
        total = flux.get("Hb_broad_1", 0.0) + flux.get("Hb_broad_2", 0.0)
        flux["Hb_broad"] = total
        flux["Hb_broad_total"] = total
        flux["Hb_broad_flux_total"] = total
    if "Ha_broad_1" in flux:
        total = flux.get("Ha_broad_1", 0.0) + flux.get("Ha_broad_2", 0.0)
        flux["Ha_broad"] = total
        flux["Ha_broad_total"] = total
        flux["Ha_broad_flux_total"] = total
    return flux


from echospec.blue_balmer import fit_blue_balmer_extended, fit_blue_balmer_window, should_enable_blue_balmer_mode
from echospec.mgii import fit_mgii_uv, fit_mgii_uvfeii, prefit_uv_feii


__all__ = [
    "EmissionFitResult",
    "FullSpectrumFitResult",
    "compute_feii_region_diagnostics",
    "fit_blue_balmer_extended",
    "fit_blue_balmer_window",
    "fit_bic",
    "fit_hb_oiii",
    "fit_hb_oiii_bic",
    "fit_ha_nii_sii",
    "fit_ha_nii_sii_bic",
    "fit_mgii_uv",
    "fit_mgii_uvfeii",
    "fit_oii",
    "joint_refit_full_spectrum",
    "measure_combined_profile_properties",
    "monte_carlo_fit",
    "prefit_feii_grid",
    "prefit_uv_feii",
    "select_broad_balmer_by_bic",
    "should_enable_blue_balmer_mode",
]
