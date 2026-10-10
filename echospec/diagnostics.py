"""Diagnostic-line and line-ratio analysis interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


def safe_log10_ratio(
    numerator: ArrayLike,
    denominator: ArrayLike,
    numerator_err: ArrayLike | None = None,
    denominator_err: ArrayLike | None = None,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """Return log10(numerator / denominator), optionally with propagated error."""

    num, den = np.broadcast_arrays(
        np.asarray(numerator, dtype=float),
        np.asarray(denominator, dtype=float),
    )
    valid = (num > 0) & (den > 0)
    log_ratio = np.full(num.shape, np.nan, dtype=float)
    log_ratio[valid] = np.log10(num[valid] / den[valid])

    if numerator_err is None and denominator_err is None:
        return _as_scalar_if_scalar(log_ratio)
    if numerator_err is None or denominator_err is None:
        raise ValueError("Both numerator_err and denominator_err must be provided.")

    num_err, den_err, _ = np.broadcast_arrays(
        np.asarray(numerator_err, dtype=float),
        np.asarray(denominator_err, dtype=float),
        num,
    )
    sigma_log = np.full_like(log_ratio, np.nan, dtype=float)
    sigma_log[valid] = (1.0 / np.log(10.0)) * np.sqrt(
        (num_err[valid] / num[valid]) ** 2 + (den_err[valid] / den[valid]) ** 2
    )
    return _as_scalar_if_scalar(log_ratio), _as_scalar_if_scalar(sigma_log)


def compute_bpt_coordinates(results: dict[str, Any]) -> tuple[float, float]:
    """Return N II BPT coordinates: log([N II]/H-alpha), log([O III]/H-beta)."""

    line_flux = _line_flux(results)
    x = safe_log10_ratio(line_flux["NII_6584"], line_flux["Ha_narrow"])
    y = safe_log10_ratio(line_flux["OIII_5007"], line_flux["Hb_narrow"])
    return float(x), float(y)


def compute_sii_bpt_coordinates(results: dict[str, Any]) -> tuple[float, float]:
    """Return S II BPT coordinates: log([S II]/H-alpha), log([O III]/H-beta)."""

    line_flux = _line_flux(results)
    sii_6717 = _get_first_available(line_flux, ("SII_6717", "SII_6716"))
    sii_6731 = line_flux["SII_6731"]
    x = safe_log10_ratio(sii_6717 + sii_6731, line_flux["Ha_narrow"])
    y = safe_log10_ratio(line_flux["OIII_5007"], line_flux["Hb_narrow"])
    return float(x), float(y)


def compute_o32(results: dict[str, Any]) -> float:
    """Return log10([O III]5007 / [O II]3727)."""

    line_flux = _line_flux(results)
    if "OII_3727_blend" in line_flux:
        oii_flux = line_flux["OII_3727_blend"]
    elif "OII_3727" in line_flux:
        oii_flux = line_flux["OII_3727"]
    else:
        oii_flux = line_flux["OII_3726"] + line_flux["OII_3729"]
    return float(safe_log10_ratio(line_flux["OIII_5007"], oii_flux))


def assess_fit_quality(result: Any) -> dict[str, Any]:
    """Return quality flags for one emission-line fit result."""

    line_snr = _line_snr(result)
    low_snr_lines = [
        name for name, snr in line_snr.items() if np.isfinite(snr) and snr < 3.0
    ]
    boundary_sigma = _boundary_params(result, suffix="_sigma")
    boundary_velocity = _boundary_params(result, suffix="_velocity")
    delta_bic_broad = _get_delta_bic_broad(result)

    fit_info = getattr(result, "fit_info", {}) or {}
    flags = {
        "flag_fit_failed": not bool(getattr(result, "success", False)),
        "flag_low_snr_line": len(low_snr_lines) > 0,
        "flag_boundary_sigma": len(boundary_sigma) > 0,
        "flag_boundary_velocity": len(boundary_velocity) > 0,
        "flag_broad_balmer_required": (
            delta_bic_broad is not None and delta_bic_broad > 10.0
        ),
        "low_snr_lines": low_snr_lines,
        "line_snr": line_snr,
        "boundary_sigma_params": boundary_sigma,
        "boundary_velocity_params": boundary_velocity,
        "delta_bic_broad": delta_bic_broad,
    }
    for key in (
        "flag_broad_hb_weak",
        "flag_broad_hb_boundary",
        "flag_broad_hb_unphysical",
        "flag_broad_hb_suspicious",
        "flag_broad_hb_flux_suspicious",
        "flag_broad_hb_continuum_degeneracy",
        "flag_broad_balmer_weak",
        "flag_broad_balmer_secure",
        "flag_broad_balmer_tentative",
        "flag_broad_balmer_unphysical",
        "flag_broad_balmer_boundary",
        "flag_feii_missing_template",
        "flag_feii_weak",
        "flag_feii_boundary",
        "flag_feii_affects_narrow_hb",
    ):
        if key in fit_info:
            flags[key] = fit_info[key]
    return flags


def assess_epoch_comparison(comparison: Any) -> dict[str, Any]:
    """Return quality flags for a repeat-epoch comparison."""

    calibration_lines = _similar_change_lines(comparison)
    differential_oiii = _has_differential_oiii_response(comparison)

    return {
        "flag_possible_calibration_offset": len(calibration_lines) >= 4,
        "flag_differential_oiii_response": differential_oiii,
        "calibration_offset_lines": calibration_lines,
    }


def _line_flux(results: dict[str, Any]) -> dict[str, float]:
    if "line_flux" in results and isinstance(results["line_flux"], dict):
        return results["line_flux"]
    return results


def _line_snr(result: Any) -> dict[str, float]:
    line_flux = getattr(result, "line_flux", {}) or {}
    line_flux_err = getattr(result, "line_flux_err", {}) or {}
    snr: dict[str, float] = {}
    for name, flux in line_flux.items():
        err = line_flux_err.get(name)
        if err is None or err <= 0:
            continue
        snr[name] = float(abs(flux) / err)
    return snr


def _boundary_params(result: Any, suffix: str) -> list[str]:
    params = getattr(result, "params", {}) or {}
    boundary_params: list[str] = []
    for name, value in params.items():
        if not name.endswith(suffix):
            continue
        bounds = _bounds_for_param(name, suffix)
        if bounds is None:
            continue
        if _near_boundary(float(value), bounds[0], bounds[1]):
            boundary_params.append(name)
    return boundary_params


def _bounds_for_param(name: str, suffix: str) -> tuple[float, float] | None:
    if suffix == "_sigma":
        if "hb_broad_1" in name:
            return (800.0, 6000.0)
        if "hb_broad_2" in name:
            return (1500.0, 12000.0)
        if "ha_broad_1" in name:
            return (800.0, 6000.0)
        if "ha_broad_2" in name:
            return (1500.0, 12000.0)
        if "broad" in name:
            return (0.0, 30000.0)
        if "feii" in name:
            return (0.0, 30000.0)
        if "wing" in name:
            return (200.0, 2000.0)
        return (50.0, 1000.0)
    if suffix == "_velocity":
        if "hb_broad_1" in name:
            return (-3000.0, 3000.0)
        if "hb_broad_2" in name:
            return (-5000.0, 5000.0)
        if "ha_broad" in name:
            return (-5000.0, 5000.0)
        if "broad" in name:
            return (-10000.0, 10000.0)
        if "feii" in name:
            return (-10000.0, 10000.0)
        if "wing" in name:
            return (-1500.0, 1000.0)
        return (-800.0, 800.0)
    return None


def _near_boundary(value: float, lower: float, upper: float) -> bool:
    if not np.isfinite(value):
        return False
    margin = 0.05 * (upper - lower)
    return value <= lower + margin or value >= upper - margin


def _get_delta_bic_broad(result: Any) -> float | None:
    for source in (
        getattr(result, "fit_info", {}) or {},
        getattr(result, "params", {}) or {},
        getattr(result, "mc_summary", {}) or {},
    ):
        for key in ("delta_bic_broad", "bic_improvement_broad"):
            if key in source:
                return float(source[key])
    return None


def _similar_change_lines(comparison: Any) -> list[str]:
    flux_ratio = getattr(comparison, "flux_ratio", {}) or {}
    strong_lines = (
        "OII_3727_blend",
        "OII_3727",
        "Hb_narrow",
        "OIII_5007",
        "Ha_narrow",
        "NII_6584",
        "SII_6716",
        "SII_6717",
        "SII_6731",
    )
    lines = [
        line
        for line in strong_lines
        if line in flux_ratio and np.isfinite(flux_ratio[line]) and flux_ratio[line] > 0
    ]
    if len(lines) < 4:
        return []
    log_ratios = np.array([np.log10(flux_ratio[line]) for line in lines])
    median_offset = float(np.nanmedian(log_ratios))
    scatter = float(1.4826 * np.nanmedian(np.abs(log_ratios - median_offset)))
    if abs(10.0**median_offset - 1.0) < 0.1:
        return []
    if scatter <= 0.08:
        return lines
    return []


def _has_differential_oiii_response(comparison: Any) -> bool:
    significance = getattr(comparison, "change_significance", {}) or {}
    oiii_sig = significance.get("OIII_5007")
    oii_sig = _first_available_optional(
        significance,
        ("OII_3727_blend", "OII_3727", "OII_3726", "OII_3729"),
    )
    if oiii_sig is not None and oii_sig is not None:
        return abs(oiii_sig) >= 3.0 and abs(oii_sig) < 3.0

    flux_ratio = getattr(comparison, "flux_ratio", {}) or {}
    oiii_ratio = flux_ratio.get("OIII_5007")
    oii_ratio = _first_available_optional(
        flux_ratio,
        ("OII_3727_blend", "OII_3727", "OII_3726", "OII_3729"),
    )
    if oiii_ratio is None or oii_ratio is None:
        return False
    return abs(oiii_ratio - 1.0) >= 0.2 and abs(oii_ratio - 1.0) < 0.1


def _first_available_optional(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def _get_first_available(values: dict[str, float], names: tuple[str, ...]) -> float:
    for name in names:
        if name in values:
            return values[name]
    raise KeyError(names[0])


def _as_scalar_if_scalar(value: np.ndarray) -> float | np.ndarray:
    if value.ndim == 0:
        return float(value)
    return value


__all__ = [
    "safe_log10_ratio",
    "compute_bpt_coordinates",
    "compute_sii_bpt_coordinates",
    "compute_o32",
    "assess_fit_quality",
    "assess_epoch_comparison",
]
