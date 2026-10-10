"""Repeat-epoch comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from echospec.diagnostics import safe_log10_ratio

if TYPE_CHECKING:
    from echospec.fitting import EmissionFitResult
else:
    EmissionFitResult = Any


@dataclass
class EpochFitSet:
    """Emission-line fit results for one observed epoch."""

    object_id: str
    survey: str
    oii: EmissionFitResult | None
    hb_oiii: EmissionFitResult | None
    ha_nii_sii: EmissionFitResult | None
    blue_balmer: EmissionFitResult | None = None
    final_refit: Any | None = None
    stellar: Any | None = None
    diagnostics: dict[str, Any] | None = None

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly detailed fit summary."""

        return {
            "object_id": self.object_id,
            "survey": self.survey,
            "diagnostics": self.diagnostics or {},
            "stellar": _result_to_dict(self.stellar, include_arrays=include_arrays),
            "oii": _result_to_dict(self.oii, include_arrays=include_arrays),
            "hb_oiii": _result_to_dict(self.hb_oiii, include_arrays=include_arrays),
            "ha_nii_sii": _result_to_dict(self.ha_nii_sii, include_arrays=include_arrays),
            "blue_balmer": _result_to_dict(self.blue_balmer, include_arrays=include_arrays),
            "final_refit": _result_to_dict(self.final_refit, include_arrays=include_arrays),
        }


@dataclass
class EpochComparisonResult:
    """Derived differential quantities between two fitted epochs."""

    object_id: str
    flux_ratio: dict[str, float]
    fractional_change: dict[str, float]
    change_significance: dict[str, float]
    log_ratio_sdss: dict[str, float]
    log_ratio_desi: dict[str, float]
    delta_log_ratio: dict[str, float]
    bpt_sdss: tuple[float, float]
    bpt_desi: tuple[float, float]
    bpt_vector: tuple[float, float]
    differential_oiii_oii_index: float | None


def compare_epochs(
    epoch1: EpochFitSet,
    epoch2: EpochFitSet,
    labels: tuple[str, str] = ("SDSS", "DESI"),
) -> EpochComparisonResult:
    """Compare line fluxes and diagnostic ratios between two fitted epochs."""

    del labels
    flux1 = _collect_line_flux(epoch1)
    flux2 = _collect_line_flux(epoch2)
    err1 = _collect_line_flux_err(epoch1)
    err2 = _collect_line_flux_err(epoch2)

    flux_ratio: dict[str, float] = {}
    fractional_change: dict[str, float] = {}
    change_significance: dict[str, float] = {}
    for line in sorted(flux1.keys() & flux2.keys()):
        f1 = flux1[line]
        f2 = flux2[line]
        flux_ratio[line] = _safe_divide(f2, f1)
        fractional_change[line] = _safe_divide(f2 - f1, f1)
        if line in err1 and line in err2:
            denom = float(np.sqrt(err1[line] ** 2 + err2[line] ** 2))
            if denom > 0:
                change_significance[line] = (f2 - f1) / denom

    log_ratio_sdss = _diagnostic_log_ratios(flux1)
    log_ratio_desi = _diagnostic_log_ratios(flux2)
    delta_log_ratio = {
        key: log_ratio_desi[key] - log_ratio_sdss[key]
        for key in sorted(log_ratio_sdss.keys() & log_ratio_desi.keys())
    }

    bpt_sdss = _bpt_coordinates(flux1)
    bpt_desi = _bpt_coordinates(flux2)
    bpt_vector = (bpt_desi[0] - bpt_sdss[0], bpt_desi[1] - bpt_sdss[1])

    return EpochComparisonResult(
        object_id=epoch1.object_id,
        flux_ratio=flux_ratio,
        fractional_change=fractional_change,
        change_significance=change_significance,
        log_ratio_sdss=log_ratio_sdss,
        log_ratio_desi=log_ratio_desi,
        delta_log_ratio=delta_log_ratio,
        bpt_sdss=bpt_sdss,
        bpt_desi=bpt_desi,
        bpt_vector=bpt_vector,
        differential_oiii_oii_index=_differential_oiii_oii_index(flux1, flux2),
    )


def _collect_line_flux(epoch: EpochFitSet) -> dict[str, float]:
    line_flux: dict[str, float] = {}
    for fit in (epoch.oii, epoch.hb_oiii, epoch.ha_nii_sii, epoch.blue_balmer):
        if fit is not None:
            line_flux.update({name: float(value) for name, value in fit.line_flux.items()})
    return line_flux


def _collect_line_flux_err(epoch: EpochFitSet) -> dict[str, float]:
    line_flux_err: dict[str, float] = {}
    for fit in (epoch.oii, epoch.hb_oiii, epoch.ha_nii_sii, epoch.blue_balmer):
        if fit is not None:
            line_flux_err.update(
                {name: float(value) for name, value in fit.line_flux_err.items()}
            )
    return line_flux_err


def _diagnostic_log_ratios(line_flux: dict[str, float]) -> dict[str, float]:
    log_ratios: dict[str, float] = {}
    if {"OIII_5007", "Hb_narrow"} <= line_flux.keys():
        log_ratios["delta_log_OIII_Hb"] = float(
            safe_log10_ratio(line_flux["OIII_5007"], line_flux["Hb_narrow"])
        )
    if {"NII_6584", "Ha_narrow"} <= line_flux.keys():
        log_ratios["delta_log_NII_Ha"] = float(
            safe_log10_ratio(line_flux["NII_6584"], line_flux["Ha_narrow"])
        )
    if {"SII_6731", "Ha_narrow"} <= line_flux.keys():
        sii_6717 = _get_sii_6717(line_flux)
        if sii_6717 is not None:
            log_ratios["delta_log_SII_Ha"] = float(
                safe_log10_ratio(sii_6717 + line_flux["SII_6731"], line_flux["Ha_narrow"])
            )
    oii_flux = _get_oii_flux(line_flux)
    if "OIII_5007" in line_flux and oii_flux is not None:
        log_ratios["delta_log_OIII_OII"] = float(
            safe_log10_ratio(line_flux["OIII_5007"], oii_flux)
        )
    return log_ratios


def _bpt_coordinates(line_flux: dict[str, float]) -> tuple[float, float]:
    if not {"NII_6584", "Ha_narrow", "OIII_5007", "Hb_narrow"} <= line_flux.keys():
        return (float("nan"), float("nan"))
    return (
        float(safe_log10_ratio(line_flux["NII_6584"], line_flux["Ha_narrow"])),
        float(safe_log10_ratio(line_flux["OIII_5007"], line_flux["Hb_narrow"])),
    )


def _differential_oiii_oii_index(
    flux1: dict[str, float],
    flux2: dict[str, float],
) -> float | None:
    oii1 = _get_oii_flux(flux1)
    oii2 = _get_oii_flux(flux2)
    if oii1 is None or oii2 is None:
        return None
    if "OIII_5007" not in flux1 or "OIII_5007" not in flux2:
        return None
    oiii_ratio = _safe_divide(flux2["OIII_5007"], flux1["OIII_5007"])
    oii_ratio = _safe_divide(oii2, oii1)
    if not np.isfinite(oiii_ratio) or not np.isfinite(oii_ratio) or oii_ratio == 0:
        return None
    return oiii_ratio / oii_ratio


def _get_oii_flux(line_flux: dict[str, float]) -> float | None:
    if "OII_3727_blend" in line_flux:
        return line_flux["OII_3727_blend"]
    if "OII_3727" in line_flux:
        return line_flux["OII_3727"]
    if "OII_3726" in line_flux and "OII_3729" in line_flux:
        return line_flux["OII_3726"] + line_flux["OII_3729"]
    return None


def _get_sii_6717(line_flux: dict[str, float]) -> float | None:
    if "SII_6717" in line_flux:
        return line_flux["SII_6717"]
    if "SII_6716" in line_flux:
        return line_flux["SII_6716"]
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def _result_to_dict(result: Any | None, include_arrays: bool = False) -> dict[str, Any] | None:
    if result is None:
        return None
    if hasattr(result, "to_dict"):
        return result.to_dict(include_arrays=include_arrays)
    return {
        "success": getattr(result, "success", None),
        "message": getattr(result, "message", ""),
        "line_flux": getattr(result, "line_flux", {}),
        "line_flux_err": getattr(result, "line_flux_err", {}),
    }


__all__ = ["EpochFitSet", "EpochComparisonResult", "compare_epochs"]
