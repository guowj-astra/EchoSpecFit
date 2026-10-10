"""Continuum components used by joint EchoSpec fits."""

from __future__ import annotations

import numpy as np


def powerlaw_continuum(
    wave: np.ndarray,
    amp: float,
    alpha: float,
    pivot: float = 5100.0,
) -> np.ndarray:
    """Return F_lambda = amp * (lambda / pivot) ** alpha."""

    wave = np.asarray(wave, dtype=float)
    return float(amp) * (wave / float(pivot)) ** float(alpha)


def powerlaw_diagnostics(
    wave: np.ndarray,
    powerlaw: np.ndarray,
    total_continuum: np.ndarray,
) -> dict[str, float | bool]:
    """Measure power-law continuum fractions at common H-beta wavelengths."""

    diagnostics: dict[str, float | bool] = {}
    for wavelength in (4200.0, 5100.0, 5600.0):
        pl = float(np.interp(wavelength, wave, powerlaw))
        total = float(np.interp(wavelength, wave, total_continuum))
        fraction = pl / total if np.isfinite(total) and total != 0.0 else float("nan")
        diagnostics[f"powerlaw_fraction_{int(wavelength)}"] = fraction
    diagnostics["flag_powerlaw_dominant"] = bool(
        np.isfinite(diagnostics["powerlaw_fraction_5100"])
        and diagnostics["powerlaw_fraction_5100"] > 0.8
    )
    return diagnostics


__all__ = ["powerlaw_continuum", "powerlaw_diagnostics"]
