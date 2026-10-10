"""Spectrum containers and related helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class AGNSpectrum:
    """One-dimensional AGN spectrum with flux uncertainties."""

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    z: float
    survey: str = ""
    object_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wave = np.asarray(self.wave, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.err = np.asarray(self.err, dtype=float)
        self.z = float(self.z)
        self.meta = deepcopy(self.meta)

        if self.wave.ndim != 1 or self.flux.ndim != 1 or self.err.ndim != 1:
            raise ValueError("wave, flux, and err must be one-dimensional arrays.")

        if not (len(self.wave) == len(self.flux) == len(self.err)):
            raise ValueError("wave, flux, and err must have the same length.")

        finite_err = np.isfinite(self.err)
        if np.any(self.err[finite_err] <= 0):
            raise ValueError("err must be positive where finite.")

        if len(self.wave) > 1 and np.any(np.diff(self.wave) < 0):
            order = np.argsort(self.wave)
            self.wave = self.wave[order]
            self.flux = self.flux[order]
            self.err = self.err[order]

    @classmethod
    def from_arrays(
        cls,
        wave: ArrayLike,
        flux: ArrayLike,
        err: ArrayLike | None = None,
        ivar: ArrayLike | None = None,
        z: float = 0.0,
        survey: str = "",
        object_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> AGNSpectrum:
        """Create a spectrum from wavelength, flux, and error or inverse variance."""

        if err is None:
            if ivar is None:
                raise ValueError("Either err or ivar must be provided.")
            ivar_array = np.asarray(ivar, dtype=float)
            err = 1.0 / np.sqrt(ivar_array)

        return cls(
            wave=np.asarray(wave, dtype=float),
            flux=np.asarray(flux, dtype=float),
            err=np.asarray(err, dtype=float),
            z=z,
            survey=survey,
            object_id=object_id,
            meta={} if meta is None else meta,
        )

    def to_rest_frame(self) -> AGNSpectrum:
        """Return a copy shifted to rest-frame wavelengths."""

        return AGNSpectrum(
            wave=self.wave / (1.0 + self.z),
            flux=self.flux.copy(),
            err=self.err.copy(),
            z=0.0,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def select_window(self, rest_min: float, rest_max: float) -> AGNSpectrum:
        """Return pixels with rest-frame wavelengths inside the requested range."""

        rest_wave = self.wave / (1.0 + self.z)
        mask = (rest_wave >= rest_min) & (rest_wave <= rest_max)
        return AGNSpectrum(
            wave=self.wave[mask],
            flux=self.flux[mask],
            err=self.err[mask],
            z=self.z,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def mask_bad_pixels(self) -> AGNSpectrum:
        """Return a spectrum with non-finite values and non-positive errors removed."""

        mask = (
            np.isfinite(self.wave)
            & np.isfinite(self.flux)
            & np.isfinite(self.err)
            & (self.err > 0)
        )
        return AGNSpectrum(
            wave=self.wave[mask],
            flux=self.flux[mask],
            err=self.err[mask],
            z=self.z,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def copy(self) -> AGNSpectrum:
        """Return a deep copy of this spectrum."""

        return deepcopy(self)

    def snr(self, window: tuple[float, float] | None = None) -> float:
        """Estimate the median flux/error globally or in a rest-frame window."""

        spectrum = self if window is None else self.select_window(*window)
        good = (
            np.isfinite(spectrum.flux)
            & np.isfinite(spectrum.err)
            & (spectrum.err > 0)
        )
        if not np.any(good):
            return float("nan")
        return float(np.nanmedian(spectrum.flux[good] / spectrum.err[good]))


__all__ = ["AGNSpectrum"]
