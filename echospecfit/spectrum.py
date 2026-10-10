"""Single one-dimensional spectrum container for EchoSpecFit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass
class H1Spectrum:
    """One-dimensional spectrum with wavelength, flux, and uncertainty arrays."""

    wave: np.ndarray
    flux: np.ndarray
    err: np.ndarray
    z: float = 0.0
    survey: str = "DESI"
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
        survey: str = "DESI",
        object_id: str = "",
        meta: dict[str, Any] | None = None,
    ) -> "H1Spectrum":
        """Create a spectrum from arrays and either error or inverse variance."""

        if err is None:
            if ivar is None:
                raise ValueError("Either err or ivar must be provided.")
            err = 1.0 / np.sqrt(np.asarray(ivar, dtype=float))
        return cls(
            wave=np.asarray(wave, dtype=float),
            flux=np.asarray(flux, dtype=float),
            err=np.asarray(err, dtype=float),
            z=z,
            survey=survey,
            object_id=object_id,
            meta={} if meta is None else meta,
        )

    @classmethod
    def from_txt(
        cls,
        path: str | Path,
        *,
        z: float = 0.0,
        survey: str = "DESI",
        object_id: str | None = None,
        delimiter: str | None = None,
        skiprows: int = 0,
        wave_col: int = 0,
        flux_col: int = 1,
        err_col: int = 2,
    ) -> "H1Spectrum":
        """Load a text spectrum with wavelength, flux, and error columns."""

        input_path = Path(path).expanduser()
        data = np.loadtxt(input_path, delimiter=delimiter, skiprows=skiprows)
        if data.ndim != 2:
            raise ValueError(f"{input_path} must contain a 2-D table.")
        max_col = max(wave_col, flux_col, err_col)
        if data.shape[1] <= max_col:
            raise ValueError(
                f"{input_path} has {data.shape[1]} columns, need column {max_col}."
            )
        return cls(
            wave=data[:, wave_col],
            flux=data[:, flux_col],
            err=data[:, err_col],
            z=z,
            survey=survey,
            object_id=object_id or input_path.stem,
            meta={"source_file": str(input_path)},
        )

    @classmethod
    def from_single(cls, spectrum: Any) -> "H1Spectrum":
        """Create an H1Spectrum from a similar single-spectrum object."""

        return cls(
            wave=np.asarray(getattr(spectrum, "wave"), dtype=float),
            flux=np.asarray(getattr(spectrum, "flux"), dtype=float),
            err=np.asarray(getattr(spectrum, "err"), dtype=float),
            z=float(getattr(spectrum, "z", 0.0)),
            survey=str(getattr(spectrum, "survey", "DESI")),
            object_id=str(getattr(spectrum, "object_id", "")),
            meta=deepcopy(getattr(spectrum, "meta", {})),
        )

    def to_rest_frame(self) -> "H1Spectrum":
        """Return a copy shifted to rest-frame wavelength."""

        return H1Spectrum(
            wave=self.wave / (1.0 + self.z),
            flux=self.flux.copy(),
            err=self.err.copy(),
            z=0.0,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def mask_bad_pixels(self) -> "H1Spectrum":
        """Drop non-finite pixels and non-positive uncertainties."""

        mask = (
            np.isfinite(self.wave)
            & np.isfinite(self.flux)
            & np.isfinite(self.err)
            & (self.err > 0)
        )
        return H1Spectrum(
            wave=self.wave[mask],
            flux=self.flux[mask],
            err=self.err[mask],
            z=self.z,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def select_window(self, rest_min: float, rest_max: float) -> "H1Spectrum":
        """Return pixels with rest wavelengths inside the requested interval."""

        rest_wave = self.wave / (1.0 + self.z)
        mask = (rest_wave >= rest_min) & (rest_wave <= rest_max)
        return H1Spectrum(
            wave=self.wave[mask],
            flux=self.flux[mask],
            err=self.err[mask],
            z=self.z,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def to_agn_spectrum(self) -> Any:
        """Convert to EchoSpec's AGNSpectrum container."""

        from echospec import AGNSpectrum

        return AGNSpectrum(
            wave=self.wave.copy(),
            flux=self.flux.copy(),
            err=self.err.copy(),
            z=self.z,
            survey=self.survey,
            object_id=self.object_id,
            meta=deepcopy(self.meta),
        )

    def copy(self) -> "H1Spectrum":
        """Return a deep copy."""

        return deepcopy(self)


SingleH1Spectrum = H1Spectrum

__all__ = ["H1Spectrum", "SingleH1Spectrum"]

