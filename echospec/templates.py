"""Discovery and loading of local stellar template libraries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from astropy.io import fits

SUPPORTED_TEMPLATE_SUFFIXES = (
    ".fits",
    ".fit",
    ".fits.gz",
    ".npy",
    ".npz",
)
TEMPLATE_CACHE_ENV_VAR = "ECHOSPECFIT_TEMPLATE_DIR"


def get_template_cache_dir() -> Path:
    """Return the configured template root, preserving the legacy cache."""

    configured_dir = os.environ.get(TEMPLATE_CACHE_ENV_VAR)
    if configured_dir:
        return Path(configured_dir).expanduser()
    return Path.home() / ".echospec" / "templates"


def find_template_files(library: str = "miles") -> list[Path]:
    """Find supported template files for a local library."""

    library_dir = _resolve_template_library_dir(library=library, local_path=None)
    if not library_dir.exists():
        return []
    return sorted(
        path
        for path in library_dir.rglob("*")
        if path.is_file() and _has_supported_suffix(path)
    )


def load_template_library(
    library: str = "miles",
    wavelength_range: tuple[float, float] | None = None,
    max_templates: int | None = None,
    local_path: str | Path | None = None,
    debug: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a local stellar template library for pPXF.

    Returns
    -------
    template_wave, template_flux
        Wavelength vector and template matrix with shape
        ``(n_pixels, n_templates)``.
    """

    files = _find_template_files_in_dir(
        _resolve_template_library_dir(library=library, local_path=local_path)
    )
    if not files:
        template_dir = _resolve_template_library_dir(
            library=library,
            local_path=local_path,
        )
        raise FileNotFoundError(
            f"No stellar template files found under {template_dir}. "
            "Run `python scripts/download_ppxf_templates.py "
            f"--library {library}` first."
        )

    numpy_files = [path for path in files if _is_numpy_template_file(path)]
    for path in numpy_files:
        try:
            wave, flux = _load_numpy_template_file(path)
        except (KeyError, ValueError, TypeError):
            continue
        template_wave, template_flux = _finalize_templates(
            wave,
            flux,
            wavelength_range=wavelength_range,
            max_templates=max_templates,
        )
        _report_template_diagnostics(
            library=library,
            template_wave=template_wave,
            template_flux=template_flux,
            files=[path],
            debug=debug,
        )
        return template_wave, template_flux

    fits_files = [path for path in files if _is_fits_template_file(path)]
    if not fits_files:
        raise ValueError(
            f"Found template files for {library}, but none could be read as "
            "npz/npy libraries or FITS templates."
        )
    if max_templates is not None:
        fits_files = fits_files[:max_templates]

    template_wave: np.ndarray | None = None
    template_fluxes: list[np.ndarray] = []
    for path in fits_files:
        wave, flux = _load_fits_template(path)
        wave, flux = _apply_wavelength_range(wave, flux, wavelength_range)
        if template_wave is None:
            template_wave = wave
        elif not _same_wavelength_grid(template_wave, wave):
            raise ValueError(
                "Template wavelength grids are not identical. EchoSpec does "
                "not interpolate templates automatically in this loader."
            )
        template_fluxes.append(flux)

    if template_wave is None or not template_fluxes:
        raise ValueError(f"No readable FITS templates found for library {library}.")
    template_flux = np.column_stack(template_fluxes)
    _report_template_diagnostics(
        library=library,
        template_wave=template_wave,
        template_flux=template_flux,
        files=fits_files,
        debug=debug,
    )
    return template_wave, template_flux


def summarize_template_library(
    library: str = "miles",
    wavelength_range: tuple[float, float] | None = None,
    max_templates: int | None = None,
    local_path: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Return basic wavelength and template-count metadata."""

    wave, flux = load_template_library(
        library=library,
        wavelength_range=wavelength_range,
        max_templates=max_templates,
        local_path=local_path,
        debug=False,
    )
    return {
        "library": library,
        "wavelength_min": float(wave.min()),
        "wavelength_max": float(wave.max()),
        "n_pixels": int(wave.size),
        "n_templates": int(flux.shape[1]),
        "median_wavelength_step": float(np.nanmedian(np.diff(wave))),
    }


def _resolve_template_library_dir(
    library: str,
    local_path: str | Path | None,
) -> Path:
    if local_path is not None:
        return Path(local_path).expanduser()
    return get_template_cache_dir() / library


def _find_template_files_in_dir(library_dir: Path) -> list[Path]:
    if not library_dir.exists():
        return []
    return sorted(
        path
        for path in library_dir.rglob("*")
        if path.is_file() and _has_supported_suffix(path)
    )


def _report_template_diagnostics(
    library: str,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    files: list[Path],
    debug: bool,
) -> None:
    n_templates = int(template_flux.shape[1]) if template_flux.ndim == 2 else 0
    diagnostics = {
        "template_wave_range": (
            float(np.nanmin(template_wave)),
            float(np.nanmax(template_wave)),
        ),
        "template_wave_median_step": float(np.nanmedian(np.diff(template_wave))),
        "template_flux_shape": tuple(template_flux.shape),
        "n_templates": n_templates,
        "first_20_template_filenames": [str(path) for path in files[:20]],
        "template_flux_min": float(np.nanmin(template_flux)),
        "template_flux_max": float(np.nanmax(template_flux)),
        "template_flux_median": float(np.nanmedian(template_flux)),
    }
    if n_templates < 20:
        warnings.warn(
            f"Template library {library!r} has only {n_templates} templates; "
            "stellar fits may be under-constrained.",
            RuntimeWarning,
            stacklevel=2,
        )
    if debug:
        print("=== EchoSpec template loading diagnostics ===")
        print(f"library: {library}")
        print(
            "template_wave range: "
            f"{diagnostics['template_wave_range'][0]:.8g} - "
            f"{diagnostics['template_wave_range'][1]:.8g}"
        )
        print(f"template_wave median step: {diagnostics['template_wave_median_step']:.8g}")
        print(f"template_flux shape: {diagnostics['template_flux_shape']}")
        print(f"n_templates: {diagnostics['n_templates']}")
        print("first 20 template filenames:")
        for filename in diagnostics["first_20_template_filenames"]:
            print(f"  {filename}")
        print(
            "template_flux min/max/median: "
            f"{diagnostics['template_flux_min']:.8g} / "
            f"{diagnostics['template_flux_max']:.8g} / "
            f"{diagnostics['template_flux_median']:.8g}"
        )


def _load_numpy_template_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as data:
            wave_key = _first_existing_key(data, ("template_wave", "wave", "lam"))
            flux_key = _first_existing_key(
                data,
                ("template_flux", "templates", "flux"),
            )
            if wave_key is None or flux_key is None:
                raise KeyError(
                    f"{path} must contain template_wave/template_flux arrays."
                )
            return np.asarray(data[wave_key], dtype=float), np.asarray(
                data[flux_key],
                dtype=float,
            )

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        item = data.item()
        if isinstance(item, dict):
            wave_key = _first_existing_key(item, ("template_wave", "wave", "lam"))
            flux_key = _first_existing_key(
                item,
                ("template_flux", "templates", "flux"),
            )
            if wave_key is None or flux_key is None:
                raise KeyError(
                    f"{path} must contain template_wave/template_flux arrays."
                )
            return np.asarray(item[wave_key], dtype=float), np.asarray(
                item[flux_key],
                dtype=float,
            )
    raise ValueError(f"Cannot infer template_wave/template_flux from {path}.")


def _load_fits_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        hdu = _first_image_hdu(hdul)
        if hdu is None or hdu.data is None:
            raise ValueError(f"No image data found in FITS template {path}.")
        flux = np.asarray(hdu.data, dtype=float).squeeze()
        if flux.ndim != 1:
            raise ValueError(f"FITS template must be one-dimensional: {path}.")
        wave = _wavelength_from_header(hdu.header, flux.size)
    return wave, flux


def _wavelength_from_header(header: fits.Header, n_pixels: int) -> np.ndarray:
    try:
        crval = float(header["CRVAL1"])
    except KeyError as exc:
        raise ValueError("FITS template header is missing CRVAL1.") from exc
    delta = _header_delta(header)
    header_naxis = int(header.get("NAXIS1", n_pixels))
    if header_naxis != n_pixels:
        raise ValueError(
            f"FITS header NAXIS1={header_naxis} does not match data length "
            f"{n_pixels}."
        )
    axis = np.arange(n_pixels, dtype=float)
    coordinate = crval + delta * axis
    if _is_log_wavelength_header(header, crval, delta):
        if str(header.get("CTYPE1", "")).upper().startswith("LN"):
            return np.exp(coordinate)
        return 10.0**coordinate
    return coordinate


def _header_delta(header: fits.Header) -> float:
    if "CDELT1" in header:
        return float(header["CDELT1"])
    if "CD1_1" in header:
        return float(header["CD1_1"])
    raise ValueError("FITS template header is missing CDELT1 or CD1_1.")


def _is_log_wavelength_header(
    header: fits.Header,
    crval: float,
    delta: float,
) -> bool:
    ctype = str(header.get("CTYPE1", "")).upper()
    if "LOG" in ctype or ctype.startswith("LN"):
        return True
    if int(header.get("DC-FLAG", 0)) == 1:
        return True
    return abs(crval) < 20.0 and abs(delta) < 1.0e-2


def _finalize_templates(
    wave: np.ndarray,
    flux: np.ndarray,
    wavelength_range: tuple[float, float] | None,
    max_templates: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.asarray(wave, dtype=float)
    flux = _as_template_matrix(flux, wave.size)
    if max_templates is not None:
        flux = flux[:, :max_templates]
    wave, flux = _apply_wavelength_range(wave, flux, wavelength_range)
    return wave, flux


def _apply_wavelength_range(
    wave: np.ndarray,
    flux: np.ndarray,
    wavelength_range: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if wavelength_range is None:
        return wave, flux
    wave_min, wave_max = wavelength_range
    mask = (wave >= wave_min) & (wave <= wave_max)
    if not np.any(mask):
        raise ValueError("wavelength_range does not overlap the templates.")
    return wave[mask], flux[mask]


def _as_template_matrix(flux: np.ndarray, n_pixels: int) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    if flux.ndim == 1:
        if flux.size != n_pixels:
            raise ValueError("Template flux length does not match wavelength grid.")
        return flux[:, None]
    if flux.ndim == 2:
        if flux.shape[0] == n_pixels:
            return flux
        if flux.shape[1] == n_pixels:
            return flux.T
    if flux.ndim == 3 and flux.shape[0] == n_pixels:
        return flux.reshape(n_pixels, -1)
    raise ValueError(
        "template_flux must have shape (n_pixels, n_templates), "
        "(n_templates, n_pixels), or a compatible 3D SPS cube."
    )


def _first_image_hdu(hdul: fits.HDUList) -> Any | None:
    for hdu in hdul:
        if hdu.data is not None:
            return hdu
    return None


def _first_existing_key(container: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in container:
            return key
    return None


def _same_wavelength_grid(wave_a: np.ndarray, wave_b: np.ndarray) -> bool:
    return wave_a.shape == wave_b.shape and np.allclose(
        wave_a,
        wave_b,
        rtol=0.0,
        atol=1.0e-8,
    )


def _has_supported_suffix(path: Path) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix) for suffix in SUPPORTED_TEMPLATE_SUFFIXES)


def _is_numpy_template_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith(".npy") or lower_name.endswith(".npz")


def _is_fits_template_file(path: Path) -> bool:
    lower_name = path.name.lower()
    return (
        lower_name.endswith(".fits")
        or lower_name.endswith(".fit")
        or lower_name.endswith(".fits.gz")
    )


__all__ = [
    "find_template_files",
    "get_template_cache_dir",
    "load_template_library",
    "summarize_template_library",
]
