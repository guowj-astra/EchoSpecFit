"""Stellar-continuum modeling interfaces."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from echospec.spectrum import AGNSpectrum
from echospec.templates import get_template_cache_dir, load_template_library

C_KMS = 299792.458
DEFAULT_TEMPLATE_CACHE_ENV_VAR = "ECHOSPEC_PPXF_TEMPLATE_CACHE"
DEFAULT_TEMPLATE_LIBRARY_ENV_VAR = "ECHOSPEC_PPXF_TEMPLATE_LIBRARY"
DEFAULT_TEMPLATE_CACHE_RELATIVE_PATH = (
    Path(".echospec") / "templates" / "ppxf_templates.npz"
)
PPXF_TEMPLATE_DOWNLOAD_HINT = (
    "Run `python scripts/download_ppxf_templates.py --library miles` or "
    "`python scripts/download_ppxf_templates.py --library emiles` to prepare "
    f"templates, set {DEFAULT_TEMPLATE_CACHE_ENV_VAR}, or pass "
    "template_wave/template_flux explicitly."
)
DEFAULT_EMISSION_LINE_CENTERS = (
    3726.03,
    3728.82,
    4101.74,
    4340.47,
    4861.33,
    4958.91,
    5006.84,
    6300.30,
    6548.05,
    6562.80,
    6583.45,
    6716.44,
    6730.82,
)


@dataclass
class StellarFitResult:
    """Result from pPXF stellar-continuum fitting."""

    stellar_model: np.ndarray
    residual_flux: np.ndarray
    velocity: float
    sigma: float
    chi2: float
    weights: np.ndarray
    goodpixels: np.ndarray
    ppxf_result: Any | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    agn_model: np.ndarray | None = None
    total_model: np.ndarray | None = None

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly stellar fit summary."""

        summary: dict[str, Any] = {
            "success": True,
            "velocity": self.velocity,
            "sigma": self.sigma,
            "velocity_err": _ppxf_formal_error(self.ppxf_result, 0),
            "sigma_err": _ppxf_formal_error(self.ppxf_result, 1),
            "chi2": self.chi2,
            "weights_nonzero": int(np.count_nonzero(np.abs(self.weights) > 0)),
            "n_goodpixels": int(len(self.goodpixels)),
            "diagnostics": self.diagnostics,
            "continuum_summary": _continuum_host_summary(
                self.stellar_model,
                self.agn_model,
                self.total_model,
            ),
        }
        if include_arrays:
            summary.update(
                {
                    "stellar_model": self.stellar_model.tolist(),
                    "residual_flux": self.residual_flux.tolist(),
                    "weights": self.weights.tolist(),
                    "goodpixels": self.goodpixels.tolist(),
                    "agn_model": (
                        None if self.agn_model is None else self.agn_model.tolist()
                    ),
                    "total_model": (
                        None if self.total_model is None else self.total_model.tolist()
                    ),
                }
            )
        return summary


def default_ppxf_template_cache_path() -> Path:
    """Return the default pPXF template cache path."""

    env_path = os.environ.get(DEFAULT_TEMPLATE_CACHE_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    return get_template_cache_dir() / DEFAULT_TEMPLATE_CACHE_RELATIVE_PATH.name


def load_default_ppxf_templates(
    cache_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load stellar templates from the default EchoSpec template location."""

    if cache_path is not None or os.environ.get(DEFAULT_TEMPLATE_CACHE_ENV_VAR):
        path = (
            Path(cache_path).expanduser()
            if cache_path is not None
            else default_ppxf_template_cache_path()
        )
        if not path.exists():
            raise FileNotFoundError(
                "No pPXF stellar templates were provided and the template cache "
                f"was not found at {path}. {PPXF_TEMPLATE_DOWNLOAD_HINT}"
            )
        return _load_template_cache_npz(path)

    library = os.environ.get(DEFAULT_TEMPLATE_LIBRARY_ENV_VAR, "miles")
    try:
        return load_template_library(library=library)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No pPXF stellar templates were provided and no local template "
            f"library was found under {get_template_cache_dir() / library}. "
            f"{PPXF_TEMPLATE_DOWNLOAD_HINT}"
        ) from exc


def _load_template_cache_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as cache:
        wave_key = _first_existing_key(cache, ("template_wave", "wave", "lam"))
        flux_key = _first_existing_key(cache, ("template_flux", "templates", "flux"))
        if wave_key is None or flux_key is None:
            raise ValueError(
                f"Template cache {path} must contain wavelength and template arrays. "
                "Expected keys like template_wave/template_flux or wave/templates."
            )
        template_wave = np.asarray(cache[wave_key], dtype=float)
        template_flux = np.asarray(cache[flux_key], dtype=float)

    if template_flux.ndim == 3:
        template_flux = template_flux.reshape(template_flux.shape[0], -1)
    return template_wave, template_flux


def fit_stellar_ppxf(
    spec: AGNSpectrum,
    template_wave: ArrayLike,
    template_flux: ArrayLike,
    mdegree: int = 10,
    degree: int = -1,
    mask_emission_lines: bool = True,
    mask_width_kms: float = 1000.0,
    template_buffer_kms: float = 10000.0,
    include_agn_powerlaw: bool = True,
    agn_powerlaw_slopes: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    debug: bool = True,
) -> StellarFitResult:
    """Fit stellar and optional AGN power-law continua with pPXF.

    If include_agn_powerlaw is True, featureless power-law templates are included
    during pPXF fitting, but the returned residual subtracts only the stellar
    component. This avoids removing the AGN continuum before local line fitting.
    """

    try:
        from ppxf.ppxf import ppxf
        from ppxf.ppxf_util import log_rebin
    except ImportError as exc:
        raise ImportError(
            "fit_stellar_ppxf requires the optional ppxf dependency to be installed."
        ) from exc

    clean_spec = spec.mask_bad_pixels()
    template_wave_array = np.asarray(template_wave, dtype=float)
    templates = _as_template_matrix(template_flux, len(template_wave_array))

    if template_wave_array.ndim != 1:
        raise ValueError("template_wave must be one-dimensional.")
    if len(template_wave_array) < 2:
        raise ValueError("template_wave must contain at least two pixels.")

    template_min = float(template_wave_array.min())
    template_max = float(template_wave_array.max())
    template_median_step = float(np.nanmedian(np.diff(template_wave_array)))
    observed_min = float(clean_spec.wave.min())
    observed_max = float(clean_spec.wave.max())

    overlap_min = max(observed_min, template_min)
    overlap_max = min(observed_max, template_max)
    if overlap_min >= overlap_max:
        raise ValueError("Spectrum and templates do not overlap in wavelength.")

    template_cut_min = max(
        template_min,
        overlap_min * (1.0 - template_buffer_kms / C_KMS),
    )
    template_cut_max = min(
        template_max,
        overlap_max * (1.0 + template_buffer_kms / C_KMS),
    )

    template_mask = (template_wave_array >= template_cut_min) & (
        template_wave_array <= template_cut_max
    )
    spec_mask = (clean_spec.wave >= overlap_min) & (clean_spec.wave <= overlap_max)
    fit_wave = clean_spec.wave[spec_mask]
    fit_flux = clean_spec.flux[spec_mask]
    fit_err = clean_spec.err[spec_mask]
    cut_template_wave = template_wave_array[template_mask]
    cut_templates = templates[template_mask, :]
    if len(fit_wave) < 2 or len(cut_template_wave) < 2:
        raise ValueError("Insufficient overlapping pixels for pPXF fitting.")

    template_prep = _prepare_templates_for_ppxf(cut_template_wave, cut_templates)
    fit_template_wave = template_prep["wave"]
    fit_templates = template_prep["flux"]
    n_stellar_templates = int(fit_templates.shape[1])
    n_agn_templates = 0
    if include_agn_powerlaw:
        powerlaw_templates = _make_powerlaw_templates(
            fit_template_wave,
            slopes=agn_powerlaw_slopes,
        )
        fit_templates = np.column_stack([fit_templates, powerlaw_templates])
        n_agn_templates = int(powerlaw_templates.shape[1])
    n_templates = int(fit_templates.shape[1])
    overlap_min = max(float(fit_wave.min()), float(fit_template_wave.min()))
    overlap_max = min(float(fit_wave.max()), float(fit_template_wave.max()))
    spec_mask = (fit_wave >= overlap_min) & (fit_wave <= overlap_max)
    fit_wave = fit_wave[spec_mask]
    fit_flux = fit_flux[spec_mask]
    fit_err = fit_err[spec_mask]
    _warn_if_ppxf_setup_suspicious(
        n_templates=n_templates,
        observed_min=observed_min,
        observed_max=observed_max,
        overlap_min=overlap_min,
        overlap_max=overlap_max,
    )

    galaxy_log, log_lam_gal, velscale = log_rebin(
        fit_wave,
        fit_flux,
    )
    noise_log, _, _ = log_rebin(
        fit_wave,
        fit_err,
        velscale=velscale,
    )
    galaxy_log, noise_log, log_lam_gal = _trim_to_common_length(
        galaxy_log,
        noise_log,
        log_lam_gal,
    )
    template_fit, log_lam_temp, _ = log_rebin(
        fit_template_wave,
        fit_templates,
        velscale=velscale,
    )

    galaxy_norm = float(
        np.nanmedian(fit_flux[np.isfinite(fit_flux) & np.isfinite(fit_err) & (fit_err > 0)])
    )
    if not np.isfinite(galaxy_norm) or galaxy_norm == 0:
        galaxy_norm = float(np.nanmedian(galaxy_log[np.isfinite(galaxy_log)]))
    if not np.isfinite(galaxy_norm) or galaxy_norm == 0:
        galaxy_norm = 1.0
    galaxy_fit = galaxy_log / galaxy_norm
    noise_fit = np.clip(noise_log / abs(galaxy_norm), 1.0e-12, None)

    lam_gal = np.exp(log_lam_gal)
    lam_temp = np.exp(log_lam_temp)
    goodpixels = _make_goodpixels(
        lam_gal,
        noise_fit,
        mask_emission_lines=mask_emission_lines,
        mask_width_kms=mask_width_kms,
    )
    if len(goodpixels) == 0:
        raise ValueError("No good pixels available for pPXF fitting.")
    diagnostics = {
        "template_wave_range": (template_min, template_max),
        "template_wave_median_step": template_median_step,
        "template_flux_shape": tuple(templates.shape),
        "template_flux_after_cut_shape": tuple(fit_templates.shape),
        "n_templates": int(templates.shape[1]),
        "n_templates_kept": n_stellar_templates,
        "n_stellar_templates": n_stellar_templates,
        "include_agn_powerlaw": bool(include_agn_powerlaw),
        "agn_powerlaw_slopes": tuple(float(slope) for slope in agn_powerlaw_slopes),
        "n_agn_powerlaw_templates": n_agn_templates,
        "n_templates_dropped": int(template_prep["n_templates_dropped"]),
        "n_bad_template_wavelength_pixels_removed": int(
            template_prep["n_bad_pixels_removed"]
        ),
        "template_flux_min": float(np.nanmin(templates)),
        "template_flux_max": float(np.nanmax(templates)),
        "template_flux_median": float(np.nanmedian(templates)),
        "ppxf_mdegree": int(mdegree),
        "ppxf_degree": int(degree),
        "template_norm_min": float(np.nanmin(template_prep["template_norms"])),
        "template_norm_median": float(np.nanmedian(template_prep["template_norms"])),
        "template_norm_max": float(np.nanmax(template_prep["template_norms"])),
        "galaxy_norm": galaxy_norm,
        "observed_spectrum_wavelength_range": (observed_min, observed_max),
        "overlap_wavelength_range": (float(overlap_min), float(overlap_max)),
        "n_pixels_used_by_ppxf": int(len(galaxy_fit)),
        "n_goodpixels_after_emission_line_masking": int(len(goodpixels)),
    }
    if debug:
        _print_ppxf_diagnostics(diagnostics)

    initial_start = [0.0, 200.0]
    initial_ppxf_result = None
    final_mdegree = mdegree
    if include_agn_powerlaw and mdegree > 0:
        initial_ppxf_result = ppxf(
            template_fit[:, :n_stellar_templates],
            galaxy_fit,
            noise_fit,
            velscale,
            start=initial_start,
            goodpixels=goodpixels,
            moments=2,
            degree=degree,
            mdegree=mdegree,
            lam=lam_gal,
            lam_temp=lam_temp,
            quiet=True,
        )
        initial_solution = np.atleast_1d(initial_ppxf_result.sol)
        if len(initial_solution) >= 2:
            initial_start = [float(initial_solution[0]), float(initial_solution[1])]
        final_mdegree = 0

    ppxf_result = ppxf(
        template_fit,
        galaxy_fit,
        noise_fit,
        velscale,
        start=initial_start,
        goodpixels=goodpixels,
        moments=2,
        degree=degree,
        mdegree=final_mdegree,
        lam=lam_gal,
        lam_temp=lam_temp,
        quiet=True,
    )

    model_log = np.asarray(ppxf_result.bestfit, dtype=float) * galaxy_norm
    weights = np.asarray(ppxf_result.weights, dtype=float)
    matrix = np.asarray(ppxf_result.matrix, dtype=float)
    stellar_log = matrix[:, :n_stellar_templates] @ weights[:n_stellar_templates]
    stellar_log *= galaxy_norm
    if n_agn_templates > 0:
        agn_log = matrix[:, n_stellar_templates:n_templates] @ weights[
            n_stellar_templates:n_templates
        ]
        agn_log *= galaxy_norm
    else:
        agn_log = np.zeros_like(stellar_log)

    total_model = _model_on_original_grid(spec.wave, fit_wave, lam_gal, model_log)
    stellar_model = _model_on_original_grid(spec.wave, fit_wave, lam_gal, stellar_log)
    agn_model = _model_on_original_grid(
        spec.wave,
        fit_wave,
        lam_gal,
        agn_log,
    )

    residual_flux = spec.flux - stellar_model
    solution = np.atleast_1d(ppxf_result.sol)
    velocity = float(solution[0])
    sigma = float(solution[1]) if len(solution) > 1 else float("nan")
    diagnostics.update(
        {
            "ppxf_chi2": float(ppxf_result.chi2),
            "ppxf_velocity": velocity,
            "ppxf_sigma": sigma,
            "ppxf_weights_nonzero_count": int(
                np.count_nonzero(weights > 0)
            ),
            "stellar_weights_nonzero_count": int(
                np.count_nonzero(weights[:n_stellar_templates] > 0)
            ),
            "agn_powerlaw_weights_nonzero_count": int(
                np.count_nonzero(weights[n_stellar_templates:n_templates] > 0)
            ),
            "agn_powerlaw_weight_sum": float(
                np.sum(weights[n_stellar_templates:n_templates])
            ),
            "initial_ppxf_mdegree": int(mdegree),
            "initial_ppxf_chi2": (
                float(initial_ppxf_result.chi2)
                if initial_ppxf_result is not None
                else float("nan")
            ),
            "initial_ppxf_velocity": (
                float(np.atleast_1d(initial_ppxf_result.sol)[0])
                if initial_ppxf_result is not None
                else float("nan")
            ),
            "initial_ppxf_sigma": (
                float(np.atleast_1d(initial_ppxf_result.sol)[1])
                if initial_ppxf_result is not None
                and len(np.atleast_1d(initial_ppxf_result.sol)) > 1
                else float("nan")
            ),
            "final_ppxf_mdegree": int(final_mdegree),
            "continuum_host_at_wavelengths": _continuum_host_at_wavelengths(
                spec.wave,
                stellar_model,
                agn_model,
                total_model,
            ),
        }
    )
    if debug:
        _print_ppxf_result_diagnostics(diagnostics)

    return StellarFitResult(
        stellar_model=stellar_model,
        residual_flux=residual_flux,
        velocity=velocity,
        sigma=sigma,
        chi2=float(ppxf_result.chi2),
        weights=np.asarray(ppxf_result.weights, dtype=float),
        goodpixels=goodpixels,
        ppxf_result=ppxf_result,
        diagnostics=diagnostics,
        agn_model=agn_model,
        total_model=total_model,
    )


def _ppxf_formal_error(ppxf_result: Any | None, index: int) -> float:
    if ppxf_result is None:
        return float("nan")
    errors = getattr(ppxf_result, "error", None)
    if errors is None:
        return float("nan")
    try:
        value = np.asarray(errors, dtype=float).ravel()[index]
    except (IndexError, TypeError, ValueError):
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _continuum_host_summary(
    stellar_model: np.ndarray,
    agn_model: np.ndarray | None,
    total_model: np.ndarray | None,
) -> dict[str, float]:
    """Summarize host and featureless-continuum contributions from pPXF."""

    stellar = np.asarray(stellar_model, dtype=float)
    agn = (
        np.zeros_like(stellar, dtype=float)
        if agn_model is None
        else np.asarray(agn_model, dtype=float)
    )
    total = (
        stellar + agn
        if total_model is None
        else np.asarray(total_model, dtype=float)
    )
    good = np.isfinite(stellar) & np.isfinite(agn) & np.isfinite(total)
    positive_total = good & (np.abs(total) > 0.0)
    host_fraction = np.full_like(total, np.nan, dtype=float)
    agn_fraction = np.full_like(total, np.nan, dtype=float)
    host_fraction[positive_total] = stellar[positive_total] / total[positive_total]
    agn_fraction[positive_total] = agn[positive_total] / total[positive_total]
    return {
        "host_flux_median": _finite_median(stellar[good]),
        "agn_powerlaw_flux_median": _finite_median(agn[good]),
        "total_continuum_flux_median": _finite_median(total[good]),
        "host_fraction_median": _finite_median(host_fraction[positive_total]),
        "agn_powerlaw_fraction_median": _finite_median(agn_fraction[positive_total]),
        "host_fraction_p16": _finite_percentile(host_fraction[positive_total], 16.0),
        "host_fraction_p84": _finite_percentile(host_fraction[positive_total], 84.0),
    }


def _continuum_host_at_wavelengths(
    wave: ArrayLike,
    stellar_model: np.ndarray,
    agn_model: np.ndarray | None,
    total_model: np.ndarray | None,
    sample_wavelengths: tuple[float, ...] = (4020.0, 5100.0, 6563.0),
) -> dict[str, dict[str, float]]:
    wave_array = np.asarray(wave, dtype=float)
    stellar = np.asarray(stellar_model, dtype=float)
    agn = (
        np.zeros_like(stellar, dtype=float)
        if agn_model is None
        else np.asarray(agn_model, dtype=float)
    )
    total = (
        stellar + agn
        if total_model is None
        else np.asarray(total_model, dtype=float)
    )
    good = np.isfinite(wave_array) & np.isfinite(stellar) & np.isfinite(agn) & np.isfinite(total)
    out: dict[str, dict[str, float]] = {}
    if np.count_nonzero(good) == 0:
        for wavelength in sample_wavelengths:
            out[f"{wavelength:.0f}"] = _empty_continuum_sample(wavelength)
        return out
    wave_good = wave_array[good]
    stellar_good = stellar[good]
    agn_good = agn[good]
    total_good = total[good]
    order = np.argsort(wave_good)
    wave_good = wave_good[order]
    stellar_good = stellar_good[order]
    agn_good = agn_good[order]
    total_good = total_good[order]
    for wavelength in sample_wavelengths:
        if wavelength < wave_good[0] or wavelength > wave_good[-1]:
            out[f"{wavelength:.0f}"] = _empty_continuum_sample(wavelength)
            continue
        stellar_flux = float(np.interp(wavelength, wave_good, stellar_good))
        agn_flux = float(np.interp(wavelength, wave_good, agn_good))
        total_flux = float(np.interp(wavelength, wave_good, total_good))
        if np.isfinite(total_flux) and abs(total_flux) > 0.0:
            host_fraction = stellar_flux / total_flux
            agn_fraction = agn_flux / total_flux
        else:
            host_fraction = float("nan")
            agn_fraction = float("nan")
        out[f"{wavelength:.0f}"] = {
            "wavelength": float(wavelength),
            "host_flux": stellar_flux,
            "agn_powerlaw_flux": agn_flux,
            "total_continuum_flux": total_flux,
            "host_fraction": float(host_fraction),
            "agn_powerlaw_fraction": float(agn_fraction),
        }
    return out


def _empty_continuum_sample(wavelength: float) -> dict[str, float]:
    return {
        "wavelength": float(wavelength),
        "host_flux": float("nan"),
        "agn_powerlaw_flux": float("nan"),
        "total_continuum_flux": float("nan"),
        "host_fraction": float("nan"),
        "agn_powerlaw_fraction": float("nan"),
    }


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanmedian(values))


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.nanpercentile(values, percentile))


def subtract_stellar_continuum(
    spec: AGNSpectrum,
    stellar_result: StellarFitResult,
) -> AGNSpectrum:
    """Return a spectrum with the fitted stellar continuum subtracted."""

    return AGNSpectrum(
        wave=spec.wave.copy(),
        flux=stellar_result.residual_flux.copy(),
        err=spec.err.copy(),
        z=spec.z,
        survey=spec.survey,
        object_id=spec.object_id,
        meta=spec.meta.copy(),
    )


def plot_stellar_fit(
    spec: AGNSpectrum,
    stellar_result: StellarFitResult,
    output_path: str | Path | None = None,
):
    """Plot observed spectrum, stellar model, and residual flux."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_spec, ax_resid) = plt.subplots(
        2,
        1,
        figsize=(8.5, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )
    ax_spec.plot(spec.wave, spec.flux, color="0.25", lw=1.0, label="Observed")
    ax_spec.plot(
        spec.wave,
        stellar_result.stellar_model,
        color="#d62728",
        lw=1.3,
        label="pPXF stellar model",
    )
    if stellar_result.agn_model is not None:
        ax_spec.plot(
            spec.wave,
            stellar_result.agn_model,
            color="#1f77b4",
            lw=1.0,
            ls="--",
            label="AGN-like featureless continuum",
        )
    if stellar_result.total_model is not None:
        ax_spec.plot(
            spec.wave,
            stellar_result.total_model,
            color="#2ca02c",
            lw=1.0,
            alpha=0.8,
            label="pPXF total model",
        )
    ax_spec.set_ylabel("Flux")
    ax_spec.legend(loc="best", frameon=False)

    ax_resid.axhline(0.0, color="0.55", lw=1.0)
    ax_resid.plot(spec.wave, stellar_result.residual_flux, color="0.15", lw=1.0)
    ax_resid.set_xlabel("Rest wavelength (Angstrom)")
    ax_resid.set_ylabel("Residual")

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
    return fig


def _as_template_matrix(template_flux: ArrayLike, n_pixels: int) -> np.ndarray:
    templates = np.asarray(template_flux, dtype=float)
    if templates.ndim == 1:
        if len(templates) != n_pixels:
            raise ValueError("1D template_flux length must match template_wave.")
        return templates[:, None]
    if templates.ndim != 2:
        raise ValueError("template_flux must be one- or two-dimensional.")
    if templates.shape[0] == n_pixels:
        return templates
    if templates.shape[1] == n_pixels:
        return templates.T
    raise ValueError(
        "template_flux must have shape (n_pixels, n_templates) or "
        "(n_templates, n_pixels)."
    )


def _prepare_templates_for_ppxf(
    template_wave: np.ndarray,
    template_flux: np.ndarray,
) -> dict[str, Any]:
    flux = np.asarray(template_flux, dtype=float).copy()
    flux[~np.isfinite(flux)] = np.nan
    bad_pixel_fraction = np.mean((~np.isfinite(flux)) | (flux == 0), axis=1)
    good_pixel_mask = bad_pixel_fraction <= 0.5
    n_bad_pixels_removed = int(np.count_nonzero(~good_pixel_mask))
    wave = template_wave[good_pixel_mask]
    flux = flux[good_pixel_mask, :]

    finite_positive = np.isfinite(flux) & (flux > 0)
    finite_fraction = np.mean(np.isfinite(flux), axis=0)
    template_norms = np.full(flux.shape[1], np.nan, dtype=float)
    keep_templates = np.zeros(flux.shape[1], dtype=bool)
    for index in range(flux.shape[1]):
        values = flux[finite_positive[:, index], index]
        if values.size:
            template_norms[index] = float(np.nanmedian(values))
        keep_templates[index] = (
            np.isfinite(template_norms[index])
            and template_norms[index] > 0
            and finite_fraction[index] >= 0.8
        )

    n_templates_dropped = int(np.count_nonzero(~keep_templates))
    if not np.any(keep_templates):
        raise ValueError("All stellar templates were rejected during cleaning.")

    flux = flux[:, keep_templates]
    template_norms = template_norms[keep_templates]

    for index in range(flux.shape[1]):
        bad = ~np.isfinite(flux[:, index])
        if np.any(bad):
            good = ~bad
            if np.count_nonzero(good) < 2:
                flux[bad, index] = template_norms[index]
            else:
                flux[bad, index] = np.interp(wave[bad], wave[good], flux[good, index])
        flux[:, index] /= template_norms[index]

    return {
        "wave": wave,
        "flux": flux,
        "template_norms": template_norms,
        "n_templates_dropped": n_templates_dropped,
        "n_bad_pixels_removed": n_bad_pixels_removed,
    }


def _make_powerlaw_templates(
    wave: np.ndarray,
    slopes: tuple[float, ...],
) -> np.ndarray:
    pivot = float(np.nanmedian(wave))
    if not np.isfinite(pivot) or pivot <= 0:
        raise ValueError("Cannot build AGN power-law templates without valid wavelengths.")
    templates = []
    for slope in slopes:
        template = (wave / pivot) ** float(slope)
        template /= np.nanmedian(template)
        templates.append(template)
    return np.column_stack(templates)


def _model_on_original_grid(
    original_wave: np.ndarray,
    fit_wave: np.ndarray,
    log_wave: np.ndarray,
    log_model: np.ndarray,
) -> np.ndarray:
    log_model, log_wave = _trim_to_common_length(log_model, log_wave)
    fit_grid_model = np.interp(fit_wave, log_wave, log_model)
    model = np.full_like(original_wave, np.nan, dtype=float)
    overlap = (original_wave >= fit_wave[0]) & (original_wave <= fit_wave[-1])
    model[overlap] = np.interp(original_wave[overlap], fit_wave, fit_grid_model)
    return model


def _trim_to_common_length(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    min_length = min(len(array) for array in arrays)
    return tuple(np.asarray(array)[:min_length] for array in arrays)


def _warn_if_ppxf_setup_suspicious(
    n_templates: int,
    observed_min: float,
    observed_max: float,
    overlap_min: float,
    overlap_max: float,
) -> None:
    if n_templates < 20:
        warnings.warn(
            f"Only {n_templates} stellar templates were provided; pPXF results "
            "may be poorly constrained.",
            RuntimeWarning,
            stacklevel=2,
        )
    observed_span = observed_max - observed_min
    overlap_span = overlap_max - overlap_min
    if observed_span <= 0 or overlap_span <= 0 or overlap_span / observed_span < 0.5:
        warnings.warn(
            "Spectrum/template wavelength overlap is small or invalid: "
            f"observed=({observed_min:.3f}, {observed_max:.3f}), "
            f"overlap=({overlap_min:.3f}, {overlap_max:.3f}).",
            RuntimeWarning,
            stacklevel=2,
        )


def _print_ppxf_diagnostics(diagnostics: dict[str, Any]) -> None:
    template_range = diagnostics["template_wave_range"]
    observed_range = diagnostics["observed_spectrum_wavelength_range"]
    overlap_range = diagnostics["overlap_wavelength_range"]
    print("=== EchoSpec pPXF diagnostics ===")
    print(f"template_wave range: {template_range[0]:.8g} - {template_range[1]:.8g}")
    print(f"template_wave median step: {diagnostics['template_wave_median_step']:.8g}")
    print(f"template_flux shape: {diagnostics['template_flux_shape']}")
    print(f"n_templates: {diagnostics['n_templates']}")
    print(f"template_flux after cut shape: {diagnostics['template_flux_after_cut_shape']}")
    print(f"number of templates kept: {diagnostics['n_templates_kept']}")
    print(f"number of stellar templates: {diagnostics['n_stellar_templates']}")
    print(
        "number of AGN power-law templates: "
        f"{diagnostics['n_agn_powerlaw_templates']}"
    )
    print(f"number of templates dropped: {diagnostics['n_templates_dropped']}")
    print(
        "number of bad wavelength pixels removed: "
        f"{diagnostics['n_bad_template_wavelength_pixels_removed']}"
    )
    print(
        "template_flux min/max/median: "
        f"{diagnostics['template_flux_min']:.8g} / "
        f"{diagnostics['template_flux_max']:.8g} / "
        f"{diagnostics['template_flux_median']:.8g}"
    )
    print(f"galaxy_norm: {diagnostics['galaxy_norm']:.8g}")
    print(
        "template_norm min/median/max: "
        f"{diagnostics['template_norm_min']:.8g} / "
        f"{diagnostics['template_norm_median']:.8g} / "
        f"{diagnostics['template_norm_max']:.8g}"
    )
    print(
        "observed spectrum wavelength range: "
        f"{observed_range[0]:.8g} - {observed_range[1]:.8g}"
    )
    print(
        "overlap wavelength range: "
        f"{overlap_range[0]:.8g} - {overlap_range[1]:.8g}"
    )
    print(f"number of pixels used by pPXF: {diagnostics['n_pixels_used_by_ppxf']}")
    print(
        "number of goodpixels after emission-line masking: "
        f"{diagnostics['n_goodpixels_after_emission_line_masking']}"
    )


def _print_ppxf_result_diagnostics(diagnostics: dict[str, Any]) -> None:
    print(f"pPXF chi2: {diagnostics['ppxf_chi2']:.8g}")
    print(f"pPXF velocity: {diagnostics['ppxf_velocity']:.8g}")
    print(f"pPXF sigma: {diagnostics['ppxf_sigma']:.8g}")
    print(
        "pPXF mdegree initial/final: "
        f"{diagnostics.get('initial_ppxf_mdegree')} / "
        f"{diagnostics.get('final_ppxf_mdegree')}"
    )
    print(f"pPXF weights nonzero count: {diagnostics['ppxf_weights_nonzero_count']}")
    print(
        "pPXF stellar / AGN nonzero weights: "
        f"{diagnostics['stellar_weights_nonzero_count']} / "
        f"{diagnostics['agn_powerlaw_weights_nonzero_count']}"
    )
    print(f"AGN power-law weight sum: {diagnostics['agn_powerlaw_weight_sum']:.8g}")


def _first_existing_key(cache: Any, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in cache:
            return key
    return None


def _normalize_templates(templates: np.ndarray) -> np.ndarray:
    normalized = templates.copy()
    for index in range(normalized.shape[1]):
        scale = float(np.nanmedian(normalized[:, index]))
        if np.isfinite(scale) and scale != 0:
            normalized[:, index] /= scale
    return normalized


def _make_goodpixels(
    wave: np.ndarray,
    noise: np.ndarray,
    mask_emission_lines: bool,
    mask_width_kms: float,
) -> np.ndarray:
    good = np.isfinite(wave) & np.isfinite(noise) & (noise > 0)
    if mask_emission_lines:
        for center in DEFAULT_EMISSION_LINE_CENTERS:
            half_width = center * mask_width_kms / C_KMS
            good &= np.abs(wave - center) > half_width
    return np.where(good)[0]


__all__ = [
    "StellarFitResult",
    "default_ppxf_template_cache_path",
    "fit_stellar_ppxf",
    "load_default_ppxf_templates",
    "subtract_stellar_continuum",
    "plot_stellar_fit",
]
