"""Fe II template loading, normalization, broadening, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import numpy as np

C_KMS = 299792.458
FEII_NORM_WINDOW = (4434.0, 4684.0)


@dataclass(frozen=True)
class FeIITemplate:
    """Normalized optical Fe II template."""

    wave: np.ndarray
    flux: np.ndarray
    diagnostics: dict[str, Any]


def load_feii_template(
    feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    feii_template_path: str | Path | None = None,
) -> FeIITemplate | None:
    """Load and normalize a Fe II template."""

    if feii_template is None and feii_template_path is None:
        return None
    if feii_template is None:
        data = np.loadtxt(Path(feii_template_path).expanduser())
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Fe II template file must contain wavelength and flux columns.")
        wave = data[:, 0]
        flux = data[:, 1]
    elif isinstance(feii_template, tuple):
        wave, flux = feii_template
    else:
        data = np.asarray(feii_template, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Fe II template array must have wavelength and flux columns.")
        wave = data[:, 0]
        flux = data[:, 1]

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0)
    wave = wave[valid]
    flux = flux[valid]
    if len(wave) < 2:
        raise ValueError("Fe II template has too few valid pixels.")

    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    wave = wave[unique]
    flux = flux[unique]
    if len(wave) < 2:
        raise ValueError("Fe II template has too few unique wavelength pixels.")

    before = _array_stats(flux)
    positive = flux > 0.0
    max_flux = float(np.nanmax(flux[positive])) if np.any(positive) else float("nan")
    if np.isfinite(max_flux) and max_flux > 0.0:
        robust_positive = positive & (flux > 1.0e-3 * max_flux)
        if np.count_nonzero(robust_positive) > 0:
            positive = robust_positive
    if np.count_nonzero(positive) == 0:
        raise ValueError("Fe II template has no positive finite flux values.")
    norm = float(np.nanmedian(flux[positive]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Fe II template normalization failed.")
    normalized = flux / norm
    diagnostics = {
        "template_wave_min": float(np.nanmin(wave)),
        "template_wave_max": float(np.nanmax(wave)),
        "template_n_pixels": int(len(wave)),
        "normalization": norm,
        "template_flux_before": before,
        "template_flux_after_normalization": _array_stats(normalized),
    }
    return FeIITemplate(wave=wave, flux=normalized, diagnostics=diagnostics)


def broaden_shift_feii_template(
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    output_wave: np.ndarray,
    width_kms: float = 0.0,
    shift_kms: float = 0.0,
    norm_region: tuple[float, float] = FEII_NORM_WINDOW,
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Return a broadened, shifted Fe II unit-flux shape.

    In the default ``extra`` mode, ``width_kms`` is the additional Gaussian
    broadening applied to the empirical Fe II template. In ``target`` mode,
    ``width_kms`` is interpreted as the final target width and the template
    intrinsic width is subtracted in quadrature.
    """

    template_wave = np.asarray(template_wave, dtype=float)
    template_flux = np.asarray(template_flux, dtype=float)
    output_wave = np.asarray(output_wave, dtype=float)
    valid = np.isfinite(template_wave) & np.isfinite(template_flux) & (template_wave > 0.0)
    template_wave = template_wave[valid]
    template_flux = template_flux[valid]
    if len(template_wave) < 2:
        result = np.zeros_like(output_wave, dtype=float)
        diagnostics = _feii_diagnostics(
            dloglam=float("nan"),
            width_kms=width_kms,
            shift_kms=shift_kms,
            feii_template_sigma_kms=feii_template_sigma_kms,
            sigma_conv_kms=float("nan"),
            sigma_pix=float("nan"),
            output=result,
            before=template_flux,
            broadened=np.array([], dtype=float),
            norm_region=norm_region,
            norm_region_integral=float("nan"),
        )
        return (result, diagnostics) if return_diagnostics else result

    order = np.argsort(template_wave)
    template_wave = template_wave[order]
    template_flux = template_flux[order]
    unique = np.concatenate(([True], np.diff(template_wave) > 0.0))
    template_wave = template_wave[unique]
    template_flux = template_flux[unique]
    template_flux = _normalize_flux(template_flux)

    loglam = np.log(template_wave)
    dloglam = float(np.nanmedian(np.diff(loglam)))
    if not np.isfinite(dloglam) or dloglam <= 0.0:
        raise ValueError("Fe II template log-lambda grid is not monotonic.")
    uniform_loglam = np.arange(loglam[0], loglam[-1] + 0.5 * dloglam, dloglam)
    template_log_flux = np.interp(uniform_loglam, loglam, template_flux, left=0.0, right=0.0)

    width_kms = float(width_kms)
    shift_kms = float(shift_kms)
    feii_template_sigma_kms = max(float(feii_template_sigma_kms), 0.0)
    if feii_broadening_mode not in {"extra", "target"}:
        raise ValueError("feii_broadening_mode must be 'extra' or 'target'.")
    if feii_broadening_mode == "extra":
        sigma_conv_kms = max(width_kms, 0.0)
    else:
        sigma_conv_kms = float(
            np.sqrt(max(width_kms * width_kms - feii_template_sigma_kms * feii_template_sigma_kms, 0.0))
        )
    if feii_broadening_mode == "target" and width_kms <= feii_template_sigma_kms:
        warnings.warn(
            "Requested Fe II sigma is not larger than the template intrinsic sigma; "
            "skipping additional broadening.",
            RuntimeWarning,
            stacklevel=2,
        )

    sigma_loglam = sigma_conv_kms / C_KMS
    sigma_pix = sigma_loglam / dloglam
    if sigma_conv_kms > 1.0e-6 and (sigma_pix < 0.5 or sigma_pix > 500.0):
        warnings.warn(
            f"Fe II broadening sigma_pix={sigma_pix:.3g} may be numerically unstable.",
            RuntimeWarning,
            stacklevel=2,
        )

    broadened = template_log_flux
    if np.isfinite(sigma_pix) and sigma_conv_kms > 1.0e-6 and sigma_pix > 0.0:
        from scipy.ndimage import gaussian_filter1d

        broadened = gaussian_filter1d(template_log_flux, sigma_pix, mode="constant")

    output = np.zeros_like(output_wave, dtype=float)
    valid_out = np.isfinite(output_wave) & (output_wave > 0.0)
    if np.any(valid_out):
        shifted_output_loglam = np.log(output_wave[valid_out]) - np.log1p(
            shift_kms / C_KMS
        )
        output[valid_out] = np.interp(
            shifted_output_loglam,
            uniform_loglam,
            broadened,
            left=0.0,
            right=0.0,
        )
    output[~np.isfinite(output)] = 0.0

    norm_mask = (
        np.isfinite(output_wave)
        & np.isfinite(output)
        & (output_wave >= norm_region[0])
        & (output_wave <= norm_region[1])
    )
    if np.count_nonzero(norm_mask) < 2:
        raise ValueError("Fe II normalization region has too few valid pixels.")
    norm_region_integral = float(np.trapz(output[norm_mask], output_wave[norm_mask]))
    if not np.isfinite(norm_region_integral) or norm_region_integral <= 0.0:
        diagnostics = _feii_diagnostics(
            dloglam=dloglam,
            width_kms=width_kms,
            shift_kms=shift_kms,
            feii_template_sigma_kms=feii_template_sigma_kms,
            feii_broadening_mode=feii_broadening_mode,
            sigma_conv_kms=sigma_conv_kms,
            sigma_pix=sigma_pix,
            output=output,
            before=template_flux,
            broadened=broadened,
            norm_region=norm_region,
            norm_region_integral=norm_region_integral,
        )
        raise ValueError(
            "Fe II normalization region integral is not positive; "
            f"diagnostics={diagnostics}"
        )
    unit_output = output / norm_region_integral

    diagnostics = _feii_diagnostics(
        dloglam=dloglam,
        width_kms=width_kms,
        shift_kms=shift_kms,
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
        sigma_conv_kms=sigma_conv_kms,
        sigma_pix=sigma_pix,
        output=unit_output,
        before=template_flux,
        broadened=broadened,
        norm_region=norm_region,
        norm_region_integral=norm_region_integral,
    )
    return (unit_output, diagnostics) if return_diagnostics else unit_output


def feii_unit_flux_template(
    output_wave: np.ndarray,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    width_kms: float,
    shift_kms: float = 0.0,
    feii_template_sigma_kms: float = 0.0,
    feii_broadening_mode: str = "extra",
    norm_window: tuple[float, float] = FEII_NORM_WINDOW,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible wrapper for DASpec-style Fe II unit normalization."""

    return broaden_shift_feii_template(
        template_wave,
        template_flux,
        output_wave,
        width_kms=width_kms,
        shift_kms=shift_kms,
        norm_region=norm_window,
        feii_template_sigma_kms=feii_template_sigma_kms,
        feii_broadening_mode=feii_broadening_mode,
        return_diagnostics=return_diagnostics,
    )


def inspect_feii_template(
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect Fe II template grid, flux values, and intrinsic feature widths."""

    wave = np.asarray(template_wave, dtype=float)
    flux = np.asarray(template_flux, dtype=float)
    finite_pair = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0)
    finite_fraction = float(np.mean(finite_pair)) if wave.size else 0.0
    wave = wave[finite_pair]
    flux = flux[finite_pair]
    if wave.size < 3:
        raise ValueError("Fe II template has too few finite pixels to inspect.")
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    wave = wave[unique]
    flux = flux[unique]

    dlambda = np.diff(wave)
    dloglam = np.diff(np.log(wave))
    median_dlambda = float(np.nanmedian(dlambda))
    median_dloglam = float(np.nanmedian(dloglam))
    grid_type = _infer_grid_type(dlambda, dloglam)
    flux_stats = _array_stats(flux)
    negative_fraction = float(np.mean(flux < 0.0)) if flux.size else 0.0
    zero_fraction = float(np.mean(flux == 0.0)) if flux.size else 0.0
    intrinsic_width = estimate_feii_template_intrinsic_width(wave, flux)

    regions = {
        "blue_4434_4684": (4434.0, 4684.0),
        "pedestal_4920_5050": (4920.0, 5050.0),
        "red_5100_5600": (5100.0, 5600.0),
    }
    region_widths: dict[str, Any] = {}
    all_fwhm_kms: list[float] = []
    all_fwhm_angstrom: list[float] = []
    for name, region in regions.items():
        widths = _estimate_region_feature_widths(wave, flux, region)
        region_widths[name] = widths
        all_fwhm_kms.extend(width["fwhm_kms"] for width in widths)
        all_fwhm_angstrom.extend(width["fwhm_angstrom"] for width in widths)

    estimated_fwhm_kms = (
        float(np.nanmedian(all_fwhm_kms)) if len(all_fwhm_kms) else float("nan")
    )
    estimated_fwhm_angstrom = (
        float(np.nanmedian(all_fwhm_angstrom)) if len(all_fwhm_angstrom) else float("nan")
    )
    estimated_sigma_kms = (
        float(estimated_fwhm_kms / 2.354820045)
        if np.isfinite(estimated_fwhm_kms)
        else float("nan")
    )
    warning_messages: list[str] = []
    if np.isfinite(estimated_sigma_kms) and estimated_sigma_kms > 500.0:
        message = (
            "Fe II template appears intrinsically broadened. Do not interpret "
            "feii_width_kms as purely additional broadening."
        )
        warning_messages.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    output_path = None
    if output_dir is not None:
        output_path = Path(output_dir) / "feii_template_inspection.png"
        _plot_feii_template_inspection(wave, flux, output_path)

    return {
        "wave_min": float(np.nanmin(wave)),
        "wave_max": float(np.nanmax(wave)),
        "template_grid_type": grid_type,
        "template_median_dlambda": median_dlambda,
        "template_median_dloglam": median_dloglam,
        "flux_min": flux_stats["min"],
        "flux_max": flux_stats["max"],
        "flux_median": flux_stats["median"],
        "finite_fraction": finite_fraction,
        "negative_fraction": negative_fraction,
        "zero_fraction": zero_fraction,
        "feature_widths": region_widths,
        "estimated_template_fwhm_angstrom": estimated_fwhm_angstrom,
        "estimated_template_fwhm_kms": estimated_fwhm_kms,
        "estimated_template_sigma_kms": estimated_sigma_kms,
        **intrinsic_width,
        "warnings": warning_messages,
        "diagnostic_plot": str(output_path) if output_path is not None else "",
    }


def estimate_feii_template_intrinsic_width(
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    regions: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Estimate intrinsic hump widths in an empirical blended Fe II template."""

    wave = np.asarray(template_wave, dtype=float)
    flux = np.asarray(template_flux, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0)
    wave = wave[valid]
    flux = flux[valid]
    if wave.size < 3:
        raise ValueError("Fe II template has too few finite pixels to estimate width.")
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    unique = np.concatenate(([True], np.diff(wave) > 0.0))
    wave = wave[unique]
    flux = flux[unique]

    if regions is None:
        regions = {
            "blue": (4434.0, 4684.0),
            "central": (4920.0, 5050.0),
            "red": (5100.0, 5600.0),
        }

    diagnostics: dict[str, Any] = {
        "warning": (
            "Fe II effective width is an approximate diagnostic for an empirical "
            "blended template, not a direct line FWHM."
        )
    }
    region_sigmas: list[float] = []
    for name, region in regions.items():
        widths = _estimate_region_feature_widths(wave, flux, region)
        fwhm_values = np.asarray([item["fwhm_kms"] for item in widths], dtype=float)
        sigma_values = np.asarray([item["sigma_kms"] for item in widths], dtype=float)
        fwhm_values = fwhm_values[np.isfinite(fwhm_values)]
        sigma_values = sigma_values[np.isfinite(sigma_values)]
        median_fwhm = float(np.nanmedian(fwhm_values)) if fwhm_values.size else float("nan")
        median_sigma = float(np.nanmedian(sigma_values)) if sigma_values.size else float("nan")
        diagnostics[f"{name}_template_fwhm_kms"] = median_fwhm
        diagnostics[f"{name}_template_sigma_kms"] = median_sigma
        diagnostics[f"{name}_template_hump_widths"] = widths
        if np.isfinite(median_sigma):
            region_sigmas.append(median_sigma)

    diagnostics["global_template_sigma_kms"] = (
        float(np.nanmedian(region_sigmas)) if region_sigmas else float("nan")
    )
    return diagnostics


def compute_feii_effective_width(
    template_sigma_kms: float,
    smooth_sigma_kms: float = 0.0,
    extra_sigma_kms: float = 0.0,
) -> dict[str, float | str]:
    """Approximate Fe II effective width from template, smoothing, and extra terms."""

    terms = np.asarray(
        [
            max(float(template_sigma_kms), 0.0),
            max(float(smooth_sigma_kms), 0.0),
            max(float(extra_sigma_kms), 0.0),
        ],
        dtype=float,
    )
    effective_sigma = float(np.sqrt(np.sum(terms**2)))
    return {
        "feii_template_intrinsic_sigma_kms": float(terms[0]),
        "feii_smooth_sigma_kms": float(terms[1]),
        "feii_extra_sigma_kms": float(terms[2]),
        "feii_effective_sigma_kms": effective_sigma,
        "feii_effective_fwhm_kms": float(2.3548 * effective_sigma),
        "warning": (
            "Fe II effective width is an approximate diagnostic for an empirical "
            "blended template, not a direct line FWHM."
        ),
    }


def _infer_grid_type(dlambda: np.ndarray, dloglam: np.ndarray) -> str:
    dlambda = dlambda[np.isfinite(dlambda) & (dlambda > 0.0)]
    dloglam = dloglam[np.isfinite(dloglam) & (dloglam > 0.0)]
    if dlambda.size == 0 or dloglam.size == 0:
        return "unknown"
    linear_scatter = float(np.nanstd(dlambda) / np.nanmedian(dlambda))
    log_scatter = float(np.nanstd(dloglam) / np.nanmedian(dloglam))
    if linear_scatter < 1.0e-3 and linear_scatter <= log_scatter:
        return "linear_wavelength"
    if log_scatter < 1.0e-3 and log_scatter < linear_scatter:
        return "log_wavelength"
    return "irregular"


def _estimate_region_feature_widths(
    wave: np.ndarray,
    flux: np.ndarray,
    region: tuple[float, float],
    max_peaks: int = 6,
) -> list[dict[str, float]]:
    mask = (wave >= region[0]) & (wave <= region[1])
    region_wave = wave[mask]
    region_flux = flux[mask]
    if region_wave.size < 5:
        return []
    continuum = float(np.nanpercentile(region_flux, 10.0))
    signal = region_flux - continuum
    signal[~np.isfinite(signal)] = 0.0
    peak_level = float(np.nanmax(signal))
    if not np.isfinite(peak_level) or peak_level <= 0.0:
        return []
    try:
        from scipy.signal import find_peaks, peak_widths

        prominence = max(0.05 * peak_level, 1.0e-12)
        peaks, props = find_peaks(signal, prominence=prominence)
        if peaks.size == 0:
            return []
        prominences = props.get("prominences", signal[peaks])
        order = np.argsort(prominences)[::-1][:max_peaks]
        peaks = peaks[order]
        widths_pix = peak_widths(signal, peaks, rel_height=0.5)[0]
    except Exception:
        peaks = _simple_local_peaks(signal, max_peaks=max_peaks)
        if peaks.size == 0:
            return []
        widths_pix = np.asarray([
            _manual_peak_fwhm_pixels(signal, int(peak)) for peak in peaks
        ])

    median_step = float(np.nanmedian(np.diff(region_wave)))
    widths: list[dict[str, float]] = []
    for peak, width_pix in zip(peaks, widths_pix):
        if not np.isfinite(width_pix) or width_pix <= 0.0:
            continue
        center = float(region_wave[int(peak)])
        fwhm_angstrom = float(width_pix * median_step)
        fwhm_kms = float(C_KMS * fwhm_angstrom / center)
        widths.append(
            {
                "center_wave": center,
                "peak_flux": float(region_flux[int(peak)]),
                "fwhm_angstrom": fwhm_angstrom,
                "fwhm_kms": fwhm_kms,
                "sigma_kms": float(fwhm_kms / 2.354820045),
            }
        )
    return widths


def _simple_local_peaks(signal: np.ndarray, max_peaks: int) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])
    peaks = np.where(is_peak)[0] + 1
    if peaks.size == 0:
        return peaks
    order = np.argsort(signal[peaks])[::-1]
    return peaks[order[:max_peaks]]


def _manual_peak_fwhm_pixels(signal: np.ndarray, peak: int) -> float:
    half = 0.5 * signal[peak]
    left = peak
    while left > 0 and signal[left] > half:
        left -= 1
    right = peak
    while right < signal.size - 1 and signal[right] > half:
        right += 1
    return float(max(right - left, 1))


def _plot_feii_template_inspection(
    wave: np.ndarray,
    flux: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = (
        ((4200.0, 5600.0), "original template 4200-5600"),
        ((4434.0, 4684.0), "blue Fe II 4434-4684"),
        ((4920.0, 5050.0), "[O III] pedestal Fe II 4920-5050"),
        ((5100.0, 5600.0), "red Fe II 5100-5600"),
    )
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10.0, 9.5),
        sharex=False,
        gridspec_kw={"hspace": 0.32},
    )
    for ax, (region, title) in zip(axes, panels):
        mask = (wave >= region[0]) & (wave <= region[1])
        ax.plot(wave[mask], flux[mask], color="0.15", lw=0.9)
        ax.set_xlim(region)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Flux")
    axes[-1].set_xlabel("Wavelength (Angstrom)")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _normalize_flux(flux: np.ndarray) -> np.ndarray:
    finite = np.asarray(flux, dtype=float)
    positive = np.isfinite(finite) & (finite > 0.0)
    if np.count_nonzero(positive) == 0:
        return np.zeros_like(finite)
    max_flux = float(np.nanmax(finite[positive]))
    if np.isfinite(max_flux) and max_flux > 0.0:
        robust_positive = positive & (finite > 1.0e-3 * max_flux)
        if np.count_nonzero(robust_positive) > 0:
            positive = robust_positive
    norm = float(np.nanmedian(finite[positive]))
    if not np.isfinite(norm) or norm <= 0.0:
        return np.zeros_like(finite)
    return finite / norm


def _feii_diagnostics(
    dloglam: float,
    width_kms: float,
    shift_kms: float,
    feii_template_sigma_kms: float,
    feii_broadening_mode: str,
    sigma_conv_kms: float,
    sigma_pix: float,
    output: np.ndarray,
    before: np.ndarray,
    broadened: np.ndarray,
    norm_region: tuple[float, float],
    norm_region_integral: float,
) -> dict[str, Any]:
    finite_output = np.isfinite(output)
    return {
        "dloglam": float(dloglam),
        "width_kms": float(width_kms),
        "feii_extra_sigma_kms": float(sigma_conv_kms),
        "feii_broadening_mode": feii_broadening_mode,
        "shift_kms": float(shift_kms),
        "sigma_kms": float(width_kms),
        "velocity_kms": float(shift_kms),
        "feii_template_sigma_kms": float(feii_template_sigma_kms),
        "sigma_conv_kms": float(sigma_conv_kms),
        "sigma_pix": float(sigma_pix),
        "norm_region": [float(norm_region[0]), float(norm_region[1])],
        "norm_region_integral": float(norm_region_integral),
        "norm_region_integral_before_normalization": float(norm_region_integral),
        "normalization_region": [float(norm_region[0]), float(norm_region[1])],
        "normalization_integral_4434_4684": float(norm_region_integral),
        "unit_template_integral_4434_4684": 1.0,
        "finite_fraction": float(np.mean(finite_output)) if output.size else 0.0,
        "output_finite_fraction": float(np.mean(finite_output)) if output.size else 0.0,
        "template_before_broadening": _array_stats(before),
        "template_after_broadening": _array_stats(broadened),
        "template_min_max_median_before": _array_stats(before),
        "template_min_max_median_after": _array_stats(broadened),
        "output": _array_stats(output),
        "unit_template": _array_stats(output),
    }


def _array_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "median": float("nan")}
    return {
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "median": float(np.nanmedian(finite)),
    }


__all__ = [
    "FEII_NORM_WINDOW",
    "FeIITemplate",
    "broaden_shift_feii_template",
    "compute_feii_effective_width",
    "estimate_feii_template_intrinsic_width",
    "feii_unit_flux_template",
    "inspect_feii_template",
    "load_feii_template",
]
