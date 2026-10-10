"""Optional Mg II + UV Fe II window fitting and model selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lmfit import Parameters, minimize

from echospec.feii import feii_unit_flux_template, load_feii_template
from echospec.spectrum import AGNSpectrum

C_KMS = 299792.458
MGII_WINDOW = (2200.0, 3090.0)
MGII_BLEND = 2798.75
MGII_2796 = 2796.35
MGII_2803 = 2803.53
UVFEII_NORM_WINDOW = (2200.0, 3090.0)
UVFEII_PREFIT_WINDOWS = ((2200.0, 2700.0), (2900.0, 3090.0))
UVFEII_MGII_MASK = (2700.0, 2900.0)


@dataclass(frozen=True)
class UVFeIIPrefitResult:
    """UV Fe II anchored-continuum prefit summary."""

    uvfeii_flux_window: float
    uvfeii_extra_sigma_kms: float
    uvfeii_shift_kms: float
    chi2: float
    snr: float
    continuum_params: dict[str, float]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uvfeii_flux_window": float(self.uvfeii_flux_window),
            "uvfeii_extra_sigma_kms": float(self.uvfeii_extra_sigma_kms),
            "uvfeii_shift_kms": float(self.uvfeii_shift_kms),
            "chi2": float(self.chi2),
            "snr": float(self.snr),
            "continuum_params": self.continuum_params,
            "diagnostics": self.diagnostics,
        }


def fit_mgii_uvfeii(
    spec: AGNSpectrum,
    window: tuple[float, float] = MGII_WINDOW,
    include_uv_feii: bool = True,
    include_mgii_broad: bool = True,
    include_mgii_narrow: bool = False,
    uv_feii_template: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    uv_feii_template_path: str | Path | None = None,
    broad_model: str = "double",
    narrow_mode: str = "blend",
    max_nfev: int | None = 3000,
) -> Any:
    """Fit Mg II + UV Fe II in a rest-frame 2200-3090 A window.

    Models are selected by BIC:
    M0 power-law only; M1 +UV Fe II; M2 +broad Mg II; M3 +narrow Mg II.
    """

    from echospec.fitting import EmissionFitResult

    if broad_model not in {"single", "double"}:
        raise ValueError("broad_model must be 'single' or 'double'.")
    if narrow_mode not in {"blend", "doublet"}:
        raise ValueError("narrow_mode must be 'blend' or 'doublet'.")
    window_spec = spec.select_window(*window).mask_bad_pixels()
    wave = np.asarray(window_spec.wave, dtype=float)
    flux = np.asarray(window_spec.flux, dtype=float)
    err = np.asarray(window_spec.err, dtype=float)
    if len(wave) < 5:
        raise ValueError("No valid pixels available in the Mg II fitting window.")

    template = load_feii_template(uv_feii_template, uv_feii_template_path)
    uv_template = None if template is None else (template.wave, template.flux)
    can_fit_uv = bool(include_uv_feii and uv_template is not None)
    prefit = (
        prefit_uv_feii(wave, flux, err, uv_template)
        if can_fit_uv
        else None
    )

    fits: dict[str, Any] = {}
    fits["M0"] = _fit_model(
        wave,
        flux,
        err,
        uv_template,
        include_uv_feii=False,
        include_mgii_broad=False,
        include_mgii_narrow=False,
        broad_model=broad_model,
        narrow_mode=narrow_mode,
        prefit=prefit,
        max_nfev=max_nfev,
    )
    fits["M1"] = _fit_model(
        wave,
        flux,
        err,
        uv_template,
        include_uv_feii=can_fit_uv,
        include_mgii_broad=False,
        include_mgii_narrow=False,
        broad_model=broad_model,
        narrow_mode=narrow_mode,
        prefit=prefit,
        max_nfev=max_nfev,
    )
    fits["M2"] = _fit_model(
        wave,
        flux,
        err,
        uv_template,
        include_uv_feii=can_fit_uv,
        include_mgii_broad=include_mgii_broad,
        include_mgii_narrow=False,
        broad_model=broad_model,
        narrow_mode=narrow_mode,
        prefit=prefit,
        max_nfev=max_nfev,
    )
    fits["M3"] = _fit_model(
        wave,
        flux,
        err,
        uv_template,
        include_uv_feii=can_fit_uv,
        include_mgii_broad=include_mgii_broad,
        include_mgii_narrow=include_mgii_narrow,
        broad_model=broad_model,
        narrow_mode=narrow_mode,
        prefit=prefit,
        max_nfev=max_nfev,
    )

    m0, m1, m2, m3 = (fits[name] for name in ("M0", "M1", "M2", "M3"))
    uv_delta_bic = _delta_bic(m0, m1)
    broad_delta_bic = _delta_bic(m1 if _accept_uv(m0, m1) else m0, m2)
    narrow_delta_bic = _delta_bic(m2, m3)
    uv_snr = _component_snr(wave, err, m1["components"].get("UV_FeII", np.zeros_like(wave)))
    broad_snr = _component_snr(wave, err, m2["components"].get("MgII_broad", np.zeros_like(wave)))
    narrow_snr = _component_snr(wave, err, m3["components"].get("MgII_narrow", np.zeros_like(wave)))
    broad_fwhm = _combined_fwhm_kms(wave, m2["components"].get("MgII_broad", np.zeros_like(wave)), MGII_BLEND)
    narrow_sigma = float(m3["params"].get("mgii_narrow_sigma", np.nan))

    accept_uv = bool(can_fit_uv and uv_delta_bic > 10.0 and uv_snr > 3.0)
    accept_broad = bool(include_mgii_broad and broad_delta_bic > 10.0 and broad_snr > 3.0 and broad_fwhm > 1000.0)
    secure_broad = bool(broad_delta_bic > 20.0 and broad_snr > 5.0 and broad_fwhm > 1200.0)
    accept_narrow = bool(include_mgii_narrow and narrow_delta_bic > 10.0 and narrow_snr > 3.0 and narrow_sigma < 1000.0)
    selected_name = "M0"
    if accept_uv:
        selected_name = "M1"
    if accept_broad:
        selected_name = "M2"
    if accept_broad and accept_narrow:
        selected_name = "M3"
    selected = fits[selected_name]

    fit_info = {
        "fit_mode": "mgii_uvfeii",
        "window": [float(window[0]), float(window[1])],
        "selected_model": selected_name,
        "model_bic": {name: float(item["bic"]) for name, item in fits.items()},
        "delta_bic_uv_feii": float(uv_delta_bic),
        "delta_bic_mgii_broad": float(broad_delta_bic),
        "delta_bic_mgii_narrow": float(narrow_delta_bic),
        "uv_feii_snr": float(uv_snr),
        "mgii_broad_snr": float(broad_snr),
        "mgii_narrow_snr": float(narrow_snr),
        "mgii_broad_fwhm": float(broad_fwhm),
        "accept_uv_feii": accept_uv,
        "accept_mgii_broad": accept_broad,
        "secure_mgii_broad": secure_broad,
        "accept_mgii_narrow": accept_narrow,
        "include_uv_feii_requested": bool(include_uv_feii),
        "include_mgii_broad_requested": bool(include_mgii_broad),
        "include_mgii_narrow_requested": bool(include_mgii_narrow),
        "flag_uv_feii_missing_template": bool(include_uv_feii and uv_template is None),
        "uv_feii_template_path": str(uv_feii_template_path) if uv_feii_template_path is not None else "",
        "broad_model": broad_model,
        "narrow_mode": narrow_mode,
        "uvfeii_prefit": None if prefit is None else prefit.to_dict(),
    }
    return EmissionFitResult(
        success=bool(selected["success"]),
        message=str(selected["message"]),
        params=selected["params"],
        line_flux=selected["line_flux"],
        line_flux_err={},
        model_flux=selected["model"],
        continuum_flux=selected["components"]["continuum"],
        residual_flux=flux - selected["model"],
        chi2=float(selected["chi2"]),
        redchi=float(selected["redchi"]),
        ndata=int(selected["ndata"]),
        nvarys=int(selected["nvarys"]),
        window_name="mgii_uv",
        fit_info=fit_info,
        component_flux=selected["components"],
    )


def fit_mgii_uv(*args: Any, **kwargs: Any) -> Any:
    """Backward-compatible alias for :func:`fit_mgii_uvfeii`."""

    if "include_narrow_mgii" in kwargs and "include_mgii_narrow" not in kwargs:
        kwargs["include_mgii_narrow"] = kwargs.pop("include_narrow_mgii")
    if "broad_mgii_model" in kwargs and "broad_model" not in kwargs:
        kwargs["broad_model"] = kwargs.pop("broad_mgii_model")
    return fit_mgii_uvfeii(*args, **kwargs)


def prefit_uv_feii(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    uv_feii_template: tuple[np.ndarray, np.ndarray],
    extra_sigma_grid: tuple[float, ...] = (0.0, 300.0, 600.0),
    shift_grid: tuple[float, ...] = (-400.0, 0.0, 400.0),
) -> UVFeIIPrefitResult:
    """Prefit UV Fe II in Fe II-dominated windows with Mg II masked."""

    continuum_params = _estimate_powerlaw_continuum(wave, flux, mask_mgii=True)
    continuum = _powerlaw(wave, continuum_params)
    residual = flux - continuum
    fit_mask = _uvfeii_prefit_mask(wave) & np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    best: dict[str, Any] | None = None
    for sigma in extra_sigma_grid:
        for shift in shift_grid:
            unit = _uv_feii_unit(wave, uv_feii_template, sigma, shift)
            denom = np.sum((unit[fit_mask] / err[fit_mask]) ** 2)
            if denom <= 0.0:
                amp = 0.0
            else:
                amp = max(float(np.sum(residual[fit_mask] * unit[fit_mask] / err[fit_mask] ** 2) / denom), 0.0)
            model = amp * unit
            chi2 = _chi2(residual[fit_mask] - model[fit_mask], err[fit_mask])
            if best is None or chi2 < best["chi2"]:
                best = {"flux": amp, "sigma": sigma, "shift": shift, "chi2": chi2, "model": model}
    if best is None:
        best = {"flux": 0.0, "sigma": 0.0, "shift": 0.0, "chi2": float("nan"), "model": np.zeros_like(wave)}
    snr = _component_snr(wave, err, best["model"])
    return UVFeIIPrefitResult(
        uvfeii_flux_window=float(best["flux"]),
        uvfeii_extra_sigma_kms=float(best["sigma"]),
        uvfeii_shift_kms=float(best["shift"]),
        chi2=float(best["chi2"]),
        snr=float(snr),
        continuum_params=continuum_params,
        diagnostics={
            "prefit_windows": [list(item) for item in UVFEII_PREFIT_WINDOWS],
            "masked_mgii_region": list(UVFEII_MGII_MASK),
            "fit_mask_npix": int(np.count_nonzero(fit_mask)),
            "extra_sigma_grid": list(extra_sigma_grid),
            "shift_grid": list(shift_grid),
        },
    )


def _fit_model(
    wave: np.ndarray,
    flux: np.ndarray,
    err: np.ndarray,
    uv_template: tuple[np.ndarray, np.ndarray] | None,
    include_uv_feii: bool,
    include_mgii_broad: bool,
    include_mgii_narrow: bool,
    broad_model: str,
    narrow_mode: str,
    prefit: UVFeIIPrefitResult | None,
    max_nfev: int | None,
) -> dict[str, Any]:
    params = _initial_params(
        wave,
        flux,
        uv_template,
        include_uv_feii,
        include_mgii_broad,
        include_mgii_narrow,
        broad_model,
        narrow_mode,
        prefit,
    )
    fit = minimize(
        lambda p: (flux - _model(wave, p, uv_template, include_uv_feii, include_mgii_broad, include_mgii_narrow, broad_model, narrow_mode)) / err,
        params,
        max_nfev=max_nfev,
    )
    model = _model(wave, fit.params, uv_template, include_uv_feii, include_mgii_broad, include_mgii_narrow, broad_model, narrow_mode)
    residual = flux - model
    chi2 = _chi2(residual, err)
    nvarys = int(getattr(fit, "nvarys", 0))
    ndata = int(len(wave))
    components = _components(wave, fit.params, uv_template, include_uv_feii, include_mgii_broad, include_mgii_narrow, broad_model, narrow_mode)
    return {
        "success": bool(fit.success),
        "message": str(fit.message),
        "params": {name: float(param.value) for name, param in fit.params.items()},
        "line_flux": _line_flux(fit.params, include_uv_feii, include_mgii_broad, include_mgii_narrow, broad_model),
        "model": model,
        "components": components,
        "chi2": chi2,
        "redchi": chi2 / max(ndata - nvarys, 1),
        "bic": chi2 + nvarys * np.log(max(ndata, 1)),
        "ndata": ndata,
        "nvarys": nvarys,
    }


def _initial_params(
    wave: np.ndarray,
    flux: np.ndarray,
    uv_template: tuple[np.ndarray, np.ndarray] | None,
    include_uv_feii: bool,
    include_mgii_broad: bool,
    include_mgii_narrow: bool,
    broad_model: str,
    narrow_mode: str,
    prefit: UVFeIIPrefitResult | None,
) -> Parameters:
    cont = _estimate_powerlaw_continuum(wave, flux, mask_mgii=True)
    continuum = _powerlaw(wave, cont)
    residual = flux - continuum
    params = Parameters()
    params.add("pl_amp", value=cont["amp"], min=0.0)
    params.add("pl_alpha", value=cont["alpha"], min=-8.0, max=4.0)
    if include_uv_feii and uv_template is not None:
        params.add("uvfeii_flux_window", value=max(0.0 if prefit is None else prefit.uvfeii_flux_window, 1.0e-12), min=0.0)
        params.add("uvfeii_shift_kms", value=0.0 if prefit is None else prefit.uvfeii_shift_kms, min=-1500.0, max=1500.0)
        params.add("uvfeii_extra_sigma_kms", value=0.0 if prefit is None else prefit.uvfeii_extra_sigma_kms, min=0.0, max=1500.0, vary=False)
        params.add("uvfeii_smooth_sigma_kms", value=300.0, min=0.0, max=1500.0, vary=False)
    if include_mgii_broad:
        broad_flux = max(_positive_integral(wave, residual, MGII_BLEND, 45.0), 1.0e-12)
        params.add("mgii_broad_1_flux", value=broad_flux, min=0.0)
        params.add("mgii_broad_1_velocity", value=0.0, min=-5000.0, max=5000.0)
        params.add("mgii_broad_1_sigma", value=1800.0, min=800.0, max=10000.0)
        if broad_model == "double":
            params.add("mgii_broad_2_flux", value=0.5 * broad_flux, min=0.0)
            params.add("mgii_broad_2_velocity", value=0.0, min=-5000.0, max=5000.0)
            params.add("mgii_broad_2_sigma", value=4500.0, min=800.0, max=10000.0)
    if include_mgii_narrow:
        params.add("mgii_narrow_flux", value=max(0.1 * _positive_integral(wave, residual, MGII_BLEND, 20.0), 1.0e-12), min=0.0)
        params.add("mgii_narrow_velocity", value=0.0, min=-800.0, max=800.0)
        params.add("mgii_narrow_sigma", value=250.0, min=50.0, max=1000.0)
    return params


def _model(
    wave: np.ndarray,
    params: Parameters,
    uv_template: tuple[np.ndarray, np.ndarray] | None,
    include_uv_feii: bool,
    include_mgii_broad: bool,
    include_mgii_narrow: bool,
    broad_model: str,
    narrow_mode: str,
) -> np.ndarray:
    model = _powerlaw(wave, {"amp": params["pl_amp"].value, "alpha": params["pl_alpha"].value})
    if include_uv_feii and uv_template is not None and "uvfeii_flux_window" in params:
        model += _uv_feii_component(wave, params, uv_template)
    if include_mgii_broad and "mgii_broad_1_flux" in params:
        model += _mgii_broad(wave, params, broad_model)
    if include_mgii_narrow and "mgii_narrow_flux" in params:
        model += _mgii_narrow(wave, params, narrow_mode)
    return model


def _components(
    wave: np.ndarray,
    params: Parameters,
    uv_template: tuple[np.ndarray, np.ndarray] | None,
    include_uv_feii: bool,
    include_mgii_broad: bool,
    include_mgii_narrow: bool,
    broad_model: str,
    narrow_mode: str,
) -> dict[str, np.ndarray]:
    components = {"continuum": _powerlaw(wave, {"amp": params["pl_amp"].value, "alpha": params["pl_alpha"].value})}
    if include_uv_feii and uv_template is not None and "uvfeii_flux_window" in params:
        components["UV_FeII"] = _uv_feii_component(wave, params, uv_template)
    if include_mgii_broad and "mgii_broad_1_flux" in params:
        components["MgII_broad"] = _mgii_broad(wave, params, broad_model)
    if include_mgii_narrow and "mgii_narrow_flux" in params:
        components["MgII_narrow"] = _mgii_narrow(wave, params, narrow_mode)
    return components


def _line_flux(
    params: Parameters,
    include_uv_feii: bool,
    include_mgii_broad: bool,
    include_mgii_narrow: bool,
    broad_model: str,
) -> dict[str, float]:
    flux: dict[str, float] = {}
    if include_uv_feii and "uvfeii_flux_window" in params:
        flux["UV_FeII_2200_3090"] = float(params["uvfeii_flux_window"].value)
    if include_mgii_broad and "mgii_broad_1_flux" in params:
        broad = float(params["mgii_broad_1_flux"].value)
        flux["MgII_broad_1"] = broad
        if broad_model == "double" and "mgii_broad_2_flux" in params:
            flux["MgII_broad_2"] = float(params["mgii_broad_2_flux"].value)
            broad += flux["MgII_broad_2"]
        flux["MgII_broad"] = broad
    if include_mgii_narrow and "mgii_narrow_flux" in params:
        flux["MgII_narrow"] = float(params["mgii_narrow_flux"].value)
    return flux


def _mgii_broad(wave: np.ndarray, params: Parameters, model: str) -> np.ndarray:
    profile = _gaussian_integrated(
        wave,
        MGII_BLEND,
        params["mgii_broad_1_flux"].value,
        params["mgii_broad_1_velocity"].value,
        params["mgii_broad_1_sigma"].value,
    )
    if model == "double" and "mgii_broad_2_flux" in params:
        profile += _gaussian_integrated(
            wave,
            MGII_BLEND,
            params["mgii_broad_2_flux"].value,
            params["mgii_broad_2_velocity"].value,
            params["mgii_broad_2_sigma"].value,
        )
    return profile


def _mgii_narrow(wave: np.ndarray, params: Parameters, mode: str) -> np.ndarray:
    if mode == "doublet":
        return _mgii_doublet(
            wave,
            params["mgii_narrow_flux"].value,
            params["mgii_narrow_velocity"].value,
            params["mgii_narrow_sigma"].value,
        )
    return _gaussian_integrated(
        wave,
        MGII_BLEND,
        params["mgii_narrow_flux"].value,
        params["mgii_narrow_velocity"].value,
        params["mgii_narrow_sigma"].value,
    )


def _mgii_doublet(wave: np.ndarray, total_flux: float, velocity: float, sigma_velocity: float) -> np.ndarray:
    return (
        _gaussian_integrated(wave, MGII_2796, 0.5 * total_flux, velocity, sigma_velocity)
        + _gaussian_integrated(wave, MGII_2803, 0.5 * total_flux, velocity, sigma_velocity)
    )


def _uv_feii_component(wave: np.ndarray, params: Parameters, uv_template: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    sigma = max(float(params["uvfeii_smooth_sigma_kms"].value), 0.0) + max(float(params["uvfeii_extra_sigma_kms"].value), 0.0)
    unit = _uv_feii_unit(wave, uv_template, sigma, float(params["uvfeii_shift_kms"].value))
    return float(params["uvfeii_flux_window"].value) * unit


def _uv_feii_unit(wave: np.ndarray, uv_template: tuple[np.ndarray, np.ndarray], sigma: float, shift: float) -> np.ndarray:
    template_wave, template_flux = uv_template
    try:
        return feii_unit_flux_template(
            wave,
            template_wave,
            template_flux,
            width_kms=sigma,
            shift_kms=shift,
            norm_window=UVFEII_NORM_WINDOW,
        )
    except Exception:
        return np.zeros_like(wave)


def _estimate_powerlaw_continuum(wave: np.ndarray, flux: np.ndarray, mask_mgii: bool = True) -> dict[str, float]:
    mask = (
        ((wave >= 2200.0) & (wave <= 2250.0))
        | ((wave >= 2650.0) & (wave <= 2675.0))
        | ((wave >= 3010.0) & (wave <= 3090.0))
    )
    if mask_mgii:
        mask &= ~((wave >= UVFEII_MGII_MASK[0]) & (wave <= UVFEII_MGII_MASK[1]))
    good = mask & np.isfinite(wave) & np.isfinite(flux) & (wave > 0.0) & (flux > 0.0)
    if np.count_nonzero(good) >= 2:
        alpha, log_amp_at_1 = np.polyfit(np.log(wave[good] / 2800.0), np.log(flux[good]), 1)
        return {"amp": float(np.exp(log_amp_at_1)), "alpha": float(alpha)}
    return {"amp": float(max(np.nanmedian(flux), 1.0e-12)), "alpha": 0.0}


def _powerlaw(wave: np.ndarray, params: dict[str, float]) -> np.ndarray:
    return float(params["amp"]) * np.power(np.clip(wave, 1.0, None) / 2800.0, float(params["alpha"]))


def _uvfeii_prefit_mask(wave: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(wave, dtype=bool)
    for start, stop in UVFEII_PREFIT_WINDOWS:
        mask |= (wave >= start) & (wave <= stop)
    mask &= ~((wave >= UVFEII_MGII_MASK[0]) & (wave <= UVFEII_MGII_MASK[1]))
    return mask


def _accept_uv(m0: dict[str, Any], m1: dict[str, Any]) -> bool:
    return _delta_bic(m0, m1) > 10.0 and _component_snr_from_fit(m1, "UV_FeII") > 3.0


def _delta_bic(base: dict[str, Any], candidate: dict[str, Any]) -> float:
    return float(base["bic"] - candidate["bic"])


def _component_snr_from_fit(fit: dict[str, Any], name: str) -> float:
    model = fit["components"].get(name)
    if model is None:
        return 0.0
    return float(np.sqrt(np.nansum(np.clip(model, 0.0, None) ** 2)))


def _component_snr(wave: np.ndarray, err: np.ndarray, component: np.ndarray) -> float:
    good = np.isfinite(wave) & np.isfinite(err) & np.isfinite(component) & (err > 0.0)
    if np.count_nonzero(good) < 2:
        return 0.0
    signal = float(np.trapz(np.clip(component[good], 0.0, None), wave[good]))
    noise = float(np.sqrt(np.sum(err[good] ** 2)) * np.nanmedian(np.diff(wave[good])))
    return signal / noise if noise > 0.0 else 0.0


def _combined_fwhm_kms(wave: np.ndarray, profile: np.ndarray, restwave: float) -> float:
    good = np.isfinite(wave) & np.isfinite(profile)
    if np.count_nonzero(good) < 3 or np.nanmax(profile[good]) <= 0.0:
        return float("nan")
    w = wave[good]
    p = profile[good]
    half = 0.5 * float(np.nanmax(p))
    above = p >= half
    if np.count_nonzero(above) < 2:
        return float("nan")
    width = float(w[above][-1] - w[above][0])
    return C_KMS * width / restwave


def _positive_integral(wave: np.ndarray, values: np.ndarray, rest_wave: float, half_width: float) -> float:
    mask = np.isfinite(wave) & np.isfinite(values) & (np.abs(wave - rest_wave) <= half_width)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(np.clip(values[mask], 0.0, None), wave[mask]))


def _chi2(residual: np.ndarray, err: np.ndarray) -> float:
    good = np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    return float(np.sum((residual[good] / err[good]) ** 2)) if np.any(good) else float("nan")


def _gaussian_integrated(wave: np.ndarray, rest_wave: float, flux: float, velocity: float, sigma_velocity: float) -> np.ndarray:
    if flux <= 0.0 or sigma_velocity <= 0.0:
        return np.zeros_like(wave)
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    return flux / (sigma_wave * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)


__all__ = ["MGII_WINDOW", "UVFeIIPrefitResult", "fit_mgii_uv", "fit_mgii_uvfeii", "prefit_uv_feii"]
