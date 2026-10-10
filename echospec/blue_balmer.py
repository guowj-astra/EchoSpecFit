"""Optional blue-Balmer extension for host-weak AGN spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from lmfit import Parameters, minimize

from echospec.spectrum import AGNSpectrum

C_KMS = 299792.458
BLUE_BALMER_WINDOW = (3850.0, 4450.0)
BLUE_TRIGGER_WINDOW = BLUE_BALMER_WINDOW
BALMER_CONTINUUM_REGION = (3000.0, 3646.0)
HGAMMA = 4340.47
HDELTA = 4101.74
HEII_4686 = 4685.70
NEIII_3869 = 3868.76
NEIII_3968 = 3967.47
NEV_3426 = 3425.86
HEPSILON = 3970.07
HBETA = 4861.33


@dataclass(frozen=True)
class BlueBalmerModeDecision:
    """Decision and diagnostics for enabling blue-Balmer extended fitting."""

    enable_blue_balmer: bool
    diagnostics: dict[str, Any]

    def __iter__(self):
        yield self.enable_blue_balmer
        yield self.diagnostics


def should_enable_blue_balmer_mode(
    spec: AGNSpectrum,
    initial_fit_result: Any | None = None,
) -> BlueBalmerModeDecision:
    """Decide whether the blue-Balmer extension should be enabled.

    The decision follows the requested permissive rule: any trigger condition is
    sufficient. Diagnostics are returned even when the mode is not enabled.
    """

    wave = np.asarray(spec.wave, dtype=float)
    flux = np.asarray(spec.flux, dtype=float)
    err = np.asarray(spec.err, dtype=float)
    residual = _initial_residual(spec, initial_fit_result)
    fit_info = getattr(initial_fit_result, "fit_info", {}) or {}

    covers_blue = bool(
        np.nanmin(wave[np.isfinite(wave)]) <= BLUE_TRIGGER_WINDOW[0]
        and np.nanmax(wave[np.isfinite(wave)]) >= BLUE_TRIGGER_WINDOW[1]
    ) if np.any(np.isfinite(wave)) else False
    residual_snr_hgamma = _integrated_residual_snr(wave, residual, err, HGAMMA)
    residual_snr_hdelta = _integrated_residual_snr(wave, residual, err, HDELTA)
    powerlaw_fraction_4200 = _first_finite(
        fit_info,
        ("powerlaw_fraction_4200", "agn_fraction_4200", "powerlaw_fraction_blue_feii"),
    )
    host_fraction_4200 = _first_finite(
        fit_info,
        ("host_fraction_4200", "stellar_fraction_4200", "stellar_fraction_blue_feii"),
    )
    broad_hbeta_snr = _first_finite(
        fit_info,
        ("broad_Hbeta_snr", "Hb_broad_snr", "broad_hb_total_snr"),
    )
    delta_bic_blue_balmer = _blue_line_delta_bic(wave, residual, err)
    triggers = {
        "covers_3800_4700": covers_blue,
        "powerlaw_fraction_4200_gt_0p5": bool(powerlaw_fraction_4200 > 0.5),
        "host_fraction_4200_lt_0p3": bool(host_fraction_4200 < 0.3),
        "broad_Hbeta_snr_gt_10": bool(broad_hbeta_snr > 10.0),
        "residual_snr_hgamma_gt_5": bool(residual_snr_hgamma > 5.0),
        "residual_snr_hdelta_gt_5": bool(residual_snr_hdelta > 5.0),
        "delta_bic_blue_balmer_gt_10": bool(delta_bic_blue_balmer > 10.0),
    }
    enable = bool(any(triggers.values()))
    diagnostics = {
        "blue_balmer_window": [float(BLUE_BALMER_WINDOW[0]), float(BLUE_BALMER_WINDOW[1])],
        "include_balmer_continuum": False,
        "skip_balmer_continuum_region": True,
        "flag_balmer_continuum_region_skipped": True,
        "warning_balmer_continuum_region": (
            "Balmer continuum region 3000-3646 A is skipped by default. "
            "Hgamma/Hdelta are still fitted in the 3850-4450 A blue-Balmer window."
        ),
        "residual_snr_hgamma": float(residual_snr_hgamma),
        "residual_snr_hdelta": float(residual_snr_hdelta),
        "powerlaw_fraction_4200": float(powerlaw_fraction_4200),
        "host_fraction_4200": float(host_fraction_4200),
        "broad_Hbeta_snr": float(broad_hbeta_snr),
        "delta_bic_blue_balmer": float(delta_bic_blue_balmer),
        "enable_blue_balmer": enable,
        "triggers": triggers,
    }
    return BlueBalmerModeDecision(enable, diagnostics)


def fit_blue_balmer_window(
    spec: AGNSpectrum,
    initial_fit_result: Any | None = None,
    force: bool = False,
    include_blue_balmer: bool | None = None,
    include_balmer_continuum: bool = False,
    skip_balmer_continuum_region: bool = True,
    include_neiii: bool = True,
    include_hepsilon: bool = False,
    force_broad_blue_balmer: bool = False,
    max_nfev: int | None = 2000,
) -> Any:
    """Fit the follow-up blue-Balmer optical window, excluding Balmer continuum.

    The fit is independent from the main H-beta solution. H-beta/[O III] results
    are used only as priors for the narrow group and broad Balmer profile.
    """

    from echospec.fitting import EmissionFitResult

    if include_balmer_continuum:
        raise NotImplementedError("Balmer continuum fitting is not implemented in this EchoSpec mode.")
    decision = should_enable_blue_balmer_mode(spec, initial_fit_result)
    covered = _window_covered(np.asarray(spec.wave, dtype=float), BLUE_BALMER_WINDOW)
    if include_blue_balmer is None:
        include_blue_balmer = covered
    wave, flux, err = _select_window_arrays(spec, BLUE_BALMER_WINDOW)
    if wave.size < 5:
        raise ValueError("No valid pixels available in the 3850-4450 A blue-Balmer fitting window.")
    if skip_balmer_continuum_region:
        keep = ~((wave >= BALMER_CONTINUUM_REGION[0]) & (wave <= BALMER_CONTINUUM_REGION[1]))
        wave, flux, err = wave[keep], flux[keep], err[keep]
    hb_constraints = _hbeta_constraints(initial_fit_result)
    diagnostics = dict(decision.diagnostics)
    diagnostics.update(
        {
            "fit_mode": "blue_balmer_window",
            "blue_balmer_window": [float(BLUE_BALMER_WINDOW[0]), float(BLUE_BALMER_WINDOW[1])],
            "include_blue_balmer": bool(include_blue_balmer),
            "include_balmer_continuum": bool(include_balmer_continuum),
            "skip_balmer_continuum_region": bool(skip_balmer_continuum_region),
            "flag_balmer_continuum_region_skipped": bool(skip_balmer_continuum_region),
            "warning_balmer_continuum_region": (
                "Balmer continuum region 3000-3646 A is skipped by default. "
                "Hgamma/Hdelta are still fitted in the 3850-4450 A blue-Balmer window."
            ),
        }
    )
    if not (force or include_blue_balmer):
        continuum = _linear_continuum(wave, _continuum_initial(wave, flux))
        residual = flux - continuum
        return EmissionFitResult(
            success=True,
            message="blue-Balmer window not covered/enabled",
            params={},
            line_flux={},
            line_flux_err={},
            model_flux=continuum,
            continuum_flux=continuum,
            residual_flux=residual,
            chi2=_chi2(residual, err),
            redchi=_chi2(residual, err) / max(len(wave), 1),
            ndata=int(len(wave)),
            nvarys=0,
            window_name="blue_balmer",
            fit_info=diagnostics,
            component_flux={"continuum": continuum},
        )

    residual0 = flux - _linear_continuum(wave, _continuum_initial(wave, flux))
    snr_hgamma = _integrated_residual_snr(wave, residual0, err, HGAMMA, half_width=18.0)
    snr_hdelta = _integrated_residual_snr(wave, residual0, err, HDELTA, half_width=18.0)
    base_params = _initial_blue_window_params(
        wave,
        flux,
        hb_constraints,
        include_neiii=include_neiii,
        include_hepsilon=include_hepsilon,
        include_broad=False,
    )
    base_fit = minimize(lambda p: (flux - _blue_window_model(wave, p)) / err, base_params, max_nfev=max_nfev)
    base_chi2 = _chi2(flux - _blue_window_model(wave, base_fit.params), err)
    broad_requested = bool(
        force_broad_blue_balmer
        or (
            hb_constraints.get("has_broad_hbeta", False)
            and (snr_hgamma > 3.0 or snr_hdelta > 3.0)
        )
    )
    selected_fit = base_fit
    selected_has_broad = False
    delta_bic_broad = float("nan")
    if broad_requested:
        broad_params = _initial_blue_window_params(
            wave,
            flux,
            hb_constraints,
            include_neiii=include_neiii,
            include_hepsilon=include_hepsilon,
            include_broad=True,
        )
        broad_fit = minimize(lambda p: (flux - _blue_window_model(wave, p)) / err, broad_params, max_nfev=max_nfev)
        broad_chi2 = _chi2(flux - _blue_window_model(wave, broad_fit.params), err)
        delta_bic_broad = _bic(base_chi2, len(wave), int(getattr(base_fit, "nvarys", 0))) - _bic(
            broad_chi2,
            len(wave),
            int(getattr(broad_fit, "nvarys", 0)),
        )
        if force_broad_blue_balmer or delta_bic_broad > 10.0:
            selected_fit = broad_fit
            selected_has_broad = True

    model = _blue_window_model(wave, selected_fit.params)
    continuum = _linear_continuum(wave, selected_fit.params)
    residual = flux - model
    components = _blue_window_components(wave, selected_fit.params)
    line_flux = _blue_window_line_flux(selected_fit.params)
    hgamma_broad = line_flux.get("Hgamma_broad_flux_total", 0.0)
    hdelta_broad = line_flux.get("Hdelta_broad_flux_total", 0.0)
    diagnostics.update(
        {
            "residual_snr_hgamma": float(snr_hgamma),
            "residual_snr_hdelta": float(snr_hdelta),
            "delta_bic_hgamma_broad": float(delta_bic_broad),
            "delta_bic_hdelta_broad": float(delta_bic_broad),
            "include_neiii": bool(include_neiii),
            "include_hepsilon": bool(include_hepsilon),
            "include_broad_blue_balmer": bool(selected_has_broad),
            "force_broad_blue_balmer": bool(force_broad_blue_balmer),
            "hbeta_constraints": hb_constraints,
            "flag_hgamma_required": bool(snr_hgamma > 5.0 and (selected_has_broad or line_flux.get("Hgamma_narrow_flux", 0.0) > 0.0)),
            "flag_hdelta_required": bool(snr_hdelta > 5.0 and (selected_has_broad or line_flux.get("Hdelta_narrow_flux", 0.0) > 0.0)),
            "flag_hgamma_broad_secure": bool(selected_has_broad and snr_hgamma > 5.0 and delta_bic_broad > 20.0),
            "flag_hdelta_broad_secure": bool(selected_has_broad and snr_hdelta > 5.0 and delta_bic_broad > 20.0),
            "flag_hgamma_weak": bool(line_flux.get("Hgamma_narrow_flux", 0.0) > 0.0 and snr_hgamma < 2.0),
            "flag_hdelta_weak": bool(line_flux.get("Hdelta_narrow_flux", 0.0) > 0.0 and snr_hdelta < 2.0),
            "flag_blue_balmer_weak": bool(max(snr_hgamma, snr_hdelta) < 2.0),
        }
    )
    diagnostics["flag_hgamma_overfit"] = bool(hgamma_broad > 0.0 and (snr_hgamma < 3.0 or hgamma_broad >= 0.8 * max(hb_constraints["hb_broad_flux_total"], 1.0e-12)))
    diagnostics["flag_hdelta_overfit"] = bool(hdelta_broad > 0.0 and (snr_hdelta < 3.0 or hdelta_broad >= 0.6 * max(hb_constraints["hb_broad_flux_total"], 1.0e-12)))
    return EmissionFitResult(
        success=bool(selected_fit.success),
        message=str(selected_fit.message),
        params={name: float(param.value) for name, param in selected_fit.params.items()},
        line_flux=line_flux,
        line_flux_err={},
        model_flux=model,
        continuum_flux=continuum,
        residual_flux=residual,
        chi2=_chi2(residual, err),
        redchi=_chi2(residual, err) / max(len(wave) - int(getattr(selected_fit, "nvarys", 0)), 1),
        ndata=int(len(wave)),
        nvarys=int(getattr(selected_fit, "nvarys", 0)),
        window_name="blue_balmer",
        fit_info=diagnostics,
        component_flux=components,
    )


def fit_blue_balmer_extended(
    spec: AGNSpectrum,
    initial_fit_result: Any | None = None,
    force: bool = False,
    include_heii: bool | None = None,
    include_neiii: bool | None = None,
    include_nev: bool | None = None,
    heii_mode: str = "narrow",
    max_nfev: int | None = 2000,
) -> Any:
    """Backward-compatible wrapper for the 3850-4450 A blue-Balmer window.

    The returned ``EmissionFitResult`` is intentionally independent from the
    default H-beta fitter. It uses the initial H-beta model where available and
    fits residual blue-end components in the 3800-5700 A range.
    """

    del include_heii, include_nev, heii_mode
    return fit_blue_balmer_window(
        spec,
        initial_fit_result=initial_fit_result,
        force=force,
        include_neiii=True if include_neiii is None else include_neiii,
        max_nfev=max_nfev,
    )


def _initial_blue_window_params(
    wave: np.ndarray,
    flux: np.ndarray,
    hb_constraints: dict[str, Any],
    include_neiii: bool,
    include_hepsilon: bool,
    include_broad: bool,
) -> Parameters:
    params = Parameters()
    cont = _continuum_initial(wave, flux)
    params.add("cont_intercept", value=cont["cont_intercept"])
    params.add("cont_slope", value=cont["cont_slope"])
    has_hbeta = bool(hb_constraints.get("has_hbeta_profile", False))
    params.add(
        "blue_velocity",
        value=float(hb_constraints["narrow_velocity"]),
        min=-600.0,
        max=600.0,
        vary=not has_hbeta,
    )
    params.add(
        "blue_narrow_sigma",
        value=float(np.clip(hb_constraints["narrow_sigma"], 50.0, 500.0)),
        min=50.0,
        max=500.0,
        vary=not has_hbeta,
    )
    residual = flux - _linear_continuum(wave, params)
    hb_narrow = float(hb_constraints.get("hb_narrow_flux", 0.0))
    params.add(
        "hgamma_narrow_flux",
        value=max(_positive_integral(wave, residual, HGAMMA, 10.0), 1.0e-12),
        min=0.0,
        max=0.8 * hb_narrow if has_hbeta and hb_narrow > 0.0 else np.inf,
    )
    params.add(
        "hdelta_narrow_flux",
        value=max(_positive_integral(wave, residual, HDELTA, 10.0), 1.0e-12),
        min=0.0,
        max=0.6 * hb_narrow if has_hbeta and hb_narrow > 0.0 else np.inf,
    )
    params.add(
        "neiii_3869_flux",
        value=max(_positive_integral(wave, residual, NEIII_3869, 8.0), 1.0e-12),
        min=0.0,
        vary=bool(include_neiii and _covered(wave, NEIII_3869)),
    )
    params.add(
        "neiii_3968_flux",
        value=max(_positive_integral(wave, residual, NEIII_3968, 8.0), 1.0e-12),
        min=0.0,
        vary=bool(include_neiii and _covered(wave, NEIII_3968)),
    )
    params.add(
        "hepsilon_narrow_flux",
        value=max(_positive_integral(wave, residual, HEPSILON, 8.0), 1.0e-12),
        min=0.0,
        max=0.5 * hb_narrow if has_hbeta and hb_narrow > 0.0 else np.inf,
        vary=bool(include_hepsilon and _covered(wave, HEPSILON)),
    )
    if include_broad and hb_constraints.get("has_broad_hbeta", False):
        ncomp = max(len(hb_constraints["broad_components"]), 1)
        hg_init = _positive_integral(wave, residual, HGAMMA, 35.0) / ncomp
        hd_init = _positive_integral(wave, residual, HDELTA, 35.0) / ncomp
        for index, comp in enumerate(hb_constraints["broad_components"], start=1):
            params.add(f"hbeta_broad_{index}_velocity", value=comp["velocity"], vary=False)
            params.add(f"hbeta_broad_{index}_sigma", value=comp["sigma"], vary=False)
            params.add(f"hbeta_broad_{index}_flux", value=comp["flux"], vary=False)
            params.add(
                f"hgamma_broad_{index}_flux",
                value=max(min(hg_init, 0.8 * comp["flux"]), 1.0e-12),
                min=0.0,
                max=max(0.8 * comp["flux"], 1.0e-12),
            )
            params.add(
                f"hdelta_broad_{index}_flux",
                value=max(min(hd_init, 0.6 * comp["flux"]), 1.0e-12),
                min=0.0,
                max=max(0.6 * comp["flux"], 1.0e-12),
            )
    return params


def _blue_window_model(wave: np.ndarray, params: Parameters) -> np.ndarray:
    model = _linear_continuum(wave, params)
    for name, component in _blue_window_components(wave, params).items():
        if name == "continuum":
            continue
        if component.shape == wave.shape:
            model = model + component
    return model


def _blue_window_components(wave: np.ndarray, params: Parameters) -> dict[str, np.ndarray]:
    v = params["blue_velocity"].value
    s = params["blue_narrow_sigma"].value
    components = {
        "continuum": _linear_continuum(wave, params),
        "Hgamma_narrow": _gaussian_integrated(wave, HGAMMA, params["hgamma_narrow_flux"].value, v, s),
        "Hdelta_narrow": _gaussian_integrated(wave, HDELTA, params["hdelta_narrow_flux"].value, v, s),
        "NeIII_3869": _gaussian_integrated(wave, NEIII_3869, params["neiii_3869_flux"].value, v, s),
        "NeIII_3968": _gaussian_integrated(wave, NEIII_3968, params["neiii_3968_flux"].value, v, s),
        "Hepsilon_narrow": _gaussian_integrated(wave, HEPSILON, params["hepsilon_narrow_flux"].value, v, s),
    }
    hgamma_broad = np.zeros_like(wave)
    hdelta_broad = np.zeros_like(wave)
    for index in _broad_indices(params):
        bv = params[f"hbeta_broad_{index}_velocity"].value
        bs = params[f"hbeta_broad_{index}_sigma"].value
        hg = _gaussian_integrated(wave, HGAMMA, params[f"hgamma_broad_{index}_flux"].value, bv, bs)
        hd = _gaussian_integrated(wave, HDELTA, params[f"hdelta_broad_{index}_flux"].value, bv, bs)
        components[f"Hgamma_broad_{index}"] = hg
        components[f"Hdelta_broad_{index}"] = hd
        hgamma_broad += hg
        hdelta_broad += hd
    components["Hgamma_broad"] = hgamma_broad
    components["Hdelta_broad"] = hdelta_broad
    return components


def _blue_window_line_flux(params: Parameters) -> dict[str, float]:
    hgamma_broad = _sum_params(params, "hgamma_broad", "flux")
    hdelta_broad = _sum_params(params, "hdelta_broad", "flux")
    return {
        "Hgamma_narrow": float(params["hgamma_narrow_flux"].value),
        "Hgamma_narrow_flux": float(params["hgamma_narrow_flux"].value),
        "Hgamma_broad": float(hgamma_broad),
        "Hgamma_broad_flux_total": float(hgamma_broad),
        "Hdelta_narrow": float(params["hdelta_narrow_flux"].value),
        "Hdelta_narrow_flux": float(params["hdelta_narrow_flux"].value),
        "Hdelta_broad": float(hdelta_broad),
        "Hdelta_broad_flux_total": float(hdelta_broad),
        "NeIII_3869": float(params["neiii_3869_flux"].value),
        "NeIII_3869_flux": float(params["neiii_3869_flux"].value),
        "NeIII_3968": float(params["neiii_3968_flux"].value),
        "NeIII_3968_flux": float(params["neiii_3968_flux"].value),
        "Hepsilon_narrow": float(params["hepsilon_narrow_flux"].value),
        "Hepsilon_narrow_flux": float(params["hepsilon_narrow_flux"].value),
    }


def _continuum_initial(wave: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    good = np.isfinite(wave) & np.isfinite(flux)
    if np.count_nonzero(good) < 2:
        return {"cont_intercept": 0.0, "cont_slope": 0.0}
    anchor = good & (
        ((wave >= 3850.0) & (wave <= 3900.0))
        | ((wave >= 4200.0) & (wave <= 4250.0))
        | ((wave >= 4410.0) & (wave <= 4450.0))
    )
    if np.count_nonzero(anchor) < 5:
        anchor = good
    x = wave[anchor] - 4200.0
    y = flux[anchor]
    try:
        slope, intercept = np.polyfit(x, y, 1)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        intercept, slope = float(np.nanmedian(y)), 0.0
    return {"cont_intercept": float(intercept), "cont_slope": float(slope)}


def _linear_continuum(wave: np.ndarray, params: Parameters | dict[str, float]) -> np.ndarray:
    intercept = params["cont_intercept"].value if hasattr(params["cont_intercept"], "value") else params["cont_intercept"]
    slope = params["cont_slope"].value if hasattr(params["cont_slope"], "value") else params["cont_slope"]
    return float(intercept) + float(slope) * (wave - 4200.0)


def _initial_blue_params(
    wave: np.ndarray,
    residual: np.ndarray,
    hb_constraints: dict[str, Any],
    optional: dict[str, Any],
    heii_mode: str = "narrow",
) -> Parameters:
    if heii_mode not in {"narrow", "narrow_broad"}:
        raise ValueError("heii_mode must be 'narrow' or 'narrow_broad'.")
    params = Parameters()
    params.add("blue_velocity", value=hb_constraints["narrow_velocity"], min=-1200.0, max=1200.0, vary=False)
    params.add("blue_narrow_sigma", value=hb_constraints["narrow_sigma"], min=50.0, max=700.0, vary=False)
    params.add(
        "hgamma_narrow_flux",
        value=max(_positive_integral(wave, residual, HGAMMA, 10.0), 1.0e-12),
        min=0.0,
        max=0.8 * hb_constraints["hb_narrow_flux"]
        if hb_constraints["has_hbeta_profile"] and hb_constraints["hb_narrow_flux"] > 0.0
        else np.inf,
    )
    params.add(
        "hdelta_narrow_flux",
        value=max(_positive_integral(wave, residual, HDELTA, 10.0), 1.0e-12),
        min=0.0,
        max=0.6 * hb_constraints["hb_narrow_flux"]
        if hb_constraints["has_hbeta_profile"] and hb_constraints["hb_narrow_flux"] > 0.0
        else np.inf,
    )
    for index, comp in enumerate(hb_constraints["broad_components"], start=1):
        params.add(f"hbeta_broad_{index}_velocity", value=comp["velocity"], vary=False)
        params.add(f"hbeta_broad_{index}_sigma", value=comp["sigma"], vary=False)
        params.add(f"hbeta_broad_{index}_flux", value=comp["flux"], vary=False)
        hgamma_initial = _positive_integral(wave, residual, HGAMMA, 45.0) / max(
            len(hb_constraints["broad_components"]), 1
        )
        hdelta_initial = _positive_integral(wave, residual, HDELTA, 45.0) / max(
            len(hb_constraints["broad_components"]), 1
        )
        params.add(
            f"hgamma_broad_{index}_flux",
            value=max(
                min(hgamma_initial, 0.4 * comp["flux"]) if comp["flux"] > 0.0 else hgamma_initial,
                1.0e-12,
            ),
            min=0.0,
            max=max(0.8 * comp["flux"], 1.0e-12)
            if hb_constraints["has_hbeta_profile"] and comp["flux"] > 0.0
            else np.inf,
        )
        params.add(
            f"hdelta_broad_{index}_flux",
            value=max(
                min(hdelta_initial, 0.25 * comp["flux"]) if comp["flux"] > 0.0 else hdelta_initial,
                1.0e-12,
            ),
            min=0.0,
            max=max(0.6 * comp["flux"], 1.0e-12)
            if hb_constraints["has_hbeta_profile"] and comp["flux"] > 0.0
            else np.inf,
        )
    params.add("heii_4686_flux", value=max(_positive_integral(wave, residual, HEII_4686, 12.0), 1.0e-12), min=0.0)
    if heii_mode == "narrow_broad":
        first = hb_constraints["broad_components"][0]
        params.add("heii_4686_broad_flux", value=max(0.5 * _positive_integral(wave, residual, HEII_4686, 35.0), 1.0e-12), min=0.0)
        params.add("heii_4686_broad_velocity", value=first["velocity"], vary=False)
        params.add("heii_4686_broad_sigma", value=first["sigma"], vary=False)
    params.add("neiii_3869_flux", value=max(_positive_integral(wave, residual, NEIII_3869, 10.0), 1.0e-12), min=0.0)
    params.add("neiii_3968_flux", value=max(_positive_integral(wave, residual, NEIII_3968, 10.0), 1.0e-12), min=0.0)
    params.add("nev_3426_flux", value=max(_positive_integral(wave, residual, NEV_3426, 10.0), 1.0e-12), min=0.0)
    return params


def _blue_model(wave: np.ndarray, params: Parameters) -> np.ndarray:
    v = params["blue_velocity"].value
    s = params["blue_narrow_sigma"].value
    model = (
        _gaussian_integrated(wave, HGAMMA, params["hgamma_narrow_flux"].value, v, s)
        + _gaussian_integrated(wave, HDELTA, params["hdelta_narrow_flux"].value, v, s)
        + _gaussian_integrated(wave, HEII_4686, params["heii_4686_flux"].value, v, s)
        + _gaussian_integrated(wave, NEIII_3869, params["neiii_3869_flux"].value, v, s)
        + _gaussian_integrated(wave, NEIII_3968, params["neiii_3968_flux"].value, v, s)
    )
    if "nev_3426_flux" in params:
        model += _gaussian_integrated(wave, NEV_3426, params["nev_3426_flux"].value, v, s)
    for index in _broad_indices(params):
        bv = params[f"hbeta_broad_{index}_velocity"].value
        bs = params[f"hbeta_broad_{index}_sigma"].value
        model += _gaussian_integrated(wave, HGAMMA, params[f"hgamma_broad_{index}_flux"].value, bv, bs)
        model += _gaussian_integrated(wave, HDELTA, params[f"hdelta_broad_{index}_flux"].value, bv, bs)
    if "heii_4686_broad_flux" in params:
        model += _gaussian_integrated(
            wave,
            HEII_4686,
            params["heii_4686_broad_flux"].value,
            params["heii_4686_broad_velocity"].value,
            params["heii_4686_broad_sigma"].value,
        )
    return model


def _blue_components(wave: np.ndarray, params: Parameters) -> dict[str, np.ndarray]:
    v = params["blue_velocity"].value
    s = params["blue_narrow_sigma"].value
    components = {
        "Hgamma_narrow": _gaussian_integrated(wave, HGAMMA, params["hgamma_narrow_flux"].value, v, s),
        "Hdelta_narrow": _gaussian_integrated(wave, HDELTA, params["hdelta_narrow_flux"].value, v, s),
        "HeII_4686": _gaussian_integrated(wave, HEII_4686, params["heii_4686_flux"].value, v, s),
        "NeIII_3869": _gaussian_integrated(wave, NEIII_3869, params["neiii_3869_flux"].value, v, s),
        "NeIII_3968": _gaussian_integrated(wave, NEIII_3968, params["neiii_3968_flux"].value, v, s),
    }
    if "nev_3426_flux" in params:
        components["NeV_3426"] = _gaussian_integrated(wave, NEV_3426, params["nev_3426_flux"].value, v, s)
    hgamma_broad = np.zeros_like(wave)
    hdelta_broad = np.zeros_like(wave)
    for index in _broad_indices(params):
        bv = params[f"hbeta_broad_{index}_velocity"].value
        bs = params[f"hbeta_broad_{index}_sigma"].value
        comp_hg = _gaussian_integrated(wave, HGAMMA, params[f"hgamma_broad_{index}_flux"].value, bv, bs)
        comp_hd = _gaussian_integrated(wave, HDELTA, params[f"hdelta_broad_{index}_flux"].value, bv, bs)
        components[f"Hgamma_broad_{index}"] = comp_hg
        components[f"Hdelta_broad_{index}"] = comp_hd
        hgamma_broad += comp_hg
        hdelta_broad += comp_hd
    components["Hgamma_broad"] = hgamma_broad
    components["Hdelta_broad"] = hdelta_broad
    if "heii_4686_broad_flux" in params:
        components["HeII_4686_broad"] = _gaussian_integrated(
            wave,
            HEII_4686,
            params["heii_4686_broad_flux"].value,
            params["heii_4686_broad_velocity"].value,
            params["heii_4686_broad_sigma"].value,
        )
    return components


def _hbeta_constraints(initial_fit_result: Any | None) -> dict[str, Any]:
    if initial_fit_result is None:
        return {
            "has_hbeta_profile": False,
            "has_broad_hbeta": False,
            "narrow_velocity": 0.0,
            "narrow_sigma": 220.0,
            "hb_narrow_flux": 0.0,
            "hb_broad_flux_total": 0.0,
            "broad_components": [],
        }
    params = getattr(initial_fit_result, "params", {}) or {}
    line_flux = getattr(initial_fit_result, "line_flux", {}) or {}
    fit_info = getattr(initial_fit_result, "fit_info", {}) or {}
    narrow_velocity = _value_from(params, "narrow_velocity", 0.0)
    narrow_sigma = _value_from(params, "narrow_sigma", 220.0)
    hb_narrow_flux = _first_positive(
        line_flux,
        ("Hb_narrow",),
        _value_from(params, "hb_narrow_flux", 1.0),
    )
    components: list[dict[str, float]] = []
    for index in (1, 2):
        flux = _first_positive(
            line_flux,
            (f"Hb_broad_{index}",),
            _value_from(params, f"hb_broad_{index}_flux", float("nan")),
        )
        sigma = _value_from(params, f"hb_broad_{index}_sigma", float("nan"))
        velocity = _value_from(params, f"hb_broad_{index}_velocity", 0.0)
        if np.isfinite(flux) and flux > 0.0 and np.isfinite(sigma) and sigma > 0.0:
            components.append({"flux": float(flux), "velocity": float(velocity), "sigma": float(sigma)})
    has_broad_hbeta = bool(components)
    if not components:
        total = _first_positive(line_flux, ("Hb_broad", "Hb_broad_total"), _first_finite(fit_info, ("Hb_broad_flux_total", "broad_hb_flux_total")))
        fallback_velocity = _first_finite(fit_info, ("Hb_broad_centroid_velocity", "broad_hb_centroid_velocity"))
        if not np.isfinite(fallback_velocity):
            fallback_velocity = 0.0
        fallback_sigma = _fwhm_to_sigma(_first_finite(fit_info, ("Hb_broad_fwhm_combined", "broad_hb_combined_fwhm")))
        if not np.isfinite(fallback_sigma) or fallback_sigma <= 0.0:
            fallback_sigma = 2200.0
        if np.isfinite(total) and total > 0.0:
            has_broad_hbeta = True
            components.append(
                {
                    "flux": float(total),
                    "velocity": float(fallback_velocity),
                    "sigma": float(max(fallback_sigma, 1200.0)),
                }
            )
    return {
        "has_hbeta_profile": True,
        "has_broad_hbeta": bool(has_broad_hbeta),
        "narrow_velocity": float(narrow_velocity),
        "narrow_sigma": float(narrow_sigma),
        "hb_narrow_flux": float(max(hb_narrow_flux, 1.0e-12)),
        "hb_broad_flux_total": float(sum(comp["flux"] for comp in components)),
        "broad_components": components,
    }


def _optional_line_plan(
    wave: np.ndarray,
    residual: np.ndarray,
    err: np.ndarray,
    include_heii: bool | None,
    include_neiii: bool | None,
    include_nev: bool | None,
) -> dict[str, Any]:
    snr_heii = _integrated_residual_snr(wave, residual, err, HEII_4686, half_width=12.0)
    snr_neiii_3869 = _integrated_residual_snr(wave, residual, err, NEIII_3869, half_width=10.0)
    snr_neiii_3968 = _integrated_residual_snr(wave, residual, err, NEIII_3968, half_width=10.0)
    snr_nev = _integrated_residual_snr(wave, residual, err, NEV_3426, half_width=10.0)
    bic_heii = _single_line_delta_bic(wave, residual, err, HEII_4686)
    bic_neiii = max(
        _single_line_delta_bic(wave, residual, err, NEIII_3869),
        _single_line_delta_bic(wave, residual, err, NEIII_3968),
    )
    bic_nev = _single_line_delta_bic(wave, residual, err, NEV_3426)
    covered = {
        "heii": _covered(wave, HEII_4686),
        "neiii": _covered(wave, NEIII_3869) or _covered(wave, NEIII_3968),
        "nev": _covered(wave, NEV_3426),
    }
    auto_heii = covered["heii"] and snr_heii > 3.0 and bic_heii > 10.0
    auto_neiii = covered["neiii"] and max(snr_neiii_3869, snr_neiii_3968) > 3.0 and bic_neiii > 10.0
    auto_nev = covered["nev"] and snr_nev > 3.0 and bic_nev > 10.0
    return {
        "include_heii": auto_heii if include_heii is None else bool(include_heii),
        "include_neiii": auto_neiii if include_neiii is None else bool(include_neiii),
        "include_nev": auto_nev if include_nev is None else bool(include_nev),
        "residual_snr_heii_4686": float(snr_heii),
        "residual_snr_neiii_3869": float(snr_neiii_3869),
        "residual_snr_neiii_3968": float(snr_neiii_3968),
        "residual_snr_nev_3426": float(snr_nev),
        "delta_bic_heii_4686": float(bic_heii),
        "delta_bic_neiii": float(bic_neiii),
        "delta_bic_nev_3426": float(bic_nev),
        "covered": covered,
    }


def _blue_quality_flags(
    wave: np.ndarray,
    residual: np.ndarray,
    err: np.ndarray,
    params: Parameters,
    hb_constraints: dict[str, Any],
    optional: dict[str, Any],
) -> dict[str, Any]:
    snr_hgamma = _integrated_residual_snr(wave, residual, err, HGAMMA, half_width=16.0)
    snr_hdelta = _integrated_residual_snr(wave, residual, err, HDELTA, half_width=16.0)
    bic_hgamma = _single_line_delta_bic(wave, residual, err, HGAMMA)
    bic_hdelta = _single_line_delta_bic(wave, residual, err, HDELTA)
    hgamma_flux = _sum_params(params, "hgamma_broad", "flux")
    hdelta_flux = _sum_params(params, "hdelta_broad", "flux")
    hb_broad_total = max(float(hb_constraints["hb_broad_flux_total"]), 1.0e-12)
    return {
        "residual_snr_hgamma_fit_window": float(snr_hgamma),
        "residual_snr_hdelta_fit_window": float(snr_hdelta),
        "delta_bic_hgamma": float(bic_hgamma),
        "delta_bic_hdelta": float(bic_hdelta),
        "flag_hgamma_required": bool(snr_hgamma > 5.0 and bic_hgamma > 10.0),
        "flag_hdelta_required": bool(snr_hdelta > 5.0 and bic_hdelta > 10.0),
        "flag_hgamma_overfit": bool(hgamma_flux >= 0.79 * hb_broad_total or (hgamma_flux > 0.0 and snr_hgamma < 3.0)),
        "flag_hdelta_overfit": bool(hdelta_flux >= 0.59 * hb_broad_total or (hdelta_flux > 0.0 and snr_hdelta < 3.0)),
        "hgamma_broad_flux_total": float(hgamma_flux),
        "hdelta_broad_flux_total": float(hdelta_flux),
        "hgamma_broad_to_hbeta_broad": float(hgamma_flux / hb_broad_total),
        "hdelta_broad_to_hbeta_broad": float(hdelta_flux / hb_broad_total),
        "optional_line_plan": optional,
    }


def _broad_indices(params: Parameters) -> list[int]:
    indices: list[int] = []
    for name in params:
        if name.startswith("hgamma_broad_") and name.endswith("_flux"):
            try:
                indices.append(int(name.split("_")[2]))
            except (IndexError, ValueError):
                continue
    return sorted(set(indices))


def _sum_params(params: Parameters, prefix: str, suffix: str) -> float:
    total = 0.0
    needle_start = f"{prefix}_"
    needle_end = f"_{suffix}"
    for name in params:
        if name.startswith(needle_start) and name.endswith(needle_end):
            total += float(params[name].value)
    return float(total)


def _initial_residual(spec: AGNSpectrum, result: Any | None) -> np.ndarray:
    if result is None:
        return np.asarray(spec.flux, dtype=float) - np.nanmedian(spec.flux)
    try:
        wave, model = _match_result_model(spec, result)
    except ValueError:
        return np.asarray(spec.flux, dtype=float) - np.nanmedian(spec.flux)
    residual = np.asarray(spec.flux, dtype=float).copy()
    matched = np.interp(spec.wave, wave, model, left=np.nan, right=np.nan)
    good = np.isfinite(matched)
    residual[good] = residual[good] - matched[good]
    return residual


def _matched_initial_model(spec: AGNSpectrum, wave: np.ndarray, result: Any | None) -> np.ndarray:
    if result is None:
        return np.full_like(wave, np.nanmedian(spec.flux), dtype=float)
    try:
        model_wave, model = _match_result_model(spec, result)
    except ValueError:
        return np.full_like(wave, np.nanmedian(spec.flux), dtype=float)
    return np.interp(wave, model_wave, model, left=0.0, right=0.0)


def _match_result_model(spec: AGNSpectrum, result: Any) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(result.model_flux, dtype=float)
    if len(model) == len(spec.wave):
        return np.asarray(spec.wave, dtype=float), model
    window_name = getattr(result, "window_name", "")
    windows = {
        "hb_oiii": (4200.0, 5700.0),
        "blue_balmer": BLUE_BALMER_WINDOW,
        "blue_balmer_extended": BLUE_BALMER_WINDOW,
    }
    if window_name in windows:
        selected = spec.select_window(*windows[window_name]).mask_bad_pixels()
        if len(selected.wave) == len(model):
            return np.asarray(selected.wave, dtype=float), model
    raise ValueError("Could not match initial fit result model to spectrum.")


def _select_window_arrays(spec: AGNSpectrum, window: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = spec.select_window(*window).mask_bad_pixels()
    return (
        np.asarray(selected.wave, dtype=float),
        np.asarray(selected.flux, dtype=float),
        np.asarray(selected.err, dtype=float),
    )


def _integrated_residual_snr(
    wave: np.ndarray,
    residual: np.ndarray,
    err: np.ndarray,
    rest_wave: float,
    half_width: float = 12.0,
) -> float:
    mask = np.isfinite(wave) & np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    mask &= np.abs(wave - rest_wave) <= half_width
    if np.count_nonzero(mask) < 2:
        return float("nan")
    signal = float(np.trapz(np.clip(residual[mask], 0.0, None), wave[mask]))
    noise = float(np.sqrt(np.sum(err[mask] ** 2)) * np.nanmedian(np.diff(wave[mask])))
    return signal / noise if noise > 0.0 else float("nan")


def _single_line_delta_bic(
    wave: np.ndarray,
    residual: np.ndarray,
    err: np.ndarray,
    rest_wave: float,
    half_width: float = 18.0,
    sigma_kms: float = 250.0,
) -> float:
    mask = np.isfinite(wave) & np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    mask &= np.abs(wave - rest_wave) <= half_width
    if np.count_nonzero(mask) < 5:
        return float("nan")
    y = residual[mask]
    e = err[mask]
    chi2_null = float(np.sum((y / e) ** 2))
    flux = _positive_integral(wave, residual, rest_wave, min(half_width, 12.0))
    candidate = _gaussian_integrated(wave[mask], rest_wave, flux, 0.0, sigma_kms)
    chi2_candidate = float(np.sum(((y - candidate) / e) ** 2))
    n = int(np.count_nonzero(mask))
    return float(chi2_null - (chi2_candidate + np.log(n)))


def _blue_line_delta_bic(wave: np.ndarray, residual: np.ndarray, err: np.ndarray) -> float:
    mask = np.isfinite(wave) & np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    mask &= ((np.abs(wave - HGAMMA) <= 25.0) | (np.abs(wave - HDELTA) <= 25.0))
    if np.count_nonzero(mask) < 8:
        return float("nan")
    y = residual[mask]
    e = err[mask]
    chi2_null = float(np.sum((y / e) ** 2))
    candidate = (
        _gaussian_integrated(wave[mask], HGAMMA, _positive_integral(wave, residual, HGAMMA, 12.0), 0.0, 250.0)
        + _gaussian_integrated(wave[mask], HDELTA, _positive_integral(wave, residual, HDELTA, 12.0), 0.0, 250.0)
    )
    chi2_candidate = float(np.sum(((y - candidate) / e) ** 2))
    n = int(np.count_nonzero(mask))
    return float((chi2_null + 0 * np.log(n)) - (chi2_candidate + 2 * np.log(n)))


def _positive_integral(wave: np.ndarray, values: np.ndarray, rest_wave: float, half_width: float) -> float:
    mask = np.isfinite(wave) & np.isfinite(values) & (np.abs(wave - rest_wave) <= half_width)
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapz(np.clip(values[mask], 0.0, None), wave[mask]))


def _covered(wave: np.ndarray, rest_wave: float, half_width: float = 12.0) -> bool:
    finite = np.isfinite(wave)
    if not np.any(finite):
        return False
    return bool(np.nanmin(wave[finite]) <= rest_wave - half_width and np.nanmax(wave[finite]) >= rest_wave + half_width)


def _window_covered(wave: np.ndarray, window: tuple[float, float]) -> bool:
    finite = np.isfinite(wave)
    if not np.any(finite):
        return False
    return bool(np.nanmin(wave[finite]) <= window[0] and np.nanmax(wave[finite]) >= window[1])


def _bic(chi2: float, ndata: int, nvarys: int) -> float:
    if not np.isfinite(chi2):
        return float("inf")
    return float(chi2 + max(nvarys, 0) * np.log(max(ndata, 1)))


def _gaussian_integrated(wave: np.ndarray, rest_wave: float, flux: float, velocity: float, sigma_velocity: float) -> np.ndarray:
    if flux <= 0.0 or sigma_velocity <= 0.0:
        return np.zeros_like(wave)
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    return flux / (sigma_wave * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)


def _chi2(residual: np.ndarray, err: np.ndarray) -> float:
    good = np.isfinite(residual) & np.isfinite(err) & (err > 0.0)
    return float(np.sum((residual[good] / err[good]) ** 2)) if np.any(good) else float("nan")


def _first_finite(values: dict[str, Any], names: tuple[str, ...]) -> float:
    for name in names:
        if name not in values:
            continue
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float("nan")


def _value_from(values: dict[str, Any], name: str, default: float) -> float:
    if name not in values:
        return float(default)
    try:
        value = values[name]
        value = getattr(value, "value", value)
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _first_positive(values: dict[str, Any], names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name not in values:
            continue
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0.0:
            return value
    return float(default)


def _fwhm_to_sigma(fwhm_kms: float) -> float:
    try:
        value = float(fwhm_kms)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(value / 2.3548)


__all__ = [
    "BALMER_CONTINUUM_REGION",
    "BLUE_BALMER_WINDOW",
    "BlueBalmerModeDecision",
    "fit_blue_balmer_extended",
    "fit_blue_balmer_window",
    "should_enable_blue_balmer_mode",
]
