"""High-level EchoSpec fitting pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from echospec.compare import EpochComparisonResult, EpochFitSet, compare_epochs
from echospec.spectrum import AGNSpectrum
from echospec.stellar import fit_stellar_ppxf, subtract_stellar_continuum

if TYPE_CHECKING:
    from echospec.fitting import EmissionFitResult
else:
    EmissionFitResult = Any

GAUSSIAN_FWHM_FACTOR = 2.3548
DEFAULT_NARROW_SIGMA_BOUNDS = (50.0, 250.0)
DEFAULT_HA_NARROW_SIGMA_BOUNDS = (50.0, 250.0)


@dataclass
class RepeatPairFitResult:
    """Fit products and comparison for a repeat-spectrum pair."""

    epoch1: EpochFitSet
    epoch2: EpochFitSet
    comparison: EpochComparisonResult

    def __iter__(self):
        yield self.epoch1
        yield self.epoch2
        yield self.comparison


@dataclass
class JointFitResult:
    """Pass-3 joint fit product."""

    success: bool
    message: str
    params: dict[str, float]
    line_flux: dict[str, float]
    line_flux_err: dict[str, float]
    total_model: np.ndarray
    components: dict[str, np.ndarray]
    residual: np.ndarray
    chi2: float
    redchi: float
    bic: float
    ndata: int
    nvarys: int
    diagnostics: dict[str, Any]
    warnings: list[str]


@dataclass
class IterativeJointFitResult:
    """Three-pass iterative joint fit result."""

    pass1_stellar_result: Any
    pass2_nonstellar_result: EpochFitSet
    pass3_joint_result: JointFitResult
    pass4_stellar_refit_result: Any | None = None
    pass5_final_joint_result: JointFitResult | None = None
    pass4_joint_stellar_result: Any | None = None
    final_result: JointFitResult | None = None
    diagnostics: dict[str, Any] | None = None
    warnings: list[str] | None = None

    @property
    def line_flux(self) -> dict[str, float]:
        return {} if self.final_result is None else self.final_result.line_flux

    @property
    def line_flux_err(self) -> dict[str, float]:
        return {} if self.final_result is None else self.final_result.line_flux_err

    @property
    def components(self) -> dict[str, np.ndarray]:
        return {} if self.final_result is None else self.final_result.components


def fit_single_epoch(
    spec: AGNSpectrum,
    do_stellar: bool = False,
    template_wave: ArrayLike | None = None,
    template_flux: ArrayLike | None = None,
    template_library: str = "miles",
    max_templates: int | None = None,
    stellar_mdegree: int = 10,
    stellar_degree: int = -1,
    stellar_mask_width_kms: float = 1000.0,
    stellar_debug: bool = False,
    include_agn_powerlaw: bool = True,
    agn_powerlaw_slopes: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    include_broad_balmer: bool = True,
    broad_bic_threshold: float = 10.0,
    include_feii: bool = True,
    feii_template_path: str | Path | None = None,
    feii_scale_mode: str = "single",
    include_blue_balmer: bool = True,
    include_balmer_continuum: bool = False,
    skip_balmer_continuum_region: bool = True,
    include_hepsilon: bool = False,
    force_broad_blue_balmer: bool = False,
    force_broad_hb: bool = False,
    do_final_flux_ratio_refit: bool = False,
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    broad_ha_model: str = "double",
    narrow_sigma_bounds: tuple[float, float] = DEFAULT_NARROW_SIGMA_BOUNDS,
    ha_narrow_sigma_bounds: tuple[float, float] = DEFAULT_HA_NARROW_SIGMA_BOUNDS,
    tie_ha_narrow_to_oiii: bool = True,
    n_mc: int = 0,
    print_report: bool = False,
    report_path: str | Path | None = None,
    fit_strategy: str = "window_joint",
    **iterative_joint_kwargs: Any,
) -> EpochFitSet | IterativeJointFitResult:
    """Clean, optionally stellar-subtract, and fit all standard EchoSpec windows.

    When include_broad_balmer=True, broad H-beta/H-alpha components are not
    forced. EchoSpec fits both narrow-only and narrow+broad models, then keeps
    the broad component only when the BIC improvement exceeds broad_bic_threshold.
    """

    if fit_strategy not in {"sequential", "window_joint", "iterative_joint"}:
        raise ValueError(
            "fit_strategy must be 'window_joint', 'sequential', or 'iterative_joint'."
        )
    if fit_strategy == "iterative_joint":
        return fit_single_epoch_iterative_joint(
            spec,
            template_library=template_library,
            do_stellar=True,
            include_powerlaw=include_agn_powerlaw,
            include_feii=include_feii,
            feii_scale_mode=feii_scale_mode,
            include_broad_balmer=include_broad_balmer,
            broad_hb_model=broad_hb_model,
            broad_ha_model=broad_ha_model,
            max_templates=max_templates,
            debug=stellar_debug,
            feii_template_path=feii_template_path,
            template_wave=template_wave,
            template_flux=template_flux,
            stellar_mdegree=stellar_mdegree,
            stellar_degree=stellar_degree,
            stellar_mask_width_kms=stellar_mask_width_kms,
            include_oiii_wing=include_oiii_wing,
            **iterative_joint_kwargs,
        )
    if fit_strategy == "window_joint":
        return fit_single_epoch_window_joint(
            spec,
            template_library=template_library,
            do_stellar=do_stellar,
            template_wave=template_wave,
            template_flux=template_flux,
            max_templates=max_templates,
            stellar_mdegree=stellar_mdegree,
            stellar_degree=stellar_degree,
            stellar_mask_width_kms=stellar_mask_width_kms,
            stellar_debug=stellar_debug,
            include_agn_powerlaw=include_agn_powerlaw,
            agn_powerlaw_slopes=agn_powerlaw_slopes,
            include_broad_balmer=include_broad_balmer,
            broad_bic_threshold=broad_bic_threshold,
            include_feii=include_feii,
            feii_template_path=feii_template_path,
            feii_scale_mode=feii_scale_mode,
            include_blue_balmer=include_blue_balmer,
            include_balmer_continuum=include_balmer_continuum,
            skip_balmer_continuum_region=skip_balmer_continuum_region,
            include_hepsilon=include_hepsilon,
            force_broad_blue_balmer=force_broad_blue_balmer,
            force_broad_hb=force_broad_hb,
            do_final_flux_ratio_refit=do_final_flux_ratio_refit,
            include_oiii_wing=include_oiii_wing,
            broad_hb_model=broad_hb_model,
            broad_ha_model=broad_ha_model,
            narrow_sigma_bounds=narrow_sigma_bounds,
            ha_narrow_sigma_bounds=ha_narrow_sigma_bounds,
            tie_ha_narrow_to_oiii=tie_ha_narrow_to_oiii,
            n_mc=n_mc,
            print_report=print_report,
            report_path=report_path,
            **iterative_joint_kwargs,
        )

    fit_spec = spec.mask_bad_pixels()
    if fit_spec.z != 0:
        fit_spec = fit_spec.to_rest_frame()

    stellar_result = None
    if do_stellar:
        template_wave, template_flux = _resolve_stellar_templates(
            template_wave,
            template_flux,
            template_library=template_library,
            max_templates=max_templates,
            debug=stellar_debug,
        )
        stellar_result = fit_stellar_ppxf(
            fit_spec,
            template_wave,
            template_flux,
            mdegree=stellar_mdegree,
            degree=stellar_degree,
            mask_width_kms=stellar_mask_width_kms,
            include_agn_powerlaw=include_agn_powerlaw,
            agn_powerlaw_slopes=agn_powerlaw_slopes,
            debug=stellar_debug,
        )
        fit_spec = subtract_stellar_continuum(fit_spec, stellar_result)

    from echospec.fitting import (
        fit_ha_nii_sii,
        fit_ha_nii_sii_bic,
        fit_hb_oiii,
        fit_hb_oiii_bic,
        fit_oii,
    )

    hb_fit_func = fit_hb_oiii_bic if include_broad_balmer else fit_hb_oiii
    if force_broad_hb:
        hb_fit_func = fit_hb_oiii
        hb_fit_kwargs: dict[str, Any] = {"include_broad_hb": True}
    else:
        hb_fit_kwargs = (
            {"bic_threshold": broad_bic_threshold}
            if include_broad_balmer
            else {"include_broad_hb": False}
        )
    hb_fit_kwargs.update(
        {
            "include_feii": include_feii,
            "feii_template_path": feii_template_path,
            "feii_scale_mode": feii_scale_mode,
            "include_oiii_wing": include_oiii_wing,
            "broad_hb_model": broad_hb_model,
        }
    )
    ha_fit_func = fit_ha_nii_sii_bic if include_broad_balmer else fit_ha_nii_sii
    ha_fit_kwargs: dict[str, Any] = (
        {"bic_threshold": broad_bic_threshold}
        if include_broad_balmer
        else {"include_broad_ha": False}
    )
    ha_fit_kwargs["broad_ha_model"] = broad_ha_model
    ha_fit_kwargs["narrow_sigma_bounds"] = ha_narrow_sigma_bounds

    epoch = EpochFitSet(
        object_id=fit_spec.object_id,
        survey=fit_spec.survey,
        oii=_fit_with_optional_mc(
            fit_spec,
            fit_oii,
            n_mc=n_mc,
            mode="doublet",
        ),
        hb_oiii=_fit_with_optional_mc(
            fit_spec,
            hb_fit_func,
            n_mc=n_mc,
            **hb_fit_kwargs,
        ),
        ha_nii_sii=_fit_with_optional_mc(
            fit_spec,
            ha_fit_func,
            n_mc=n_mc,
            **ha_fit_kwargs,
        ),
        stellar=stellar_result,
    )
    if print_report:
        print(format_epoch_fit_report(epoch))
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(_json_safe(epoch.to_dict()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return epoch


def fit_single_epoch_window_joint(
    spec: AGNSpectrum,
    template_library: str = "miles",
    do_stellar: bool = True,
    template_wave: ArrayLike | None = None,
    template_flux: ArrayLike | None = None,
    max_templates: int | None = None,
    stellar_mdegree: int = 10,
    stellar_degree: int = -1,
    stellar_mask_width_kms: float = 1000.0,
    stellar_debug: bool = False,
    include_agn_powerlaw: bool = True,
    agn_powerlaw_slopes: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    include_broad_balmer: bool = True,
    broad_bic_threshold: float = 10.0,
    include_feii: bool = True,
    feii_template_path: str | Path | None = None,
    feii_scale_mode: str = "single",
    include_blue_balmer: bool = True,
    include_balmer_continuum: bool = False,
    skip_balmer_continuum_region: bool = True,
    include_hepsilon: bool = False,
    force_broad_blue_balmer: bool = False,
    force_broad_hb: bool = False,
    do_final_flux_ratio_refit: bool = False,
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    broad_ha_model: str = "double",
    narrow_sigma_bounds: tuple[float, float] = DEFAULT_NARROW_SIGMA_BOUNDS,
    ha_narrow_sigma_bounds: tuple[float, float] = DEFAULT_HA_NARROW_SIGMA_BOUNDS,
    tie_ha_narrow_to_oiii: bool = True,
    n_mc: int = 0,
    print_report: bool = False,
    report_path: str | Path | None = None,
    do_stellar_refit: bool = False,
    do_pass4_stellar_refit: bool | None = None,
) -> EpochFitSet:
    """Fit one epoch with global stellar pPXF and independent emission windows.

    This is the default non-stellar strategy. It avoids a single global
    all-window chi2 for H-beta/Fe II/[O III], H-alpha/[N II]/[S II], and [O II].
    Window results are merged at the catalog/result layer.
    """

    fit_spec = spec.mask_bad_pixels()
    if fit_spec.z != 0:
        fit_spec = fit_spec.to_rest_frame()

    stellar_result = None
    if do_stellar:
        template_wave, template_flux = _resolve_stellar_templates(
            template_wave,
            template_flux,
            template_library=template_library,
            max_templates=max_templates,
            debug=stellar_debug,
        )
        stellar_result = fit_stellar_ppxf(
            fit_spec,
            template_wave,
            template_flux,
            mdegree=stellar_mdegree,
            degree=stellar_degree,
            mask_width_kms=stellar_mask_width_kms,
            include_agn_powerlaw=include_agn_powerlaw,
            agn_powerlaw_slopes=agn_powerlaw_slopes,
            debug=stellar_debug,
        )
        emission_spec = subtract_stellar_continuum(fit_spec, stellar_result)
    else:
        emission_spec = fit_spec

    epoch = _fit_emission_windows(
        emission_spec,
        stellar_result=stellar_result,
        include_broad_balmer=include_broad_balmer,
        broad_bic_threshold=broad_bic_threshold,
        include_feii=include_feii,
        feii_template_path=feii_template_path,
        feii_scale_mode=feii_scale_mode,
        include_blue_balmer=include_blue_balmer,
        include_balmer_continuum=include_balmer_continuum,
        skip_balmer_continuum_region=skip_balmer_continuum_region,
        include_hepsilon=include_hepsilon,
        force_broad_blue_balmer=force_broad_blue_balmer,
        force_broad_hb=force_broad_hb,
        do_final_flux_ratio_refit=do_final_flux_ratio_refit,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        broad_ha_model=broad_ha_model,
        narrow_sigma_bounds=narrow_sigma_bounds,
        ha_narrow_sigma_bounds=ha_narrow_sigma_bounds,
        tie_ha_narrow_to_oiii=tie_ha_narrow_to_oiii,
        n_mc=n_mc,
    )

    if do_pass4_stellar_refit is not None:
        do_stellar_refit = bool(do_pass4_stellar_refit)
    if do_stellar and do_stellar_refit and template_wave is not None and template_flux is not None:
        cleaned_spec = _subtract_window_nonstellar_components(fit_spec, epoch)
        stellar_result = fit_stellar_ppxf(
            cleaned_spec,
            template_wave,
            template_flux,
            mdegree=stellar_mdegree,
            degree=stellar_degree,
            mask_width_kms=stellar_mask_width_kms,
            include_agn_powerlaw=include_agn_powerlaw,
            agn_powerlaw_slopes=agn_powerlaw_slopes,
            debug=stellar_debug,
        )
        emission_spec = subtract_stellar_continuum(fit_spec, stellar_result)
        epoch = _fit_emission_windows(
            emission_spec,
            stellar_result=stellar_result,
            include_broad_balmer=include_broad_balmer,
            broad_bic_threshold=broad_bic_threshold,
            include_feii=include_feii,
            feii_template_path=feii_template_path,
            feii_scale_mode=feii_scale_mode,
            include_blue_balmer=include_blue_balmer,
            include_balmer_continuum=include_balmer_continuum,
            skip_balmer_continuum_region=skip_balmer_continuum_region,
            include_hepsilon=include_hepsilon,
            force_broad_blue_balmer=force_broad_blue_balmer,
            force_broad_hb=force_broad_hb,
            do_final_flux_ratio_refit=do_final_flux_ratio_refit,
            include_oiii_wing=include_oiii_wing,
            broad_hb_model=broad_hb_model,
            broad_ha_model=broad_ha_model,
            narrow_sigma_bounds=narrow_sigma_bounds,
            ha_narrow_sigma_bounds=ha_narrow_sigma_bounds,
            tie_ha_narrow_to_oiii=tie_ha_narrow_to_oiii,
            n_mc=n_mc,
        )

    diagnostics = _window_joint_diagnostics(epoch, do_stellar_refit=do_stellar_refit)
    epoch.diagnostics = diagnostics
    if print_report:
        print(format_epoch_fit_report(epoch))
    if report_path is not None:
        Path(report_path).write_text(
            json.dumps(_json_safe(epoch.to_dict()), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return epoch


def _fit_emission_windows(
    fit_spec: AGNSpectrum,
    stellar_result: Any | None,
    include_broad_balmer: bool,
    broad_bic_threshold: float,
    include_feii: bool,
    feii_template_path: str | Path | None,
    feii_scale_mode: str,
    include_blue_balmer: bool,
    include_balmer_continuum: bool,
    skip_balmer_continuum_region: bool,
    include_hepsilon: bool,
    force_broad_blue_balmer: bool,
    force_broad_hb: bool,
    do_final_flux_ratio_refit: bool,
    include_oiii_wing: bool,
    broad_hb_model: str,
    broad_ha_model: str,
    narrow_sigma_bounds: tuple[float, float],
    ha_narrow_sigma_bounds: tuple[float, float],
    tie_ha_narrow_to_oiii: bool,
    n_mc: int,
) -> EpochFitSet:
    from echospec.fitting import (
        fit_ha_nii_sii,
        fit_ha_nii_sii_bic,
        fit_hb_oiii,
        fit_hb_oiii_bic,
        fit_oii,
        fit_blue_balmer_window,
    )

    if force_broad_hb:
        hb_fit_func = fit_hb_oiii
        hb_fit_kwargs: dict[str, Any] = {"include_broad_hb": True}
    else:
        hb_fit_func = fit_hb_oiii_bic if include_broad_balmer else fit_hb_oiii
        hb_fit_kwargs = (
            {"bic_threshold": broad_bic_threshold}
            if include_broad_balmer
            else {"include_broad_hb": False}
        )
    hb_fit_kwargs.update(
        {
            "include_feii": include_feii,
            "feii_template_path": feii_template_path,
            "feii_scale_mode": feii_scale_mode,
            "include_oiii_wing": include_oiii_wing,
            "broad_hb_model": broad_hb_model,
            "narrow_sigma_bounds": narrow_sigma_bounds,
        }
    )
    ha_fit_func = fit_ha_nii_sii_bic if include_broad_balmer else fit_ha_nii_sii
    ha_fit_kwargs: dict[str, Any] = (
        {"bic_threshold": broad_bic_threshold}
        if include_broad_balmer
        else {"include_broad_ha": False}
    )
    ha_fit_kwargs["broad_ha_model"] = broad_ha_model
    ha_fit_kwargs["narrow_sigma_bounds"] = ha_narrow_sigma_bounds

    hb_oiii = _fit_with_optional_mc(fit_spec, hb_fit_func, n_mc=n_mc, **hb_fit_kwargs)
    narrow_reference, narrow_reference_info = _strongest_narrow_reference_from_result(
        hb_oiii,
        label="hb_oiii",
        narrow_sigma_bounds=narrow_sigma_bounds,
        enabled=tie_ha_narrow_to_oiii,
    )
    if hb_oiii is not None:
        hb_oiii.fit_info.update(
            {
                "staged_fit_order": "stellar_ppxf_then_powerlaw_feii_then_narrow_then_broad",
                "force_broad_hb": bool(force_broad_hb),
                "narrow_reference_policy": narrow_reference_info["policy"],
                "narrow_reference_used_for_ha": bool(narrow_reference_info["used"]),
                "narrow_reference_reason": narrow_reference_info["reason"],
                "strongest_narrow_line": narrow_reference_info["strongest_line"],
                "strongest_narrow_flux": narrow_reference_info["strongest_flux"],
                "strongest_narrow_sigma_kms": narrow_reference_info["sigma"],
                "strongest_narrow_fwhm_kms": narrow_reference_info["fwhm"],
            }
        )
    if narrow_reference is not None:
        ha_fit_kwargs["narrow_kinematics"] = narrow_reference
        ha_fit_kwargs["freeze_narrow_kinematics"] = False
        reference_sigma = float(narrow_reference["sigma"])
        base_min, base_max = _bounds_including_reference_sigma(
            ha_narrow_sigma_bounds,
            reference_sigma,
        )
        reference_cap = max(reference_sigma * 1.25, reference_sigma + 40.0)
        ha_fit_kwargs["narrow_sigma_bounds"] = (
            float(base_min),
            float(min(base_max, reference_cap)),
        )
    ha_nii_sii = _fit_with_optional_mc(
        fit_spec,
        ha_fit_func,
        n_mc=n_mc,
        **ha_fit_kwargs,
    )
    if ha_nii_sii is not None:
        ha_reference, ha_reference_info = _strongest_narrow_reference_from_result(
            ha_nii_sii,
            label="ha_nii_sii",
            narrow_sigma_bounds=ha_fit_kwargs.get("narrow_sigma_bounds", ha_narrow_sigma_bounds),
            enabled=True,
        )
        strongest_reference_info = _choose_strongest_narrow_reference_info(
            narrow_reference_info,
            ha_reference_info,
        )
        ha_nii_sii.fit_info.update(
            {
                "staged_fit_order": "stellar_ppxf_then_powerlaw_feii_then_narrow_then_broad",
                "narrow_reference_policy": narrow_reference_info["policy"],
                "narrow_reference_used": bool(narrow_reference_info["used"]),
                "narrow_reference_reason": narrow_reference_info["reason"],
                "narrow_reference_source": narrow_reference_info["source"],
                "narrow_reference_strongest_line": narrow_reference_info["strongest_line"],
                "narrow_reference_strongest_flux": narrow_reference_info["strongest_flux"],
                "narrow_reference_oiii_5007_flux": narrow_reference_info["oiii_5007_flux"],
                "narrow_reference_hb_flux": narrow_reference_info["hb_narrow_flux"],
                "narrow_reference_velocity_kms": narrow_reference_info["velocity"],
                "narrow_reference_sigma_kms": narrow_reference_info["sigma"],
                "ha_narrow_frozen_to_strong_oiii": bool(
                    narrow_reference_info["used"]
                    and str(narrow_reference_info["strongest_line"]).startswith("OIII")
                ),
                "ha_narrow_frozen_to_strongest_narrow": bool(narrow_reference_info["used"]),
                "ha_window_strongest_narrow_line": ha_reference_info["strongest_line"],
                "ha_window_strongest_narrow_flux": ha_reference_info["strongest_flux"],
                "ha_window_strongest_narrow_sigma_kms": ha_reference_info["sigma"],
                "ha_window_strongest_narrow_fwhm_kms": ha_reference_info["fwhm"],
                "global_strongest_narrow_line": strongest_reference_info["strongest_line"],
                "global_strongest_narrow_source": strongest_reference_info["source"],
                "global_strongest_narrow_flux": strongest_reference_info["strongest_flux"],
                "global_strongest_narrow_sigma_kms": strongest_reference_info["sigma"],
                "global_strongest_narrow_fwhm_kms": strongest_reference_info["fwhm"],
            }
        )
    epoch = EpochFitSet(
        object_id=fit_spec.object_id,
        survey=fit_spec.survey,
        oii=_fit_with_optional_mc(fit_spec, fit_oii, n_mc=n_mc, mode="doublet"),
        hb_oiii=hb_oiii,
        ha_nii_sii=ha_nii_sii,
        blue_balmer=_fit_with_optional_mc(
            fit_spec,
            fit_blue_balmer_window,
            n_mc=n_mc,
            initial_fit_result=hb_oiii,
            include_blue_balmer=include_blue_balmer,
            include_balmer_continuum=include_balmer_continuum,
            skip_balmer_continuum_region=skip_balmer_continuum_region,
            include_hepsilon=include_hepsilon,
            force_broad_blue_balmer=force_broad_blue_balmer,
        ),
        stellar=stellar_result,
    )
    if do_final_flux_ratio_refit and stellar_result is not None:
        try:
            epoch.final_refit = final_flux_ratio_refit(fit_spec, stellar_result, epoch)
        except Exception as exc:
            epoch.final_refit = None
            epoch.diagnostics = {
                **(epoch.diagnostics or {}),
                "final_flux_ratio_refit_error": repr(exc),
            }
    return epoch


def _strongest_narrow_reference_from_result(
    result: EmissionFitResult | None,
    label: str,
    narrow_sigma_bounds: tuple[float, float],
    enabled: bool = True,
    max_redchi: float = 25.0,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    """Return the shared narrow-line kinematics from the strongest fitted narrow line.

    The H-beta/[O III] and H-alpha/[N II]/[S II] models tie all narrow
    components inside each window to one velocity and sigma.  The strongest
    narrow line therefore supplies the best empirical anchor for that shared
    profile without hard-coding [O III] as the only possible reference.
    """

    info = _empty_narrow_reference_info(label=label)
    if not enabled:
        info["reason"] = "disabled"
        return None, info
    if result is None:
        info["reason"] = "missing_result"
        return None, info
    if not bool(result.success):
        info["reason"] = "fit_failed"
        return None, info
    if np.isfinite(result.redchi) and float(result.redchi) > max_redchi:
        info["reason"] = "high_redchi"
        return None, info

    params = result.params or {}
    line_flux = result.line_flux or {}
    sigma_key = "narrow_sigma"
    velocity_key = "narrow_velocity"
    candidate_names: tuple[str, ...]
    if label == "hb_oiii":
        candidate_names = ("OIII_5007", "OIII_4959", "Hb_narrow")
    elif label == "ha_nii_sii":
        candidate_names = ("Ha_narrow", "NII_6584", "NII_6548", "SII_6716", "SII_6731")
    elif label == "oii":
        sigma_key = "oii_sigma"
        velocity_key = "oii_velocity"
        candidate_names = ("OII_3727_blend", "OII_3726", "OII_3729")
    else:
        candidate_names = tuple(
            name for name in line_flux if "broad" not in name.lower() and "FeII" not in name
        )

    candidates: list[tuple[str, float]] = []
    for name in candidate_names:
        value = _finite_float(line_flux.get(name, np.nan))
        if np.isfinite(value) and value > 0.0:
            candidates.append((name, value))
    if not candidates:
        info["reason"] = "no_positive_narrow_flux"
        return None, info
    strongest_line, strongest_flux = max(candidates, key=lambda item: item[1])

    sigma = _finite_float(params.get(sigma_key, np.nan))
    velocity = _finite_float(params.get(velocity_key, 0.0))
    if not np.isfinite(sigma) or sigma <= 0.0 or sigma > 1000.0:
        info.update(
            {
                "reason": "invalid_sigma",
                "strongest_line": strongest_line,
                "strongest_flux": strongest_flux,
                "sigma": sigma,
                "velocity": velocity,
            }
        )
        return None, info

    low, high = narrow_sigma_bounds
    near_boundary = bool(sigma <= float(low) + 1.0e-3 or sigma >= float(high) - 1.0e-3)
    fwhm = float(GAUSSIAN_FWHM_FACTOR * sigma)
    source = f"{label}:{strongest_line}"
    info.update(
        {
            "used": True,
            "reason": "selected_strongest_narrow_line",
            "source": source,
            "strongest_line": strongest_line,
            "strongest_flux": float(strongest_flux),
            "velocity": float(velocity) if np.isfinite(velocity) else 0.0,
            "sigma": float(sigma),
            "fwhm": fwhm,
            "near_sigma_boundary": near_boundary,
            "oiii_5007_flux": float(_finite_float(line_flux.get("OIII_5007", np.nan))),
            "hb_narrow_flux": float(_finite_float(line_flux.get("Hb_narrow", np.nan))),
        }
    )
    return (
        {
            "velocity": float(velocity) if np.isfinite(velocity) else 0.0,
            "sigma": float(sigma),
            "source": source,
            "line": strongest_line,
            "flux": float(strongest_flux),
            "fwhm": fwhm,
        },
        info,
    )


def _empty_narrow_reference_info(label: str) -> dict[str, Any]:
    return {
        "policy": "strongest_narrow_line",
        "used": False,
        "reason": "",
        "source": label,
        "strongest_line": "",
        "strongest_flux": float("nan"),
        "velocity": float("nan"),
        "sigma": float("nan"),
        "fwhm": float("nan"),
        "near_sigma_boundary": False,
        "oiii_5007_flux": float("nan"),
        "hb_narrow_flux": float("nan"),
    }


def _choose_strongest_narrow_reference_info(
    *infos: dict[str, Any],
) -> dict[str, Any]:
    usable = [
        info
        for info in infos
        if bool(info.get("used", False))
        and np.isfinite(_finite_float(info.get("strongest_flux", np.nan)))
    ]
    if not usable:
        return _empty_narrow_reference_info(label="none")
    return max(usable, key=lambda item: _finite_float(item.get("strongest_flux", np.nan)))


def _bounds_including_reference_sigma(
    bounds: tuple[float, float],
    sigma: float,
) -> tuple[float, float]:
    low, high = float(bounds[0]), float(bounds[1])
    if not np.isfinite(sigma):
        return (low, high)
    return (min(low, float(sigma)), max(high, float(sigma)))


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _subtract_window_nonstellar_components(
    fit_spec: AGNSpectrum,
    epoch: EpochFitSet,
) -> AGNSpectrum:
    cleaned_flux = fit_spec.flux.copy()
    for result, window in (
        (epoch.hb_oiii, (4200.0, 5700.0)),
        (epoch.ha_nii_sii, (6400.0, 6800.0)),
        (epoch.oii, (3650.0, 3800.0)),
    ):
        if result is None:
            continue
        mask = (fit_spec.wave >= window[0]) & (fit_spec.wave <= window[1])
        if np.count_nonzero(mask) != len(result.model_flux):
            continue
        nonstellar = result.model_flux - result.continuum_flux
        cleaned_flux[mask] = cleaned_flux[mask] - nonstellar
    return AGNSpectrum(
        wave=fit_spec.wave.copy(),
        flux=cleaned_flux,
        err=fit_spec.err.copy(),
        z=fit_spec.z,
        survey=fit_spec.survey,
        object_id=fit_spec.object_id,
        meta={**fit_spec.meta, "window_joint_pass": "stellar_refit_input"},
    ).mask_bad_pixels()


def final_flux_ratio_refit(
    spec: AGNSpectrum,
    stellar_result: Any,
    epoch: EpochFitSet,
    continuum_scale_bounds: tuple[float, float] = (0.5, 2.0),
    emission_scale_bounds: tuple[float, float] = (0.0, 3.0),
    global_flux_scale_bounds: tuple[float, float] = (0.3, 3.0),
    continuum_tilt_bounds: tuple[float, float] = (-0.5, 0.5),
    continuum_curvature_bounds: tuple[float, float] = (-0.25, 0.25),
    additive_continuum_order: int = 0,
    skip_balmer_continuum_region: bool = True,
    fit_full_valid_spectrum: bool = True,
    max_nfev: int | None = 2000,
) -> Any:
    """Final calibrated joint refit using fixed window-fit component shapes.

    pPXF stellar/power-law shapes and local emission-window shapes are fixed.
    The last pass is allowed to solve a smooth full-spectrum calibration
    correction plus component scale factors, which keeps emission profiles fixed
    while preventing a simple flux-scale mismatch from dominating the overview
    diagnostic plot.
    """

    from lmfit import Parameters, minimize

    from echospec.fitting import FullSpectrumFitResult

    fit_spec = spec.mask_bad_pixels()
    stellar_shape = np.asarray(stellar_result.stellar_model, dtype=float)
    if len(stellar_shape) != len(fit_spec.wave):
        raise ValueError("stellar_result.stellar_model must match the spectrum length.")
    powerlaw_shape = getattr(stellar_result, "agn_model", None)
    if powerlaw_shape is None:
        powerlaw_shape = np.zeros_like(stellar_shape)
    powerlaw_shape = np.asarray(powerlaw_shape, dtype=float)
    if len(powerlaw_shape) != len(fit_spec.wave):
        powerlaw_shape = np.zeros_like(stellar_shape)

    component_shapes: dict[str, np.ndarray] = {}
    for name, result in (
        ("oii", epoch.oii),
        ("blue_balmer", epoch.blue_balmer),
        ("hb_oiii", epoch.hb_oiii),
        ("ha_nii_sii", epoch.ha_nii_sii),
    ):
        shape = _window_emission_shape_on_full_grid(fit_spec, result)
        if shape is not None and np.any(np.isfinite(shape) & (np.abs(shape) > 0.0)):
            component_shapes[name] = shape
    if not component_shapes:
        raise ValueError("No emission-window component shapes are available for final refit.")

    valid = (
        np.isfinite(fit_spec.wave)
        & np.isfinite(fit_spec.flux)
        & np.isfinite(fit_spec.err)
        & (fit_spec.err > 0.0)
        & np.isfinite(stellar_shape)
        & np.isfinite(powerlaw_shape)
    )
    if skip_balmer_continuum_region:
        valid &= ~((fit_spec.wave >= 3000.0) & (fit_spec.wave <= 3646.0))
    if not fit_full_valid_spectrum:
        window_mask = np.zeros_like(valid)
        for shape in component_shapes.values():
            window_mask |= np.isfinite(shape) & (np.abs(shape) > 0.0)
        valid &= window_mask
    if np.count_nonzero(valid) < 10:
        raise ValueError("Insufficient valid pixels for final flux-ratio refit.")

    wave_scale = _final_refit_wave_scale(fit_spec.wave, valid)
    additive_bounds = _final_refit_additive_bounds(fit_spec.flux, valid)
    params = Parameters()
    _add_scale_parameter(params, "global_flux_scale", 1.0, global_flux_scale_bounds)
    params.add(
        "continuum_tilt",
        value=0.0,
        min=float(continuum_tilt_bounds[0]),
        max=float(continuum_tilt_bounds[1]),
    )
    params.add(
        "continuum_curvature",
        value=0.0,
        min=float(continuum_curvature_bounds[0]),
        max=float(continuum_curvature_bounds[1]),
    )
    _add_scale_parameter(params, "stellar_scale", 1.0, continuum_scale_bounds)
    _add_scale_parameter(params, "powerlaw_scale", 1.0, continuum_scale_bounds)
    _add_additive_continuum_parameters(
        params,
        fit_spec.flux,
        stellar_shape + powerlaw_shape,
        valid,
        additive_bounds,
        additive_continuum_order,
    )
    for name in component_shapes:
        _add_scale_parameter(params, f"{name}_scale", 1.0, emission_scale_bounds)

    def build_model(local_params: Parameters) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        calibration = _final_refit_calibration(local_params, wave_scale)
        continuum_correction = _final_refit_additive_continuum(
            local_params,
            wave_scale,
            additive_continuum_order,
        )
        components: dict[str, np.ndarray] = {
            "stellar_raw": local_params["stellar_scale"].value * stellar_shape,
            "powerlaw_raw": local_params["powerlaw_scale"].value * powerlaw_shape,
            "calibration": calibration,
            "continuum_correction": continuum_correction,
        }
        total_raw = components["stellar_raw"] + components["powerlaw_raw"]
        emission_total = np.zeros_like(stellar_shape)
        for comp_name, shape in component_shapes.items():
            scaled = local_params[f"{comp_name}_scale"].value * shape
            components[f"{comp_name}_raw"] = scaled
            emission_total = emission_total + scaled
            total_raw = total_raw + scaled
        components["emission_total_raw"] = emission_total
        total = calibration * total_raw
        components["stellar"] = calibration * components["stellar_raw"]
        components["powerlaw"] = calibration * components["powerlaw_raw"]
        components["emission_total"] = calibration * emission_total
        for comp_name in component_shapes:
            components[comp_name] = calibration * components[f"{comp_name}_raw"]
        total = total + continuum_correction
        return total, components

    def objective(local_params: Parameters) -> np.ndarray:
        model, _ = build_model(local_params)
        return (fit_spec.flux[valid] - model[valid]) / fit_spec.err[valid]

    fit = minimize(objective, params, max_nfev=max_nfev)
    model, components = build_model(fit.params)
    residual = fit_spec.flux - model
    chi2 = float(np.sum(((fit_spec.flux[valid] - model[valid]) / fit_spec.err[valid]) ** 2))
    ndata = int(np.count_nonzero(valid))
    nvarys = int(getattr(fit, "nvarys", 0))
    line_flux = _scaled_final_line_flux(epoch, fit.params)
    scale_params = {name: float(param.value) for name, param in fit.params.items()}
    calibration = np.asarray(components["calibration"], dtype=float)
    component_scales = {
        name: value
        for name, value in scale_params.items()
        if name.endswith("_scale") or name in {"continuum_tilt", "continuum_curvature"}
    }
    return FullSpectrumFitResult(
        success=bool(fit.success),
        message=str(fit.message),
        params=scale_params,
        line_flux=line_flux,
        model_flux=model,
        stellar_flux=components["stellar"],
        agn_flux=components["powerlaw"] + components["continuum_correction"],
        emission_flux=components["emission_total"],
        residual_flux=residual,
        chi2=chi2,
        redchi=chi2 / max(ndata - nvarys, 1),
        ndata=ndata,
        nvarys=nvarys,
        fit_info={
            "fit_type": "final_flux_ratio_refit",
            "scale_only": False,
            "calibrated_scale_only_emission": True,
            "fixed_component_shapes": True,
            "component_scales": component_scales,
            "calibration_median": float(np.nanmedian(calibration[valid])),
            "calibration_min": float(np.nanmin(calibration[valid])),
            "calibration_max": float(np.nanmax(calibration[valid])),
            "additive_continuum_order": int(additive_continuum_order),
            "additive_continuum_bounds": tuple(float(v) for v in additive_bounds),
            "additive_continuum_median": float(np.nanmedian(components["continuum_correction"][valid])),
            "additive_continuum_min": float(np.nanmin(components["continuum_correction"][valid])),
            "additive_continuum_max": float(np.nanmax(components["continuum_correction"][valid])),
            "component_names": sorted(component_shapes.keys()),
            "continuum_scale_bounds": tuple(float(v) for v in continuum_scale_bounds),
            "emission_scale_bounds": tuple(float(v) for v in emission_scale_bounds),
            "global_flux_scale_bounds": tuple(float(v) for v in global_flux_scale_bounds),
            "continuum_tilt_bounds": tuple(float(v) for v in continuum_tilt_bounds),
            "continuum_curvature_bounds": tuple(float(v) for v in continuum_curvature_bounds),
            "skip_balmer_continuum_region": bool(skip_balmer_continuum_region),
            "fit_full_valid_spectrum": bool(fit_full_valid_spectrum),
            "n_fit_pixels": ndata,
        },
    )


def _final_refit_wave_scale(wave: np.ndarray, valid: np.ndarray) -> np.ndarray:
    good_wave = np.asarray(wave, dtype=float)[valid]
    if good_wave.size < 2:
        return np.zeros_like(wave, dtype=float)
    pivot = float(np.nanmedian(good_wave))
    span = float(np.nanpercentile(good_wave, 95) - np.nanpercentile(good_wave, 5))
    if not np.isfinite(span) or span <= 0.0:
        span = max(float(np.nanmax(good_wave) - np.nanmin(good_wave)), 1.0)
    return (np.asarray(wave, dtype=float) - pivot) / span


def _final_refit_calibration(params: Any, wave_scale: np.ndarray) -> np.ndarray:
    scale = float(params["global_flux_scale"].value)
    tilt = float(params["continuum_tilt"].value)
    curvature = float(params["continuum_curvature"].value)
    calibration = scale * (1.0 + tilt * wave_scale + curvature * (wave_scale**2))
    return np.clip(calibration, 0.05, np.inf)


def _final_refit_additive_bounds(flux: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    values = np.asarray(flux, dtype=float)[valid]
    if values.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(values, [5, 95])
    scale = float(max(abs(lo), abs(hi), np.nanstd(values), 1.0))
    return (0.0, 2.0 * scale)


def _add_additive_continuum_parameters(
    params: Any,
    flux: np.ndarray,
    initial_model: np.ndarray,
    valid: np.ndarray,
    bounds: tuple[float, float],
    order: int,
) -> None:
    residual = np.asarray(flux, dtype=float) - np.asarray(initial_model, dtype=float)
    initial_offset = float(np.nanmedian(residual[valid])) if np.count_nonzero(valid) else 0.0
    initial_offset = max(initial_offset, 0.0)
    lower, upper = bounds
    params.add(
        "continuum_offset",
        value=float(np.clip(initial_offset, lower, upper)),
        min=float(lower),
        max=float(upper),
    )
    if order >= 1:
        params.add("continuum_slope", value=0.0, min=float(lower), max=float(upper))
    if order >= 2:
        params.add("continuum_quadratic", value=0.0, min=float(lower), max=float(upper))


def _final_refit_additive_continuum(
    params: Any,
    wave_scale: np.ndarray,
    order: int,
) -> np.ndarray:
    correction = np.full_like(wave_scale, float(params["continuum_offset"].value), dtype=float)
    if order >= 1 and "continuum_slope" in params:
        correction = correction + float(params["continuum_slope"].value) * wave_scale
    if order >= 2 and "continuum_quadratic" in params:
        correction = correction + float(params["continuum_quadratic"].value) * (wave_scale**2)
    return correction


def _add_scale_parameter(
    params: Any,
    name: str,
    value: float,
    bounds: tuple[float, float],
) -> None:
    lower, upper = float(bounds[0]), float(bounds[1])
    if np.isclose(lower, upper):
        params.add(name, value=float(value), vary=False)
    else:
        params.add(name, value=float(value), min=lower, max=upper)


def _window_emission_shape_on_full_grid(
    spec: AGNSpectrum,
    result: Any | None,
) -> np.ndarray | None:
    if result is None or not getattr(result, "success", False):
        return None
    model = np.asarray(getattr(result, "model_flux", []), dtype=float)
    continuum = np.asarray(getattr(result, "continuum_flux", np.zeros_like(model)), dtype=float)
    if model.size == 0 or model.size != continuum.size:
        return None
    emission = model - continuum
    selected_wave = _matched_window_wave(spec, result, len(model))
    if selected_wave is None:
        return None
    shape = np.zeros_like(spec.wave, dtype=float)
    mask = (spec.wave >= float(np.nanmin(selected_wave))) & (spec.wave <= float(np.nanmax(selected_wave)))
    if np.count_nonzero(mask) == len(emission):
        shape[mask] = emission
    else:
        shape[mask] = np.interp(spec.wave[mask], selected_wave, emission, left=0.0, right=0.0)
    return shape


def _matched_window_wave(spec: AGNSpectrum, result: Any, n_model: int) -> np.ndarray | None:
    candidates = []
    info = getattr(result, "fit_info", {}) or {}
    if "blue_balmer_window" in info:
        try:
            candidates.append(tuple(float(v) for v in info["blue_balmer_window"]))
        except (TypeError, ValueError):
            pass
    window_name = getattr(result, "window_name", "")
    candidates.extend(
        {
            "oii_blend": [(3650.0, 3800.0)],
            "oii_doublet": [(3650.0, 3800.0)],
            "blue_balmer": [(3850.0, 4450.0)],
            "blue_balmer_extended": [(3850.0, 4450.0), (3800.0, 5700.0)],
            "hb_oiii": [(4400.0, 5600.0), (4200.0, 5700.0)],
            "ha_nii_sii": [(6400.0, 6800.0)],
        }.get(window_name, [])
    )
    for start, stop in candidates:
        selected = spec.select_window(start, stop).mask_bad_pixels()
        if len(selected.wave) == n_model:
            return np.asarray(selected.wave, dtype=float)
    return None


def _scaled_final_line_flux(epoch: EpochFitSet, params: Any) -> dict[str, float]:
    line_flux: dict[str, float] = {}
    global_scale = float(params["global_flux_scale"].value) if "global_flux_scale" in params else 1.0
    for name, result in (
        ("oii", epoch.oii),
        ("blue_balmer", epoch.blue_balmer),
        ("hb_oiii", epoch.hb_oiii),
        ("ha_nii_sii", epoch.ha_nii_sii),
    ):
        if result is None or not getattr(result, "success", False):
            continue
        scale_name = f"{name}_scale"
        scale = float(params[scale_name].value) if scale_name in params else 1.0
        for line_name, value in (getattr(result, "line_flux", {}) or {}).items():
            try:
                line_flux[line_name] = float(value) * scale * global_scale
            except (TypeError, ValueError):
                continue
    return line_flux


def fit_single_epoch_iterative_joint(
    spec: AGNSpectrum,
    template_library: str = "emiles_padova00_ch",
    do_stellar: bool = True,
    include_powerlaw: bool = True,
    include_feii: bool = True,
    feii_scale_mode: str = "single",
    include_broad_balmer: bool = True,
    broad_hb_model: str = "double",
    broad_ha_model: str = "double",
    hb_window: tuple[float, float] = (4200.0, 5700.0),
    ha_window: tuple[float, float] = (6400.0, 6800.0),
    oii_window: tuple[float, float] = (3650.0, 3800.0),
    max_templates: int | None = None,
    debug: bool = True,
    template_wave: ArrayLike | None = None,
    template_flux: ArrayLike | None = None,
    stellar_mdegree: int = 10,
    stellar_degree: int = -1,
    stellar_mask_width_kms: float = 8000.0,
    include_oiii_wing: bool = True,
    feii_template_path: str | Path | None = None,
    do_stellar_refit: bool = True,
    do_final_joint_refit: bool = True,
    do_pass4_stellar_refit: bool | None = None,
) -> IterativeJointFitResult:
    """Fit one epoch with pPXF initialization followed by observed-frame joint fit."""

    del ha_window, oii_window
    warnings: list[str] = []
    fit_spec = spec.mask_bad_pixels()
    if fit_spec.z != 0:
        fit_spec = fit_spec.to_rest_frame()

    if not do_stellar:
        raise ValueError("iterative_joint currently requires do_stellar=True.")
    template_wave, template_flux = _resolve_stellar_templates(
        template_wave,
        template_flux,
        template_library=template_library,
        max_templates=max_templates,
        debug=debug,
    )
    pass1_stellar = fit_stellar_ppxf(
        fit_spec,
        template_wave,
        template_flux,
        mdegree=stellar_mdegree,
        degree=stellar_degree,
        mask_width_kms=stellar_mask_width_kms,
        include_agn_powerlaw=False,
        debug=debug,
    )
    residual_spec = subtract_stellar_continuum(fit_spec, pass1_stellar)

    from echospec.fitting import fit_ha_nii_sii, fit_hb_oiii, fit_oii

    pass2_hb = fit_hb_oiii(
        residual_spec,
        include_broad_hb=include_broad_balmer,
        include_feii=include_feii,
        feii_template_path=feii_template_path,
        feii_scale_mode=feii_scale_mode,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        hb_window=hb_window,
    )
    pass2_ha = fit_ha_nii_sii(
        residual_spec,
        include_broad_ha=include_broad_balmer,
        broad_ha_model=broad_ha_model,
    )
    pass2_oii = fit_oii(residual_spec, mode="doublet")
    pass2 = EpochFitSet(
        object_id=fit_spec.object_id,
        survey=fit_spec.survey,
        oii=pass2_oii,
        hb_oiii=pass2_hb,
        ha_nii_sii=pass2_ha,
        stellar=pass1_stellar,
    )

    if do_pass4_stellar_refit is not None:
        do_stellar_refit = bool(do_pass4_stellar_refit)

    pass3 = _fit_joint_observed(
        fit_spec,
        pass1_stellar,
        pass2,
        include_powerlaw=include_powerlaw,
        include_feii=include_feii,
        include_broad_balmer=include_broad_balmer,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        broad_ha_model=broad_ha_model,
        feii_template_path=feii_template_path,
        pass_name="pass3",
    )
    pass4_result = None
    pass5 = pass3
    if do_stellar_refit:
        nonstellar_model_1 = _nonstellar_model_from_joint(pass3)
        stellar_input_2 = AGNSpectrum(
            wave=fit_spec.wave.copy(),
            flux=fit_spec.flux - nonstellar_model_1,
            err=fit_spec.err.copy(),
            z=fit_spec.z,
            survey=fit_spec.survey,
            object_id=fit_spec.object_id,
            meta={**fit_spec.meta, "iterative_joint_pass": "pass4_stellar_refit"},
        ).mask_bad_pixels()
        pass4_result = fit_stellar_ppxf(
            stellar_input_2,
            template_wave,
            template_flux,
            mdegree=stellar_mdegree,
            degree=stellar_degree,
            mask_width_kms=stellar_mask_width_kms,
            include_agn_powerlaw=False,
            debug=debug,
        )
        if do_final_joint_refit:
            pass5 = _fit_joint_observed(
                fit_spec,
                pass4_result,
                pass2,
                include_powerlaw=include_powerlaw,
                include_feii=include_feii,
                include_broad_balmer=include_broad_balmer,
                include_oiii_wing=include_oiii_wing,
                broad_hb_model=broad_hb_model,
                broad_ha_model=broad_ha_model,
                feii_template_path=feii_template_path,
                initial_joint_params=pass3.params,
                pass_name="pass5",
            )
    diagnostics = dict(pass5.diagnostics)
    diagnostics["fit_strategy"] = "iterative_joint"
    diagnostics["experimental"] = True
    diagnostics["global_nonstellar_joint_fit"] = True
    diagnostics.update(_iterative_change_diagnostics(pass1_stellar, pass4_result, pass3, pass5))
    if diagnostics.get("flag_final_joint_failed"):
        warnings.append("Final Pass 5 joint fit failed.")
    if diagnostics.get("flag_final_joint_unstable"):
        warnings.append("Line fluxes changed by more than 20% between Pass 3 and Pass 5.")
    if diagnostics.get("flag_stellar_refit_changed_too_much"):
        warnings.append("Pass 4 stellar model changed by more than 30% relative to Pass 1.")
    diagnostics["flag_needs_manual_inspection"] = bool(
        diagnostics.get("flag_final_joint_failed")
        or diagnostics.get("flag_final_joint_unstable")
        or diagnostics.get("flag_stellar_refit_changed_too_much")
        or diagnostics.get("fit_quality") == "needs_inspection"
    )
    result = IterativeJointFitResult(
        pass1_stellar_result=pass1_stellar,
        pass2_nonstellar_result=pass2,
        pass3_joint_result=pass3,
        pass4_stellar_refit_result=pass4_result,
        pass5_final_joint_result=pass5,
        pass4_joint_stellar_result=pass4_result,
        final_result=pass5,
        diagnostics=diagnostics,
        warnings=warnings + pass5.warnings,
    )
    return result


def fit_repeat_pair(
    spec1: AGNSpectrum,
    spec2: AGNSpectrum,
    do_stellar: bool = False,
    templates: Any | None = None,
    template_library: str = "miles",
    max_templates: int | None = None,
    stellar_mdegree: int = 10,
    stellar_degree: int = -1,
    stellar_mask_width_kms: float = 1000.0,
    stellar_debug: bool = False,
    include_agn_powerlaw: bool = True,
    agn_powerlaw_slopes: tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    include_broad_balmer: bool = True,
    broad_bic_threshold: float = 10.0,
    include_feii: bool = True,
    feii_template_path: str | Path | None = None,
    feii_scale_mode: str = "single",
    include_oiii_wing: bool = True,
    broad_hb_model: str = "double",
    broad_ha_model: str = "double",
    n_mc: int = 0,
) -> RepeatPairFitResult:
    """Fit two repeat spectra and compare their emission-line changes."""

    template_wave, template_flux = _resolve_templates(templates)
    epoch1 = fit_single_epoch(
        spec1,
        do_stellar=do_stellar,
        template_wave=template_wave,
        template_flux=template_flux,
        template_library=template_library,
        max_templates=max_templates,
        stellar_mdegree=stellar_mdegree,
        stellar_degree=stellar_degree,
        stellar_mask_width_kms=stellar_mask_width_kms,
        stellar_debug=stellar_debug,
        include_agn_powerlaw=include_agn_powerlaw,
        agn_powerlaw_slopes=agn_powerlaw_slopes,
        include_broad_balmer=include_broad_balmer,
        broad_bic_threshold=broad_bic_threshold,
        include_feii=include_feii,
        feii_template_path=feii_template_path,
        feii_scale_mode=feii_scale_mode,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        broad_ha_model=broad_ha_model,
        n_mc=n_mc,
    )
    epoch2 = fit_single_epoch(
        spec2,
        do_stellar=do_stellar,
        template_wave=template_wave,
        template_flux=template_flux,
        template_library=template_library,
        max_templates=max_templates,
        stellar_mdegree=stellar_mdegree,
        stellar_degree=stellar_degree,
        stellar_mask_width_kms=stellar_mask_width_kms,
        stellar_debug=stellar_debug,
        include_agn_powerlaw=include_agn_powerlaw,
        agn_powerlaw_slopes=agn_powerlaw_slopes,
        include_broad_balmer=include_broad_balmer,
        broad_bic_threshold=broad_bic_threshold,
        include_feii=include_feii,
        feii_template_path=feii_template_path,
        feii_scale_mode=feii_scale_mode,
        include_oiii_wing=include_oiii_wing,
        broad_hb_model=broad_hb_model,
        broad_ha_model=broad_ha_model,
        n_mc=n_mc,
    )
    comparison = compare_epochs(epoch1, epoch2)
    return RepeatPairFitResult(epoch1=epoch1, epoch2=epoch2, comparison=comparison)


def _fit_joint_observed(
    spec: AGNSpectrum,
    stellar_result: Any,
    pass2: EpochFitSet,
    include_powerlaw: bool,
    include_feii: bool,
    include_broad_balmer: bool,
    include_oiii_wing: bool,
    broad_hb_model: str,
    broad_ha_model: str,
    feii_template_path: str | Path | None,
    initial_joint_params: dict[str, float] | None = None,
    pass_name: str = "joint",
) -> JointFitResult:
    from lmfit import Parameters, minimize

    from echospec.continuum import powerlaw_diagnostics
    from echospec.fitting import _load_feii_template
    from echospec.joint_model import build_joint_model

    wave = spec.wave
    fit_mask = _joint_fit_mask(wave)
    feii_template = _load_feii_template(None, feii_template_path) if include_feii else None
    params = _initial_joint_parameters(
        spec,
        stellar_result,
        pass2,
        include_powerlaw,
        include_feii,
        include_broad_balmer,
    )
    if initial_joint_params:
        _seed_joint_parameters(params, initial_joint_params)
    components_config = {
        "feii_template": feii_template,
        "include_feii": include_feii and feii_template is not None,
        "include_broad_hb": include_broad_balmer,
        "include_broad_ha": include_broad_balmer,
        "include_oiii_wing": include_oiii_wing,
        "broad_hb_model": broad_hb_model,
        "broad_ha_model": broad_ha_model,
    }

    def objective(local_params: Parameters) -> np.ndarray:
        model, _ = build_joint_model(
            wave,
            stellar_result.stellar_model,
            local_params,
            components_config,
        )
        return (spec.flux[fit_mask] - model[fit_mask]) / spec.err[fit_mask]

    fit = minimize(objective, params, max_nfev=6000)
    total_model, components = build_joint_model(
        wave,
        stellar_result.stellar_model,
        fit.params,
        components_config,
    )
    residual = spec.flux - total_model
    chi2 = float(np.sum(((spec.flux[fit_mask] - total_model[fit_mask]) / spec.err[fit_mask]) ** 2))
    ndata = int(np.count_nonzero(fit_mask))
    nvarys = int(fit.nvarys)
    redchi = chi2 / max(ndata - nvarys, 1)
    bic = float(chi2 + nvarys * np.log(max(ndata, 1)))
    line_flux = _joint_line_flux_from_params(fit.params)
    diagnostics = _joint_diagnostics(
        wave,
        spec.flux,
        total_model,
        components,
        fit.params,
        chi2,
        bic,
    )
    pl = components.get("powerlaw", np.zeros_like(wave))
    cont_total = components.get("stellar", np.zeros_like(wave)) + pl
    diagnostics.update(powerlaw_diagnostics(wave, pl, cont_total))
    diagnostics["wave"] = wave.copy()
    diagnostics["joint_pass_name"] = pass_name
    flags = [
        diagnostics.get("flag_powerlaw_dominant", False),
        diagnostics.get("flag_stellar_scale_large_change", False),
        diagnostics.get("flag_feii_underfit", False),
        diagnostics.get("flag_broad_hb_eats_feii", False),
        diagnostics.get("flag_broad_hb_eats_oiii_feii_pedestal", False),
        diagnostics.get("flag_oiii_wing_eats_feii_pedestal", False),
    ]
    if not fit.success:
        fit_quality = "bad"
    elif any(bool(flag) for flag in flags):
        fit_quality = "needs_inspection"
    else:
        fit_quality = "good"
    diagnostics["fit_quality"] = fit_quality
    return JointFitResult(
        success=bool(fit.success),
        message=str(fit.message),
        params={name: float(param.value) for name, param in fit.params.items()},
        line_flux=line_flux,
        line_flux_err={},
        total_model=total_model,
        components=components,
        residual=residual,
        chi2=chi2,
        redchi=redchi,
        bic=bic,
        ndata=ndata,
        nvarys=nvarys,
        diagnostics=diagnostics,
        warnings=[],
    )


def _joint_fit_mask(wave: np.ndarray) -> np.ndarray:
    windows = ((3600.0, 3800.0), (4200.0, 5700.0), (6400.0, 6800.0))
    mask = np.zeros_like(wave, dtype=bool)
    for start, stop in windows:
        mask |= (wave >= start) & (wave <= stop)
    return mask


def _nonstellar_model_from_joint(joint: JointFitResult) -> np.ndarray:
    nonstellar = np.zeros_like(joint.total_model)
    for name, values in joint.components.items():
        if name == "stellar":
            continue
        nonstellar = nonstellar + np.asarray(values, dtype=float)
    return nonstellar


def _iterative_change_diagnostics(
    pass1_stellar: Any,
    pass4_stellar: Any | None,
    pass3: JointFitResult,
    pass5: JointFitResult,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    if pass4_stellar is not None:
        s1 = np.asarray(pass1_stellar.stellar_model, dtype=float)
        s4 = np.asarray(pass4_stellar.stellar_model, dtype=float)
        good = np.isfinite(s1) & np.isfinite(s4) & (np.abs(s1) > 0)
        frac = (s4[good] - s1[good]) / s1[good] if np.any(good) else np.array([])
        median_change = float(np.nanmedian(np.abs(frac))) if frac.size else float("nan")
        rms_change = float(np.sqrt(np.nanmean(frac**2))) if frac.size else float("nan")
    else:
        median_change = 0.0
        rms_change = 0.0
    diagnostics["stellar_change_between_pass1_pass4"] = {
        "median_fractional_change": median_change,
        "rms_fractional_change": rms_change,
        "A_star_1": pass3.params.get("stellar_scale", float("nan")),
        "A_star_2": pass5.params.get("stellar_scale", float("nan")),
    }
    diagnostics["stellar_median_fractional_change_pass1_pass4"] = median_change
    diagnostics["stellar_rms_fractional_change_pass1_pass4"] = rms_change
    diagnostics["A_star_1"] = pass3.params.get("stellar_scale", float("nan"))
    diagnostics["A_star_2"] = pass5.params.get("stellar_scale", float("nan"))
    key_lines = {
        "OII_3727_blend": "[O II]",
        "Hb_narrow": "Hbeta narrow",
        "OIII_5007": "[O III] 5007",
        "Ha_narrow": "Halpha narrow",
        "NII_6584": "[N II] 6584",
        "Hb_broad": "broad Hbeta total",
        "Ha_broad": "broad Halpha total",
    }
    line_changes: dict[str, float] = {}
    unstable = False
    for line in key_lines:
        old = pass3.line_flux.get(line, float("nan"))
        new = pass5.line_flux.get(line, float("nan"))
        if np.isfinite(old) and old != 0 and np.isfinite(new):
            change = float((new - old) / old)
            if abs(change) > 0.20:
                unstable = True
        else:
            change = float("nan")
        line_changes[line] = change
        diagnostics[f"line_flux_change_pass3_pass5_{line}"] = change
    diagnostics["line_flux_change_between_pass3_pass5"] = line_changes
    diagnostics["flag_stellar_refit_changed_too_much"] = bool(
        np.isfinite(median_change) and median_change > 0.30
    )
    diagnostics["flag_final_joint_unstable"] = bool(unstable)
    diagnostics["flag_final_joint_failed"] = bool(not pass5.success)
    return diagnostics


def _initial_joint_parameters(
    spec: AGNSpectrum,
    stellar_result: Any,
    pass2: EpochFitSet,
    include_powerlaw: bool,
    include_feii: bool,
    include_broad_balmer: bool,
):
    from lmfit import Parameters

    params = Parameters()
    params.add("stellar_scale", value=1.0, min=0.5, max=1.5)
    pl_amp = _estimate_powerlaw_amp(spec, stellar_result)
    params.add("pl_amp", value=pl_amp if include_powerlaw else 0.0, min=0.0, vary=include_powerlaw)
    params.add("pl_alpha", value=-1.0, min=-3.0, max=1.0, vary=include_powerlaw)

    hb_info = getattr(pass2.hb_oiii, "fit_info", {}) or {}
    hb_params = getattr(pass2.hb_oiii, "params", {}) or {}
    ha_params = getattr(pass2.ha_nii_sii, "params", {}) or {}
    oii_params = getattr(pass2.oii, "params", {}) or {}
    line_flux = {}
    for result in (pass2.hb_oiii, pass2.ha_nii_sii, pass2.oii):
        if result is not None:
            line_flux.update(result.line_flux)

    feii_flux = _first_finite(
        hb_info,
        (
            "feii_flux_4434_4684_final",
            "feii_flux_4434_4684",
            "feii_flux_4434_4684_prefit",
            "feii_amp",
        ),
        1.0,
    )
    feii_shift = _first_finite(
        hb_info,
        ("feii_shift_kms_final", "feii_shift_kms", "feii_shift_kms_prefit", "feii_velocity"),
        0.0,
    )
    feii_extra_sigma = _first_finite(
        hb_info,
        ("feii_extra_sigma_kms_final", "feii_extra_sigma_kms", "feii_extra_sigma_kms_prefit"),
        0.0,
    )
    feii_smooth_sigma = _first_finite(
        hb_info,
        ("feii_smooth_sigma_kms_final", "feii_smooth_sigma_kms", "feii_smooth_sigma_kms_prefit"),
        500.0,
    )
    params.add("feii_flux_4434_4684", value=max(float(feii_flux), 0.0), min=0.0, vary=include_feii)
    params.add("feii_shift_kms", value=float(feii_shift), min=-1500.0, max=1500.0, vary=include_feii)
    params.add("feii_extra_sigma_kms", value=max(float(feii_extra_sigma), 0.0), min=0.0, max=1500.0, vary=False)
    params.add("feii_smooth_sigma_kms", value=max(float(feii_smooth_sigma), 0.0), min=0.0, max=1500.0, vary=False)
    params.add("feii_amp", expr="feii_flux_4434_4684")
    params.add("feii_velocity", expr="feii_shift_kms")
    params.add("feii_sigma", expr="feii_extra_sigma_kms")

    params.add("hb_velocity", value=float(hb_params.get("narrow_velocity", 0.0)), min=-800.0, max=800.0)
    params.add("hb_sigma", value=float(hb_params.get("narrow_sigma", 200.0)), min=50.0, max=1000.0)
    params.add("hb_narrow_flux", value=max(float(line_flux.get("Hb_narrow", 1.0)), 0.0), min=0.0)
    params.add("oiii_4959_flux", value=max(float(line_flux.get("OIII_4959", 1.0)), 0.0), min=0.0)
    params.add("oiii_5007_flux", expr="2.98 * oiii_4959_flux")
    params.add("oiii_wing_velocity", value=float(hb_params.get("oiii_wing_velocity", -200.0)), min=-1500.0, max=1000.0)
    params.add("oiii_wing_sigma", value=float(hb_params.get("oiii_wing_sigma", 600.0)), min=200.0, max=2000.0)
    params.add("oiii_4959_wing_flux", value=max(float(line_flux.get("OIII_4959_wing", 0.0)), 0.0), min=0.0)
    params.add("oiii_5007_wing_flux", expr="2.98 * oiii_4959_wing_flux")
    _add_joint_broad_params(params, "hb", line_flux, hb_params, include_broad_balmer)

    params.add("ha_velocity", value=float(ha_params.get("narrow_velocity", 0.0)), min=-800.0, max=800.0)
    params.add("ha_sigma", value=float(ha_params.get("narrow_sigma", 200.0)), min=50.0, max=1000.0)
    params.add("ha_narrow_flux", value=max(float(line_flux.get("Ha_narrow", 1.0)), 0.0), min=0.0)
    params.add("nii_6548_flux", value=max(float(line_flux.get("NII_6548", 1.0)), 0.0), min=0.0)
    params.add("nii_6584_flux", expr="2.96 * nii_6548_flux")
    params.add("sii_6716_flux", value=max(float(line_flux.get("SII_6716", 1.0)), 0.0), min=0.0)
    params.add("sii_6731_flux", value=max(float(line_flux.get("SII_6731", 1.0)), 0.0), min=0.0)
    _add_joint_broad_params(params, "ha", line_flux, ha_params, include_broad_balmer)
    params.add("oii_velocity", value=float(oii_params.get("oii_velocity", 0.0)), min=-800.0, max=800.0)
    params.add("oii_sigma", value=float(oii_params.get("oii_sigma", 180.0)), min=50.0, max=1000.0)
    params.add("oii_3727_blend_flux", value=max(float(line_flux.get("OII_3727_blend", 1.0)), 0.0), min=0.0)
    return params


def _seed_joint_parameters(params: Any, values: dict[str, float]) -> None:
    seed_values = dict(values)
    if "feii_flux_4434_4684" not in seed_values and "feii_amp" in seed_values:
        seed_values["feii_flux_4434_4684"] = seed_values["feii_amp"]
    if "feii_shift_kms" not in seed_values and "feii_velocity" in seed_values:
        seed_values["feii_shift_kms"] = seed_values["feii_velocity"]
    if "feii_extra_sigma_kms" not in seed_values and "feii_sigma" in seed_values:
        seed_values["feii_extra_sigma_kms"] = seed_values["feii_sigma"]
    for name, value in seed_values.items():
        if name not in params:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        param = params[name]
        if getattr(param, "expr", None):
            continue
        lower = getattr(param, "min", -np.inf)
        upper = getattr(param, "max", np.inf)
        if np.isfinite(lower):
            numeric = max(numeric, float(lower))
        if np.isfinite(upper):
            numeric = min(numeric, float(upper))
        param.set(value=numeric)


def _first_finite(values: dict[str, Any], names: tuple[str, ...], default: float) -> float:
    for name in names:
        if name not in values:
            continue
        try:
            value = float(values[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return float(default)


def _add_joint_broad_params(params: Any, prefix: str, line_flux: dict[str, float], source_params: dict[str, float], include: bool) -> None:
    label = "Hb" if prefix == "hb" else "Ha"
    params.add(f"{prefix}_broad_1_flux", value=max(float(line_flux.get(f"{label}_broad_1", 1.0 if include else 0.0)), 0.0), min=0.0, vary=include)
    params.add(f"{prefix}_broad_1_velocity", value=float(source_params.get(f"{prefix}_broad_1_velocity", 0.0)), min=-5000.0, max=5000.0, vary=include)
    params.add(f"{prefix}_broad_1_sigma", value=float(source_params.get(f"{prefix}_broad_1_sigma", 1500.0)), min=800.0, max=6000.0, vary=include)
    params.add(f"{prefix}_broad_2_flux", value=max(float(line_flux.get(f"{label}_broad_2", 1.0 if include else 0.0)), 0.0), min=0.0, vary=include)
    params.add(f"{prefix}_broad_2_velocity", value=float(source_params.get(f"{prefix}_broad_2_velocity", 0.0)), min=-5000.0, max=5000.0, vary=include)
    params.add(f"{prefix}_broad_2_sigma", value=float(source_params.get(f"{prefix}_broad_2_sigma", 4000.0)), min=1500.0, max=12000.0, vary=include)


def _estimate_powerlaw_amp(spec: AGNSpectrum, stellar_result: Any) -> float:
    residual = spec.flux - np.asarray(stellar_result.stellar_model, dtype=float)
    mask = (spec.wave >= 5080.0) & (spec.wave <= 5120.0) & np.isfinite(residual)
    if np.count_nonzero(mask) < 3:
        mask = np.isfinite(residual)
    return max(float(np.nanmedian(residual[mask])), 0.0) if np.any(mask) else 0.0


def _joint_line_flux_from_params(params: Any) -> dict[str, float]:
    flux = {
        "OII_3727_blend": float(params["oii_3727_blend_flux"].value),
        "Hb_narrow": float(params["hb_narrow_flux"].value),
        "OIII_4959": float(params["oiii_4959_flux"].value),
        "OIII_5007": float(params["oiii_5007_flux"].value),
        "OIII_4959_wing": float(params["oiii_4959_wing_flux"].value),
        "OIII_5007_wing": float(params["oiii_5007_wing_flux"].value),
        "Hb_broad_1": float(params["hb_broad_1_flux"].value),
        "Hb_broad_2": float(params["hb_broad_2_flux"].value),
        "Ha_narrow": float(params["ha_narrow_flux"].value),
        "NII_6548": float(params["nii_6548_flux"].value),
        "NII_6584": float(params["nii_6584_flux"].value),
        "SII_6716": float(params["sii_6716_flux"].value),
        "SII_6731": float(params["sii_6731_flux"].value),
        "Ha_broad_1": float(params["ha_broad_1_flux"].value),
        "Ha_broad_2": float(params["ha_broad_2_flux"].value),
    }
    flux["Hb_broad"] = flux["Hb_broad_1"] + flux["Hb_broad_2"]
    flux["Hb_broad_total"] = flux["Hb_broad"]
    flux["Hb_broad_flux_total"] = flux["Hb_broad"]
    flux["Ha_broad"] = flux["Ha_broad_1"] + flux["Ha_broad_2"]
    flux["Ha_broad_total"] = flux["Ha_broad"]
    flux["Ha_broad_flux_total"] = flux["Ha_broad"]
    if "feii_flux_4434_4684" in params:
        flux["FeII_4434_4684"] = float(params["feii_flux_4434_4684"].value)
    return flux


def _joint_diagnostics(
    wave: np.ndarray,
    flux: np.ndarray,
    total_model: np.ndarray,
    components: dict[str, np.ndarray],
    params: Any,
    chi2: float,
    bic: float,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "fit_strategy": "iterative_joint",
        "final_joint_chi2": chi2,
        "final_joint_bic": bic,
        "stellar_scale": float(params["stellar_scale"].value),
        "powerlaw_amp": float(params["pl_amp"].value),
        "powerlaw_alpha": float(params["pl_alpha"].value),
        "feii_flux_4434_4684": float(params["feii_flux_4434_4684"].value),
        "feii_shift_kms": float(params["feii_shift_kms"].value),
        "feii_extra_sigma_kms": float(params["feii_extra_sigma_kms"].value),
        "feii_smooth_sigma_kms": float(params["feii_smooth_sigma_kms"].value),
        "feii_broadening_mode": "extra",
        "feii_amp": float(params["feii_amp"].value),
        "feii_velocity": float(params["feii_velocity"].value),
        "feii_sigma": float(params["feii_sigma"].value),
    }
    diagnostics["flag_stellar_scale_large_change"] = bool(abs(diagnostics["stellar_scale"] - 1.0) > 0.30)
    region_defs = {
        "blue_feii": (4434.0, 4684.0),
        "hb_blue_wing": (4700.0, 4820.0),
        "hb_core": (4820.0, 4905.0),
        "oiii_pedestal": (4920.0, 5050.0),
        "red_feii": (5100.0, 5600.0),
    }
    for label, region in region_defs.items():
        mask = (wave >= region[0]) & (wave <= region[1])
        total = _trapz_positive(wave, total_model, mask)
        for comp_name in ("stellar", "powerlaw", "feii", "hbeta_broad_total", "oiii_wing"):
            value = _trapz_positive(wave, components.get(comp_name, np.zeros_like(wave)), mask)
            diagnostics[f"{comp_name}_fraction_{label}"] = value / total if total > 0 else float("nan")
        resid = flux[mask] - total_model[mask]
        diagnostics[f"residual_rms_{label}"] = float(np.sqrt(np.nanmean(resid**2))) if resid.size else float("nan")
    diagnostics["flag_stellar_eats_feii"] = bool(
        diagnostics.get("stellar_fraction_blue_feii", 0.0) > 0.7
        or diagnostics.get("stellar_fraction_red_feii", 0.0) > 0.7
    )
    diagnostics["flag_powerlaw_eats_feii"] = bool(
        diagnostics.get("powerlaw_fraction_blue_feii", 0.0) > 0.7
        or diagnostics.get("powerlaw_fraction_red_feii", 0.0) > 0.7
    )
    diagnostics["flag_broad_hb_eats_feii"] = bool(
        diagnostics.get("hbeta_broad_total_fraction_blue_feii", 0.0) > 0.5
        or diagnostics.get("hbeta_broad_total_fraction_red_feii", 0.0) > 0.3
    )
    diagnostics["flag_broad_hb_eats_oiii_feii_pedestal"] = bool(
        diagnostics.get("hbeta_broad_total_fraction_oiii_pedestal", 0.0) > 0.3
    )
    diagnostics["flag_oiii_wing_eats_feii_pedestal"] = bool(
        diagnostics.get("oiii_wing_fraction_oiii_pedestal", 0.0) > 0.3
    )
    diagnostics["flag_feii_underfit"] = bool(
        diagnostics.get("feii_fraction_blue_feii", 0.0) < 0.05
        and diagnostics.get("feii_fraction_red_feii", 0.0) < 0.05
    )
    return diagnostics


def _window_joint_diagnostics(
    epoch: EpochFitSet,
    do_stellar_refit: bool = False,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "fit_strategy": "window_joint",
        "global_nonstellar_joint_fit": False,
        "do_stellar_refit": bool(do_stellar_refit),
        "fit_quality": "good",
        "flag_needs_manual_inspection": False,
    }
    inspection_flags: list[str] = []
    for label, result in (
        ("oii", epoch.oii),
        ("hb_oiii", epoch.hb_oiii),
        ("ha_nii_sii", epoch.ha_nii_sii),
        ("blue_balmer", epoch.blue_balmer),
    ):
        if result is None:
            diagnostics[f"{label}_success"] = False
            if label != "blue_balmer":
                inspection_flags.append(f"{label}_missing")
            continue
        diagnostics[f"{label}_success"] = bool(result.success)
        diagnostics[f"{label}_chi2"] = float(result.chi2)
        diagnostics[f"{label}_redchi"] = float(result.redchi)
        diagnostics[f"{label}_bic"] = float(result.bic)
        diagnostics[f"{label}_ndata"] = int(result.ndata)
        diagnostics[f"{label}_nvarys"] = int(result.nvarys)
        residual = np.asarray(result.residual_flux, dtype=float)
        scaled_residual = _window_scaled_residual(result)
        diagnostics[f"{label}_residual_median_scaled"] = float(
            np.nanmedian(scaled_residual)
        )
        diagnostics[f"{label}_residual_rms_scaled"] = float(
            np.sqrt(np.nanmean(scaled_residual**2))
        )
        diagnostics[f"{label}_max_abs_residual_scaled"] = float(
            np.nanmax(np.abs(scaled_residual))
        )
        diagnostics[f"{label}_model_nan_count"] = int(
            np.count_nonzero(~np.isfinite(result.model_flux))
        )
        diagnostics[f"{label}_residual_nan_count"] = int(
            np.count_nonzero(~np.isfinite(residual))
        )
        if not result.success:
            if label != "blue_balmer":
                inspection_flags.append(f"{label}_fit_failed")
        if not np.isfinite(result.chi2) or not np.isfinite(result.redchi):
            if label != "blue_balmer":
                inspection_flags.append(f"{label}_nonfinite_chi2")
        if np.isfinite(result.redchi) and result.redchi > 25.0:
            inspection_flags.append(f"{label}_large_redchi")
        if abs(diagnostics[f"{label}_residual_median_scaled"]) > 3.0:
            inspection_flags.append(f"{label}_systematic_residual")

    final_refit = getattr(epoch, "final_refit", None)
    if final_refit is not None:
        diagnostics["final_flux_ratio_refit_success"] = bool(final_refit.success)
        diagnostics["final_flux_ratio_refit_chi2"] = float(final_refit.chi2)
        diagnostics["final_flux_ratio_refit_redchi"] = float(final_refit.redchi)
        diagnostics["final_flux_ratio_refit_bic"] = float(final_refit.bic)
        diagnostics["final_flux_ratio_refit_nvarys"] = int(final_refit.nvarys)
        for name, value in (final_refit.fit_info.get("component_scales", {}) or {}).items():
            diagnostics[f"final_flux_ratio_refit_{name}"] = float(value)
        if not final_refit.success:
            inspection_flags.append("final_flux_ratio_refit_failed")
    else:
        diagnostics["final_flux_ratio_refit_success"] = False
        if "final_flux_ratio_refit_error" in diagnostics:
            inspection_flags.append("final_flux_ratio_refit_error")
        if diagnostics[f"{label}_model_nan_count"] or diagnostics[f"{label}_residual_nan_count"]:
            inspection_flags.append(f"{label}_nan_values")
        for key, value in (result.fit_info or {}).items():
            if key.startswith("flag_") and bool(value):
                diagnostics[f"{label}_{key}"] = True
                if key not in {"flag_broad_balmer_weak", "flag_feii_weak"}:
                    inspection_flags.append(f"{label}_{key}")

    if inspection_flags:
        diagnostics["fit_quality"] = "needs_inspection"
        diagnostics["flag_needs_manual_inspection"] = True
    diagnostics["inspection_flags"] = sorted(set(inspection_flags))
    return diagnostics


def _window_scaled_residual(result: Any) -> np.ndarray:
    residual = np.asarray(result.residual_flux, dtype=float)
    if residual.size == 0:
        return np.array([], dtype=float)
    redchi = float(getattr(result, "redchi", np.nan))
    if np.isfinite(redchi) and redchi >= 0.0:
        scale = float(np.sqrt(max(redchi, 1.0e-12)))
        return residual / scale
    return residual


def _trapz_positive(wave: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float:
    good = mask & np.isfinite(wave) & np.isfinite(values)
    if np.count_nonzero(good) < 2:
        return 0.0
    return float(np.trapz(np.clip(values[good], 0.0, None), wave[good]))


def _fit_with_optional_mc(
    spec: AGNSpectrum,
    fit_func,
    n_mc: int,
    **fit_kwargs: Any,
) -> EmissionFitResult:
    from echospec.fitting import monte_carlo_fit

    if n_mc > 0:
        return monte_carlo_fit(spec, fit_func, n_mc=n_mc, **fit_kwargs)
    return fit_func(spec, **fit_kwargs)


def _resolve_templates(
    templates: Any | None,
) -> tuple[ArrayLike | None, ArrayLike | None]:
    if templates is None:
        return None, None
    if isinstance(templates, dict):
        return templates.get("template_wave"), templates.get("template_flux")
    if isinstance(templates, tuple) and len(templates) == 2:
        return templates
    raise ValueError(
        "templates must be None, a (template_wave, template_flux) tuple, or a dict."
    )


def _resolve_stellar_templates(
    template_wave: ArrayLike | None,
    template_flux: ArrayLike | None,
    template_library: str = "miles",
    max_templates: int | None = None,
    debug: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    if template_wave is not None and template_flux is not None:
        return template_wave, template_flux

    from echospec.templates import load_template_library

    try:
        return load_template_library(
            library=template_library,
            max_templates=max_templates,
            debug=debug,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No stellar templates found. Please run: "
            "python scripts/download_ppxf_templates.py "
            f"--library {template_library}"
        ) from exc


def format_epoch_fit_report(epoch: EpochFitSet) -> str:
    """Format a detailed human-readable report for one fitted epoch."""

    lines = [
        f"EchoSpec fit report: {epoch.object_id or '(unknown object)'}"
        f" [{epoch.survey or 'unknown survey'}]"
    ]
    if epoch.diagnostics:
        lines.extend(
            [
                "strategy:",
                f"  fit_strategy: {epoch.diagnostics.get('fit_strategy')}",
                f"  fit_quality: {epoch.diagnostics.get('fit_quality')}",
                "  manual inspection: "
                f"{epoch.diagnostics.get('flag_needs_manual_inspection', False)}",
            ]
        )
        flags = epoch.diagnostics.get("inspection_flags") or []
        if flags:
            lines.append(f"  inspection flags: {', '.join(map(str, flags))}")
    if epoch.stellar is not None:
        stellar = epoch.stellar
        diagnostics = getattr(stellar, "diagnostics", {}) or {}
        lines.extend(
            [
                "stellar:",
                f"  success: True",
                f"  velocity: {_fmt(getattr(stellar, 'velocity', np.nan))} km/s",
                f"  sigma: {_fmt(getattr(stellar, 'sigma', np.nan))} km/s",
                f"  chi2: {_fmt(getattr(stellar, 'chi2', np.nan))}",
                f"  goodpixels: {len(getattr(stellar, 'goodpixels', []))}",
                "  AGN power-law: "
                f"enabled={diagnostics.get('include_agn_powerlaw', False)} "
                f"templates={diagnostics.get('n_agn_powerlaw_templates', 0)} "
                f"nonzero={diagnostics.get('agn_powerlaw_weights_nonzero_count', 0)} "
                f"weight_sum={_fmt(diagnostics.get('agn_powerlaw_weight_sum', np.nan))}",
                "  pPXF mdegree: "
                f"initial={diagnostics.get('initial_ppxf_mdegree', diagnostics.get('ppxf_mdegree'))} "
                f"final={diagnostics.get('final_ppxf_mdegree', diagnostics.get('ppxf_mdegree'))}",
            ]
        )
    for label, result in (
        ("[O II]", epoch.oii),
        ("Hbeta + [O III]", epoch.hb_oiii),
        ("Halpha + [N II] + [S II]", epoch.ha_nii_sii),
    ):
        lines.append(f"{label}:")
        if result is None:
            lines.append("  not fitted")
            continue
        lines.extend(
            [
                f"  success: {result.success}",
                f"  message: {result.message}",
                f"  chi2/redchi/BIC: {_fmt(result.chi2)} / {_fmt(result.redchi)} / {_fmt(result.bic)}",
                f"  ndata/nvarys: {result.ndata} / {result.nvarys}",
            ]
        )
        if result.fit_info:
            lines.append(
                "  broad Balmer BIC: "
                f"selected={result.fit_info.get('broad_balmer_selected')} "
                f"delta={_fmt(result.fit_info.get('delta_bic_broad'))} "
                f"threshold={result.fit_info.get('bic_threshold')}"
            )
        lines.append("  line fluxes:")
        for name, value in sorted(result.line_flux.items()):
            err = result.line_flux_err.get(name)
            err_text = f" +/- {_fmt(err)}" if err is not None else ""
            lines.append(f"    {name}: {_fmt(value)}{err_text}")
    return "\n".join(lines)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.6g}"


__all__ = [
    "IterativeJointFitResult",
    "JointFitResult",
    "RepeatPairFitResult",
    "final_flux_ratio_refit",
    "fit_single_epoch_iterative_joint",
    "fit_single_epoch_window_joint",
    "fit_single_epoch",
    "fit_repeat_pair",
    "format_epoch_fit_report",
]
