"""DESI single-spectrum pPXF + Fe II fitting pipeline."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .spectrum import H1Spectrum
from .staged import (
    DEFAULT_FEII_SHIFT_GRID,
    DEFAULT_FEII_SIGMA_GRID,
    DEFAULT_HOST_POWERLAW_SLOPES,
    DEFAULT_PPXF_FEII_WINDOWS,
    DEFAULT_STAGE2_EMISSION_MASKS,
    DEFAULT_STAGE2_POWERLAW_GRID,
    DEFAULT_STAGE2_WINDOW,
    continuum_subtracted_spectrum,
    downweight_spectrum_for_host,
    fit_fixed_host_feii,
    load_representative_stellar_templates,
    make_staged_diagnostic_figure,
    write_staged_fits,
)

C_KMS = 299792.458
HB_FEII_WINDOW = (4400.0, 5600.0)
HB_FEII_MASKS = ((4670.0, 4705.0), (4810.0, 4910.0), (4935.0, 5035.0))
MGII_WINDOW = (2200.0, 3090.0)
BLUE_BALMER_WINDOW = (3850.0, 4450.0)
HA_NII_SII_WINDOW = (6250.0, 6800.0)
OII_BLEND = 3727.42
HGAMMA = 4340.47
HDELTA = 4101.74
NEIII_3869 = 3868.76
NEIII_3968 = 3967.47
HEPSILON = 3970.07
HBETA = 4861.33
OIII_4959 = 4958.91
OIII_5007 = 5006.84
HALPHA = 6562.80
OI_6300 = 6300.30
OI_6363 = 6363.78
NII_6548 = 6548.05
NII_6584 = 6583.45
SII_6716 = 6716.44
SII_6731 = 6730.82
DEFAULT_PPXF_FEII_MASK_WINDOWS = DEFAULT_PPXF_FEII_WINDOWS
DEFAULT_LINES = (
    "UV_FeII_2200_3090",
    "MgII_broad",
    "MgII_broad_1",
    "MgII_broad_2",
    "MgII_narrow",
    "OII_3727_blend",
    "Hgamma_narrow",
    "Hgamma_narrow_flux",
    "Hgamma_broad",
    "Hgamma_broad_total",
    "Hgamma_broad_flux_total",
    "Hdelta_narrow",
    "Hdelta_narrow_flux",
    "Hdelta_broad",
    "Hdelta_broad_total",
    "Hdelta_broad_flux_total",
    "NeIII_3869",
    "NeIII_3869_flux",
    "NeIII_3968",
    "NeIII_3968_flux",
    "Hepsilon_narrow",
    "Hepsilon_narrow_flux",
    "Hb_narrow",
    "Hb_broad",
    "Hb_broad_total",
    "Hb_broad_1",
    "Hb_broad_2",
    "Hb_broad_flux_total",
    "OIII_4959",
    "OIII_5007",
    "Ha_narrow",
    "Ha_broad",
    "Ha_broad_total",
    "Ha_broad_1",
    "Ha_broad_2",
    "Ha_broad_flux_total",
    "NII_6548",
    "NII_6584",
    "OI_6300",
    "OI_6363",
    "SII_6716",
    "SII_6731",
)


@dataclass
class EchoSpecFitConfig:
    """Configuration for one EchoSpecFit single-spectrum run."""

    template_library: str = "emiles_basti_ch"
    max_templates: int | None = None
    representative_stellar_templates: bool = True
    stellar_template_path: str | Path | None = None
    representative_n_metallicities: int = 6
    stellar_mdegree: int = 12
    stellar_degree: int = -1
    stellar_mask_width_kms: float = 8000.0
    ppxf_mask_feii_windows: bool = True
    ppxf_feii_mask_windows: tuple[tuple[float, float], ...] = DEFAULT_PPXF_FEII_MASK_WINDOWS
    ppxf_feii_error_scale: float = 1.0e6
    include_agn_powerlaw: bool = True
    agn_powerlaw_slopes: tuple[float, ...] = DEFAULT_HOST_POWERLAW_SLOPES
    staged_continuum: bool = True
    staged_fit_window: tuple[float, float] = DEFAULT_STAGE2_WINDOW
    staged_emission_masks: tuple[tuple[float, float], ...] = DEFAULT_STAGE2_EMISSION_MASKS
    staged_powerlaw_grid: tuple[float, ...] = DEFAULT_STAGE2_POWERLAW_GRID
    staged_feii_sigma_grid: tuple[float, ...] = DEFAULT_FEII_SIGMA_GRID
    staged_feii_shift_grid: tuple[float, ...] = DEFAULT_FEII_SHIFT_GRID
    staged_host_scale_max: float = 5.0
    staged_subtract_feii_before_lines: bool = True
    include_feii: bool = True
    feii_template_path: str | Path | None = None
    feii_scale_mode: str = "single"
    include_mgii: bool = True
    include_uv_feii: bool = True
    uv_feii_template_path: str | Path | None = None
    include_mgii_broad: bool = True
    force_mgii_broad: bool = True
    include_mgii_narrow: bool = False
    mgii_broad_model: str = "double"
    mgii_narrow_mode: str = "blend"
    mgii_broad_snr_threshold: float = 0.5
    mgii_window: tuple[float, float] = MGII_WINDOW
    low_host_refit_threshold: float = 0.30
    include_oiii_wing: bool = True
    broad_bic_threshold: float = 10.0
    force_broad_hb: bool = False
    broad_hb_model: str = "double"
    hb_broad_max_component_fwhm_kms: float = 20000.0
    broad_ha_model: str = "double"
    hb_window: tuple[float, float] = HB_FEII_WINDOW
    include_blue_balmer: bool = True
    include_blue_balmer_neiii: bool = True
    include_blue_balmer_hepsilon: bool = False
    force_broad_blue_balmer: bool = False
    joint_refit: bool = False
    no_local_continuum_refit: bool = True
    no_local_continuum_err_floor_fraction: float = 0.0
    no_local_forced_blue_broad_min_scale: float = 1.0
    no_local_blue_broad_profile: str = "filtered_hbeta"
    no_local_blue_broad_max_sigma_kms: float = 3000.0
    no_local_blue_broad_max_abs_velocity_kms: float = 2000.0
    make_plots: bool = True
    make_line_plots: bool = True
    write_array_products: bool = False
    write_fits_products: bool = True


@dataclass
class EchoSpecFitResult:
    """In-memory result from one EchoSpecFit run."""

    spectrum: H1Spectrum
    spec: Any
    stellar: Any
    low_host_refit: dict[str, Any]
    residual_spec: Any
    fits: dict[str, dict[str, Any]]
    joint: dict[str, Any]
    no_local_continuum: dict[str, Any]
    summary: dict[str, Any]
    report_text: str

    @property
    def staged_continuum(self) -> dict[str, Any]:
        """Return the staged host + power-law + Fe II result.

        ``low_host_refit`` remains the stored field for compatibility with
        result files and user scripts created before the staged workflow.
        """

        return self.low_host_refit

    @property
    def oii(self) -> Any | None:
        return self.fits["oii"]["result"]

    @property
    def blue_balmer(self) -> Any | None:
        return self.fits["blue_balmer"]["result"]

    @property
    def hb_oiii(self) -> Any | None:
        return self.fits["hb_oiii"]["result"]

    @property
    def ha_nii_sii(self) -> Any | None:
        return self.fits["ha_nii_sii"]["result"]

    @property
    def joint_result(self) -> Any | None:
        return self.joint["result"]

    def to_detail_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Return a JSON-friendly detailed result."""

        staged_detail = _json_safe(
            {
                key: value
                for key, value in self.staged_continuum.items()
                if include_arrays
                or key
                not in {
                    "valid_mask",
                    "fit_mask",
                    "emission_mask",
                    "ppxf_feii_ignored_mask",
                    "host_model",
                    "powerlaw_model",
                    "feii_model",
                    "total_model",
                    "residual",
                    "pull",
                }
            }
        )
        return {
            "object_id": self.spectrum.object_id,
            "survey": self.spectrum.survey,
            "stellar": self.stellar.to_dict(include_arrays=include_arrays),
            "staged_continuum": staged_detail,
            "low_host_refit": staged_detail,
            "fits": {
                key: {
                    "error": payload["error"],
                    "result": (
                        None
                        if payload["result"] is None
                        else payload["result"].to_dict(include_arrays=include_arrays)
                    ),
                }
                for key, payload in self.fits.items()
            },
            "joint": {
                "error": self.joint["error"],
                "result": (
                    None
                    if self.joint["result"] is None
                    else self.joint["result"].to_dict(include_arrays=include_arrays)
                ),
            },
            "no_local_continuum": _no_local_continuum_detail(
                self.no_local_continuum,
                include_arrays=include_arrays,
            ),
            "summary": _json_safe(self.summary),
        }


# Preserve scripts written before the public EchoSpecFit rename.
DesiPecFitConfig = EchoSpecFitConfig
DesiPecFitResult = EchoSpecFitResult


def fit_h1spectrum(
    spectrum: H1Spectrum | Any,
    config: EchoSpecFitConfig | None = None,
    *,
    template_wave: np.ndarray | None = None,
    template_flux: np.ndarray | None = None,
) -> EchoSpecFitResult:
    """Fit one H1Spectrum with pPXF host+power law, Fe II, and line windows."""

    cfg = EchoSpecFitConfig() if config is None else config
    h1 = spectrum if isinstance(spectrum, H1Spectrum) else H1Spectrum.from_single(spectrum)
    fit_h1 = h1.mask_bad_pixels()
    spec = fit_h1.to_agn_spectrum()
    if spec.z != 0:
        spec = spec.to_rest_frame()

    if template_wave is None or template_flux is None:
        if cfg.representative_stellar_templates:
            template_wave, template_flux, template_names = (
                load_representative_stellar_templates(
                    library=cfg.template_library,
                    local_path=cfg.stellar_template_path,
                    n_metallicities=cfg.representative_n_metallicities,
                )
            )
        else:
            from echospec.templates import load_template_library

            template_wave, template_flux = load_template_library(
                library=cfg.template_library,
                max_templates=cfg.max_templates,
                local_path=cfg.stellar_template_path,
                debug=False,
            )
            template_names = []
    else:
        template_names = []

    from echospec.fitting import (
        fit_blue_balmer_window,
        fit_ha_nii_sii,
        fit_hb_oiii,
        fit_hb_oiii_bic,
        fit_oii,
    )
    from echospec.mgii import fit_mgii_uvfeii
    from echospec.stellar import fit_stellar_ppxf

    if cfg.ppxf_mask_feii_windows:
        ppxf_spec, ppxf_feii_ignored = downweight_spectrum_for_host(
            spec,
            windows=cfg.ppxf_feii_mask_windows,
            error_scale=cfg.ppxf_feii_error_scale,
        )
    else:
        ppxf_spec = spec
        ppxf_feii_ignored = np.zeros_like(spec.wave, dtype=bool)
    stellar = fit_stellar_ppxf(
        ppxf_spec,
        template_wave,
        template_flux,
        mdegree=cfg.stellar_mdegree,
        degree=cfg.stellar_degree,
        mask_width_kms=cfg.stellar_mask_width_kms,
        include_agn_powerlaw=cfg.include_agn_powerlaw,
        agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
        debug=False,
    )
    stellar.diagnostics.update(
        {
            "desipecfit_ppxf_feii_windows_masked": bool(
                cfg.ppxf_mask_feii_windows
            ),
            "desipecfit_ppxf_feii_windows_downweighted": bool(
                cfg.ppxf_mask_feii_windows
            ),
            "desipecfit_ppxf_feii_mask_windows": [
                list(window) for window in cfg.ppxf_feii_mask_windows
            ],
            "desipecfit_ppxf_feii_error_scale": float(cfg.ppxf_feii_error_scale),
            "desipecfit_ppxf_n_pixels_after_feii_window_mask": int(
                np.count_nonzero(~ppxf_feii_ignored)
            ),
            "desipecfit_stellar_template_names": template_names,
            "desipecfit_stellar_template_count": int(template_flux.shape[1]),
        }
    )
    host_fraction_hb = _host_fraction(
        spec.wave,
        stellar.stellar_model,
        stellar.agn_model,
        region=cfg.hb_window,
    )
    low_host_refit = _safe_staged_continuum_feii_refit(
        spec,
        stellar,
        cfg,
        host_fraction_hb=host_fraction_hb,
    )
    low_host_refit["ppxf_feii_ignored_mask"] = ppxf_feii_ignored
    residual_spec = _residual_after_staged_refit(
        spec,
        stellar,
        low_host_refit,
        subtract_feii=cfg.staged_subtract_feii_before_lines,
    )
    staged_feii_subtracted = bool(
        low_host_refit.get("usable") and cfg.staged_subtract_feii_before_lines
    )

    hb_fit_func = fit_hb_oiii if cfg.force_broad_hb else fit_hb_oiii_bic
    hb_fit_kwargs: dict[str, Any] = {
        "include_feii": bool(cfg.include_feii and not staged_feii_subtracted),
        "feii_template_path": cfg.feii_template_path,
        "feii_scale_mode": cfg.feii_scale_mode,
        "include_oiii_wing": cfg.include_oiii_wing,
        "broad_hb_model": cfg.broad_hb_model,
        "hb_window": cfg.hb_window,
    }
    if cfg.force_broad_hb:
        hb_fit_kwargs["include_broad_hb"] = True
    else:
        hb_fit_kwargs["bic_threshold"] = cfg.broad_bic_threshold
    uv_feii_template_path, uv_feii_template_source = _resolve_uv_feii_template_path(
        cfg.uv_feii_template_path
    )

    fits = {
        "mgii_uv": _safe_mgii_fit(
            residual_spec,
            cfg,
            uv_feii_template_path=uv_feii_template_path,
            max_component_fwhm_kms=cfg.hb_broad_max_component_fwhm_kms,
        )
        if cfg.include_mgii
        else {"result": None, "error": "Mg II fitting disabled by config."},
        "oii": _safe_fit(fit_oii, residual_spec, mode="blend"),
        "hb_oiii": _safe_hb_fit(
            hb_fit_func,
            residual_spec,
            hb_fit_kwargs,
            max_component_fwhm_kms=cfg.hb_broad_max_component_fwhm_kms,
        ),
        "blue_balmer": {"result": None, "error": "filled after Hbeta fit"},
        "ha_nii_sii": _safe_ha_fit(
            residual_spec,
            broad_ha_model=cfg.broad_ha_model,
            max_component_fwhm_kms=cfg.hb_broad_max_component_fwhm_kms,
        ),
    }
    effective_force_broad_blue_balmer = bool(
        cfg.force_broad_blue_balmer or cfg.force_broad_hb
    )
    fits["blue_balmer"] = _safe_fit(
        fit_blue_balmer_window,
        residual_spec,
        initial_fit_result=fits["hb_oiii"]["result"],
        include_blue_balmer=cfg.include_blue_balmer,
        include_balmer_continuum=False,
        skip_balmer_continuum_region=True,
        include_neiii=cfg.include_blue_balmer_neiii,
        include_hepsilon=cfg.include_blue_balmer_hepsilon,
        force_broad_blue_balmer=effective_force_broad_blue_balmer,
    )
    blue_result = fits["blue_balmer"]["result"]
    if blue_result is not None:
        blue_result.fit_info.update(
            {
                "desipecfit_force_broad_blue_balmer": effective_force_broad_blue_balmer,
                "desipecfit_force_broad_blue_balmer_reason": (
                    "forced_explicitly"
                    if cfg.force_broad_blue_balmer
                    else (
                        "forced_with_broad_hb"
                        if cfg.force_broad_hb
                        else "automatic_blue_balmer_selection"
                    )
                ),
            }
        )
        if not cfg.include_blue_balmer_neiii:
            _remove_fit_components(blue_result, ("NeIII_3869", "NeIII_3968"))
        if not cfg.include_blue_balmer_hepsilon:
            _remove_fit_components(blue_result, ("Hepsilon_narrow",))
    if fits["mgii_uv"]["result"] is not None:
        fits["mgii_uv"]["result"].fit_info.update(
            {
                "desipecfit_uv_feii_template_source": uv_feii_template_source,
                "desipecfit_uv_feii_template_path": (
                    "" if uv_feii_template_path is None else str(uv_feii_template_path)
                ),
            }
        )
    if fits["hb_oiii"]["result"] is not None:
        fits["hb_oiii"]["result"].fit_info.update(
            {
                "desipecfit_force_broad_hb": bool(cfg.force_broad_hb),
                "desipecfit_hb_model_mode": (
                    "forced_broad_hb" if cfg.force_broad_hb else "bic_selected"
                ),
            }
        )
    joint = _safe_joint_fit(
        spec,
        stellar,
        fits,
        agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
        enabled=cfg.joint_refit,
    )
    no_local_continuum = _safe_no_local_continuum_refit(
        spec,
        stellar,
        low_host_refit,
        fits,
        cfg,
    )
    summary = _summary_row(h1, stellar, low_host_refit, fits, joint, no_local_continuum)
    report_text = _format_report(
        h1,
        stellar,
        low_host_refit,
        fits,
        joint,
        no_local_continuum,
    )
    return EchoSpecFitResult(
        spectrum=h1,
        spec=spec,
        stellar=stellar,
        low_host_refit=low_host_refit,
        residual_spec=residual_spec,
        fits=fits,
        joint=joint,
        no_local_continuum=no_local_continuum,
        summary=summary,
        report_text=report_text,
    )


def fit_spectrum_file(
    path: str | Path,
    config: EchoSpecFitConfig | None = None,
    *,
    z: float = 0.0,
    survey: str = "DESI",
    object_id: str | None = None,
    template_wave: np.ndarray | None = None,
    template_flux: np.ndarray | None = None,
) -> EchoSpecFitResult:
    """Load and fit one text spectrum."""

    spectrum = H1Spectrum.from_txt(path, z=z, survey=survey, object_id=object_id)
    return fit_h1spectrum(
        spectrum,
        config,
        template_wave=template_wave,
        template_flux=template_flux,
    )


def fit_directory(
    input_path: str | Path,
    outdir: str | Path,
    config: EchoSpecFitConfig | None = None,
    *,
    pattern: str = "*_desi.txt",
    z: float = 0.0,
    survey: str = "DESI",
) -> list[dict[str, Any]]:
    """Fit one file or all matching text spectra in a directory."""

    cfg = EchoSpecFitConfig() if config is None else config
    source = Path(input_path).expanduser()
    output_dir = Path(outdir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(source.glob(pattern)) if source.is_dir() else [source]
    if not paths:
        raise FileNotFoundError(f"No input spectra matched {source}/{pattern}.")

    if cfg.representative_stellar_templates:
        template_wave, template_flux, _ = load_representative_stellar_templates(
            library=cfg.template_library,
            local_path=cfg.stellar_template_path,
            n_metallicities=cfg.representative_n_metallicities,
        )
    else:
        from echospec.templates import load_template_library

        template_wave, template_flux = load_template_library(
            library=cfg.template_library,
            max_templates=cfg.max_templates,
            local_path=cfg.stellar_template_path,
            debug=False,
        )

    rows: list[dict[str, Any]] = []
    for path in paths:
        start = time.time()
        row: dict[str, Any] = {
            "object_id": path.stem,
            "survey": survey,
            "source_file": str(path),
            "elapsed_s": np.nan,
            "error": "",
        }
        try:
            result = fit_spectrum_file(
                path,
                cfg,
                z=z,
                survey=survey,
                object_id=path.stem,
                template_wave=template_wave,
                template_flux=template_flux,
            )
            product_paths = save_fit_result(output_dir, result, cfg)
            row.update(result.summary)
            row.update(product_paths)
        except Exception as exc:
            row["error"] = repr(exc)
        row["elapsed_s"] = time.time() - start
        rows.append(row)

    _write_csv(output_dir / "desipecfit_epoch_fit.csv", rows)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "input": str(source),
                "pattern": pattern,
                "survey": survey,
                "z": z,
                "template_library": cfg.template_library,
                "max_templates": cfg.max_templates,
                "representative_stellar_templates": cfg.representative_stellar_templates,
                "representative_n_metallicities": cfg.representative_n_metallicities,
                "stellar_mask_width_kms": cfg.stellar_mask_width_kms,
                "ppxf_feii_error_scale": cfg.ppxf_feii_error_scale,
                "staged_continuum": cfg.staged_continuum,
                "staged_subtract_feii_before_lines": cfg.staged_subtract_feii_before_lines,
                "joint_refit": cfg.joint_refit,
                "no_local_continuum_refit": cfg.no_local_continuum_refit,
                "agn_powerlaw_slopes": cfg.agn_powerlaw_slopes,
                "feii_template_path": (
                    None if cfg.feii_template_path is None else str(cfg.feii_template_path)
                ),
                "n_spectra": len(rows),
                "n_success_stellar": int(
                    sum(bool(row.get("stellar_success")) for row in rows)
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return rows


def save_fit_result(
    outdir: str | Path,
    result: EchoSpecFitResult,
    config: EchoSpecFitConfig | None = None,
) -> dict[str, str]:
    """Write reports, staged FITS products, and diagnostic plots for one fit."""

    cfg = EchoSpecFitConfig() if config is None else config
    output_dir = Path(outdir).expanduser()
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"
    products_dir = output_dir / "fits"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if cfg.make_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)
    if cfg.write_fits_products:
        products_dir.mkdir(parents=True, exist_ok=True)

    object_id = result.spectrum.object_id
    report_txt = reports_dir / f"{object_id}_fit_report.txt"
    report_json = reports_dir / f"{object_id}_fit_detail.json"
    report_txt.write_text(result.report_text, encoding="utf-8")
    report_json.write_text(
        json.dumps(
            result.to_detail_dict(include_arrays=cfg.write_array_products),
            indent=2,
        ),
        encoding="utf-8",
    )
    paths = {
        "report_txt": str(report_txt),
        "report_json": str(report_json),
    }
    if cfg.write_fits_products and result.low_host_refit.get("usable"):
        product_fits = products_dir / f"{object_id}_staged_host_feii.fits"
        write_staged_fits(
            product_fits,
            spectrum=result.spectrum,
            rest_spec=result.spec,
            stellar=result.stellar,
            staged=result.low_host_refit,
        )
        paths["staged_fits"] = str(product_fits)
    if cfg.make_plots:
        paths.update(
            _save_plots(
                plots_dir,
                result,
                include_line_plots=cfg.make_line_plots,
            )
        )
    return paths


def _safe_fit(fit_func: Any, spec: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return {"result": fit_func(spec, **kwargs), "error": ""}
    except Exception as exc:
        return {"result": None, "error": repr(exc)}


def _safe_hb_fit(
    fit_func: Any,
    spec: Any,
    fit_kwargs: dict[str, Any],
    *,
    max_component_fwhm_kms: float,
) -> dict[str, Any]:
    payload = _safe_fit(fit_func, spec, **fit_kwargs)
    result = payload.get("result")
    if result is None:
        primary_error = str(payload.get("error", ""))

        # The multi-start Hbeta solver can occasionally lose every final trial
        # under concurrent BLAS load even though an identical retry converges.
        retry_payload = _safe_fit(fit_func, spec, **fit_kwargs)
        result = retry_payload.get("result")
        if result is not None:
            payload = retry_payload
            _ensure_fit_info(result).update(
                {
                    "desipecfit_hb_recovery": "identical_retry",
                    "desipecfit_hb_initial_error": primary_error,
                }
            )

        if result is None and fit_kwargs.get("include_oiii_wing", False):
            core_kwargs = dict(fit_kwargs)
            core_kwargs["include_oiii_wing"] = False
            core_payload = _safe_fit(fit_func, spec, **core_kwargs)
            result = core_payload.get("result")
            if result is not None:
                payload = core_payload
                _ensure_fit_info(result).update(
                    {
                        "desipecfit_hb_recovery": "retry_without_oiii_wing",
                        "desipecfit_hb_initial_error": primary_error,
                    }
                )

        if result is None:
            from echospec.fitting import fit_hb_oiii

            narrow_kwargs = dict(fit_kwargs)
            narrow_kwargs.pop("bic_threshold", None)
            narrow_kwargs.update(
                {
                    "include_broad_hb": False,
                    "include_oiii_wing": False,
                    "broad_hb_model": "single",
                }
            )
            narrow_payload = _safe_fit(fit_hb_oiii, spec, **narrow_kwargs)
            result = narrow_payload.get("result")
            if result is not None:
                payload = narrow_payload
                _ensure_fit_info(result).update(
                    {
                        "desipecfit_hb_recovery": "narrow_core_fallback",
                        "desipecfit_hb_initial_error": primary_error,
                        "selected_model": "narrow_core",
                        "broad_balmer_selected": False,
                    }
                )

        if result is None:
            return payload

    if (
        "bic_threshold" in fit_kwargs
        and fit_kwargs.get("broad_hb_model") == "double"
        and not bool((getattr(result, "fit_info", {}) or {}).get("broad_balmer_selected"))
        and _rejected_double_hb_has_bic_support(result, fit_kwargs["bic_threshold"])
    ):
        single_kwargs = dict(fit_kwargs)
        single_kwargs["broad_hb_model"] = "single"
        single_payload = _safe_fit(fit_func, spec, **single_kwargs)
        single_result = single_payload.get("result")
        if _single_hb_fallback_is_better(
            result,
            single_result,
            fit_kwargs["bic_threshold"],
        ):
            single_info = _ensure_fit_info(single_result)
            single_info.update(
                {
                    "desipecfit_hb_double_to_single_fallback": True,
                    "desipecfit_hb_fallback_reason": (
                        "double_broad_bic_supported_but_pathological"
                    ),
                    "desipecfit_hb_pre_fallback_selected_model": (
                        (getattr(result, "fit_info", {}) or {}).get("selected_model", "")
                    ),
                    "desipecfit_hb_pre_fallback_bic": float(result.bic),
                    "desipecfit_hb_single_fallback_bic": float(single_result.bic),
                }
            )
            payload = single_payload
            result = single_result
        else:
            _ensure_fit_info(result)["desipecfit_hb_single_fallback_attempted"] = True

    result = _apply_broad_component_limit_metadata(
        result,
        prefix="hb",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="Hbeta",
    )
    if not _has_broad_fwhm_violation(result):
        payload["result"] = result
        return payload

    sequence = ["double"]
    if fit_kwargs.get("broad_hb_model") == "double":
        single_kwargs = dict(fit_kwargs)
        single_kwargs["broad_hb_model"] = "single"
        single_payload = _safe_fit(fit_func, spec, **single_kwargs)
        single_result = _apply_broad_refit_history(
            single_payload.get("result"),
            result,
            sequence + ["single"],
        )
        single_result = _apply_broad_component_limit_metadata(
            single_result,
            prefix="hb",
            max_fwhm_kms=max_component_fwhm_kms,
            line_label="Hbeta",
        )
        if single_result is not None and not _has_broad_fwhm_violation(single_result):
            single_payload["result"] = single_result
            return single_payload
        if single_result is not None:
            result = single_result
            payload = single_payload
            sequence.append("single")

    from echospec.fitting import fit_hb_oiii

    narrow_kwargs = dict(fit_kwargs)
    narrow_kwargs.pop("bic_threshold", None)
    narrow_kwargs["include_broad_hb"] = False
    narrow_kwargs["broad_hb_model"] = "single"
    narrow_payload = _safe_fit(fit_hb_oiii, spec, **narrow_kwargs)
    narrow_result = _apply_broad_refit_history(
        narrow_payload.get("result"),
        result,
        sequence + ["no_broad"],
    )
    _mark_broad_removed_by_limit(
        narrow_result,
        previous_result=result,
        prefix="hb",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="Hbeta",
    )
    if narrow_result is not None:
        narrow_payload["result"] = narrow_result
    return narrow_payload


def _rejected_double_hb_has_bic_support(result: Any, threshold: float) -> bool:
    info = getattr(result, "fit_info", {}) or {}
    narrow_bic = min(
        (
            value
            for value in (
                _as_float(info.get("bic_narrow_core")),
                _as_float(info.get("bic_narrow_wing")),
            )
            if np.isfinite(value)
        ),
        default=float("inf"),
    )
    broad_bic = min(
        (
            value
            for value in (
                _as_float(info.get("bic_broad_core")),
                _as_float(info.get("bic_broad_wing")),
            )
            if np.isfinite(value)
        ),
        default=float("inf"),
    )
    return bool(
        np.isfinite(narrow_bic)
        and np.isfinite(broad_bic)
        and narrow_bic - broad_bic > float(threshold)
    )


def _single_hb_fallback_is_better(
    original_result: Any,
    single_result: Any | None,
    threshold: float,
) -> bool:
    if single_result is None or not bool(getattr(single_result, "success", False)):
        return False
    info = getattr(single_result, "fit_info", {}) or {}
    if not bool(info.get("broad_balmer_selected")):
        return False
    original_bic = _as_float(getattr(original_result, "bic", np.nan))
    single_bic = _as_float(getattr(single_result, "bic", np.nan))
    return bool(
        np.isfinite(original_bic)
        and np.isfinite(single_bic)
        and original_bic - single_bic > float(threshold)
    )


def _safe_ha_fit(
    spec: Any,
    *,
    broad_ha_model: str,
    max_component_fwhm_kms: float,
) -> dict[str, Any]:
    from echospec.fitting import fit_ha_nii_sii

    payload = _safe_fit(
        fit_ha_nii_sii,
        spec,
        include_broad_ha=True,
        broad_ha_model=broad_ha_model,
    )
    result = _apply_broad_component_limit_metadata(
        payload.get("result"),
        prefix="ha",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="Halpha",
    )
    if result is None or not _has_broad_fwhm_violation(result):
        payload["result"] = result
        return payload

    sequence = ["double"]
    if broad_ha_model == "double":
        single_payload = _safe_fit(
            fit_ha_nii_sii,
            spec,
            include_broad_ha=True,
            broad_ha_model="single",
        )
        single_result = _apply_broad_refit_history(
            single_payload.get("result"),
            result,
            sequence + ["single"],
        )
        single_result = _apply_broad_component_limit_metadata(
            single_result,
            prefix="ha",
            max_fwhm_kms=max_component_fwhm_kms,
            line_label="Halpha",
        )
        if single_result is not None and not _has_broad_fwhm_violation(single_result):
            single_payload["result"] = single_result
            return single_payload
        if single_result is not None:
            result = single_result
            payload = single_payload
            sequence.append("single")

    narrow_payload = _safe_fit(
        fit_ha_nii_sii,
        spec,
        include_broad_ha=False,
        broad_ha_model="single",
    )
    narrow_result = _apply_broad_refit_history(
        narrow_payload.get("result"),
        result,
        sequence + ["no_broad"],
    )
    _mark_broad_removed_by_limit(
        narrow_result,
        previous_result=result,
        prefix="ha",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="Halpha",
    )
    if narrow_result is not None:
        narrow_payload["result"] = narrow_result
    return narrow_payload


def _safe_mgii_fit(
    spec: Any,
    cfg: EchoSpecFitConfig,
    *,
    uv_feii_template_path: str | Path | None,
    max_component_fwhm_kms: float,
) -> dict[str, Any]:
    payload = _safe_mgii_fit_once(
        spec,
        cfg,
        uv_feii_template_path=uv_feii_template_path,
        broad_model=cfg.mgii_broad_model,
        include_mgii_broad=cfg.include_mgii_broad,
    )
    result = _apply_broad_component_limit_metadata(
        payload.get("result"),
        prefix="mgii",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="MgII",
    )
    if result is None or not _has_broad_fwhm_violation(result):
        payload["result"] = result
        return payload

    sequence = [cfg.mgii_broad_model]
    if cfg.mgii_broad_model == "double":
        single_payload = _safe_mgii_fit_once(
            spec,
            cfg,
            uv_feii_template_path=uv_feii_template_path,
            broad_model="single",
            include_mgii_broad=True,
        )
        single_result = _apply_broad_refit_history(
            single_payload.get("result"),
            result,
            sequence + ["single"],
        )
        single_result = _apply_broad_component_limit_metadata(
            single_result,
            prefix="mgii",
            max_fwhm_kms=max_component_fwhm_kms,
            line_label="MgII",
        )
        if single_result is not None and not _has_broad_fwhm_violation(single_result):
            single_payload["result"] = single_result
            return single_payload
        if single_result is not None:
            result = single_result
            payload = single_payload
            sequence.append("single")

    narrow_payload = _safe_mgii_fit_once(
        spec,
        cfg,
        uv_feii_template_path=uv_feii_template_path,
        broad_model="single",
        include_mgii_broad=False,
    )
    narrow_result = _apply_broad_refit_history(
        narrow_payload.get("result"),
        result,
        sequence + ["no_broad"],
    )
    _mark_broad_removed_by_limit(
        narrow_result,
        previous_result=result,
        prefix="mgii",
        max_fwhm_kms=max_component_fwhm_kms,
        line_label="MgII",
    )
    if narrow_result is not None:
        narrow_payload["result"] = narrow_result
    return narrow_payload


def _safe_mgii_fit_once(
    spec: Any,
    cfg: EchoSpecFitConfig,
    *,
    uv_feii_template_path: str | Path | None,
    broad_model: str,
    include_mgii_broad: bool,
) -> dict[str, Any]:
    from echospec.mgii import fit_mgii_uvfeii

    payload = _safe_fit(
        fit_mgii_uvfeii,
        spec,
        window=cfg.mgii_window,
        include_uv_feii=cfg.include_uv_feii,
        include_mgii_broad=include_mgii_broad,
        include_mgii_narrow=cfg.include_mgii_narrow,
        uv_feii_template_path=uv_feii_template_path if cfg.include_uv_feii else None,
        broad_model=broad_model,
        narrow_mode=cfg.mgii_narrow_mode,
    )
    result = payload.get("result")
    if result is None:
        return payload
    info = _ensure_fit_info(result)
    info["desipecfit_force_mgii_broad"] = bool(cfg.force_mgii_broad)
    info["desipecfit_mgii_broad_snr_threshold"] = float(
        cfg.mgii_broad_snr_threshold
    )
    if not include_mgii_broad or bool(info.get("accept_mgii_broad", False)):
        return payload

    force_reason = ""
    if cfg.force_mgii_broad:
        force_reason = "default_force_mgii_broad"
    elif _mgii_broad_passes_threshold(info, cfg.mgii_broad_snr_threshold):
        force_reason = "snr_threshold"
    else:
        return payload

    forced = _force_mgii_broad_model(
        spec,
        cfg,
        uv_feii_template_path=uv_feii_template_path,
        broad_model=broad_model,
        original_result=result,
        force_reason=force_reason,
    )
    if forced is not None:
        payload["result"] = forced
    return payload


def _mgii_broad_passes_threshold(info: dict[str, Any], snr_threshold: float) -> bool:
    delta_bic = _as_float(info.get("delta_bic_mgii_broad"))
    snr = _as_float(info.get("mgii_broad_snr"))
    fwhm = _as_float(info.get("mgii_broad_fwhm"))
    return bool(
        np.isfinite(delta_bic)
        and delta_bic > 10.0
        and np.isfinite(snr)
        and snr >= snr_threshold
        and np.isfinite(fwhm)
        and fwhm > 1000.0
    )


def _force_mgii_broad_model(
    spec: Any,
    cfg: EchoSpecFitConfig,
    *,
    uv_feii_template_path: str | Path | None,
    broad_model: str,
    original_result: Any,
    force_reason: str,
) -> Any | None:
    try:
        from echospec.fitting import EmissionFitResult
        from echospec.mgii import _fit_model, load_feii_template, prefit_uv_feii

        window_spec = spec.select_window(*cfg.mgii_window).mask_bad_pixels()
        wave = np.asarray(window_spec.wave, dtype=float)
        flux = np.asarray(window_spec.flux, dtype=float)
        err = np.asarray(window_spec.err, dtype=float)
        if len(wave) < 5:
            return None
        template = load_feii_template(None, uv_feii_template_path)
        uv_template = None if template is None else (template.wave, template.flux)
        can_fit_uv = bool(cfg.include_uv_feii and uv_template is not None)
        prefit = prefit_uv_feii(wave, flux, err, uv_template) if can_fit_uv else None
        forced = _fit_model(
            wave,
            flux,
            err,
            uv_template,
            include_uv_feii=can_fit_uv,
            include_mgii_broad=True,
            include_mgii_narrow=False,
            broad_model=broad_model,
            narrow_mode=cfg.mgii_narrow_mode,
            prefit=prefit,
            max_nfev=3000,
        )
        info = dict(getattr(original_result, "fit_info", {}) or {})
        info.update(
            {
                "selected_model": "M2",
                "accept_mgii_broad": True,
                "secure_mgii_broad": bool(
                    _as_float(info.get("delta_bic_mgii_broad")) > 20.0
                    and _as_float(info.get("mgii_broad_snr")) > 5.0
                    and _as_float(info.get("mgii_broad_fwhm")) > 1200.0
                ),
                "desipecfit_original_selected_model": info.get("selected_model"),
                "desipecfit_original_accept_mgii_broad": info.get(
                    "accept_mgii_broad"
                ),
                "desipecfit_force_mgii_broad": True,
                "desipecfit_mgii_broad_force_reason": force_reason,
                "desipecfit_force_mgii_broad_by_default": (
                    force_reason == "default_force_mgii_broad"
                ),
                "desipecfit_force_mgii_broad_by_snr_threshold": (
                    force_reason == "snr_threshold"
                ),
                "desipecfit_mgii_broad_snr_threshold": float(
                    cfg.mgii_broad_snr_threshold
                ),
            }
        )
        return EmissionFitResult(
            success=bool(forced["success"]),
            message=(
                f"{forced['message']} "
                + (
                    "DESIPecFit kept broad MgII by default."
                    if force_reason == "default_force_mgii_broad"
                    else (
                        "DESIPecFit kept broad MgII with "
                        f"SNR threshold {cfg.mgii_broad_snr_threshold:g}."
                    )
                )
            ),
            params=forced["params"],
            line_flux=forced["line_flux"],
            line_flux_err={},
            model_flux=forced["model"],
            continuum_flux=forced["components"]["continuum"],
            residual_flux=flux - forced["model"],
            chi2=float(forced["chi2"]),
            redchi=float(forced["redchi"]),
            ndata=int(forced["ndata"]),
            nvarys=int(forced["nvarys"]),
            window_name="mgii_uv",
            fit_info=info,
            component_flux=forced["components"],
        )
    except Exception:
        return None


def _as_float(value: Any) -> float:
    try:
        if hasattr(value, "value"):
            value = value.value
        value = float(value)
    except Exception:
        return float("nan")
    return value if np.isfinite(value) else float("nan")


def _apply_broad_component_limit_metadata(
    result: Any | None,
    *,
    prefix: str,
    max_fwhm_kms: float,
    line_label: str,
) -> Any | None:
    if result is None:
        return None
    params = getattr(result, "params", {}) or {}
    if not hasattr(result, "fit_info"):
        return result
    info = _ensure_fit_info(result)
    component_fwhm: dict[int, float] = {}
    for index in _broad_component_indices(params, prefix):
        sigma = _as_float(params.get(f"{prefix}_broad_{index}_sigma"))
        if not (np.isfinite(sigma) and sigma > 0.0):
            continue
        fwhm = 2.35482 * sigma
        component_fwhm[index] = float(fwhm)
        info[f"desipecfit_{prefix}_broad_{index}_sigma_kms"] = float(sigma)
        info[f"desipecfit_{prefix}_broad_{index}_fwhm_kms"] = float(fwhm)

    max_component_fwhm = (
        max(component_fwhm.values()) if component_fwhm else float("nan")
    )
    violating = [
        index for index, fwhm in component_fwhm.items() if fwhm > max_fwhm_kms
    ]
    flagged = bool(
        np.isfinite(max_component_fwhm)
        and np.isfinite(max_fwhm_kms)
        and bool(violating)
    )
    info[f"desipecfit_{prefix}_broad_max_component_fwhm_kms"] = float(
        max_component_fwhm
    )
    info[f"desipecfit_{prefix}_broad_component_fwhm_limit_kms"] = float(
        max_fwhm_kms
    )
    info[f"desipecfit_flag_{prefix}_broad_component_fwhm_gt_limit"] = flagged
    info["desipecfit_flag_broad_component_fwhm_gt_limit"] = flagged
    info["desipecfit_flag_continuum_unreasonable"] = flagged
    info["desipecfit_broad_fwhm_violating_components"] = ",".join(
        f"{prefix}_broad_{index}" for index in violating
    )
    if flagged:
        offending = ", ".join(
            f"{prefix}_broad_{index} FWHM={fwhm:.0f} km/s"
            for index, fwhm in component_fwhm.items()
            if fwhm > max_fwhm_kms
        )
        warning = (
            f"{line_label} broad component FWHM exceeds "
            f"{max_fwhm_kms:.0f} km/s ({offending}); continuum solution is "
            "likely unreasonable, so DESIPecFit will refit without that broad "
            "component."
        )
        warnings = getattr(result, "warnings", None)
        if isinstance(warnings, list) and warning not in warnings:
            warnings.append(warning)
    return result


def _apply_broad_refit_history(
    result: Any | None,
    previous_result: Any | None,
    sequence: list[str],
) -> Any | None:
    if result is None:
        return None
    info = _ensure_fit_info(result)
    previous_info = getattr(previous_result, "fit_info", {}) or {}
    if previous_info.get("desipecfit_broad_fwhm_violating_components"):
        info["desipecfit_removed_broad_components"] = previous_info.get(
            "desipecfit_broad_fwhm_violating_components"
        )
    info["desipecfit_broad_refit_sequence"] = "->".join(sequence)
    info["desipecfit_refit_after_broad_fwhm_limit"] = True
    warnings = getattr(result, "warnings", None)
    previous_warnings = getattr(previous_result, "warnings", None)
    if isinstance(warnings, list) and isinstance(previous_warnings, list):
        for warning in previous_warnings:
            if "FWHM exceeds" in warning and warning not in warnings:
                warnings.append(warning)
    return result


def _mark_broad_removed_by_limit(
    result: Any | None,
    *,
    previous_result: Any | None,
    prefix: str,
    max_fwhm_kms: float,
    line_label: str,
) -> None:
    if result is None:
        return
    info = _ensure_fit_info(result)
    previous_info = getattr(previous_result, "fit_info", {}) or {}
    removed = previous_info.get("desipecfit_broad_fwhm_violating_components", "")
    info[f"desipecfit_flag_{prefix}_broad_component_fwhm_gt_limit"] = True
    info["desipecfit_flag_broad_component_fwhm_gt_limit"] = True
    info["desipecfit_flag_continuum_unreasonable"] = True
    info["desipecfit_removed_broad_components"] = removed
    info[f"desipecfit_{prefix}_broad_component_fwhm_limit_kms"] = float(
        max_fwhm_kms
    )
    warning = (
        f"{line_label} broad component(s) {removed or prefix + '_broad'} exceeded "
        f"{max_fwhm_kms:.0f} km/s and were removed before the final refit."
    )
    warnings = getattr(result, "warnings", None)
    if isinstance(warnings, list) and warning not in warnings:
        warnings.append(warning)


def _has_broad_fwhm_violation(result: Any | None) -> bool:
    if result is None:
        return False
    info = getattr(result, "fit_info", {}) or {}
    return bool(info.get("desipecfit_flag_broad_component_fwhm_gt_limit", False))


def _ensure_fit_info(result: Any) -> dict[str, Any]:
    info = getattr(result, "fit_info", None)
    if not isinstance(info, dict):
        info = {}
        try:
            result.fit_info = info
        except Exception:
            return info
    return info


def _broad_component_indices(params: dict[str, Any], prefix: str) -> tuple[int, ...]:
    indices: list[int] = []
    for key in params:
        if key.startswith(f"{prefix}_broad_") and key.endswith("_sigma"):
            try:
                indices.append(int(key.split("_")[2]))
            except Exception:
                pass
    return tuple(sorted(set(indices)))


def _remove_fit_components(result: Any, names: tuple[str, ...]) -> None:
    line_flux = getattr(result, "line_flux", {}) or {}
    component_flux = getattr(result, "component_flux", {}) or {}
    params = getattr(result, "params", {}) or {}
    for name in names:
        for key in list(line_flux):
            if key == name or key.startswith(f"{name}_"):
                line_flux.pop(key, None)
        for key in list(component_flux):
            if key == name or key.startswith(f"{name}_"):
                component_flux.pop(key, None)
        for key in list(params):
            normalized = name.lower().replace("neiii", "neiii").replace("hepsilon", "hepsilon")
            if key.lower().startswith(normalized.lower()):
                params.pop(key, None)


def _mask_ppxf_feii_windows(
    spec: Any,
    windows: tuple[tuple[float, float], ...],
) -> Any:
    from echospec import AGNSpectrum

    mask = np.isfinite(spec.wave) & np.isfinite(spec.flux) & np.isfinite(spec.err) & (spec.err > 0)
    for start, stop in windows:
        mask &= ~((spec.wave >= start) & (spec.wave <= stop))
    if np.count_nonzero(mask) < 50:
        return spec
    meta = spec.meta.copy()
    meta["desipecfit_ppxf_feii_mask_windows"] = [list(window) for window in windows]
    return AGNSpectrum(
        wave=spec.wave[mask],
        flux=spec.flux[mask],
        err=spec.err[mask],
        z=spec.z,
        survey=spec.survey,
        object_id=spec.object_id,
        meta=meta,
    )


def _expand_stellar_result_to_full_spectrum(
    full_spec: Any,
    ppxf_spec: Any,
    stellar: Any,
    masked_windows: tuple[tuple[float, float], ...],
) -> Any:
    from echospec.stellar import StellarFitResult

    if len(ppxf_spec.wave) == len(full_spec.wave) and np.allclose(ppxf_spec.wave, full_spec.wave):
        diagnostics = dict(getattr(stellar, "diagnostics", {}) or {})
        diagnostics["desipecfit_ppxf_feii_windows_masked"] = False
        return StellarFitResult(
            stellar_model=stellar.stellar_model,
            residual_flux=stellar.residual_flux,
            velocity=stellar.velocity,
            sigma=stellar.sigma,
            chi2=stellar.chi2,
            weights=stellar.weights,
            goodpixels=stellar.goodpixels,
            ppxf_result=stellar.ppxf_result,
            diagnostics=diagnostics,
            agn_model=stellar.agn_model,
            total_model=stellar.total_model,
        )

    stellar_model = _interp_model_to_full_grid(full_spec.wave, ppxf_spec.wave, stellar.stellar_model)
    agn_model = (
        None
        if stellar.agn_model is None
        else _interp_model_to_full_grid(full_spec.wave, ppxf_spec.wave, stellar.agn_model)
    )
    total_model = (
        None
        if stellar.total_model is None
        else _interp_model_to_full_grid(full_spec.wave, ppxf_spec.wave, stellar.total_model)
    )
    diagnostics = dict(getattr(stellar, "diagnostics", {}) or {})
    diagnostics.update(
        {
            "desipecfit_ppxf_feii_windows_masked": True,
            "desipecfit_ppxf_feii_mask_windows": [list(window) for window in masked_windows],
            "desipecfit_ppxf_n_pixels_before_feii_window_mask": int(len(full_spec.wave)),
            "desipecfit_ppxf_n_pixels_after_feii_window_mask": int(len(ppxf_spec.wave)),
            "desipecfit_ppxf_model_interpolated_over_feii_masks": True,
        }
    )
    return StellarFitResult(
        stellar_model=stellar_model,
        residual_flux=full_spec.flux - stellar_model,
        velocity=stellar.velocity,
        sigma=stellar.sigma,
        chi2=stellar.chi2,
        weights=stellar.weights,
        goodpixels=stellar.goodpixels,
        ppxf_result=stellar.ppxf_result,
        diagnostics=diagnostics,
        agn_model=agn_model,
        total_model=total_model,
    )


def _interp_model_to_full_grid(
    full_wave: np.ndarray,
    fit_wave: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return np.interp(full_wave, fit_wave, values, left=values[0], right=values[-1])


def _host_fraction(
    wave: np.ndarray,
    host_model: np.ndarray,
    agn_model: np.ndarray | None,
    region: tuple[float, float],
) -> float:
    mask = (wave >= region[0]) & (wave <= region[1])
    if agn_model is None:
        agn_model = np.zeros_like(host_model)
    good = mask & np.isfinite(host_model) & np.isfinite(agn_model)
    if np.count_nonzero(good) < 3:
        return float("nan")
    host_level = float(np.nanmedian(np.clip(np.asarray(host_model)[good], 0.0, None)))
    agn_level = float(np.nanmedian(np.clip(np.asarray(agn_model)[good], 0.0, None)))
    denom = host_level + agn_level
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return host_level / denom


def _safe_staged_continuum_feii_refit(
    spec: Any,
    stellar: Any,
    cfg: EchoSpecFitConfig,
    host_fraction_hb: float,
) -> dict[str, Any]:
    if not cfg.staged_continuum:
        return {
            "enabled": False,
            "success": False,
            "usable": False,
            "reason": "staged_continuum_disabled",
            "host_fraction_hb": host_fraction_hb,
        }
    if not cfg.include_feii:
        return {
            "enabled": False,
            "success": False,
            "usable": False,
            "reason": "optical_feii_disabled",
            "host_fraction_hb": host_fraction_hb,
        }
    path = _resolve_feii_template_path(cfg.feii_template_path)
    if path is None:
        return {
            "enabled": True,
            "success": False,
            "usable": False,
            "reason": "missing_feii_template",
            "host_fraction_hb": host_fraction_hb,
        }
    try:
        result = fit_fixed_host_feii(
            spec,
            stellar,
            feii_template_path=path,
            fit_window=cfg.staged_fit_window,
            emission_masks=cfg.staged_emission_masks,
            powerlaw_grid=cfg.staged_powerlaw_grid,
            feii_sigma_grid=cfg.staged_feii_sigma_grid,
            feii_shift_grid=cfg.staged_feii_shift_grid,
            host_scale_max=cfg.staged_host_scale_max,
        )
        result["ppxf_host_fraction_hb"] = host_fraction_hb
        return result
    except Exception as exc:
        return {
            "enabled": True,
            "success": False,
            "usable": False,
            "reason": repr(exc),
            "host_fraction_hb": host_fraction_hb,
        }


def _resolve_feii_template_path(path: str | Path | None) -> Path | None:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    try:
        import echospec

        candidates.append(Path(echospec.__file__).resolve().parents[1] / "templates" / "fetemplate_no3")
    except Exception:
        pass
    candidates.append(Path.cwd() / "templates" / "fetemplate_no3")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_uv_feii_template_path(path: str | Path | None) -> tuple[Path | None, str]:
    if path:
        candidate = Path(path).expanduser()
        return (candidate, "user") if candidate.exists() else (candidate, "user_missing")
    default = _resolve_feii_template_path(None)
    if default is not None:
        return default, "default_fetemplate_no3"
    return None, "missing"


def _low_host_powerlaw_feii_refit(
    spec: Any,
    stellar: Any,
    agn_powerlaw_slopes: tuple[float, ...],
    feii_template_path: Path,
) -> dict[str, Any]:
    from lmfit import Parameters, minimize

    template_wave, template_flux = _load_normalized_feii_template(feii_template_path)
    wave = spec.wave
    mask = (wave >= HB_FEII_WINDOW[0]) & (wave <= HB_FEII_WINDOW[1])
    fit_mask = mask & _outside_intervals(wave, HB_FEII_MASKS)
    fit_mask &= np.isfinite(spec.flux) & np.isfinite(spec.err) & (spec.err > 0)
    if np.count_nonzero(fit_mask) < 20:
        raise ValueError("too few pixels for low-host power-law+FeII refit")

    basis = _powerlaw_basis(wave, agn_powerlaw_slopes)
    params = Parameters()
    params.add("host_scale", value=1.0, min=0.0, max=5.0)
    for index in range(basis.shape[1]):
        params.add(f"pl_{index}", value=0.0, min=0.0)
    params.add(
        "feii_amp",
        value=_initial_low_host_feii_amp(spec, stellar, template_wave, template_flux),
        min=0.0,
    )
    params.add("feii_velocity", value=0.0, min=-1500.0, max=1500.0)
    params.add("feii_sigma", value=2000.0, min=800.0, max=6000.0)

    fit = minimize(
        _low_host_powerlaw_feii_residual,
        params,
        args=(spec, stellar, basis, template_wave, template_flux, fit_mask),
        max_nfev=5000,
    )
    host_model = fit.params["host_scale"].value * np.asarray(
        stellar.stellar_model,
        dtype=float,
    )
    powerlaw_model = _powerlaw_model(fit.params, basis)
    feii_model = _script_feii_component(wave, fit.params, template_wave, template_flux)
    model = host_model + powerlaw_model + feii_model
    residual = spec.flux[fit_mask] - model[fit_mask]
    chi2 = float(np.sum((residual / spec.err[fit_mask]) ** 2))
    usable = bool(
        np.all(np.isfinite(host_model[fit_mask]))
        and np.all(np.isfinite(powerlaw_model[fit_mask]))
        and np.all(np.isfinite(feii_model[fit_mask]))
        and np.isfinite(chi2)
    )
    return {
        "enabled": True,
        "success": bool(fit.success),
        "usable": usable,
        "reason": str(fit.message),
        "host_fraction_hb": _host_fraction(wave, host_model, powerlaw_model, HB_FEII_WINDOW),
        "host_scale": float(fit.params["host_scale"].value),
        "feii_amp": float(fit.params["feii_amp"].value),
        "feii_velocity": float(fit.params["feii_velocity"].value),
        "feii_sigma": float(fit.params["feii_sigma"].value),
        "chi2": chi2,
        "redchi": float(fit.redchi),
        "host_model": host_model,
        "powerlaw_model": powerlaw_model,
        "feii_model": feii_model,
    }


def _residual_after_staged_refit(
    spec: Any,
    stellar: Any,
    low_host_refit: dict[str, Any],
    *,
    subtract_feii: bool,
) -> Any:
    if low_host_refit.get("usable"):
        return continuum_subtracted_spectrum(
            spec,
            low_host_refit,
            subtract_feii=subtract_feii,
        )

    from echospec.stellar import subtract_stellar_continuum

    return subtract_stellar_continuum(spec, stellar)


def _low_host_powerlaw_feii_residual(
    params: Any,
    spec: Any,
    stellar: Any,
    basis: np.ndarray,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    host = params["host_scale"].value * np.asarray(stellar.stellar_model, dtype=float)
    model = host + _powerlaw_model(params, basis)
    model += _script_feii_component(spec.wave, params, template_wave, template_flux)
    return (spec.flux[fit_mask] - model[fit_mask]) / spec.err[fit_mask]


def _powerlaw_basis(wave: np.ndarray, slopes: tuple[float, ...]) -> np.ndarray:
    pivot = 5100.0
    columns = []
    for slope in slopes:
        column = (wave / pivot) ** float(slope)
        finite = np.isfinite(column)
        norm = float(np.nanmedian(column[finite])) if np.any(finite) else 1.0
        if np.isfinite(norm) and norm != 0:
            column = column / norm
        columns.append(column)
    return np.column_stack(columns) if columns else np.empty((len(wave), 0))


def _powerlaw_model(params: Any, basis: np.ndarray) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros(basis.shape[0])
    coeff = np.array([params[f"pl_{index}"].value for index in range(basis.shape[1])])
    return basis @ coeff


def _load_normalized_feii_template(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    wave = np.asarray(data[:, 0], dtype=float)
    flux = np.asarray(data[:, 1], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]
    positive = np.isfinite(flux) & (flux > 0)
    norm = float(np.nanmedian(flux[positive])) if np.any(positive) else float("nan")
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("invalid FeII template normalization")
    return wave, flux / norm


def _script_feii_component(
    wave: np.ndarray,
    params: Any,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
) -> np.ndarray:
    shifted_wave = wave / (1.0 + params["feii_velocity"].value / C_KMS)
    sampled = np.interp(shifted_wave, template_wave, template_flux, left=0.0, right=0.0)
    step = float(np.nanmedian(np.diff(wave))) if len(wave) > 2 else 1.0
    sigma_wave = 4861.33 * params["feii_sigma"].value / C_KMS
    if np.isfinite(step) and step > 0 and sigma_wave > 0:
        try:
            from scipy.ndimage import gaussian_filter1d

            sampled = gaussian_filter1d(sampled, sigma_wave / step, mode="constant")
        except Exception:
            pass
    return params["feii_amp"].value * sampled


def _initial_low_host_feii_amp(
    spec: Any,
    stellar: Any,
    template_wave: np.ndarray,
    template_flux: np.ndarray,
) -> float:
    from lmfit import Parameters

    params = Parameters()
    params.add("feii_amp", value=1.0)
    params.add("feii_velocity", value=0.0)
    params.add("feii_sigma", value=2000.0)
    template = _script_feii_component(spec.wave, params, template_wave, template_flux)
    mask = (spec.wave >= HB_FEII_WINDOW[0]) & (spec.wave <= HB_FEII_WINDOW[1])
    mask &= _outside_intervals(spec.wave, HB_FEII_MASKS)
    residual = spec.flux - np.asarray(stellar.stellar_model, dtype=float)
    good = mask & np.isfinite(residual) & np.isfinite(template) & (template > 0)
    if np.count_nonzero(good) < 3:
        return 1.0
    numerator = float(np.nansum(residual[good] * template[good]))
    denominator = float(np.nansum(template[good] ** 2))
    if denominator <= 0 or not np.isfinite(denominator):
        return 1.0
    return max(0.0, numerator / denominator)


def _outside_intervals(
    wave: np.ndarray,
    intervals: tuple[tuple[float, float], ...],
) -> np.ndarray:
    mask = np.ones_like(wave, dtype=bool)
    for start, stop in intervals:
        mask &= ~((wave >= start) & (wave <= stop))
    return mask


def _safe_joint_fit(
    spec: Any,
    stellar: Any,
    fits: dict[str, dict[str, Any]],
    agn_powerlaw_slopes: tuple[float, ...],
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"result": None, "error": "Joint refit disabled by config."}
    try:
        from echospec.fitting import joint_refit_full_spectrum

        result = joint_refit_full_spectrum(
            spec,
            stellar,
            oii=fits["oii"]["result"],
            hb_oiii=fits["hb_oiii"]["result"],
            ha_nii_sii=fits["ha_nii_sii"]["result"],
            agn_powerlaw_slopes=agn_powerlaw_slopes,
            continuum_mask_width_kms=8000.0,
            line_prior_sigma_frac=0.30,
            fit_mask_mode="line_windows",
            line_window_half_width_kms=12000.0,
        )
        return {"result": result, "error": ""}
    except Exception as exc:
        return {"result": None, "error": repr(exc)}


def _safe_no_local_continuum_refit(
    spec: Any,
    stellar: Any,
    staged_continuum: dict[str, Any],
    fits: dict[str, dict[str, Any]],
    cfg: EchoSpecFitConfig,
) -> dict[str, Any]:
    if not cfg.no_local_continuum_refit:
        return {
            "enabled": False,
            "success": False,
            "error": "No-local-continuum refit disabled by config.",
        }
    try:
        return _no_local_continuum_refit(
            spec,
            stellar,
            staged_continuum,
            fits,
            agn_powerlaw_slopes=cfg.agn_powerlaw_slopes,
            err_floor_fraction=cfg.no_local_continuum_err_floor_fraction,
            force_blue_balmer_broad=bool(
                cfg.force_broad_blue_balmer or cfg.force_broad_hb
            ),
            forced_blue_broad_min_scale=cfg.no_local_forced_blue_broad_min_scale,
            blue_broad_profile=cfg.no_local_blue_broad_profile,
            blue_broad_max_sigma_kms=cfg.no_local_blue_broad_max_sigma_kms,
            blue_broad_max_abs_velocity_kms=cfg.no_local_blue_broad_max_abs_velocity_kms,
        )
    except Exception as exc:
        return {"enabled": True, "success": False, "error": repr(exc)}


def _no_local_continuum_refit(
    spec: Any,
    stellar: Any,
    staged_continuum: dict[str, Any],
    fits: dict[str, dict[str, Any]],
    *,
    agn_powerlaw_slopes: tuple[float, ...],
    err_floor_fraction: float,
    force_blue_balmer_broad: bool,
    forced_blue_broad_min_scale: float,
    blue_broad_profile: str,
    blue_broad_max_sigma_kms: float,
    blue_broad_max_abs_velocity_kms: float,
) -> dict[str, Any]:
    wave = np.asarray(spec.wave, dtype=float)
    flux = np.asarray(spec.flux, dtype=float)
    err = np.asarray(spec.err, dtype=float)
    valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Insufficient valid pixels for no-local-continuum refit.")

    columns: list[np.ndarray] = []
    column_names: list[str] = []
    groups: dict[str, list[str]] = {"continuum": [], "uv": [], "optical": []}

    host = np.asarray(stellar.stellar_model, dtype=float)
    if len(host) == len(wave) and _finite_absmax(host) > 0:
        columns.append(host)
        column_names.append("pPXF host")
        groups["continuum"].append("pPXF host")
    powerlaw_insert_index = len(column_names)

    initial_line_flux: dict[str, float] = {}
    for payload in fits.values():
        result = payload.get("result")
        if result is not None:
            initial_line_flux.update(result.line_flux)

    if staged_continuum.get("usable"):
        staged_feii = np.asarray(
            staged_continuum.get("feii_model", np.zeros_like(wave)),
            dtype=float,
        )
        if len(staged_feii) == len(wave):
            _append_component_column(
                columns,
                column_names,
                groups,
                "optical",
                "Optical FeII",
                staged_feii,
            )

    _add_fit_component_columns(
        spec,
        fits,
        columns,
        column_names,
        groups,
        fit_key="mgii_uv",
        window=MGII_WINDOW,
        group="uv",
        components=(
            ("UV FeII 2200-3090", "UV_FeII"),
            ("MgII broad", "MgII_broad"),
        ),
    )
    _add_oii_component_columns(spec, fits, columns, column_names, groups)
    blue_initial_flux = _add_blue_balmer_component_columns(
        spec,
        fits,
        columns,
        column_names,
        groups,
        broad_profile=blue_broad_profile,
        max_sigma_kms=blue_broad_max_sigma_kms,
        max_abs_velocity_kms=blue_broad_max_abs_velocity_kms,
    )
    initial_line_flux.update(blue_initial_flux)
    _add_hb_oiii_component_columns(spec, fits, columns, column_names, groups)
    _add_ha_nii_sii_component_columns(spec, fits, columns, column_names, groups)

    candidate_slopes = _unique_finite_slopes(agn_powerlaw_slopes)
    if not candidate_slopes:
        raise ValueError("At least one finite power-law slope is required.")

    fit_err = err.copy()
    median_err = float(np.nanmedian(fit_err[valid]))
    if np.isfinite(median_err) and median_err > 0:
        fit_err[valid] = np.maximum(fit_err[valid], err_floor_fraction * median_err)

    candidate_results: list[dict[str, Any]] = []
    basis = _powerlaw_basis(wave, candidate_slopes)
    for index, slope in enumerate(candidate_slopes):
        powerlaw_name = f"powerlaw slope {slope:g}"
        candidate_columns = (
            columns[:powerlaw_insert_index]
            + [basis[:, index]]
            + columns[powerlaw_insert_index:]
        )
        candidate_names = (
            column_names[:powerlaw_insert_index]
            + [powerlaw_name]
            + column_names[powerlaw_insert_index:]
        )
        candidate_groups = {
            "continuum": groups["continuum"] + [powerlaw_name],
            "uv": list(groups["uv"]),
            "optical": list(groups["optical"]),
        }
        payload = _solve_no_local_linear_model(
            wave,
            flux,
            fit_err,
            valid,
            candidate_columns,
            candidate_names,
            candidate_groups,
            initial_line_flux,
            force_blue_balmer_broad=force_blue_balmer_broad,
            forced_blue_broad_min_scale=forced_blue_broad_min_scale,
        )
        payload["selected_powerlaw_slope"] = float(slope)
        payload["selected_powerlaw_name"] = powerlaw_name
        candidate_results.append(payload)

    selected = min(
        candidate_results,
        key=lambda item: (
            not bool(item.get("success")),
            float(item.get("bic", np.inf)),
            float(item.get("chi2", np.inf)),
        ),
    )
    return {
        "enabled": True,
        "success": bool(selected["success"]),
        "error": "",
        "message": str(selected["message"]),
        "local_continuum_terms_included": False,
        "powerlaw_mode": "single",
        "candidate_powerlaw_slopes": [float(slope) for slope in candidate_slopes],
        "selected_powerlaw_slope": float(selected["selected_powerlaw_slope"]),
        "selected_powerlaw_name": str(selected["selected_powerlaw_name"]),
        "single_powerlaw_candidate_fits": [
            {
                "slope": float(item["selected_powerlaw_slope"]),
                "success": bool(item["success"]),
                "chi2": float(item["chi2"]),
                "redchi": float(item["redchi"]),
                "bic": float(item["bic"]),
                "nvarys": int(item["nvarys"]),
                "message": str(item["message"]),
            }
            for item in candidate_results
        ],
        "force_blue_balmer_broad": bool(force_blue_balmer_broad),
        "forced_blue_broad_min_scale": float(forced_blue_broad_min_scale),
        "blue_broad_profile": str(blue_broad_profile),
        "blue_broad_max_sigma_kms": float(blue_broad_max_sigma_kms),
        "blue_broad_max_abs_velocity_kms": float(blue_broad_max_abs_velocity_kms),
        "chi2": selected["chi2"],
        "redchi": selected["redchi"],
        "bic": selected["bic"],
        "ndata": selected["ndata"],
        "nvarys": selected["nvarys"],
        "column_names": selected["column_names"],
        "groups": selected["groups"],
        "coefficients": selected["coefficients"],
        "scaled_flux": selected["scaled_flux"],
        "initial_window_flux": initial_line_flux,
        "model_flux": selected["model_flux"],
        "continuum_flux": selected["continuum_flux"],
        "line_flux": selected["line_flux"],
        "residual_flux": selected["residual_flux"],
        "component_flux": selected["component_flux"],
        "valid_mask": valid,
    }


def _unique_finite_slopes(slopes: tuple[float, ...]) -> tuple[float, ...]:
    values: list[float] = []
    seen: set[float] = set()
    for slope in slopes:
        value = float(slope)
        if not np.isfinite(value):
            continue
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    return tuple(values)


def _solve_no_local_linear_model(
    wave: np.ndarray,
    flux: np.ndarray,
    fit_err: np.ndarray,
    valid: np.ndarray,
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
    initial_line_flux: dict[str, float],
    *,
    force_blue_balmer_broad: bool,
    forced_blue_broad_min_scale: float,
) -> dict[str, Any]:
    from scipy.optimize import lsq_linear

    matrix = np.column_stack(columns)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    weighted_matrix = matrix[valid] / fit_err[valid, None]
    weighted_flux = flux[valid] / fit_err[valid]
    lower_bounds = np.zeros(len(column_names), dtype=float)
    if force_blue_balmer_broad:
        for index, name in enumerate(column_names):
            if name in {"Hgamma broad", "Hdelta broad"}:
                lower_bounds[index] = max(0.0, float(forced_blue_broad_min_scale))

    fit = lsq_linear(
        weighted_matrix,
        weighted_flux,
        bounds=(lower_bounds, np.inf),
        lsmr_tol="auto",
        max_iter=3000,
    )
    coefficients = np.asarray(fit.x, dtype=float)
    model = matrix @ coefficients
    components = {
        name: matrix[:, index] * coefficients[index]
        for index, name in enumerate(column_names)
    }
    continuum_model = np.zeros_like(wave, dtype=float)
    for name in groups["continuum"]:
        continuum_model += components[name]
    line_model = model - continuum_model
    residual = flux - model
    chi2 = float(np.nansum((residual[valid] / fit_err[valid]) ** 2))
    ndata = int(np.count_nonzero(valid))
    nvarys = int(np.count_nonzero(coefficients > 1.0e-12))
    nparams = len(column_names)
    scaled_flux = _scaled_no_local_line_flux(
        wave,
        components,
        coefficients,
        column_names,
        initial_line_flux,
        valid,
    )
    return {
        "success": bool(fit.success),
        "message": str(fit.message),
        "chi2": chi2,
        "redchi": chi2 / max(ndata - nvarys, 1),
        "bic": chi2 + nparams * np.log(max(ndata, 1)),
        "ndata": ndata,
        "nvarys": nvarys,
        "column_names": list(column_names),
        "groups": {name: list(values) for name, values in groups.items()},
        "coefficients": {
            name: float(coefficients[index])
            for index, name in enumerate(column_names)
        },
        "scaled_flux": scaled_flux,
        "model_flux": model,
        "continuum_flux": continuum_model,
        "line_flux": line_model,
        "residual_flux": residual,
        "component_flux": components,
    }


def _fit_params(fits: dict[str, dict[str, Any]], fit_key: str) -> dict[str, Any]:
    payload = fits.get(fit_key, {})
    fit = payload.get("result")
    return {} if fit is None else (getattr(fit, "params", {}) or {})


def _add_oii_component_columns(
    spec: Any,
    fits: dict[str, dict[str, Any]],
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
) -> None:
    params = _fit_params(fits, "oii")
    if not params:
        return
    wave = np.asarray(spec.wave, dtype=float)
    velocity = _param_float(params, "oii_velocity", 0.0)
    sigma = _param_float(params, "oii_sigma", np.nan)
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[OII] 3727",
        _gaussian_integrated_flux(
            wave,
            OII_BLEND,
            _param_float(params, "oii_3727_blend_flux", np.nan),
            velocity,
            sigma,
        ),
    )


def _add_hb_oiii_component_columns(
    spec: Any,
    fits: dict[str, dict[str, Any]],
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
) -> None:
    params = _fit_params(fits, "hb_oiii")
    if not params:
        return
    wave = np.asarray(spec.wave, dtype=float)
    velocity = _param_float(params, "narrow_velocity", 0.0)
    sigma = _param_float(params, "narrow_sigma", np.nan)
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Hbeta narrow",
        _gaussian_integrated_flux(
            wave,
            HBETA,
            _param_float(params, "hb_narrow_flux", np.nan),
            velocity,
            sigma,
        ),
    )
    oiii_core = _gaussian_integrated_flux(
        wave,
        OIII_4959,
        _param_float(params, "oiii_4959_flux", np.nan),
        velocity,
        sigma,
    ) + _gaussian_integrated_flux(
        wave,
        OIII_5007,
        _param_float(params, "oiii_5007_flux", np.nan),
        velocity,
        sigma,
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[OIII] core doublet",
        oiii_core,
    )
    for index in (1, 2):
        flux_name = f"hb_broad_{index}_flux"
        if flux_name not in params:
            continue
        _append_component_column(
            columns,
            column_names,
            groups,
            "optical",
            f"Hbeta broad {index}",
            _gaussian_integrated_flux(
                wave,
                HBETA,
                _param_float(params, flux_name, np.nan),
                _param_float(params, f"hb_broad_{index}_velocity", 0.0),
                _param_float(params, f"hb_broad_{index}_sigma", np.nan),
            ),
        )
    wing_velocity = _param_float(params, "oiii_wing_velocity", np.nan)
    wing_sigma = _param_float(params, "oiii_wing_sigma", np.nan)
    if "oiii_4959_wing_flux" in params or "oiii_5007_wing_flux" in params:
        oiii_wing = _gaussian_integrated_flux(
            wave,
            OIII_4959,
            _param_float(params, "oiii_4959_wing_flux", np.nan),
            wing_velocity,
            wing_sigma,
        ) + _gaussian_integrated_flux(
            wave,
            OIII_5007,
            _param_float(params, "oiii_5007_wing_flux", np.nan),
            wing_velocity,
            wing_sigma,
        )
        _append_component_column(
            columns,
            column_names,
            groups,
            "optical",
            "[OIII] wing doublet",
            oiii_wing,
        )


def _add_ha_nii_sii_component_columns(
    spec: Any,
    fits: dict[str, dict[str, Any]],
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
) -> None:
    params = _fit_params(fits, "ha_nii_sii")
    if not params:
        return
    wave = np.asarray(spec.wave, dtype=float)
    velocity = _param_float(params, "narrow_velocity", 0.0)
    sigma = _param_float(params, "narrow_sigma", np.nan)
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Halpha narrow",
        _gaussian_integrated_flux(
            wave,
            HALPHA,
            _param_float(params, "ha_narrow_flux", np.nan),
            velocity,
            sigma,
        ),
    )
    oi = _gaussian_integrated_flux(
        wave,
        OI_6300,
        _param_float(params, "oi_6300_flux", np.nan),
        velocity,
        sigma,
    ) + _gaussian_integrated_flux(
        wave,
        OI_6363,
        _param_float(params, "oi_6363_flux", np.nan),
        velocity,
        sigma,
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[OI] doublet",
        oi,
    )
    nii = _gaussian_integrated_flux(
        wave,
        NII_6548,
        _param_float(params, "nii_6548_flux", np.nan),
        velocity,
        sigma,
    ) + _gaussian_integrated_flux(
        wave,
        NII_6584,
        _param_float(params, "nii_6584_flux", np.nan),
        velocity,
        sigma,
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[NII] doublet",
        nii,
    )
    sii = _gaussian_integrated_flux(
        wave,
        SII_6716,
        _param_float(params, "sii_6716_flux", np.nan),
        velocity,
        sigma,
    ) + _gaussian_integrated_flux(
        wave,
        SII_6731,
        _param_float(params, "sii_6731_flux", np.nan),
        velocity,
        sigma,
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[SII] doublet",
        sii,
    )
    for index in (1, 2):
        flux_name = f"ha_broad_{index}_flux"
        if flux_name not in params:
            continue
        _append_component_column(
            columns,
            column_names,
            groups,
            "optical",
            f"Halpha broad {index}",
            _gaussian_integrated_flux(
                wave,
                HALPHA,
                _param_float(params, flux_name, np.nan),
                _param_float(params, f"ha_broad_{index}_velocity", 0.0),
                _param_float(params, f"ha_broad_{index}_sigma", np.nan),
            ),
        )


def _add_fit_component_columns(
    spec: Any,
    fits: dict[str, dict[str, Any]],
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
    *,
    fit_key: str,
    window: tuple[float, float],
    group: str,
    components: tuple[tuple[str, str], ...],
) -> None:
    payload = fits.get(fit_key, {})
    fit = payload.get("result")
    if fit is None:
        return
    window_spec = spec.select_window(*window).mask_bad_pixels()
    window_wave = np.asarray(window_spec.wave, dtype=float)
    comp_map = getattr(fit, "component_flux", {}) or {}
    for label, key in components:
        values = comp_map.get(key)
        if values is None:
            continue
        values = np.asarray(values, dtype=float)
        good = np.isfinite(window_wave) & np.isfinite(values)
        if np.count_nonzero(good) < 2 or _finite_absmax(values[good]) <= 0:
            continue
        full = np.zeros_like(spec.wave, dtype=float)
        mask = (spec.wave >= np.nanmin(window_wave[good])) & (
            spec.wave <= np.nanmax(window_wave[good])
        )
        full[mask] = np.interp(
            spec.wave[mask],
            window_wave[good],
            values[good],
            left=0.0,
            right=0.0,
        )
        columns.append(full)
        column_names.append(label)
        groups[group].append(label)


def _add_blue_balmer_component_columns(
    spec: Any,
    fits: dict[str, dict[str, Any]],
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
    *,
    broad_profile: str,
    max_sigma_kms: float,
    max_abs_velocity_kms: float,
) -> dict[str, float]:
    payload = fits.get("blue_balmer", {})
    fit = payload.get("result")
    if fit is None:
        return {}
    wave = np.asarray(spec.wave, dtype=float)
    params = getattr(fit, "params", {}) or {}
    component_flux = getattr(fit, "component_flux", {}) or {}
    initial_flux: dict[str, float] = {}

    narrow_velocity = _param_float(params, "blue_velocity", 0.0)
    narrow_sigma = _param_float(params, "blue_narrow_sigma", 220.0)
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Hgamma narrow",
        _gaussian_integrated_flux(
            wave,
            HGAMMA,
            _param_float(params, "hgamma_narrow_flux", np.nan),
            narrow_velocity,
            narrow_sigma,
        ),
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Hdelta narrow",
        _gaussian_integrated_flux(
            wave,
            HDELTA,
            _param_float(params, "hdelta_narrow_flux", np.nan),
            narrow_velocity,
            narrow_sigma,
        ),
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[NeIII] 3869",
        _gaussian_integrated_flux(
            wave,
            NEIII_3869,
            _param_float(params, "neiii_3869_flux", np.nan),
            narrow_velocity,
            narrow_sigma,
        ),
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "[NeIII] 3968",
        _gaussian_integrated_flux(
            wave,
            NEIII_3968,
            _param_float(params, "neiii_3968_flux", np.nan),
            narrow_velocity,
            narrow_sigma,
        ),
    )
    if "hepsilon_narrow_flux" in params:
        _append_component_column(
            columns,
            column_names,
            groups,
            "optical",
            "Hepsilon narrow",
            _gaussian_integrated_flux(
                wave,
                HEPSILON,
                _param_float(params, "hepsilon_narrow_flux", np.nan),
                narrow_velocity,
                narrow_sigma,
            ),
        )

    hgamma_broad = np.zeros_like(wave, dtype=float)
    hdelta_broad = np.zeros_like(wave, dtype=float)
    hgamma_broad_flux = 0.0
    hdelta_broad_flux = 0.0
    for index in _selected_blue_broad_indices(
        params,
        broad_profile=broad_profile,
        max_sigma_kms=max_sigma_kms,
        max_abs_velocity_kms=max_abs_velocity_kms,
    ):
        velocity = _param_float(params, f"hbeta_broad_{index}_velocity", 0.0)
        sigma = _param_float(params, f"hbeta_broad_{index}_sigma", np.nan)
        hgamma_flux = _param_float(params, f"hgamma_broad_{index}_flux", np.nan)
        hdelta_flux = _param_float(params, f"hdelta_broad_{index}_flux", np.nan)
        hgamma_broad += _gaussian_integrated_flux(
            wave,
            HGAMMA,
            hgamma_flux,
            velocity,
            sigma,
        )
        hdelta_broad += _gaussian_integrated_flux(
            wave,
            HDELTA,
            hdelta_flux,
            velocity,
            sigma,
        )
        if np.isfinite(hgamma_flux) and hgamma_flux > 0:
            hgamma_broad_flux += float(hgamma_flux)
        if np.isfinite(hdelta_flux) and hdelta_flux > 0:
            hdelta_broad_flux += float(hdelta_flux)

    if _finite_absmax(hgamma_broad) <= 0:
        hgamma_broad = _component_on_full_grid(spec, fit, component_flux, "Hgamma_broad")
        hgamma_broad_flux = _param_sum(params, "hgamma_broad", "flux")
    if _finite_absmax(hdelta_broad) <= 0:
        hdelta_broad = _component_on_full_grid(spec, fit, component_flux, "Hdelta_broad")
        hdelta_broad_flux = _param_sum(params, "hdelta_broad", "flux")
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Hgamma broad",
        hgamma_broad,
    )
    _append_component_column(
        columns,
        column_names,
        groups,
        "optical",
        "Hdelta broad",
        hdelta_broad,
    )
    if hgamma_broad_flux > 0:
        initial_flux["Hgamma_broad"] = float(hgamma_broad_flux)
    if hdelta_broad_flux > 0:
        initial_flux["Hdelta_broad"] = float(hdelta_broad_flux)
    return initial_flux


def _append_component_column(
    columns: list[np.ndarray],
    column_names: list[str],
    groups: dict[str, list[str]],
    group: str,
    label: str,
    values: np.ndarray,
) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or _finite_absmax(values) <= 0:
        return
    columns.append(np.where(np.isfinite(values), values, 0.0))
    column_names.append(label)
    groups[group].append(label)


def _component_on_full_grid(
    spec: Any,
    fit: Any,
    component_flux: dict[str, np.ndarray],
    key: str,
) -> np.ndarray:
    values = component_flux.get(key)
    if values is None:
        return np.zeros_like(spec.wave, dtype=float)
    window_spec = spec.select_window(*BLUE_BALMER_WINDOW).mask_bad_pixels()
    window_wave = np.asarray(window_spec.wave, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(window_wave) & np.isfinite(values)
    full = np.zeros_like(spec.wave, dtype=float)
    if np.count_nonzero(good) < 2:
        return full
    mask = (spec.wave >= np.nanmin(window_wave[good])) & (
        spec.wave <= np.nanmax(window_wave[good])
    )
    full[mask] = np.interp(
        spec.wave[mask],
        window_wave[good],
        values[good],
        left=0.0,
        right=0.0,
    )
    return full


def _blue_broad_indices(params: dict[str, Any]) -> tuple[int, ...]:
    indices: list[int] = []
    for key in params:
        if key.startswith("hgamma_broad_") and key.endswith("_flux"):
            try:
                indices.append(int(key.split("_")[2]))
            except Exception:
                pass
    return tuple(sorted(set(indices)))


def _selected_blue_broad_indices(
    params: dict[str, Any],
    *,
    broad_profile: str,
    max_sigma_kms: float,
    max_abs_velocity_kms: float,
) -> tuple[int, ...]:
    indices = _blue_broad_indices(params)
    if broad_profile == "hbeta":
        return indices
    selected: list[int] = []
    for index in indices:
        velocity = _param_float(params, f"hbeta_broad_{index}_velocity", np.nan)
        sigma = _param_float(params, f"hbeta_broad_{index}_sigma", np.nan)
        if not (np.isfinite(velocity) and np.isfinite(sigma) and sigma > 0.0):
            continue
        if sigma <= max_sigma_kms and abs(velocity) <= max_abs_velocity_kms:
            selected.append(index)
    return tuple(selected or indices[:1])


def _param_sum(params: dict[str, Any], prefix: str, suffix: str) -> float:
    total = 0.0
    for key in params:
        if key.startswith(f"{prefix}_") and key.endswith(f"_{suffix}"):
            value = _param_float(params, key, np.nan)
            if np.isfinite(value) and value > 0.0:
                total += float(value)
    return total


def _param_float(params: dict[str, Any], name: str, default: float) -> float:
    try:
        value = params[name]
        if hasattr(value, "value"):
            value = value.value
        value = float(value)
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _gaussian_integrated_flux(
    wave: np.ndarray,
    rest_wave: float,
    flux: float,
    velocity: float,
    sigma_velocity: float,
) -> np.ndarray:
    if not (
        np.isfinite(flux)
        and flux > 0.0
        and np.isfinite(sigma_velocity)
        and sigma_velocity > 0.0
    ):
        return np.zeros_like(wave, dtype=float)
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    if not np.isfinite(sigma_wave) or sigma_wave <= 0:
        return np.zeros_like(wave, dtype=float)
    return (
        flux
        / (sigma_wave * np.sqrt(2.0 * np.pi))
        * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)
    )


def _scaled_no_local_line_flux(
    wave: np.ndarray,
    components: dict[str, np.ndarray],
    coefficients: np.ndarray,
    column_names: list[str],
    initial_line_flux: dict[str, float],
    valid: np.ndarray,
) -> dict[str, float]:
    flux_map = {
        "UV FeII 2200-3090": "UV_FeII_2200_3090",
        "MgII broad": "MgII_broad",
        "[OII] 3727": "OII_3727_blend",
        "Hgamma narrow": "Hgamma_narrow",
        "Hgamma broad": "Hgamma_broad",
        "Hdelta narrow": "Hdelta_narrow",
        "Hdelta broad": "Hdelta_broad",
        "[NeIII] 3869": "NeIII_3869",
        "[NeIII] 3968": "NeIII_3968",
        "Hepsilon narrow": "Hepsilon_narrow",
        "Hbeta narrow": "Hb_narrow",
        "Hbeta broad 1": "Hb_broad_1",
        "Hbeta broad 2": "Hb_broad_2",
        "Halpha narrow": "Ha_narrow",
        "Halpha broad 1": "Ha_broad_1",
        "Halpha broad 2": "Ha_broad_2",
        "[OI] 6300": "OI_6300",
        "[OI] 6363": "OI_6363",
        "[NII] 6548": "NII_6548",
        "[NII] 6584": "NII_6584",
        "[SII] 6716": "SII_6716",
        "[SII] 6731": "SII_6731",
    }
    tied_flux_map = {
        "[OIII] core doublet": ("OIII_4959", "OIII_5007"),
        "[OIII] wing doublet": ("OIII_4959_wing", "OIII_5007_wing"),
        "[OI] doublet": ("OI_6300", "OI_6363"),
        "[NII] doublet": ("NII_6548", "NII_6584"),
        "[SII] doublet": ("SII_6716", "SII_6731"),
    }
    scaled: dict[str, float] = {}
    for index, name in enumerate(column_names):
        tied_lines = tied_flux_map.get(name, ())
        if tied_lines:
            for line_name in tied_lines:
                initial = initial_line_flux.get(line_name)
                if initial is not None and np.isfinite(initial):
                    scaled[line_name] = float(initial) * float(coefficients[index])
            continue
        line_name = flux_map.get(name)
        if line_name is None:
            continue
        initial = initial_line_flux.get(line_name)
        if initial is not None and np.isfinite(initial):
            scaled[line_name] = float(initial) * float(coefficients[index])
    if "Optical FeII" in components:
        comp = components["Optical FeII"]
        mask = valid & (wave >= 4434.0) & (wave <= 4684.0)
        if np.count_nonzero(mask) >= 2:
            scaled["FeII_4434_4684"] = float(np.trapz(comp[mask], wave[mask]))
    for line_name in ("NeIII_3869", "NeIII_3968", "Hepsilon_narrow"):
        if line_name in scaled:
            scaled[f"{line_name}_flux"] = scaled[line_name]
    _add_broad_total_aliases(scaled)
    return scaled


def _finite_absmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(finite)))


def _summary_row(
    h1: H1Spectrum,
    stellar: Any,
    low_host_refit: dict[str, Any],
    fits: dict[str, dict[str, Any]],
    joint: dict[str, Any],
    no_local_continuum: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = getattr(stellar, "diagnostics", {}) or {}
    row: dict[str, Any] = {
        "object_id": h1.object_id,
        "survey": h1.survey,
        "stellar_success": True,
        "stellar_chi2": stellar.chi2,
        "stellar_velocity": stellar.velocity,
        "stellar_sigma": stellar.sigma,
        "ppxf_host_fraction_hb": low_host_refit.get(
            "ppxf_host_fraction_hb",
            low_host_refit.get("host_fraction_hb", np.nan),
        ),
        "initial_ppxf_mdegree": diagnostics.get("initial_ppxf_mdegree"),
        "final_ppxf_mdegree": diagnostics.get("final_ppxf_mdegree"),
        "agn_powerlaw_enabled": diagnostics.get("include_agn_powerlaw", True),
        "agn_powerlaw_weights_nonzero": diagnostics.get(
            "agn_powerlaw_weights_nonzero_count"
        ),
        "agn_powerlaw_weight_sum": diagnostics.get("agn_powerlaw_weight_sum"),
        "ppxf_weights_nonzero": int(np.count_nonzero(np.abs(stellar.weights) > 0)),
        "ppxf_feii_windows_masked": bool(
            diagnostics.get("desipecfit_ppxf_feii_windows_masked", False)
        ),
        "ppxf_feii_mask_fallback_to_unmasked": bool(
            diagnostics.get("desipecfit_ppxf_feii_mask_fallback_to_unmasked", False)
        ),
        "ppxf_feii_mask_fallback_error": diagnostics.get(
            "desipecfit_ppxf_feii_mask_fallback_error", ""
        ),
        "ppxf_n_pixels_after_feii_window_mask": diagnostics.get(
            "desipecfit_ppxf_n_pixels_after_feii_window_mask", np.nan
        ),
    }
    row.update(_low_host_refit_summary(low_host_refit))

    line_flux: dict[str, float] = {}
    for prefix, payload in fits.items():
        result = payload["result"]
        row[f"{prefix}_error"] = payload["error"]
        row.update(_fit_summary(prefix, result))
        if result is not None:
            line_flux.update(result.line_flux)
    _add_broad_total_aliases(line_flux)

    for line in DEFAULT_LINES:
        row[line] = line_flux.get(line, np.nan)
    row["log_OIII_Hb"] = _log_ratio(
        line_flux.get("OIII_5007"),
        line_flux.get("Hb_narrow"),
    )
    row["log_NII_Ha"] = _log_ratio(
        line_flux.get("NII_6584"),
        line_flux.get("Ha_narrow"),
    )
    row["log_OIII_OII"] = _log_ratio(
        line_flux.get("OIII_5007"),
        line_flux.get("OII_3727_blend"),
    )

    row["joint_error"] = joint["error"]
    joint_result = joint["result"]
    final_flux = dict(line_flux)
    if joint_result is not None:
        joint_usable = _joint_result_usable(joint_result)
        row.update(
            {
                "joint_success": bool(joint_result.success),
                "joint_usable": joint_usable,
                "joint_chi2": joint_result.chi2,
                "joint_redchi": joint_result.redchi,
                "joint_bic": joint_result.bic,
                "joint_ndata": joint_result.ndata,
                "joint_nvarys": joint_result.nvarys,
                "joint_stellar_scale": joint_result.params.get("stellar_scale", np.nan),
                "joint_fit_mask_mode": joint_result.fit_info.get("fit_mask_mode"),
                "joint_line_prior_sigma_frac": joint_result.fit_info.get(
                    "line_prior_sigma_frac"
                ),
                "joint_n_valid_pixels": joint_result.fit_info.get("n_valid_pixels"),
                "joint_n_fit_pixels": joint_result.fit_info.get("n_fit_pixels"),
                "joint_n_line_flux_boundary_hits": joint_result.fit_info.get(
                    "n_line_flux_boundary_hits"
                ),
            }
        )
        if joint_usable:
            final_flux.update(joint_result.line_flux)
            _add_broad_total_aliases(final_flux)
        for line, value in joint_result.line_flux.items():
            row[f"joint_{line}"] = value
    else:
        row["joint_success"] = False
        row["joint_usable"] = False

    row.update(_no_local_continuum_summary(no_local_continuum))
    if no_local_continuum.get("success"):
        final_flux.update(no_local_continuum.get("scaled_flux", {}))
        _add_broad_total_aliases(final_flux)

    for line in DEFAULT_LINES:
        row[f"final_{line}"] = final_flux.get(line, np.nan)
    row["final_log_OIII_Hb"] = _log_ratio(
        final_flux.get("OIII_5007"),
        final_flux.get("Hb_narrow"),
    )
    row["final_log_NII_Ha"] = _log_ratio(
        final_flux.get("NII_6584"),
        final_flux.get("Ha_narrow"),
    )
    row["final_log_OIII_OII"] = _log_ratio(
        final_flux.get("OIII_5007"),
        final_flux.get("OII_3727_blend"),
    )
    return row


def _no_local_continuum_summary(no_local_continuum: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "no_local_continuum_refit_enabled": bool(no_local_continuum.get("enabled")),
        "no_local_continuum_refit_success": bool(no_local_continuum.get("success")),
        "no_local_continuum_refit_error": no_local_continuum.get("error", ""),
        "no_local_continuum_refit_message": no_local_continuum.get("message", ""),
        "no_local_continuum_terms_included": bool(
            no_local_continuum.get("local_continuum_terms_included", False)
        ),
        "no_local_continuum_chi2": no_local_continuum.get("chi2", np.nan),
        "no_local_continuum_redchi": no_local_continuum.get("redchi", np.nan),
        "no_local_continuum_bic": no_local_continuum.get("bic", np.nan),
        "no_local_continuum_ndata": no_local_continuum.get("ndata", np.nan),
        "no_local_continuum_nvarys": no_local_continuum.get("nvarys", np.nan),
        "no_local_powerlaw_mode": no_local_continuum.get("powerlaw_mode", ""),
        "no_local_selected_powerlaw_slope": no_local_continuum.get(
            "selected_powerlaw_slope", np.nan
        ),
        "no_local_selected_powerlaw_name": no_local_continuum.get(
            "selected_powerlaw_name", ""
        ),
        "no_local_powerlaw_candidate_count": len(
            no_local_continuum.get("candidate_powerlaw_slopes", []) or []
        ),
        "no_local_force_blue_balmer_broad": bool(
            no_local_continuum.get("force_blue_balmer_broad", False)
        ),
        "no_local_forced_blue_broad_min_scale": no_local_continuum.get(
            "forced_blue_broad_min_scale",
            np.nan,
        ),
        "no_local_blue_broad_profile": no_local_continuum.get(
            "blue_broad_profile",
            "",
        ),
        "no_local_blue_broad_max_sigma_kms": no_local_continuum.get(
            "blue_broad_max_sigma_kms",
            np.nan,
        ),
        "no_local_blue_broad_max_abs_velocity_kms": no_local_continuum.get(
            "blue_broad_max_abs_velocity_kms",
            np.nan,
        ),
    }
    for name, value in (no_local_continuum.get("scaled_flux", {}) or {}).items():
        row[f"no_local_{name}"] = value
    return row


def _low_host_refit_summary(low_host_refit: dict[str, Any]) -> dict[str, Any]:
    return {
        "staged_continuum_enabled": bool(low_host_refit.get("enabled")),
        "staged_continuum_success": bool(low_host_refit.get("success")),
        "staged_continuum_usable": bool(low_host_refit.get("usable")),
        "staged_continuum_reason": low_host_refit.get("reason", ""),
        "staged_errors": low_host_refit.get("errors", ""),
        "staged_host_kinematics": low_host_refit.get("host_kinematics", ""),
        "staged_host_scale": low_host_refit.get("host_scale", np.nan),
        "staged_host_fraction_5100": low_host_refit.get(
            "host_fraction_5100", np.nan
        ),
        "staged_powerlaw_amp_5100": low_host_refit.get(
            "powerlaw_amp_5100", np.nan
        ),
        "staged_powerlaw_alpha": low_host_refit.get("powerlaw_alpha", np.nan),
        "staged_feii_flux_4434_4684": low_host_refit.get(
            "feii_flux_4434_4684", np.nan
        ),
        "staged_feii_ew_4434_4684": low_host_refit.get(
            "feii_ew_4434_4684", np.nan
        ),
        "staged_feii_sigma_kms": low_host_refit.get("feii_sigma_kms", np.nan),
        "staged_feii_fwhm_kms": low_host_refit.get("feii_fwhm_kms", np.nan),
        "staged_feii_shift_kms": low_host_refit.get("feii_shift_kms", np.nan),
        "staged_feii_sigma_at_bound": bool(
            low_host_refit.get("feii_sigma_at_bound", False)
        ),
        "staged_feii_shift_at_bound": bool(
            low_host_refit.get("feii_shift_at_bound", False)
        ),
        "staged_feii_detected": bool(low_host_refit.get("feii_detected", False)),
        "staged_feii_well_constrained": bool(
            low_host_refit.get("feii_well_constrained", False)
        ),
        "staged_feii_secure": bool(low_host_refit.get("feii_secure", False)),
        "staged_quality": low_host_refit.get("quality", ""),
        "staged_host_kinematics_reliable": bool(
            low_host_refit.get("host_kinematics_reliable", False)
        ),
        "staged_host_decomposition_reliable": bool(
            low_host_refit.get("host_decomposition_reliable", False)
        ),
        "staged_delta_bic_feii": low_host_refit.get("delta_bic_feii", np.nan),
        "staged_chi2": low_host_refit.get("chi2", np.nan),
        "staged_redchi": low_host_refit.get("redchi", np.nan),
        "staged_ndata": low_host_refit.get("ndata", np.nan),
        "low_host_powerlaw_feii_enabled": bool(low_host_refit.get("enabled")),
        "low_host_powerlaw_feii_success": bool(low_host_refit.get("success")),
        "low_host_powerlaw_feii_usable": bool(low_host_refit.get("usable")),
        "low_host_powerlaw_feii_reason": low_host_refit.get("reason", ""),
        "low_host_fraction_hb": low_host_refit.get("host_fraction_hb", np.nan),
        "low_host_refit_host_scale": low_host_refit.get("host_scale", np.nan),
        "low_host_refit_feii_amp": low_host_refit.get("feii_amp", np.nan),
        "low_host_refit_feii_velocity": low_host_refit.get("feii_velocity", np.nan),
        "low_host_refit_feii_sigma": low_host_refit.get("feii_sigma", np.nan),
        "low_host_refit_chi2": low_host_refit.get("chi2", np.nan),
        "low_host_refit_redchi": low_host_refit.get("redchi", np.nan),
    }


def _add_broad_total_aliases(line_flux: dict[str, Any]) -> None:
    for prefix in ("Hgamma", "Hdelta"):
        if f"{prefix}_narrow" in line_flux:
            line_flux[f"{prefix}_narrow_flux"] = line_flux[f"{prefix}_narrow"]
        if f"{prefix}_broad" in line_flux:
            line_flux[f"{prefix}_broad_total"] = line_flux[f"{prefix}_broad"]
            line_flux[f"{prefix}_broad_flux_total"] = line_flux[f"{prefix}_broad"]
    if "Hb_broad_1" in line_flux:
        total = float(line_flux.get("Hb_broad_1", 0.0)) + float(
            line_flux.get("Hb_broad_2", 0.0)
        )
        line_flux["Hb_broad"] = total
        line_flux["Hb_broad_total"] = total
        line_flux["Hb_broad_flux_total"] = total
    if "Ha_broad_1" in line_flux:
        total = float(line_flux.get("Ha_broad_1", 0.0)) + float(
            line_flux.get("Ha_broad_2", 0.0)
        )
        line_flux["Ha_broad"] = total
        line_flux["Ha_broad_total"] = total
        line_flux["Ha_broad_flux_total"] = total


def _joint_result_usable(result: Any) -> bool:
    if result is None:
        return False
    if not (np.isfinite(result.chi2) and np.isfinite(result.redchi)):
        return False
    if result.fit_info.get("n_line_flux_boundary_hits", 0):
        return False
    return True


def _fit_summary(prefix: str, result: Any | None) -> dict[str, Any]:
    if result is None:
        return {
            f"{prefix}_success": False,
            f"{prefix}_chi2": np.nan,
            f"{prefix}_redchi": np.nan,
            f"{prefix}_bic": np.nan,
        }
    info = getattr(result, "fit_info", {}) or {}
    summary = {
        f"{prefix}_success": bool(result.success),
        f"{prefix}_chi2": result.chi2,
        f"{prefix}_redchi": result.redchi,
        f"{prefix}_bic": result.bic,
        f"{prefix}_ndata": result.ndata,
        f"{prefix}_nvarys": result.nvarys,
        f"{prefix}_broad_selected": info.get(
            "broad_balmer_selected",
            info.get("include_broad_hb", False),
        ),
        f"{prefix}_delta_bic_broad": info.get("delta_bic_broad", np.nan),
        f"{prefix}_bic_narrow": info.get("bic_narrow", np.nan),
        f"{prefix}_bic_broad": info.get("bic_broad", np.nan),
        f"{prefix}_include_broad_hb": bool(info.get("include_broad_hb", False)),
    }
    for key in (
        "Hb_broad_fwhm_combined",
        "Hb_broad_line_dispersion",
        "Hb_broad_centroid_velocity",
        "Hb_broad_snr",
        "Ha_broad_fwhm_combined",
        "Ha_broad_line_dispersion",
        "Ha_broad_centroid_velocity",
        "Ha_broad_snr",
        "feii_amp",
        "feii_velocity",
        "feii_sigma",
        "feii_flux_integrated",
        "feii_snr",
        "flag_feii_missing_template",
        "flag_feii_boundary",
        "flag_feii_weak",
        "flag_feii_underfit",
        "bic_with_feii",
        "bic_without_feii",
        "delta_bic_feii",
        "narrow_hb_flux_change_percent_feii",
        "flag_broad_hb_weak",
        "flag_broad_hb_boundary",
        "flag_broad_hb_unphysical",
        "flag_broad_hb_suspicious",
        "flag_broad_balmer_secure",
        "flag_broad_balmer_tentative",
        "flag_broad_balmer_unphysical",
        "flag_broad_balmer_boundary",
        "include_broad_hb",
        "broad_hb_model",
        "desipecfit_force_broad_hb",
        "desipecfit_hb_model_mode",
        "desipecfit_hb_broad_1_sigma_kms",
        "desipecfit_hb_broad_1_fwhm_kms",
        "desipecfit_hb_broad_2_sigma_kms",
        "desipecfit_hb_broad_2_fwhm_kms",
        "desipecfit_hb_broad_max_component_fwhm_kms",
        "desipecfit_hb_broad_component_fwhm_limit_kms",
        "desipecfit_flag_hb_broad_component_fwhm_gt_limit",
        "desipecfit_ha_broad_1_sigma_kms",
        "desipecfit_ha_broad_1_fwhm_kms",
        "desipecfit_ha_broad_2_sigma_kms",
        "desipecfit_ha_broad_2_fwhm_kms",
        "desipecfit_ha_broad_max_component_fwhm_kms",
        "desipecfit_ha_broad_component_fwhm_limit_kms",
        "desipecfit_flag_ha_broad_component_fwhm_gt_limit",
        "desipecfit_mgii_broad_1_sigma_kms",
        "desipecfit_mgii_broad_1_fwhm_kms",
        "desipecfit_mgii_broad_2_sigma_kms",
        "desipecfit_mgii_broad_2_fwhm_kms",
        "desipecfit_mgii_broad_max_component_fwhm_kms",
        "desipecfit_mgii_broad_component_fwhm_limit_kms",
        "desipecfit_flag_mgii_broad_component_fwhm_gt_limit",
        "desipecfit_flag_broad_component_fwhm_gt_limit",
        "desipecfit_flag_continuum_unreasonable",
        "desipecfit_broad_fwhm_violating_components",
        "desipecfit_removed_broad_components",
        "desipecfit_broad_refit_sequence",
        "desipecfit_refit_after_broad_fwhm_limit",
        "selected_model",
        "delta_bic_uv_feii",
        "delta_bic_mgii_broad",
        "delta_bic_mgii_narrow",
        "uv_feii_snr",
        "mgii_broad_snr",
        "mgii_narrow_snr",
        "mgii_broad_fwhm",
        "accept_uv_feii",
        "accept_mgii_broad",
        "secure_mgii_broad",
        "accept_mgii_narrow",
        "desipecfit_original_selected_model",
        "desipecfit_original_accept_mgii_broad",
        "desipecfit_force_mgii_broad",
        "desipecfit_mgii_broad_force_reason",
        "desipecfit_force_mgii_broad_by_default",
        "desipecfit_force_mgii_broad_by_snr_threshold",
        "desipecfit_mgii_broad_snr_threshold",
        "flag_uv_feii_missing_template",
        "desipecfit_uv_feii_template_source",
        "desipecfit_uv_feii_template_path",
        "broad_model",
        "narrow_mode",
    ):
        if key in info:
            summary[f"{prefix}_{key}"] = info[key]
    if getattr(result, "warnings", None):
        summary[f"{prefix}_warnings"] = " | ".join(result.warnings)
    return summary


def _format_report(
    h1: H1Spectrum,
    stellar: Any,
    low_host_refit: dict[str, Any],
    fits: dict[str, dict[str, Any]],
    joint: dict[str, Any],
    no_local_continuum: dict[str, Any],
) -> str:
    d = getattr(stellar, "diagnostics", {}) or {}
    lines = [
        f"EchoSpecFit report: {h1.object_id} [{h1.survey}]",
        "stage 1: pPXF host + power-law",
        "  success: True",
        f"  velocity: {stellar.velocity:.6g} km/s",
        f"  sigma: {stellar.sigma:.6g} km/s",
        f"  chi2: {stellar.chi2:.6g}",
        f"  goodpixels: {len(stellar.goodpixels)}",
        "  pPXF mdegree: "
        f"initial={d.get('initial_ppxf_mdegree')} final={d.get('final_ppxf_mdegree')}",
        "  AGN power law: "
        f"templates={d.get('n_agn_powerlaw_templates')} "
        f"nonzero={d.get('agn_powerlaw_weights_nonzero_count')} "
        f"weight_sum={_fmt_float(d.get('agn_powerlaw_weight_sum'))}",
        "stage 2: fixed-host power-law + optical Fe II",
        f"  enabled: {bool(low_host_refit.get('enabled'))}",
        f"  success: {bool(low_host_refit.get('success'))}",
        f"  usable: {bool(low_host_refit.get('usable'))}",
        f"  reason: {low_host_refit.get('reason', '')}",
        f"  pPXF host fraction Hbeta/FeII: {_fmt_float(low_host_refit.get('ppxf_host_fraction_hb', low_host_refit.get('host_fraction_hb')))}",
        f"  host kinematics: {low_host_refit.get('host_kinematics', '')}",
        f"  likelihood errors: {low_host_refit.get('errors', '')}",
        f"  staged host fraction at 5100 A: {_fmt_float(low_host_refit.get('host_fraction_5100'))}",
        f"  host scale / power-law slope: {_fmt_float(low_host_refit.get('host_scale'))} / {_fmt_float(low_host_refit.get('powerlaw_alpha'))}",
        f"  FeII flux/EW: {_fmt_float(low_host_refit.get('feii_flux_4434_4684'))} / {_fmt_float(low_host_refit.get('feii_ew_4434_4684'))}",
        f"  FeII shift/sigma/FWHM: {_fmt_float(low_host_refit.get('feii_shift_kms'))} / {_fmt_float(low_host_refit.get('feii_sigma_kms'))} / {_fmt_float(low_host_refit.get('feii_fwhm_kms'))}",
        f"  chi2/redchi/delta-BIC(FeII): {_fmt_float(low_host_refit.get('chi2'))} / {_fmt_float(low_host_refit.get('redchi'))} / {_fmt_float(low_host_refit.get('delta_bic_feii'))}",
    ]
    for label, key in (
        ("Mg II + UV Fe II", "mgii_uv"),
        ("[O II]", "oii"),
        ("Hγ + Hδ + [Ne III]", "blue_balmer"),
        ("Hbeta + [O III] after staged Fe II subtraction", "hb_oiii"),
        ("Halpha + [N II] + [S II]", "ha_nii_sii"),
    ):
        payload = fits[key]
        result = payload["result"]
        lines.append(f"{label}:")
        if result is None:
            lines.append(f"  failed: {payload['error']}")
            continue
        lines.extend(
            [
                f"  success: {result.success}",
                f"  message: {result.message}",
                f"  chi2/redchi/BIC: {result.chi2:.6g} / {result.redchi:.6g} / {result.bic:.6g}",
                f"  ndata/nvarys: {result.ndata} / {result.nvarys}",
            ]
        )
        if result.fit_info and "delta_bic_broad" in result.fit_info:
            lines.append(
                "  broad Balmer BIC: "
                f"selected={result.fit_info.get('broad_balmer_selected')} "
                f"delta={_fmt_float(result.fit_info.get('delta_bic_broad'))}"
            )
        if getattr(result, "warnings", None):
            lines.append("  warnings:")
            for warning in result.warnings:
                lines.append(f"    {warning}")
        lines.append("  line fluxes:")
        for name, value in sorted(result.line_flux.items()):
            lines.append(f"    {name}: {value:.6g}")
    lines.append("stage 4: full-spectrum joint refinement")
    joint_result = joint["result"]
    if joint_result is None:
        lines.append(f"  failed: {joint['error']}")
    else:
        lines.extend(
            [
                f"  success: {joint_result.success}",
                f"  message: {joint_result.message}",
                f"  chi2/redchi/BIC: {joint_result.chi2:.6g} / {joint_result.redchi:.6g} / {joint_result.bic:.6g}",
                f"  ndata/nvarys: {joint_result.ndata} / {joint_result.nvarys}",
                f"  stellar_scale: {joint_result.params.get('stellar_scale', float('nan')):.6g}",
                f"  usable_final_columns: {_joint_result_usable(joint_result)}",
                "  line fluxes:",
            ]
        )
        for name, value in sorted(joint_result.line_flux.items()):
            lines.append(f"    {name}: {value:.6g}")
    lines.append("stage 5: global no-local-continuum refit")
    if not no_local_continuum.get("enabled"):
        lines.append(f"  disabled: {no_local_continuum.get('error', '')}")
    elif not no_local_continuum.get("success"):
        lines.append(f"  failed: {no_local_continuum.get('error', '')}")
    else:
        lines.extend(
            [
                f"  success: {bool(no_local_continuum.get('success'))}",
                f"  message: {no_local_continuum.get('message', '')}",
                "  local continuum terms included: False",
                f"  power-law mode: {no_local_continuum.get('powerlaw_mode', '')}",
                f"  selected power-law slope: {_fmt_float(no_local_continuum.get('selected_powerlaw_slope'))}",
                f"  force broad Hγ/Hδ: {bool(no_local_continuum.get('force_blue_balmer_broad', False))}",
                f"  forced broad Hγ/Hδ min scale: {_fmt_float(no_local_continuum.get('forced_blue_broad_min_scale'))}",
                f"  Hγ/Hδ broad profile: {no_local_continuum.get('blue_broad_profile', '')}",
                f"  chi2/redchi/BIC: {_fmt_float(no_local_continuum.get('chi2'))} / {_fmt_float(no_local_continuum.get('redchi'))} / {_fmt_float(no_local_continuum.get('bic'))}",
                f"  ndata/nvarys: {no_local_continuum.get('ndata')} / {no_local_continuum.get('nvarys')}",
                "  final line/component fluxes:",
            ]
        )
        for name, value in sorted((no_local_continuum.get("scaled_flux", {}) or {}).items()):
            lines.append(f"    {name}: {value:.6g}")
    return "\n".join(lines)


def _save_plots(
    outdir: Path,
    result: EchoSpecFitResult,
    *,
    include_line_plots: bool = True,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from echospec.plotting import (
        plot_blue_balmer_fit,
        plot_emission_fit,
        plot_hb_oiii_feii_diagnostic,
        plot_mgii_fit,
    )
    from echospec.stellar import plot_stellar_fit

    object_id = result.spectrum.object_id
    plot_warnings: list[str] = []
    generated: dict[str, str] = {}
    if result.low_host_refit.get("usable"):
        try:
            staged_path = outdir / f"{object_id}_staged_host_feii_diagnostic.png"
            fig = make_staged_diagnostic_figure(
                spectrum=result.spectrum,
                rest_spec=result.spec,
                stellar=result.stellar,
                staged=result.low_host_refit,
            )
            fig.savefig(
                staged_path,
                dpi=180,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close(fig)
            generated["staged_diagnostic_plot"] = str(staged_path)
        except Exception as exc:
            plot_warnings.append(f"staged_host_feii_diagnostic: {exc!r}")
    if not include_line_plots:
        if plot_warnings:
            result.summary["plot_warnings"] = " | ".join(plot_warnings)
        return generated
    try:
        stellar_path = outdir / f"{object_id}_stellar_fit.png"
        fig = plot_stellar_fit(
            result.spec,
            result.stellar,
            stellar_path,
        )
        plt.close(fig)
        generated["stellar_plot"] = str(stellar_path)
    except Exception as exc:
        plot_warnings.append(f"stellar_fit: {exc!r}")
    for key in ("mgii_uv", "oii", "blue_balmer", "hb_oiii", "ha_nii_sii"):
        fit = result.fits[key]["result"]
        if fit is None:
            continue
        fig = None
        output_path = outdir / f"{object_id}_{key}_fit.png"
        try:
            if key == "mgii_uv":
                mgii_spec = result.residual_spec.select_window(*MGII_WINDOW).mask_bad_pixels()
                fig = plot_mgii_fit(
                    mgii_spec,
                    fit,
                    output_path,
                    title=None,
                )
            elif key == "blue_balmer":
                blue_spec = result.residual_spec.select_window(*BLUE_BALMER_WINDOW).mask_bad_pixels()
                fig = plot_blue_balmer_fit(
                    blue_spec,
                    fit,
                    output_path,
                    title=None,
                )
            elif key == "hb_oiii":
                hb_spec = result.residual_spec.select_window(*HB_FEII_WINDOW).mask_bad_pixels()
                fig = plot_hb_oiii_feii_diagnostic(
                    hb_spec,
                    fit,
                    output_path,
                    title=None,
                )
            elif key == "ha_nii_sii":
                ha_spec = result.residual_spec.select_window(*HA_NII_SII_WINDOW).mask_bad_pixels()
                fig = plot_emission_fit(
                    ha_spec,
                    fit,
                    output_path,
                    title=f"{object_id} {key}",
                )
            else:
                fig = plot_emission_fit(
                    result.residual_spec,
                    fit,
                    output_path,
                    title=f"{object_id} {key}",
                )
            if output_path.is_file():
                generated[f"{key}_plot"] = str(output_path)
        except Exception as exc:
            plot_warnings.append(f"{key}_fit: {exc!r}")
        finally:
            if fig is not None:
                plt.close(fig)
    if result.no_local_continuum.get("success"):
        fig = None
        try:
            no_local_path = outdir / f"{object_id}_no_local_continuum_refit_components.png"
            fig = _plot_no_local_continuum_refit(
                result,
                no_local_path,
            )
            generated["no_local_continuum_plot"] = str(no_local_path)
        except Exception as exc:
            plot_warnings.append(f"no_local_continuum_refit_components: {exc!r}")
        finally:
            if fig is not None:
                plt.close(fig)
    if plot_warnings:
        warning_path = outdir / f"{object_id}_plot_warnings.txt"
        warning_path.write_text(
            "\n".join(plot_warnings) + "\n",
            encoding="utf-8",
        )
        generated["plot_warnings_file"] = str(warning_path)
        result.summary["plot_warnings"] = " | ".join(plot_warnings)
    return generated


def _plot_no_local_continuum_refit(
    result: EchoSpecFitResult,
    output_path: Path,
) -> Any:
    import matplotlib.pyplot as plt

    payload = result.no_local_continuum
    wave = np.asarray(result.spec.wave, dtype=float)
    flux = np.asarray(result.spec.flux, dtype=float)
    valid = np.asarray(payload.get("valid_mask"), dtype=bool)
    model = np.asarray(payload["model_flux"], dtype=float)
    continuum = np.asarray(payload["continuum_flux"], dtype=float)
    components = payload.get("component_flux", {}) or {}
    groups = payload.get("groups", {}) or {}

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(15.8, 11.2),
        dpi=180,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.35, 1.05, 1.35, 1.55]},
    )
    ax, axc, axuv, axopt = axes
    ax.plot(wave[valid], flux[valid], color="0.17", lw=0.82, alpha=0.88, label="Observed")
    ax.plot(wave[valid], model[valid], color="#d62728", lw=1.85, label="No-local-continuum refit")
    ax.plot(wave[valid], continuum[valid], color="#1f2937", lw=1.05, ls="--", alpha=0.82, label="Global continuum")
    y = flux[valid]
    lo, hi = np.nanpercentile(y, [1.0, 99.3])
    pad = 0.12 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("Flux")
    ax.set_title(f"{result.spectrum.object_id} no-local-continuum refit")
    ax.legend(loc="upper right", frameon=False, ncol=3)

    host = np.asarray(components.get("pPXF host", np.zeros_like(wave)), dtype=float)
    powerlaw = continuum - host
    axc.plot(wave[valid], host[valid], color="#8c564b", lw=1.25, label="pPXF host")
    axc.plot(wave[valid], powerlaw[valid], color="#1f77b4", lw=1.25, label="Refit power law")
    axc.plot(wave[valid], continuum[valid], color="#111827", lw=1.0, ls="--", label="Host + refit power law")
    axc.set_ylabel("Continuum")
    axc.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.5)

    uv_colors = {"UV FeII 2200-3090": "#9467bd", "MgII broad": "#ff7f0e"}
    for name in groups.get("uv", []):
        values = np.asarray(components[name], dtype=float)
        axuv.plot(
            wave[valid],
            values[valid],
            lw=1.65,
            color=uv_colors.get(name),
            label=_component_display_label(name),
        )
    axuv.axhline(0, color="0.55", lw=0.65)
    axuv.set_ylabel("UV comp.")
    handles, _ = axuv.get_legend_handles_labels()
    if handles:
        axuv.legend(loc="upper right", frameon=False, ncol=2, fontsize=8.5)

    opt_colors = {
        "[OII] 3727": "#2ca02c",
        "Hdelta narrow": "#7f7f7f",
        "Hdelta broad": "#6a3d9a",
        "Hgamma narrow": "#b15928",
        "Hgamma broad": "#fb9a99",
        "[NeIII] 3869": "#33a02c",
        "[NeIII] 3968": "#a6cee3",
        "Hepsilon narrow": "#cab2d6",
        "Optical FeII": "#17becf",
        "Hbeta narrow": "#bcbd22",
        "Hbeta broad 1": "#e377c2",
        "Hbeta broad 2": "#c51b7d",
        "[OIII] 4959": "#4c78a8",
        "[OIII] 5007": "#1f4e79",
        "Halpha narrow": "#d62728",
        "Halpha broad 1": "#ff7f0e",
        "Halpha broad 2": "#b45f06",
        "[OI] doublet": "#009E73",
        "[OI] 6300": "#8dd3c7",
        "[OI] 6363": "#66c2a5",
        "[NII] 6548": "#f781bf",
        "[NII] 6584": "#e7298a",
        "[SII] 6716": "#a6761d",
        "[SII] 6731": "#666666",
    }
    for name in groups.get("optical", []):
        values = np.asarray(components[name], dtype=float)
        axopt.plot(
            wave[valid],
            values[valid],
            lw=1.55,
            color=opt_colors.get(name),
            label=_component_display_label(name),
        )
    axopt.axhline(0, color="0.55", lw=0.65)
    axopt.set_ylabel("Optical comp.")
    axopt.set_xlabel("Rest wavelength (Angstrom)")
    axopt.legend(loc="upper right", frameon=False, ncol=5, fontsize=7.6)

    xmin = float(np.nanmin(wave[valid]))
    xmax = float(np.nanmax(wave[valid]))
    for axis in axes:
        axis.set_xlim(xmin, xmax)
        axis.grid(True, color="0.90", lw=0.5, alpha=0.55)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    _set_component_ylim(axuv, wave, valid, components, groups.get("uv", []))
    _set_component_ylim(axopt, wave, valid, components, groups.get("optical", []))
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def _set_component_ylim(
    axis: Any,
    wave: np.ndarray,
    valid: np.ndarray,
    components: dict[str, np.ndarray],
    names: list[str],
) -> None:
    values: list[np.ndarray] = []
    for name in names:
        component = np.asarray(components[name], dtype=float)
        finite = component[valid & np.isfinite(wave) & np.isfinite(component)]
        if finite.size:
            values.append(finite)
    if not values:
        axis.set_ylim(-0.1, 1.0)
        return
    combined = np.concatenate(values)
    ymax = float(np.nanmax(combined))
    ymin = float(np.nanmin(combined))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    floor = min(-0.05 * ymax, ymin - 0.08 * (ymax - ymin))
    axis.set_ylim(floor, ymax * 1.12)


def _component_display_label(name: str) -> str:
    return (
        name.replace("Hgamma", "Hγ")
        .replace("Hdelta", "Hδ")
        .replace("Hbeta", "Hβ")
        .replace("Halpha", "Hα")
        .replace("Hepsilon", "Hε")
    )


def _fmt_float(value: Any) -> str:
    try:
        value = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def _log_ratio(num: float | None, den: float | None) -> float:
    if num is None or den is None or num <= 0 or den <= 0:
        return float("nan")
    return float(np.log10(num / den))


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


def _no_local_continuum_detail(
    no_local_continuum: dict[str, Any],
    *,
    include_arrays: bool,
) -> dict[str, Any]:
    array_keys = {
        "model_flux",
        "continuum_flux",
        "line_flux",
        "residual_flux",
        "component_flux",
        "valid_mask",
    }
    return _json_safe(
        {
            key: value
            for key, value in no_local_continuum.items()
            if include_arrays or key not in array_keys
        }
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "EchoSpecFitConfig",
    "EchoSpecFitResult",
    "DesiPecFitConfig",
    "DesiPecFitResult",
    "fit_directory",
    "fit_h1spectrum",
    "fit_spectrum_file",
]
