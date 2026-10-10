"""FITS persistence and replotting for EchoSpec fit results.

The functions in this module are deliberately independent from the fitting
machinery. A saved ``result.fits`` should contain enough arrays, parameters,
diagnostics, and configuration metadata to recreate diagnostic plots without
rerunning pPXF or any emission-line fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import hashlib
import json
import re

import numpy as np

C_KMS = 299792.458

WINDOW_RESULTS: tuple[tuple[str, str, str], ...] = (
    ("hb_oiii", "Hbeta + [O III] + Fe II", "hb_oiii"),
    ("ha_nii_sii", "Halpha + [O I] + [N II] + [S II]", "ha_nii_sii"),
    ("oii", "[O II]", "oii"),
    ("blue_balmer", "Blue Balmer", "blue_balmer"),
)

COMPONENT_COLUMNS: tuple[str, ...] = (
    "host_stellar",
    "galaxy_host",
    "stellar_model",
    "powerlaw",
    "feii_total",
    "feii_blue",
    "feii_central",
    "feii_red",
    "Hb_narrow",
    "Hb_broad_1",
    "Hb_broad_2",
    "Hb_broad_total",
    "OIII_4959_core",
    "OIII_5007_core",
    "OIII_4959_wing",
    "OIII_5007_wing",
    "Ha_narrow",
    "Ha_broad_1",
    "Ha_broad_2",
    "Ha_broad_total",
    "OI_6300",
    "OI_6363",
    "NII_6548",
    "NII_6584",
    "SII_6717",
    "SII_6731",
    "OII_3726",
    "OII_3729",
    "OII_model",
    "Hgamma_narrow",
    "Hgamma_broad_total",
    "Hdelta_narrow",
    "Hdelta_broad_total",
    "NeIII_3869",
    "NeIII_3968",
)

LINE_PARAM_COLUMNS: tuple[str, ...] = (
    "name",
    "flux",
    "flux_err",
    "flux_p16",
    "flux_p50",
    "flux_p84",
    "luminosity",
    "luminosity_err",
    "ew",
    "ew_err",
    "velocity",
    "velocity_err",
    "sigma_kms",
    "sigma_kms_err",
    "fwhm_kms",
    "fwhm_kms_err",
    "snr",
    "tied_group",
    "component_type",
    "detected_flag",
    "quality_flag",
)

DEFAULT_FLAGS: tuple[str, ...] = (
    "flag_feii_shape_mismatch",
    "flag_single_feii_template_shape_mismatch",
    "flag_feii_boundary",
    "flag_narrow_hbeta_too_broad",
    "flag_ha_overfit",
    "flag_oii_overfit",
    "flag_balmer_continuum_region_skipped",
    "flag_needs_manual_inspection",
)

AGN_AUXILIARY_COMPONENT_COLUMNS: frozenset[str] = frozenset(
    {
        "wave",
        "host_stellar",
        "galaxy_host",
        "stellar_model",
        "powerlaw",
        "feii_total",
        "feii_blue",
        "feii_central",
        "feii_red",
        "local_continuum_model",
        "extra_local_continuum_model",
    }
)

AGN_EMISSION_SUM_EXCLUDE: frozenset[str] = frozenset(
    {
        "Hb_broad_total",
        "Ha_broad_total",
        "Hgamma_broad_total",
        "Hdelta_broad_total",
        "OII_model",
    }
)

AGN_KNOWN_LINE_POSITIONS: tuple[tuple[float, str], ...] = (
    (3426.85, "[Ne V]"),
    (3726.03, "[O II] 3726"),
    (3728.82, "[O II] 3729"),
    (3868.76, "[Ne III]"),
    (3967.79, "[Ne III]"),
    (3970.07, "Hepsilon"),
    (4101.74, "Hdelta"),
    (4340.47, "Hgamma"),
    (4685.68, "He II"),
    (4861.33, "Hbeta"),
    (4958.91, "[O III] 4959"),
    (5006.84, "[O III] 5007"),
    (6300.30, "[O I] 6300"),
    (6363.78, "[O I] 6363"),
    (6548.05, "[N II] 6548"),
    (6562.80, "Halpha"),
    (6583.45, "[N II] 6584"),
    (6716.44, "[S II] 6716"),
    (6730.82, "[S II] 6731"),
)


@dataclass
class SavedFitResult:
    """FitResult-like object reconstructed from a durable FITS product."""

    object_id: str = ""
    redshift: float = float("nan")
    survey: str = ""
    fitdate: str = ""
    code_version: str = ""
    strategy: str = ""
    status: str = ""
    quality: str = ""
    mc_n: int = 0
    random_seed: int | None = None
    template_info: str = ""
    config_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    spectrum: dict[str, np.ndarray] = field(default_factory=dict)
    model_total: dict[str, np.ndarray] = field(default_factory=dict)
    components: dict[str, np.ndarray] = field(default_factory=dict)
    line_params: list[dict[str, Any]] = field(default_factory=list)
    continuum_params: list[dict[str, Any]] = field(default_factory=list)
    stellar_params: list[dict[str, Any]] = field(default_factory=list)
    mc_line_samples: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def save_fit_result_to_fits(
    result: Any,
    output_path: str | Path,
    overwrite: bool = True,
) -> Path:
    """Save an EchoSpec fit result to a self-contained FITS file.

    ``result`` may be a :class:`SavedFitResult`, a mapping with equivalent
    fields, or an EpochFitSet-like object carrying a full input spectrum on one
    of these attributes: ``spectrum``, ``spec``, ``input_spectrum`` or
    ``_spectrum``. The full spectrum is required because ``EpochFitSet`` alone
    stores window arrays but not the parent wavelength grid.
    """

    from astropy.io import fits
    from astropy.table import Table

    saved = ensure_saved_fit_result(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing FITS file: {output}")

    hdus: list[Any] = [fits.PrimaryHDU(header=_primary_header(saved))]
    hdus.append(fits.BinTableHDU(Table(_spectrum_table(saved)), name="SPECTRUM"))
    hdus.append(fits.BinTableHDU(Table(_model_total_table(saved)), name="MODEL_TOTAL"))
    hdus.append(fits.BinTableHDU(Table(_components_table(saved)), name="COMPONENTS"))
    hdus.append(fits.BinTableHDU(Table(rows=_line_param_rows(saved)), name="LINE_PARAMS"))
    hdus.append(
        fits.BinTableHDU(
            Table(rows=_continuum_param_rows(saved)),
            name="CONTINUUM_PARAMS",
        )
    )
    hdus.append(
        fits.BinTableHDU(
            Table(rows=_stellar_param_rows(saved)),
            name="STELLAR_PARAMS",
        )
    )
    if saved.mc_line_samples:
        hdus.append(
            fits.BinTableHDU(
                Table(rows=_mc_line_sample_rows(saved)),
                name="MC_LINE_SAMPLES",
            )
        )
    hdus.append(fits.BinTableHDU(Table(rows=_diagnostic_rows(saved)), name="DIAGNOSTICS"))
    hdus.append(fits.BinTableHDU(Table(rows=_config_rows(saved)), name="CONFIG"))

    tmp_path = output.with_name(f"{output.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    fits.HDUList(hdus).writeto(tmp_path, overwrite=True)
    tmp_path.replace(output)
    return output


def load_fit_result_from_fits(fits_path: str | Path) -> SavedFitResult:
    """Load a saved EchoSpec FITS product into a FitResult-like object."""

    from astropy.io import fits

    path = Path(fits_path)
    required = ("SPECTRUM", "MODEL_TOTAL", "COMPONENTS", "LINE_PARAMS", "DIAGNOSTICS", "CONFIG")
    with fits.open(path) as hdul:
        missing = [name for name in required if name not in hdul]
        if missing:
            raise ValueError(f"{path} is missing required HDU(s): {', '.join(missing)}")
        header = hdul[0].header
        saved = SavedFitResult(
            object_id=str(header.get("OBJECTID", "")),
            redshift=_to_float(header.get("REDSHIFT", np.nan)),
            survey=str(header.get("SURVEY", "")),
            fitdate=str(header.get("FITDATE", "")),
            code_version=str(header.get("CODEVER", "")),
            strategy=str(header.get("STRATEGY", "")),
            status=str(header.get("STATUS", "")),
            quality=str(header.get("QUALITY", "")),
            mc_n=int(header.get("MC_N", 0) or 0),
            random_seed=_header_optional_int(header.get("RANDOMSD")),
            template_info=str(header.get("TEMPLATE", "")),
            config_hash=str(header.get("CONFIGHS", "")),
            spectrum=_table_to_dict(hdul["SPECTRUM"].data),
            model_total=_table_to_dict(hdul["MODEL_TOTAL"].data),
            components=_table_to_dict(hdul["COMPONENTS"].data),
            line_params=_table_to_rows(hdul["LINE_PARAMS"].data),
            continuum_params=_table_to_rows(hdul["CONTINUUM_PARAMS"].data)
            if "CONTINUUM_PARAMS" in hdul
            else [],
            stellar_params=_table_to_rows(hdul["STELLAR_PARAMS"].data)
            if "STELLAR_PARAMS" in hdul
            else [],
            mc_line_samples=_table_to_rows(hdul["MC_LINE_SAMPLES"].data)
            if "MC_LINE_SAMPLES" in hdul
            else [],
            diagnostics=_key_value_table_to_dict(hdul["DIAGNOSTICS"].data),
            config=_key_value_table_to_dict(hdul["CONFIG"].data),
        )
    saved.warnings = _collect_warnings(saved.diagnostics)
    return saved


def plot_fit_result_pdf(
    result: Any,
    output_pdf: str | Path,
    plot_style: str = "full",
    show_components: str | list[str] | tuple[str, ...] | None = None,
    xlim: tuple[float, float] | None = None,
) -> Path:
    """Write a multi-page diagnostic PDF from a saved result object or FITS path."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    saved = load_fit_result_from_fits(result) if isinstance(result, (str, Path)) else ensure_saved_fit_result(result)
    output = Path(output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    groups = _parse_component_filter(show_components)

    with PdfPages(output) as pdf:
        fig = _plot_full_overview(saved, groups, xlim=xlim)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_hb_oiii_page(saved, groups, plot_style=plot_style)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_feii_diagnostic_page(saved, groups)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_halpha_page(saved, groups, plot_style=plot_style)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _plot_oii_page(saved, groups)
        pdf.savefig(fig)
        plt.close(fig)

        if _has_any_component(saved, ("Hgamma_narrow", "Hgamma_broad_total", "Hdelta_narrow", "Hdelta_broad_total", "NeIII_3869", "NeIII_3968")):
            fig = _plot_blue_balmer_page(saved, groups)
            pdf.savefig(fig)
            plt.close(fig)
    return output


def plot_agn_fit_from_fits(
    fits_path: str | Path,
    output_pdf: str | Path | None = None,
    rest_frame: bool = True,
    show_all_components: bool = True,
    annotate_all_lines: bool = True,
    min_component_peak: float | None = None,
) -> Path | list[Path]:
    """Create a multi-page AGN fit-quality PDF directly from ``result.fits``.

    The function is deliberately FITS-structure driven. Emission-line
    components are discovered dynamically from HDU 3 ``COMPONENTS`` by
    excluding the continuum/template/helper columns in
    ``AGN_AUXILIARY_COMPONENT_COLUMNS``. If a line is present in
    ``LINE_PARAMS`` but the corresponding component array is missing or empty,
    a Gaussian is reconstructed for plotting and the condition is written to
    the PDF/log warnings.
    """

    path = Path(fits_path)
    if path.is_dir():
        out_dir = Path(output_pdf) if output_pdf is not None else path / "agn_fit_pdfs"
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for child in sorted(path.rglob("*_result.fits")):
            outputs.append(
                plot_agn_fit_from_fits(
                    child,
                    output_pdf=out_dir,
                    rest_frame=rest_frame,
                    show_all_components=show_all_components,
                    annotate_all_lines=annotate_all_lines,
                    min_component_peak=min_component_peak,
                )
            )
        return outputs

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    payload = _agn_read_fits_payload(path, rest_frame=rest_frame)
    if output_pdf is None or (Path(output_pdf).suffix.lower() != ".pdf"):
        out_dir = Path(output_pdf) if output_pdf is not None else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / _agn_default_pdf_name(payload, path)
    else:
        output = Path(output_pdf)
        output.parent.mkdir(parents=True, exist_ok=True)

    warnings = _agn_sanity_checks(payload)
    log_path = output.with_suffix(".log")
    _agn_write_log(log_path, path, payload, warnings)

    with PdfPages(output) as pdf:
        if payload["fatal_missing"]:
            fig = _agn_warning_only_page(payload, warnings)
            pdf.savefig(fig)
            plt.close(fig)
            return output

        fig = _agn_page_full_overview(payload, warnings, annotate_all_lines=annotate_all_lines)
        pdf.savefig(fig)
        plt.close(fig)

        fig = _agn_page_window(
            payload,
            "Blue / [O II] region",
            (3600.0, 4100.0),
            annotate_all_lines=annotate_all_lines,
            include_feii=False,
            min_component_peak=min_component_peak,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _agn_page_window(
            payload,
            r"H$\beta$ + [O III] + Fe II region",
            (4400.0, 5200.0),
            annotate_all_lines=annotate_all_lines,
            include_feii=True,
            min_component_peak=min_component_peak,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig = _agn_page_window(
            payload,
            r"H$\alpha$ + low-ionization region",
            (6200.0, 6900.0),
            annotate_all_lines=annotate_all_lines,
            include_feii=False,
            min_component_peak=min_component_peak,
            broad_first=True,
        )
        pdf.savefig(fig)
        plt.close(fig)

        if show_all_components:
            fig = _agn_page_all_components(payload, min_component_peak=min_component_peak)
            pdf.savefig(fig)
            plt.close(fig)

        for fig in _agn_line_param_table_pages(payload, warnings):
            pdf.savefig(fig)
            plt.close(fig)

    return output


def ensure_saved_fit_result(result: Any) -> SavedFitResult:
    """Normalize supported fit result objects to :class:`SavedFitResult`."""

    if isinstance(result, SavedFitResult):
        return _ensure_complete_saved_result(result)
    if isinstance(result, Mapping):
        return _ensure_complete_saved_result(_saved_from_mapping(result))
    return _ensure_complete_saved_result(_saved_from_epoch_fit_set(result))


def attach_spectrum_to_result(result: Any, spectrum: Any) -> Any:
    """Attach a full input spectrum to an existing EpochFitSet-like result."""

    setattr(result, "spectrum", spectrum)
    return result


def _saved_from_mapping(data: Mapping[str, Any]) -> SavedFitResult:
    return SavedFitResult(
        object_id=str(data.get("object_id", "")),
        redshift=_to_float(data.get("redshift", data.get("z", np.nan))),
        survey=str(data.get("survey", "")),
        fitdate=str(data.get("fitdate", "")),
        code_version=str(data.get("code_version", "")),
        strategy=str(data.get("strategy", "")),
        status=str(data.get("status", "")),
        quality=str(data.get("quality", "")),
        mc_n=int(data.get("mc_n", 0) or 0),
        random_seed=data.get("random_seed"),
        template_info=str(data.get("template_info", "")),
        config_hash=str(data.get("config_hash", "")),
        config=dict(data.get("config", {}) or {}),
        spectrum={key: np.asarray(value) for key, value in (data.get("spectrum", {}) or {}).items()},
        model_total={key: np.asarray(value) for key, value in (data.get("model_total", {}) or {}).items()},
        components={key: np.asarray(value) for key, value in (data.get("components", {}) or {}).items()},
        line_params=[dict(row) for row in data.get("line_params", []) or []],
        continuum_params=[dict(row) for row in data.get("continuum_params", []) or []],
        stellar_params=[dict(row) for row in data.get("stellar_params", []) or []],
        mc_line_samples=[dict(row) for row in data.get("mc_line_samples", []) or []],
        diagnostics=dict(data.get("diagnostics", {}) or {}),
        warnings=list(data.get("warnings", []) or []),
    )


def _saved_from_epoch_fit_set(epoch: Any) -> SavedFitResult:
    spec = _extract_spectrum(epoch)
    if spec is None:
        raise ValueError(
            "save_fit_result_to_fits needs a full spectrum. Attach it first with "
            "`result.spectrum = spec` or `attach_spectrum_to_result(result, spec)`."
        )

    wave, flux, err, rest_wave, original_flux, mask_good, z = _spectrum_arrays(spec)
    n = len(wave)
    components = _zero_components(n)
    model_total = {
        "wave": wave.copy(),
        "total_model": np.zeros(n, dtype=float),
        "residual": np.full(n, np.nan, dtype=float),
        "chi": np.full(n, np.nan, dtype=float),
        "stellar_model": np.zeros(n, dtype=float),
        "powerlaw_model": np.zeros(n, dtype=float),
        "feii_model": np.zeros(n, dtype=float),
        "local_continuum_model": np.zeros(n, dtype=float),
        "emission_total_model": np.zeros(n, dtype=float),
    }

    stellar = getattr(epoch, "stellar", None)
    stellar_params = _stellar_rows_from_result(stellar)
    if stellar is not None:
        stellar_model = _array_or_zero(getattr(stellar, "stellar_model", None), n)
        powerlaw_model = _array_or_zero(getattr(stellar, "agn_model", None), n)
        components["host_stellar"] = stellar_model.copy()
        components["galaxy_host"] = stellar_model.copy()
        components["stellar_model"] = stellar_model.copy()
        components["powerlaw"] = powerlaw_model.copy()
        model_total["stellar_model"] = stellar_model.copy()
        model_total["powerlaw_model"] = powerlaw_model.copy()

    line_params: list[dict[str, Any]] = []
    continuum_params: list[dict[str, Any]] = []
    diagnostics = dict(getattr(epoch, "diagnostics", {}) or {})

    for attr, _, _ in WINDOW_RESULTS:
        fit = getattr(epoch, attr, None)
        if fit is None:
            continue
        _add_window_to_saved_arrays(wave, components, model_total, fit)
        line_params.extend(_line_rows_from_fit(attr, fit))
        continuum_params.extend(_continuum_rows_from_fit(attr, fit))
        diagnostics.update(_diagnostics_from_fit(attr, fit))

    final_refit = getattr(epoch, "final_refit", None)
    if final_refit is not None and len(np.asarray(getattr(final_refit, "model_flux", []))) == n:
        model_total["total_model"] = np.asarray(final_refit.model_flux, dtype=float)
        model_total["residual"] = np.asarray(final_refit.residual_flux, dtype=float)
        model_total["stellar_model"] = np.asarray(getattr(final_refit, "stellar_flux", model_total["stellar_model"]), dtype=float)
        model_total["powerlaw_model"] = np.asarray(getattr(final_refit, "agn_flux", model_total["powerlaw_model"]), dtype=float)
        model_total["emission_total_model"] = np.asarray(getattr(final_refit, "emission_flux", model_total["emission_total_model"]), dtype=float)
    else:
        feii_groups = components["feii_blue"] + components["feii_central"] + components["feii_red"]
        model_total["feii_model"] = components["feii_total"] if np.any(components["feii_total"]) else feii_groups
        model_total["emission_total_model"] = _sum_emission_components(components)
        model_total["total_model"] = (
            model_total["stellar_model"]
            + model_total["powerlaw_model"]
            + model_total["emission_total_model"]
        )
        model_total["residual"] = flux - model_total["total_model"]

    with np.errstate(invalid="ignore", divide="ignore"):
        model_total["chi"] = model_total["residual"] / err

    config = _config_from_epoch(epoch)
    return SavedFitResult(
        object_id=str(getattr(epoch, "object_id", getattr(spec, "object_id", ""))),
        redshift=float(getattr(spec, "z", z)),
        survey=str(getattr(epoch, "survey", getattr(spec, "survey", ""))),
        fitdate=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        code_version=_code_version(),
        strategy=str(diagnostics.get("fit_strategy", config.get("fit_strategy", "window_joint"))),
        status=str(diagnostics.get("status", "ok")),
        quality=str(diagnostics.get("fit_quality", "")),
        mc_n=int(_first_finite_config(config, ("mc_n", "n_mc"), default=0)),
        random_seed=_optional_int(config.get("random_seed")),
        template_info=_template_info(stellar),
        config_hash=_config_hash(config),
        config=config,
        spectrum={
            "wave": wave,
            "flux": flux,
            "err": err,
            "mask_good": mask_good.astype(bool),
            "rest_wave": rest_wave,
            "original_flux": original_flux,
        },
        model_total=model_total,
        components=components,
        line_params=line_params,
        continuum_params=continuum_params,
        stellar_params=stellar_params,
        mc_line_samples=_mc_rows_from_epoch(epoch),
        diagnostics=diagnostics,
        warnings=_warnings_from_epoch(epoch),
    )


def _ensure_complete_saved_result(saved: SavedFitResult) -> SavedFitResult:
    wave = np.asarray(saved.spectrum.get("wave", []), dtype=float)
    if wave.size == 0:
        wave = np.asarray(saved.model_total.get("wave", []), dtype=float)
    if wave.size == 0:
        raise ValueError("SavedFitResult requires a non-empty SPECTRUM wave array.")
    n = len(wave)
    saved.spectrum.setdefault("wave", wave)
    saved.spectrum.setdefault("rest_wave", wave.copy())
    saved.spectrum.setdefault("flux", np.full(n, np.nan))
    saved.spectrum.setdefault("err", np.full(n, np.nan))
    saved.spectrum.setdefault("mask_good", np.isfinite(saved.spectrum["wave"]))
    saved.spectrum.setdefault("original_flux", np.asarray(saved.spectrum.get("flux"), dtype=float).copy())

    for key in ("wave", "total_model", "residual", "chi", "stellar_model", "powerlaw_model", "feii_model", "local_continuum_model", "emission_total_model"):
        if key == "wave":
            saved.model_total[key] = _array_or_default(saved.model_total.get(key), wave, n)
        else:
            saved.model_total[key] = _array_or_default(saved.model_total.get(key), np.zeros(n), n)

    for key in COMPONENT_COLUMNS:
        saved.components[key] = _array_or_default(saved.components.get(key), np.zeros(n), n)

    if not saved.fitdate:
        saved.fitdate = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not saved.code_version:
        saved.code_version = _code_version()
    if not saved.config_hash:
        saved.config_hash = _config_hash(saved.config)
    for flag in DEFAULT_FLAGS:
        saved.diagnostics.setdefault(flag, False)
    return saved


def _primary_header(saved: SavedFitResult) -> Any:
    from astropy.io import fits

    header = fits.Header()
    header["OBJECTID"] = str(saved.object_id)
    header["REDSHIFT"] = _finite_or_nan(saved.redshift)
    header["SURVEY"] = str(saved.survey)
    header["FITDATE"] = str(saved.fitdate)
    header["CODEVER"] = str(saved.code_version)
    header["STRATEGY"] = str(saved.strategy)
    header["STATUS"] = str(saved.status)
    header["QUALITY"] = str(saved.quality)
    header["MC_N"] = int(saved.mc_n or 0)
    header["RANDOMSD"] = -1 if saved.random_seed is None else int(saved.random_seed)
    header["TEMPLATE"] = str(saved.template_info)[:68]
    header["CONFIGHS"] = str(saved.config_hash)
    config_json = _json_dumps(saved.config)
    header["CONFIG"] = config_json[:68]
    if len(config_json) > 68:
        header["CONFLONG"] = True
    return header


def _spectrum_table(saved: SavedFitResult) -> dict[str, np.ndarray]:
    n = len(saved.spectrum["wave"])
    return {
        "wave": np.asarray(saved.spectrum.get("wave"), dtype=float),
        "flux": np.asarray(saved.spectrum.get("flux"), dtype=float),
        "err": np.asarray(saved.spectrum.get("err"), dtype=float),
        "mask_good": np.asarray(saved.spectrum.get("mask_good", np.ones(n, dtype=bool)), dtype=bool),
        "rest_wave": np.asarray(saved.spectrum.get("rest_wave", saved.spectrum["wave"]), dtype=float),
        "original_flux": np.asarray(saved.spectrum.get("original_flux", saved.spectrum["flux"]), dtype=float),
    }


def _model_total_table(saved: SavedFitResult) -> dict[str, np.ndarray]:
    n = len(saved.spectrum["wave"])
    table = {}
    for key in ("wave", "total_model", "residual", "chi", "stellar_model", "powerlaw_model", "feii_model", "local_continuum_model", "emission_total_model"):
        table[key] = _array_or_default(saved.model_total.get(key), np.zeros(n), n)
    table["wave"] = _array_or_default(saved.model_total.get("wave"), saved.spectrum["wave"], n)
    return table


def _components_table(saved: SavedFitResult) -> dict[str, np.ndarray]:
    n = len(saved.spectrum["wave"])
    table = {"wave": np.asarray(saved.spectrum["wave"], dtype=float)}
    used = {"wave"}
    for name in COMPONENT_COLUMNS:
        table[name] = _array_or_default(saved.components.get(name), np.zeros(n), n)
        used.add(name.lower())
    for name, values in sorted(saved.components.items()):
        if name in COMPONENT_COLUMNS:
            continue
        col = _unique_component_column(f"extra_{_safe_column(name)}", used)
        table[col] = _array_or_default(values, np.zeros(n), n)
    return table


def _line_param_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    rows = []
    for row in saved.line_params:
        clean = {key: _default_line_value(key) for key in LINE_PARAM_COLUMNS}
        for key, value in row.items():
            if key in clean:
                clean[key] = _fits_scalar(value)
        rows.append(clean)
    if not rows:
        rows.append({key: _default_line_value(key) for key in LINE_PARAM_COLUMNS})
    return rows


def _continuum_param_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    rows = []
    for row in saved.continuum_params:
        rows.append(
            {
                "component": _fits_text(row.get("component", "")),
                "parameter": _fits_text(row.get("parameter", "")),
                "value": _to_float(row.get("value", np.nan)),
                "err": _to_float(row.get("err", np.nan)),
                "p16": _to_float(row.get("p16", np.nan)),
                "p50": _to_float(row.get("p50", np.nan)),
                "p84": _to_float(row.get("p84", np.nan)),
            }
        )
    if not rows:
        rows.append({"component": "", "parameter": "", "value": np.nan, "err": np.nan, "p16": np.nan, "p50": np.nan, "p84": np.nan})
    return rows


def _stellar_param_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    rows = []
    for row in saved.stellar_params:
        rows.append(
            {
                "stellar_velocity": _to_float(row.get("stellar_velocity", np.nan)),
                "stellar_sigma": _to_float(row.get("stellar_sigma", np.nan)),
                "stellar_chi2": _to_float(row.get("stellar_chi2", np.nan)),
                "stellar_scale": _to_float(row.get("stellar_scale", np.nan)),
                "template_library": _fits_text(row.get("template_library", "")),
                "n_templates": _safe_int(row.get("n_templates", -1), default=-1),
                "ppxf_goodpixels": _safe_int(row.get("ppxf_goodpixels", -1), default=-1),
                "host_fraction_4000": _to_float(row.get("host_fraction_4000", np.nan)),
                "host_fraction_5100": _to_float(row.get("host_fraction_5100", np.nan)),
                "template_index": _safe_int(row.get("template_index", -1), default=-1),
                "weight": _to_float(row.get("weight", np.nan)),
            }
        )
    if not rows:
        rows.append(
            {
                "stellar_velocity": np.nan,
                "stellar_sigma": np.nan,
                "stellar_chi2": np.nan,
                "stellar_scale": np.nan,
                "template_library": "",
                "n_templates": -1,
                "ppxf_goodpixels": -1,
                "host_fraction_4000": np.nan,
                "host_fraction_5100": np.nan,
                "template_index": -1,
                "weight": np.nan,
            }
        )
    return rows


def _mc_line_sample_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    rows = []
    for row in saved.mc_line_samples:
        rows.append(
            {
                "mc_id": _safe_int(row.get("mc_id", -1), default=-1),
                "line_name": _fits_text(row.get("line_name", row.get("name", ""))),
                "flux": _to_float(row.get("flux", np.nan)),
                "velocity": _to_float(row.get("velocity", np.nan)),
                "sigma_kms": _to_float(row.get("sigma_kms", np.nan)),
                "fwhm_kms": _to_float(row.get("fwhm_kms", np.nan)),
            }
        )
    return rows


def _diagnostic_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    diagnostics = dict(saved.diagnostics)
    for flag in DEFAULT_FLAGS:
        diagnostics.setdefault(flag, False)
    if saved.warnings:
        diagnostics.setdefault("warnings", saved.warnings)
    rows = [
        {
            "key": _fits_text(key),
            "value": _fits_text(_json_dumps(value)),
            "description": _fits_text(_diagnostic_description(str(key))),
        }
        for key, value in sorted(diagnostics.items())
    ]
    if not rows:
        rows.append({"key": "", "value": "", "description": ""})
    return rows


def _config_rows(saved: SavedFitResult) -> list[dict[str, Any]]:
    config = dict(saved.config)
    config.setdefault("fit_strategy", saved.strategy)
    config.setdefault("mc_n", saved.mc_n)
    rows = [
        {"key": _fits_text(key), "value": _fits_text(_json_dumps(_metadata_json_value(value)))}
        for key, value in sorted(config.items())
    ]
    if not rows:
        rows.append({"key": "", "value": ""})
    return rows


def _add_window_to_saved_arrays(
    full_wave: np.ndarray,
    components: dict[str, np.ndarray],
    model_total: dict[str, np.ndarray],
    fit: Any,
) -> None:
    model = np.asarray(getattr(fit, "model_flux", []), dtype=float)
    if model.size == 0:
        return
    continuum = _array_or_zero(getattr(fit, "continuum_flux", None), len(model))
    window_wave = _matched_window_wave(full_wave, fit, len(model))
    if window_wave is None:
        return
    slc = _window_indices(full_wave, window_wave)
    if slc is None:
        return

    emission = model - continuum
    _add_window_array(model_total["local_continuum_model"], full_wave, window_wave, continuum, slc)
    _add_window_array(model_total["emission_total_model"], full_wave, window_wave, emission, slc)

    comp_flux = getattr(fit, "component_flux", {}) or {}
    for comp_name, values in comp_flux.items():
        array = np.asarray(values, dtype=float)
        if array.size != len(model):
            continue
        target = _component_target_name(str(comp_name), getattr(fit, "window_name", ""))
        if target == "__skip__":
            continue
        if target not in components:
            components[target] = np.zeros_like(full_wave, dtype=float)
        _add_window_array(components[target], full_wave, window_wave, array, slc)

    _add_fallback_component_from_window(components, full_wave, window_wave, emission, fit, slc)
    _refresh_aggregate_components(components)


def _line_rows_from_fit(window: str, fit: Any) -> list[dict[str, Any]]:
    rows = []
    fluxes = getattr(fit, "line_flux", {}) or {}
    errors = getattr(fit, "line_flux_err", {}) or {}
    mc_summary = getattr(fit, "mc_summary", {}) or {}
    line_mc = mc_summary.get("line_flux", {}) if isinstance(mc_summary, Mapping) else {}
    params = getattr(fit, "params", {}) or {}
    param_err = getattr(fit, "param_err", {}) or {}
    info = getattr(fit, "fit_info", {}) or {}
    for name, flux in sorted(fluxes.items()):
        velocity, velocity_err = _line_velocity(name, params, param_err, info)
        sigma, sigma_err = _line_sigma(name, params, param_err, info)
        fwhm, fwhm_err = _line_fwhm(name, sigma, sigma_err, info)
        flux_err = _to_float(errors.get(name, np.nan))
        snr = _safe_ratio(_to_float(flux), flux_err)
        percentiles = line_mc.get(name, {}) if isinstance(line_mc, Mapping) else {}
        rows.append(
            {
                "name": str(name),
                "flux": _to_float(flux),
                "flux_err": flux_err,
                "flux_p16": _to_float(percentiles.get("p16", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
                "flux_p50": _to_float(percentiles.get("p50", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
                "flux_p84": _to_float(percentiles.get("p84", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
                "luminosity": np.nan,
                "luminosity_err": np.nan,
                "ew": np.nan,
                "ew_err": np.nan,
                "velocity": velocity,
                "velocity_err": velocity_err,
                "sigma_kms": sigma,
                "sigma_kms_err": sigma_err,
                "fwhm_kms": fwhm,
                "fwhm_kms_err": fwhm_err,
                "snr": snr,
                "tied_group": _line_tied_group(name),
                "component_type": _line_component_type(name, window),
                "detected_flag": bool(np.isfinite(snr) and snr >= 3.0 and _to_float(flux) > 0.0),
                "quality_flag": _quality_for_line(name, fit),
            }
        )
    return rows


def _continuum_rows_from_fit(window: str, fit: Any) -> list[dict[str, Any]]:
    rows = []
    params = getattr(fit, "params", {}) or {}
    errors = getattr(fit, "param_err", {}) or {}
    mc_summary = getattr(fit, "mc_summary", {}) or {}
    param_summary = mc_summary.get("params", {}) if isinstance(mc_summary, Mapping) else {}
    info = getattr(fit, "fit_info", {}) or {}
    for name, value in sorted(params.items()):
        if not _is_continuum_or_feii_param(name):
            continue
        percentiles = param_summary.get(name, {}) if isinstance(param_summary, Mapping) else {}
        rows.append(
            {
                "component": _continuum_component_name(name, window),
                "parameter": str(name),
                "value": _to_float(value),
                "err": _to_float(errors.get(name, np.nan)),
                "p16": _to_float(percentiles.get("p16", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
                "p50": _to_float(percentiles.get("p50", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
                "p84": _to_float(percentiles.get("p84", np.nan)) if isinstance(percentiles, Mapping) else np.nan,
            }
        )
    for key, value in sorted(info.items()):
        if _is_continuum_or_feii_param(key) or "fraction" in str(key).lower():
            rows.append(
                {
                    "component": _continuum_component_name(key, window),
                    "parameter": str(key),
                    "value": _to_float(value),
                    "err": np.nan,
                    "p16": np.nan,
                    "p50": np.nan,
                    "p84": np.nan,
                }
            )
    return rows


def _diagnostics_from_fit(window: str, fit: Any) -> dict[str, Any]:
    diagnostics = {
        f"{window}_chi2": _to_float(getattr(fit, "chi2", np.nan)),
        f"{window}_redchi": _to_float(getattr(fit, "redchi", np.nan)),
        f"{window}_bic": _to_float(getattr(fit, "bic", np.nan)),
        f"{window}_success": bool(getattr(fit, "success", False)),
    }
    for key, value in (getattr(fit, "fit_info", {}) or {}).items():
        out_key = str(key)
        if out_key.startswith("flag_"):
            diagnostics[out_key] = bool(value)
        diagnostics[f"{window}_{out_key}"] = _metadata_json_value(value)
    warnings = list(getattr(fit, "warnings", []) or [])
    if warnings:
        diagnostics[f"{window}_warnings"] = warnings
    return diagnostics


def _stellar_rows_from_result(stellar: Any | None) -> list[dict[str, Any]]:
    if stellar is None:
        return []
    diagnostics = getattr(stellar, "diagnostics", {}) or {}
    weights = np.asarray(getattr(stellar, "weights", []), dtype=float)
    n_templates = int(weights.size) if weights.size else int(diagnostics.get("n_templates", -1) or -1)
    goodpixels = np.asarray(getattr(stellar, "goodpixels", []))
    template_library = str(
        diagnostics.get("template_library")
        or diagnostics.get("library")
        or diagnostics.get("ppxf_template_library")
        or ""
    )
    base = {
        "stellar_velocity": _to_float(getattr(stellar, "velocity", np.nan)),
        "stellar_sigma": _to_float(getattr(stellar, "sigma", np.nan)),
        "stellar_chi2": _to_float(getattr(stellar, "chi2", np.nan)),
        "stellar_scale": _to_float(diagnostics.get("stellar_scale", np.nan)),
        "template_library": template_library,
        "n_templates": n_templates,
        "ppxf_goodpixels": int(goodpixels.size),
        "host_fraction_4000": _to_float(_first_existing(diagnostics, ("host_fraction_4000", "stellar_fraction_4000"))),
        "host_fraction_5100": _to_float(_first_existing(diagnostics, ("host_fraction_5100", "stellar_fraction_5100"))),
    }
    rows = []
    if weights.size:
        for idx, weight in enumerate(weights):
            if np.isfinite(weight) and weight != 0.0:
                rows.append({**base, "template_index": int(idx), "weight": float(weight)})
    if not rows:
        rows.append({**base, "template_index": -1, "weight": np.nan})
    return rows


def _mc_rows_from_epoch(epoch: Any) -> list[dict[str, Any]]:
    rows = []
    for attr, _, _ in WINDOW_RESULTS:
        fit = getattr(epoch, attr, None)
        if fit is None:
            continue
        mc_summary = getattr(fit, "mc_summary", {}) or {}
        line_samples = mc_summary.get("line_samples") or mc_summary.get("samples") or {}
        if not isinstance(line_samples, Mapping):
            continue
        for line_name, samples in line_samples.items():
            for idx, sample in enumerate(samples):
                rows.append(
                    {
                        "mc_id": idx,
                        "line_name": str(line_name),
                        "flux": _to_float(sample),
                        "velocity": np.nan,
                        "sigma_kms": np.nan,
                        "fwhm_kms": np.nan,
                    }
                )
    return rows


def _config_from_epoch(epoch: Any) -> dict[str, Any]:
    diagnostics = dict(getattr(epoch, "diagnostics", {}) or {})
    config = dict(diagnostics.get("config", {}) or {}) if isinstance(diagnostics.get("config"), Mapping) else {}
    config.setdefault("fit_strategy", diagnostics.get("fit_strategy", "window_joint"))
    config.setdefault("include_feii", _has_fit(getattr(epoch, "hb_oiii", None)))
    hb = getattr(epoch, "hb_oiii", None)
    ha = getattr(epoch, "ha_nii_sii", None)
    for fit in (hb, ha, getattr(epoch, "blue_balmer", None)):
        if fit is None:
            continue
        for key, value in (getattr(fit, "fit_info", {}) or {}).items():
            if key in {
                "feii_template_method_final",
                "feii_scale_mode",
                "include_broad_balmer",
                "broad_hb_model",
                "broad_ha_model",
                "mc_n",
                "template_library",
            }:
                config.setdefault(key, value)
    return config


def _extract_spectrum(result: Any) -> Any | None:
    for name in ("spectrum", "spec", "input_spectrum", "_spectrum"):
        value = getattr(result, name, None)
        if value is not None:
            return value
    diagnostics = getattr(result, "diagnostics", {}) or {}
    if isinstance(diagnostics, Mapping):
        for key in ("spectrum", "spec", "input_spectrum"):
            if diagnostics.get(key) is not None:
                return diagnostics[key]
    return None


def _spectrum_arrays(spec: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    wave = np.asarray(getattr(spec, "wave", spec.get("wave") if isinstance(spec, Mapping) else []), dtype=float)
    flux = np.asarray(getattr(spec, "flux", spec.get("flux") if isinstance(spec, Mapping) else []), dtype=float)
    err = np.asarray(getattr(spec, "err", spec.get("err") if isinstance(spec, Mapping) else []), dtype=float)
    if wave.size == 0 or flux.size != wave.size or err.size != wave.size:
        raise ValueError("Attached spectrum must provide 1D wave, flux, and err arrays of the same length.")
    z = _to_float(getattr(spec, "z", spec.get("z", 0.0) if isinstance(spec, Mapping) else 0.0))
    mask_good = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
    wave = wave[mask_good]
    flux = flux[mask_good]
    err = err[mask_good]
    rest_wave = wave / (1.0 + z) if np.isfinite(z) and z != 0.0 else wave.copy()
    return rest_wave, flux, err, rest_wave.copy(), flux.copy(), np.ones_like(rest_wave, dtype=bool), z


def _matched_window_wave(full_wave: np.ndarray, fit: Any, n_model: int) -> np.ndarray | None:
    info = getattr(fit, "fit_info", {}) or {}
    candidates: list[tuple[float, float]] = []
    if "blue_balmer_window" in info:
        try:
            candidates.append(tuple(float(v) for v in info["blue_balmer_window"]))
        except Exception:
            pass
    window_name = str(getattr(fit, "window_name", ""))
    candidates.extend(
        {
            "oii_blend": [(3650.0, 3800.0)],
            "oii_doublet": [(3650.0, 3800.0)],
            "blue_balmer": [(3850.0, 4450.0)],
            "blue_balmer_extended": [(3850.0, 4450.0), (3800.0, 5700.0)],
            "hb_oiii": [(4400.0, 5600.0), (4200.0, 5700.0)],
            "ha_nii_sii": [(6250.0, 6800.0), (6400.0, 6800.0)],
        }.get(window_name, [])
    )
    for start, stop in candidates:
        mask = (full_wave >= start) & (full_wave <= stop)
        if np.count_nonzero(mask) == n_model:
            return np.asarray(full_wave[mask], dtype=float)
    return None


def _window_indices(full_wave: np.ndarray, window_wave: np.ndarray) -> np.ndarray | None:
    mask = (full_wave >= np.nanmin(window_wave)) & (full_wave <= np.nanmax(window_wave))
    if np.count_nonzero(mask) == len(window_wave):
        return mask
    return mask if np.count_nonzero(mask) >= 2 else None


def _add_window_array(
    target: np.ndarray,
    full_wave: np.ndarray,
    window_wave: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> None:
    if np.count_nonzero(mask) == len(values):
        target[mask] += values
    else:
        target[mask] += np.interp(full_wave[mask], window_wave, values, left=0.0, right=0.0)


def _component_target_name(name: str, window_name: str) -> str:
    clean = name.strip()
    lookup = {
        "continuum": "local_continuum_model",
        "FeII": "feii_total",
        "FeII_total": "feii_total",
        "FeII_blue": "feii_blue",
        "FeII_central": "feii_central",
        "FeII_red": "feii_red",
        "Hb_narrow": "Hb_narrow",
        "Hb_broad_1": "Hb_broad_1",
        "Hb_broad_2": "Hb_broad_2",
        "Hb_broad": "Hb_broad_total",
        "Hb_broad_total": "Hb_broad_total",
        "Hb_broad_combined": "Hb_broad_total",
        "OIII_4959": "OIII_4959_core",
        "OIII_5007": "OIII_5007_core",
        "OIII_4959_core": "OIII_4959_core",
        "OIII_5007_core": "OIII_5007_core",
        "OIII_4959_wing": "OIII_4959_wing",
        "OIII_5007_wing": "OIII_5007_wing",
        "OIII_core": "__skip__",
        "OIII_wing": "__skip__",
        "Ha_narrow": "Ha_narrow",
        "Ha_broad_1": "Ha_broad_1",
        "Ha_broad_2": "Ha_broad_2",
        "Ha_broad": "Ha_broad_total",
        "Ha_broad_total": "Ha_broad_total",
        "Ha_broad_combined": "Ha_broad_total",
        "OI_6300": "OI_6300",
        "OI_6363": "OI_6363",
        "NII_6548": "NII_6548",
        "NII_6584": "NII_6584",
        "SII_6716": "SII_6717",
        "SII_6717": "SII_6717",
        "SII_6731": "SII_6731",
        "OII_3726": "OII_3726",
        "OII_3729": "OII_3729",
        "OII": "OII_model",
        "OII_model": "OII_model",
        "Hgamma_narrow": "Hgamma_narrow",
        "Hgamma_broad": "Hgamma_broad_total",
        "Hgamma_broad_total": "Hgamma_broad_total",
        "Hdelta_narrow": "Hdelta_narrow",
        "Hdelta_broad": "Hdelta_broad_total",
        "Hdelta_broad_total": "Hdelta_broad_total",
        "NeIII_3869": "NeIII_3869",
        "NeIII_3968": "NeIII_3968",
    }
    if clean in lookup:
        return lookup[clean]
    if "oii" in str(window_name).lower() and clean.lower() != "continuum":
        return "OII_model"
    return f"extra_{_safe_column(clean)}"


def _add_fallback_component_from_window(
    components: dict[str, np.ndarray],
    full_wave: np.ndarray,
    window_wave: np.ndarray,
    emission: np.ndarray,
    fit: Any,
    mask: np.ndarray,
) -> None:
    name = str(getattr(fit, "window_name", "")).lower()
    target = None
    if "oii" in name and not np.any(components["OII_model"]):
        target = "OII_model"
    if target is not None:
        _add_window_array(components[target], full_wave, window_wave, emission, mask)


def _refresh_aggregate_components(components: dict[str, np.ndarray]) -> None:
    hb_parts = _sum_existing(components, ("Hb_broad_1", "Hb_broad_2"))
    if hb_parts.size and np.any(np.abs(hb_parts) > 0):
        components["Hb_broad_total"] = hb_parts
    ha_parts = _sum_existing(components, ("Ha_broad_1", "Ha_broad_2"))
    if ha_parts.size and np.any(np.abs(ha_parts) > 0):
        components["Ha_broad_total"] = ha_parts
    if not np.any(components["feii_total"]):
        components["feii_total"] = _sum_existing(components, ("feii_blue", "feii_central", "feii_red"))
    oii_parts = _sum_existing(components, ("OII_3726", "OII_3729"))
    if oii_parts.size and np.any(np.abs(oii_parts) > 0):
        components["OII_model"] = oii_parts


def _sum_emission_components(components: dict[str, np.ndarray]) -> np.ndarray:
    exclude = {"host_stellar", "galaxy_host", "stellar_model", "powerlaw"}
    total = np.zeros_like(next(iter(components.values())), dtype=float)
    for name, values in components.items():
        lname = name.lower()
        if name in exclude or "continuum" in lname:
            continue
        if name.endswith("_total") and name in {"Hb_broad_total", "Ha_broad_total"}:
            continue
        if name in {"feii_blue", "feii_central", "feii_red"} and np.any(components.get("feii_total", 0.0)):
            continue
        if name == "OII_model" and (
            np.any(components.get("OII_3726", 0.0))
            or np.any(components.get("OII_3729", 0.0))
        ):
            continue
        total += np.asarray(values, dtype=float)
    return total


def _plot_full_overview(saved: SavedFitResult, groups: set[str] | None, xlim: tuple[float, float] | None = None):
    import matplotlib.pyplot as plt

    wave = np.asarray(saved.spectrum["wave"], dtype=float)
    flux = np.asarray(saved.spectrum["flux"], dtype=float)
    mt = saved.model_total
    total_model = _overview_total_model(saved)
    residual = flux - total_model
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(wave, flux, color="0.15", lw=0.8, label="observed")
    axes[0].plot(wave, total_model, color="#d62728", lw=1.1, label="overview model")
    _plot_if_group(axes[0], wave, mt["stellar_model"], "stellar host", "#2ca02c", groups, "stellar")
    _plot_if_group(axes[0], wave, mt["powerlaw_model"], "power-law", "#1f77b4", groups, "stellar", ls="--")
    _plot_if_group(axes[0], wave, mt["feii_model"], "Fe II", "#9467bd", groups, "feii")
    axes[0].set_title(_page_title(saved, "Full spectrum overview"))
    axes[0].set_ylabel("Flux")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(wave, residual, color="0.25", lw=0.75)
    axes[1].axhline(0.0, color="0.8", lw=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Rest wavelength [A]")
    if xlim is not None:
        axes[0].set_xlim(*xlim)
    return fig


def _plot_hb_oiii_page(saved: SavedFitResult, groups: set[str] | None, plot_style: str):
    del plot_style
    return _plot_component_window(
        saved,
        title="Hbeta + [O III] + Fe II",
        xlim=(4200, 5700),
        component_names=(
            "feii_total",
            "Hb_narrow",
            "Hb_broad_1",
            "Hb_broad_2",
            "Hb_broad_total",
            "OIII_4959_core",
            "OIII_5007_core",
            "OIII_4959_wing",
            "OIII_5007_wing",
        ),
        groups=groups,
    )


def _plot_feii_diagnostic_page(saved: SavedFitResult, groups: set[str] | None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), constrained_layout=True)
    for ax, region, label in zip(
        axes,
        ((4434, 4684), (4920, 5050), (5100, 5600)),
        ("blue Fe II 4434-4684", "[O III] pedestal 4920-5050", "red Fe II 5100-5600"),
        strict=True,
    ):
        _plot_window_on_axis(saved, ax, region, ("feii_total", "feii_blue", "feii_central", "feii_red"), groups)
        ax.set_title(label)
    fig.suptitle(_page_title(saved, "Fe II diagnostics"))
    return fig


def _plot_halpha_page(saved: SavedFitResult, groups: set[str] | None, plot_style: str):
    del plot_style
    return _plot_component_window(
        saved,
        title="Halpha + [O I] + [N II] + [S II]",
        xlim=(6250, 6800),
        component_names=(
            "OI_6300",
            "OI_6363",
            "Ha_narrow",
            "Ha_broad_1",
            "Ha_broad_2",
            "Ha_broad_total",
            "NII_6548",
            "NII_6584",
            "SII_6717",
            "SII_6731",
        ),
        groups=groups,
    )


def _plot_oii_page(saved: SavedFitResult, groups: set[str] | None):
    return _plot_component_window(
        saved,
        title="[O II]",
        xlim=(3650, 3800),
        component_names=("OII_3726", "OII_3729", "OII_model"),
        groups=groups,
    )


def _plot_blue_balmer_page(saved: SavedFitResult, groups: set[str] | None):
    fig = _plot_component_window(
        saved,
        title="Blue Balmer",
        xlim=(3850, 4450),
        component_names=(
            "Hgamma_narrow",
            "Hgamma_broad_total",
            "Hdelta_narrow",
            "Hdelta_broad_total",
            "NeIII_3869",
            "NeIII_3968",
        ),
        groups=groups,
    )
    if bool(saved.diagnostics.get("flag_balmer_continuum_region_skipped", False)):
        fig.axes[0].text(
            0.01,
            0.95,
            "Balmer continuum region 3000-3646 A skipped",
            transform=fig.axes[0].transAxes,
            va="top",
            fontsize=9,
            color="0.25",
        )
    return fig


def _plot_component_window(
    saved: SavedFitResult,
    title: str,
    xlim: tuple[float, float],
    component_names: tuple[str, ...],
    groups: set[str] | None,
):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    _plot_window_on_axis(saved, axes[0], xlim, component_names, groups)
    wave = np.asarray(saved.spectrum["wave"], dtype=float)
    flux = np.asarray(saved.spectrum["flux"], dtype=float)
    mt = saved.model_total
    continuum_model = _array_or_default(mt.get("stellar_model"), np.zeros_like(wave), wave.size)
    window_total = _window_total_model(saved)
    residual = flux - continuum_model - window_total
    mask = (wave >= xlim[0]) & (wave <= xlim[1])
    axes[0].set_title(_page_title(saved, title))
    axes[1].plot(wave[mask], residual[mask], color="0.25", lw=0.8)
    axes[1].axhline(0.0, color="0.8", lw=0.8)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Rest wavelength [A]")
    return fig


def _plot_window_on_axis(
    saved: SavedFitResult,
    ax: Any,
    xlim: tuple[float, float],
    component_names: tuple[str, ...],
    groups: set[str] | None,
) -> None:
    wave = np.asarray(saved.spectrum["wave"], dtype=float)
    flux = np.asarray(saved.spectrum["flux"], dtype=float)
    mt = saved.model_total
    continuum_model = _array_or_default(mt.get("stellar_model"), np.zeros_like(wave), wave.size)
    residual_flux = flux - continuum_model
    local_continuum = _array_or_default(mt.get("local_continuum_model"), np.zeros_like(wave), wave.size)
    emission_total = _array_or_default(mt.get("emission_total_model"), np.zeros_like(wave), wave.size)
    window_model = local_continuum + emission_total
    mask = (wave >= xlim[0]) & (wave <= xlim[1])
    ax.plot(wave[mask], residual_flux[mask], color="0.2", lw=0.8, label="stellar-sub residual")
    ax.plot(wave[mask], window_model[mask], color="#d62728", lw=1.1, label="window total")
    ax.plot(wave[mask], local_continuum[mask], color="#1f77b4", ls="--", lw=0.9, label="local continuum")
    colors = _component_colors()
    for name in component_names:
        values = np.asarray(saved.components.get(name, np.zeros_like(wave)), dtype=float)
        if not _show_component(name, groups) or not np.any(np.isfinite(values[mask]) & (np.abs(values[mask]) > 0)):
            continue
        ax.plot(wave[mask], values[mask], lw=0.9, alpha=0.8, label=name, color=colors.get(name))
    ax.set_xlim(*xlim)
    ax.set_ylabel("Flux")
    ax.legend(fontsize=7, ncol=3)


def _overview_total_model(saved: SavedFitResult) -> np.ndarray:
    """Full-spectrum display model without local window continuum boxes."""
    mt = saved.model_total
    wave = np.asarray(saved.spectrum["wave"], dtype=float)
    n = wave.size
    return (
        _array_or_default(mt.get("stellar_model"), np.zeros(n), n)
        + _array_or_default(mt.get("powerlaw_model"), np.zeros(n), n)
        + _array_or_default(mt.get("emission_total_model"), np.zeros(n), n)
    )


def _window_total_model(saved: SavedFitResult) -> np.ndarray:
    """Local-window model, including the per-window continuum used by fits."""
    mt = saved.model_total
    wave = np.asarray(saved.spectrum["wave"], dtype=float)
    n = wave.size
    return (
        _array_or_default(mt.get("local_continuum_model"), np.zeros(n), n)
        + _array_or_default(mt.get("emission_total_model"), np.zeros(n), n)
    )


def _plot_if_group(ax: Any, wave: np.ndarray, values: np.ndarray, label: str, color: str, groups: set[str] | None, group: str, ls: str = "-") -> None:
    if groups is not None and group not in groups and "all" not in groups:
        return
    if np.any(np.isfinite(values) & (np.abs(values) > 0)):
        ax.plot(wave, values, color=color, lw=1.0, ls=ls, label=label)


def _show_component(name: str, groups: set[str] | None) -> bool:
    if groups is None or "all" in groups:
        return True
    lname = name.lower()
    if "feii" in lname and "feii" in groups:
        return True
    if "broad" in lname and "broad" in groups:
        return True
    if (
        "narrow" in lname
        or "core" in lname
        or "nii" in lname
        or "sii" in lname
        or "oi_" in lname
        or "neiii" in lname
        or "oii" in lname
    ) and "narrow" in groups:
        return True
    if ("stellar" in lname or "powerlaw" in lname or "continuum" in lname) and "stellar" in groups:
        return True
    return False


def _page_title(saved: SavedFitResult, page: str) -> str:
    flags = [key for key, value in sorted(saved.diagnostics.items()) if key.startswith("flag_") and bool(value)]
    flags_text = ",".join(flags[:4]) if flags else "no major flags"
    chi2 = _first_existing(saved.diagnostics, ("chi2", "total_chi2", "hb_oiii_chi2"))
    redchi = _first_existing(saved.diagnostics, ("redchi", "total_redchi", "hb_oiii_redchi"))
    bic = _first_existing(saved.diagnostics, ("bic", "total_bic", "hb_oiii_bic"))
    return (
        f"{saved.object_id} | {page} | quality={saved.quality or 'unknown'} | "
        f"z={_fmt(saved.redshift)} | chi2/redchi/BIC={_fmt(chi2)}/{_fmt(redchi)}/{_fmt(bic)} | "
        f"MC_N={saved.mc_n} | {flags_text}"
    )


def _has_any_component(saved: SavedFitResult, names: tuple[str, ...]) -> bool:
    return any(np.any(np.abs(np.asarray(saved.components.get(name, []), dtype=float)) > 0) for name in names)


def _parse_component_filter(show_components: str | list[str] | tuple[str, ...] | None) -> set[str] | None:
    if show_components is None:
        return None
    if isinstance(show_components, str):
        values = [item.strip().lower() for item in show_components.split(",") if item.strip()]
    else:
        values = [str(item).strip().lower() for item in show_components if str(item).strip()]
    return set(values) if values else None


def _component_colors() -> dict[str, str]:
    return {
        "feii_total": "#9467bd",
        "feii_blue": "#9467bd",
        "feii_central": "#8c6bb1",
        "feii_red": "#756bb1",
        "Hb_narrow": "#ff7f0e",
        "Hb_broad_1": "#bcbd22",
        "Hb_broad_2": "#17becf",
        "Hb_broad_total": "#e377c2",
        "OIII_4959_core": "#2ca02c",
        "OIII_5007_core": "#2ca02c",
        "OIII_4959_wing": "#98df8a",
        "OIII_5007_wing": "#98df8a",
        "Ha_narrow": "#ff7f0e",
        "Ha_broad_1": "#bcbd22",
        "Ha_broad_2": "#17becf",
        "Ha_broad_total": "#e377c2",
        "OI_6300": "#8c564b",
        "OI_6363": "#8c564b",
        "NII_6548": "#2ca02c",
        "NII_6584": "#2ca02c",
        "SII_6717": "#1f77b4",
        "SII_6731": "#1f77b4",
        "OII_3726": "#9467bd",
        "OII_3729": "#c5b0d5",
        "OII_model": "#9467bd",
    }


def _table_to_dict(data: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in data.names:
        values = data[name]
        key = str(name)
        if getattr(values, "dtype", None) is not None and values.dtype.kind in {"S", "U", "O"}:
            result[key] = np.asarray([_decode_string(item) for item in values])
        else:
            result[key] = np.asarray(values)
    return result


def _table_to_rows(data: Any) -> list[dict[str, Any]]:
    rows = []
    names = list(data.names)
    for idx in range(len(data)):
        rows.append({name: _decode_scalar(data[name][idx]) for name in names})
    return rows


def _key_value_table_to_dict(data: Any) -> dict[str, Any]:
    result = {}
    if data is None or "key" not in data.names or "value" not in data.names:
        return result
    for row in data:
        key = _decode_string(row["key"])
        value = _decode_string(row["value"])
        if key == "":
            continue
        try:
            result[key] = json.loads(value)
        except Exception:
            result[key] = value
    return result


def _zero_components(n: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(n, dtype=float) for name in COMPONENT_COLUMNS}


def _array_or_zero(values: Any, n: int) -> np.ndarray:
    return _array_or_default(values, np.zeros(n), n)


def _array_or_default(values: Any, default: np.ndarray, n: int) -> np.ndarray:
    if values is None:
        return np.asarray(default, dtype=float).copy()
    array = np.asarray(values, dtype=float)
    if array.size != n:
        return np.asarray(default, dtype=float).copy()
    return array.copy()


def _sum_existing(components: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    arrays = [np.asarray(components.get(name, 0.0), dtype=float) for name in names if name in components]
    if not arrays:
        return np.array([], dtype=float)
    total = np.zeros_like(arrays[0], dtype=float)
    for array in arrays:
        total += array
    return total


def _line_velocity(name: str, params: Mapping[str, Any], errors: Mapping[str, Any], info: Mapping[str, Any]) -> tuple[float, float]:
    keys = _line_param_keys(name, "velocity")
    value = _first_existing(params, keys)
    err = _first_existing(errors, keys)
    if value is None:
        value = _first_existing(info, tuple(f"{key}" for key in keys))
    return _to_float(value), _to_float(err)


def _line_sigma(name: str, params: Mapping[str, Any], errors: Mapping[str, Any], info: Mapping[str, Any]) -> tuple[float, float]:
    keys = _line_param_keys(name, "sigma")
    value = _first_existing(params, keys)
    err = _first_existing(errors, keys)
    if value is None:
        value = _first_existing(info, keys)
    return _to_float(value), _to_float(err)


def _line_fwhm(name: str, sigma: float, sigma_err: float, info: Mapping[str, Any]) -> tuple[float, float]:
    lname = name.lower()
    if "hb_broad" in lname:
        value = _first_existing(info, ("Hb_broad_fwhm_combined", "broad_hb_combined_fwhm"))
    elif "ha_broad" in lname:
        value = _first_existing(info, ("Ha_broad_fwhm_combined", "broad_ha_combined_fwhm"))
    else:
        value = None
    fwhm = _to_float(value)
    if not np.isfinite(fwhm):
        fwhm = 2.3548 * sigma if np.isfinite(sigma) else np.nan
    fwhm_err = 2.3548 * sigma_err if np.isfinite(sigma_err) else np.nan
    return fwhm, fwhm_err


def _line_param_keys(name: str, suffix: str) -> tuple[str, ...]:
    lname = name.lower()
    keys = []
    if "hb_broad_1" in lname:
        keys.extend((f"hb_broad_1_{suffix}", f"broad1_{suffix}", f"broad_hb_1_{suffix}"))
    elif "hb_broad_2" in lname:
        keys.extend((f"hb_broad_2_{suffix}", f"broad2_{suffix}", f"broad_hb_2_{suffix}"))
    elif "hb_broad" in lname:
        keys.extend((f"hb_broad_1_{suffix}", f"broad_hb_{suffix}", f"broad1_{suffix}"))
    elif "ha_broad_1" in lname:
        keys.extend((f"ha_broad_1_{suffix}", f"broad1_{suffix}", f"broad_ha_1_{suffix}"))
    elif "ha_broad_2" in lname:
        keys.extend((f"ha_broad_2_{suffix}", f"broad2_{suffix}", f"broad_ha_2_{suffix}"))
    elif "ha_broad" in lname:
        keys.extend((f"ha_broad_1_{suffix}", f"broad_ha_{suffix}", f"broad1_{suffix}"))
    elif "feii" in lname:
        keys.extend((f"feii_shift_kms" if suffix == "velocity" else f"feii_extra_sigma_kms", f"feii_{suffix}"))
    elif any(token in lname for token in ("oiii", "nii", "sii", "oi_", "narrow", "neiii", "oii")):
        keys.extend((f"narrow_{suffix}", f"core_{suffix}", f"line_{suffix}"))
    return tuple(keys)


def _line_tied_group(name: str) -> str:
    lname = name.lower()
    if "broad" in lname:
        return "broad_balmer"
    if any(token in lname for token in ("oiii", "nii", "sii", "oi_", "narrow", "neiii", "oii")):
        return "narrow"
    if "feii" in lname:
        return "feii"
    return ""


def _line_component_type(name: str, window: str) -> str:
    lname = name.lower()
    if "feii" in lname:
        return "feii"
    if "wing" in lname:
        return "wing"
    if "broad" in lname:
        return "broad"
    if any(token in lname for token in ("oiii", "nii", "sii", "oi_", "oii", "neiii")):
        return "forbidden"
    if "narrow" in lname:
        return "narrow"
    return str(window)


def _quality_for_line(name: str, fit: Any) -> str:
    flags = []
    for key, value in (getattr(fit, "fit_info", {}) or {}).items():
        if str(key).startswith("flag_") and bool(value):
            flags.append(str(key))
    return ",".join(flags)


def _is_continuum_or_feii_param(name: str) -> bool:
    lname = str(name).lower()
    return any(token in lname for token in ("continuum", "slope", "intercept", "powerlaw", "power_law", "stellar", "host", "feii", "fraction"))


def _continuum_component_name(name: str, window: str) -> str:
    lname = str(name).lower()
    if "feii" in lname:
        return "feii"
    if "power" in lname:
        return "powerlaw"
    if "stellar" in lname or "host" in lname:
        return "stellar"
    return str(window)


def _warnings_from_epoch(epoch: Any) -> list[str]:
    warnings = []
    for attr, _, _ in WINDOW_RESULTS:
        fit = getattr(epoch, attr, None)
        if fit is not None:
            warnings.extend(str(item) for item in (getattr(fit, "warnings", []) or []))
    diagnostics = getattr(epoch, "diagnostics", {}) or {}
    if isinstance(diagnostics, Mapping) and diagnostics.get("warnings"):
        warnings.extend(str(item) for item in diagnostics.get("warnings"))
    return warnings


def _collect_warnings(diagnostics: Mapping[str, Any]) -> list[str]:
    warnings = diagnostics.get("warnings", [])
    if isinstance(warnings, str):
        return [warnings]
    if isinstance(warnings, list):
        return [str(item) for item in warnings]
    return []


def _template_info(stellar: Any | None) -> str:
    if stellar is None:
        return ""
    diagnostics = getattr(stellar, "diagnostics", {}) or {}
    return str(
        diagnostics.get("template_library")
        or diagnostics.get("ppxf_template_library")
        or diagnostics.get("library")
        or ""
    )


def _has_fit(fit: Any | None) -> bool:
    return fit is not None and bool(getattr(fit, "success", True))


def _first_existing(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _first_finite_config(config: Mapping[str, Any], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        value = _to_float(config.get(key, np.nan))
        if np.isfinite(value):
            return value
    return default


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
        return np.nan
    return float(num / den)


def _default_line_value(key: str) -> Any:
    if key in {"name", "tied_group", "component_type", "quality_flag"}:
        return ""
    if key == "detected_flag":
        return False
    return np.nan


def _fits_scalar(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return _fits_text(value)


def _fits_text(value: Any) -> str:
    """Return an ASCII-safe string for FITS table text columns."""
    if value is None:
        return ""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def _finite_or_nan(value: Any) -> float:
    numeric = _to_float(value)
    return numeric if np.isfinite(numeric) else np.nan


def _to_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _optional_int(value: Any) -> int | None:
    numeric = _to_float(value)
    if np.isfinite(numeric) and numeric >= 0:
        return int(numeric)
    return None


def _safe_int(value: Any, default: int = -1) -> int:
    numeric = _to_float(value)
    if np.isfinite(numeric):
        return int(numeric)
    return int(default)


def _header_optional_int(value: Any) -> int | None:
    out = _optional_int(value)
    return None if out == -1 else out


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=True, sort_keys=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _metadata_json_value(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _metadata_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _metadata_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and any(isinstance(item, np.ndarray) for item in value):
        return [_metadata_json_value(item) for item in value]
    return _json_safe(value)


def _config_hash(config: Mapping[str, Any]) -> str:
    payload = _json_dumps(config)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _code_version() -> str:
    try:
        import echospec

        return str(getattr(echospec, "__version__", ""))
    except Exception:
        return ""


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _safe_column(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if not clean:
        clean = "component"
    if clean[0].isdigit():
        clean = f"v_{clean}"
    return clean[:48]


def _unique_component_column(name: str, used: set[str]) -> str:
    base = name[:48]
    candidate = base
    idx = 1
    while candidate.lower() in used:
        suffix = f"_{idx}"
        candidate = f"{base[:48 - len(suffix)]}{suffix}"
        idx += 1
    used.add(candidate.lower())
    return candidate


def _diagnostic_description(key: str) -> str:
    descriptions = {
        "chi2": "Global chi-square if available.",
        "redchi": "Global reduced chi-square if available.",
        "bic": "Global Bayesian information criterion if available.",
        "fit_quality": "EchoSpec fit-quality label.",
        "flag_feii_shape_mismatch": "Fe II single-template shape mismatch flag.",
        "flag_feii_boundary": "Fe II parameter reached or approached fitting boundary.",
        "flag_narrow_hbeta_too_broad": "Narrow Hbeta is broader than expected.",
        "flag_ha_overfit": "Halpha fit may be overfit.",
        "flag_oii_overfit": "[O II] fit may be overfit.",
        "flag_balmer_continuum_region_skipped": "3000-3646 A Balmer continuum region skipped.",
    }
    return descriptions.get(key, "")


def _fmt(value: Any) -> str:
    numeric = _to_float(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.4g}"


def _agn_read_fits_payload(path: Path, rest_frame: bool = True) -> dict[str, Any]:
    from astropy.io import fits

    required = ("SPECTRUM", "MODEL_TOTAL", "COMPONENTS", "LINE_PARAMS")
    payload: dict[str, Any] = {
        "path": path,
        "header": {},
        "spectrum": {},
        "model_total": {},
        "components": {},
        "line_params": [],
        "diagnostics": {},
        "missing_hdus": [],
        "fatal_missing": False,
        "rest_frame": rest_frame,
    }
    with fits.open(path) as hdul:
        payload["header"] = dict(hdul[0].header)
        names = {hdu.name for hdu in hdul}
        payload["missing_hdus"] = [name for name in required if name not in names]
        payload["fatal_missing"] = bool(payload["missing_hdus"])
        if "SPECTRUM" in names:
            payload["spectrum"] = _agn_table_dict(hdul["SPECTRUM"].data)
        if "MODEL_TOTAL" in names:
            payload["model_total"] = _agn_table_dict(hdul["MODEL_TOTAL"].data)
        if "COMPONENTS" in names:
            payload["components"] = _agn_table_dict(hdul["COMPONENTS"].data)
        if "LINE_PARAMS" in names:
            payload["line_params"] = _agn_table_rows(hdul["LINE_PARAMS"].data)
        if "DIAGNOSTICS" in names:
            payload["diagnostics"] = _key_value_table_to_dict(hdul["DIAGNOSTICS"].data)

    wave = _agn_choose_wave(payload, rest_frame=rest_frame)
    payload["plot_wave"] = wave
    payload["wave_label"] = "Rest-frame wavelength (Angstrom)" if rest_frame else "Observed wavelength (Angstrom)"
    payload["emission_components"] = _agn_emission_component_names(payload["components"])
    payload["plot_components"], fallback_warnings = _agn_plot_component_arrays(payload)
    payload["fallback_warnings"] = fallback_warnings
    return payload


def _agn_table_dict(data: Any) -> dict[str, np.ndarray]:
    if data is None:
        return {}
    return {name: np.asarray(data[name]).copy() for name in data.names}


def _agn_table_rows(data: Any) -> list[dict[str, Any]]:
    if data is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in data:
        rows.append({name: _decode_scalar(row[name]) for name in data.names})
    return rows


def _agn_choose_wave(payload: dict[str, Any], rest_frame: bool = True) -> np.ndarray:
    spec = payload.get("spectrum", {})
    model = payload.get("model_total", {})
    wave = np.asarray(spec.get("wave", model.get("wave", [])), dtype=float)
    if not rest_frame:
        return wave
    rest_wave = np.asarray(spec.get("rest_wave", []), dtype=float)
    if rest_wave.size == wave.size and np.isfinite(rest_wave).any():
        return rest_wave
    z = _to_float(payload.get("header", {}).get("REDSHIFT", np.nan))
    if np.isfinite(z):
        return wave / (1.0 + z)
    return wave


def _agn_default_pdf_name(payload: dict[str, Any], path: Path) -> str:
    header = payload.get("header", {})
    obj = str(header.get("OBJECTID", path.stem.replace("_result", ""))) or "unknown"
    survey = str(header.get("SURVEY", "")) or "unknown"
    z = _to_float(header.get("REDSHIFT", np.nan))
    z_text = f"{z:.4f}" if np.isfinite(z) else "nan"
    return f"{_agn_safe_filename(obj)}_{_agn_safe_filename(survey)}_z{z_text}_fit.pdf"


def _agn_safe_filename(value: str) -> str:
    return re.sub(r"[\\/\\s]+", "_", str(value)).strip("_") or "unknown"


def _agn_emission_component_names(components: dict[str, np.ndarray]) -> list[str]:
    names = []
    for name in components:
        lower = name.lower()
        if lower in {item.lower() for item in AGN_AUXILIARY_COMPONENT_COLUMNS}:
            continue
        if "local_continuum" in lower:
            continue
        names.append(name)
    return names


def _agn_nonzero(values: np.ndarray, min_component_peak: float | None = None) -> bool:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return False
    peak = np.nanmax(np.abs(arr[finite]))
    threshold = 0.0 if min_component_peak is None else float(min_component_peak)
    return bool(np.isfinite(peak) and peak > max(threshold, 1.0e-12))


def _agn_plot_component_arrays(payload: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    components = payload.get("components", {})
    wave = np.asarray(payload.get("plot_wave", []), dtype=float)
    plot_components: dict[str, np.ndarray] = {}
    warnings: list[str] = []

    for name in payload.get("emission_components", []):
        arr = np.asarray(components.get(name, []), dtype=float)
        if arr.size == wave.size and _agn_nonzero(arr):
            plot_components[name] = arr

    component_names = set(components)
    for row in payload.get("line_params", []):
        raw_name = str(row.get("name", "")).strip()
        target = _agn_component_name_from_line_param(raw_name)
        if not target:
            continue
        existing = np.asarray(components.get(target, []), dtype=float)
        existing_ok = existing.size == wave.size and _agn_nonzero(existing)
        if existing_ok or target in plot_components:
            continue
        model = _agn_gaussian_from_line_param(wave, row)
        if _agn_nonzero(model):
            plot_components[target] = model
            if target in component_names:
                warnings.append(
                    f"COMPONENTS column {target!r} is empty; plotted Gaussian fallback from LINE_PARAMS {raw_name!r}."
                )
            else:
                warnings.append(
                    f"LINE_PARAMS {raw_name!r} has no COMPONENTS column; plotted Gaussian fallback as {target!r}."
                )
    return plot_components, warnings


def _agn_component_name_from_line_param(name: str) -> str:
    if not name or name.endswith("_flux") or name.endswith("_flux_total"):
        return ""
    mapping = {
        "OIII_4959": "OIII_4959_core",
        "OIII_5007": "OIII_5007_core",
        "SII_6716": "SII_6717",
        "Hgamma_broad": "Hgamma_broad_total",
        "Hdelta_broad": "Hdelta_broad_total",
        "Ha_broad": "Ha_broad_total",
        "Hb_broad": "Hb_broad_total",
        "Hepsilon_narrow": "extra_extra_Hepsilon_narrow",
    }
    return mapping.get(name, name)


def _agn_rest_wave_for_name(name: str) -> float:
    clean = _agn_component_name_from_line_param(name) or name
    rest = {
        "OII_3726": 3726.03,
        "OII_3729": 3728.82,
        "OII_model": 3727.50,
        "NeIII_3869": 3868.76,
        "NeIII_3968": 3967.79,
        "extra_extra_Hepsilon_narrow": 3970.07,
        "Hepsilon_narrow": 3970.07,
        "Hdelta_narrow": 4101.74,
        "Hdelta_broad_total": 4101.74,
        "Hgamma_narrow": 4340.47,
        "Hgamma_broad_total": 4340.47,
        "Hb_narrow": 4861.33,
        "Hb_broad_1": 4861.33,
        "Hb_broad_2": 4861.33,
        "Hb_broad_total": 4861.33,
        "OIII_4959_core": 4958.91,
        "OIII_5007_core": 5006.84,
        "OIII_4959_wing": 4958.91,
        "OIII_5007_wing": 5006.84,
        "OI_6300": 6300.30,
        "OI_6363": 6363.78,
        "Ha_narrow": 6562.80,
        "Ha_broad_1": 6562.80,
        "Ha_broad_2": 6562.80,
        "Ha_broad_total": 6562.80,
        "NII_6548": 6548.05,
        "NII_6584": 6583.45,
        "SII_6717": 6716.44,
        "SII_6716": 6716.44,
        "SII_6731": 6730.82,
    }
    return rest.get(clean, np.nan)


def _agn_gaussian_from_line_param(wave: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    name = str(row.get("name", "")).strip()
    rest = _agn_rest_wave_for_name(name)
    flux = _to_float(row.get("flux", np.nan))
    velocity = _to_float(row.get("velocity", 0.0))
    sigma_kms = _to_float(row.get("sigma_kms", np.nan))
    if not np.isfinite(rest) or not np.isfinite(flux) or flux <= 0.0:
        return np.zeros_like(wave, dtype=float)
    if not np.isfinite(sigma_kms) or sigma_kms <= 0.0:
        fwhm = _to_float(row.get("fwhm_kms", np.nan))
        sigma_kms = fwhm / 2.3548 if np.isfinite(fwhm) and fwhm > 0.0 else np.nan
    if not np.isfinite(sigma_kms) or sigma_kms <= 0.0:
        return np.zeros_like(wave, dtype=float)
    if not np.isfinite(velocity):
        velocity = 0.0
    center = rest * (1.0 + velocity / C_KMS)
    sigma_a = max(center * sigma_kms / C_KMS, 1.0e-6)
    return flux / (np.sqrt(2.0 * np.pi) * sigma_a) * np.exp(-0.5 * ((wave - center) / sigma_a) ** 2)


def _agn_sanity_checks(payload: dict[str, Any]) -> list[str]:
    warnings = list(payload.get("fallback_warnings", []))
    missing = payload.get("missing_hdus", [])
    if missing:
        warnings.append(f"Missing required HDU(s): {', '.join(missing)}")
        return warnings

    spec = payload.get("spectrum", {})
    err = np.asarray(spec.get("err", []), dtype=float)
    if err.size:
        bad = np.count_nonzero(~np.isfinite(err))
        nonpos = np.count_nonzero(np.isfinite(err) & (err <= 0.0))
        if bad:
            warnings.append(f"SPECTRUM err contains {bad} non-finite values.")
        if nonpos:
            warnings.append(f"SPECTRUM err contains {nonpos} non-positive values.")

    component_keys = {_agn_normalize_name(name) for name in payload.get("emission_components", [])}
    line_keys = set()
    for row in payload.get("line_params", []):
        raw = str(row.get("name", "")).strip()
        if raw.endswith("_flux") or raw.endswith("_flux_total"):
            continue
        key = _agn_normalize_name(_agn_component_name_from_line_param(raw) or raw)
        line_keys.add(key)
        if key and key not in component_keys:
            warnings.append(f"LINE_PARAMS row {raw!r} has no matching non-auxiliary COMPONENTS column.")

    for name in payload.get("emission_components", []):
        key = _agn_normalize_name(name)
        if key and key not in line_keys:
            warnings.append(f"COMPONENTS emission column {name!r} has no matching LINE_PARAMS row.")

    components = payload.get("components", {})
    model_total = payload.get("model_total", {})
    wave = np.asarray(payload.get("plot_wave", []), dtype=float)
    emission_model = np.asarray(model_total.get("emission_total_model", []), dtype=float)
    if emission_model.size == wave.size and wave.size:
        emission_sum = _agn_sum_emission_components_from_columns(components, payload.get("emission_components", []))
        finite = np.isfinite(emission_model) & np.isfinite(emission_sum)
        if finite.any():
            diff = emission_model[finite] - emission_sum[finite]
            scale = np.nanpercentile(np.abs(emission_model[finite]), 95)
            diff95 = np.nanpercentile(np.abs(diff), 95)
            if np.isfinite(diff95) and diff95 > max(1.0e-6, 0.05 * max(scale, 1.0e-12)):
                warnings.append("emission_total_model differs from sum of individual emission components.")
    return warnings


def _agn_normalize_name(name: str) -> str:
    clean = str(name).strip()
    clean = clean.replace("_core", "")
    clean = clean.replace("SII_6717", "SII_6716")
    clean = clean.replace("extra_extra_Hepsilon_narrow", "Hepsilon_narrow")
    return clean


def _agn_sum_emission_components_from_columns(
    components: dict[str, np.ndarray],
    emission_names: list[str],
) -> np.ndarray:
    wave = np.asarray(components.get("wave", []), dtype=float)
    if wave.size == 0:
        for values in components.values():
            arr = np.asarray(values)
            if arr.ndim == 1:
                wave = np.zeros(arr.size, dtype=float)
                break
    total = np.zeros_like(wave, dtype=float)
    for name in emission_names:
        if name in AGN_EMISSION_SUM_EXCLUDE:
            continue
        arr = np.asarray(components.get(name, []), dtype=float)
        if arr.size != total.size or not _agn_nonzero(arr):
            continue
        total += np.nan_to_num(arr, nan=0.0)
    return total


def _agn_write_log(log_path: Path, fits_path: Path, payload: dict[str, Any], warnings: list[str]) -> None:
    header = payload.get("header", {})
    lines = [
        f"fits_path: {fits_path}",
        f"object_id: {header.get('OBJECTID', '')}",
        f"survey: {header.get('SURVEY', '')}",
        f"redshift: {header.get('REDSHIFT', np.nan)}",
        f"n_emission_components: {len(payload.get('emission_components', []))}",
        "warnings:",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    log_path.write_text("\n".join(lines) + "\n")


def _agn_warning_only_page(payload: dict[str, Any], warnings: list[str]) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    header = payload.get("header", {})
    title = f"{header.get('OBJECTID', '')} {header.get('SURVEY', '')} FITS warnings"
    ax.text(0.02, 0.96, title, va="top", ha="left", fontsize=16, weight="bold")
    text = "\n".join(warnings) if warnings else "No warnings."
    ax.text(0.02, 0.88, text, va="top", ha="left", fontsize=10, wrap=True)
    return fig


def _agn_page_full_overview(
    payload: dict[str, Any],
    warnings: list[str],
    annotate_all_lines: bool = True,
) -> Any:
    import matplotlib.pyplot as plt

    wave = np.asarray(payload["plot_wave"], dtype=float)
    spec = payload["spectrum"]
    model = payload["model_total"]
    flux = np.asarray(spec.get("flux", np.full_like(wave, np.nan)), dtype=float)
    err = np.asarray(spec.get("err", np.full_like(wave, np.nan)), dtype=float)
    total = np.asarray(model.get("total_model", np.full_like(wave, np.nan)), dtype=float)
    residual = np.asarray(model.get("residual", flux - total), dtype=float)
    chi = np.asarray(model.get("chi", np.divide(residual, err, out=np.full_like(residual, np.nan), where=err > 0)), dtype=float)
    header = payload.get("header", {})

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12.5, 10.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0, 1.0, 0.9], "hspace": 0.08},
    )
    ax = axes[0]
    ax.plot(wave, flux, color="0.25", lw=0.55, label="data flux")
    if err.size == wave.size and np.isfinite(err).any():
        ax.fill_between(wave, flux - err, flux + err, color="0.6", alpha=0.18, lw=0, label="error")
    _agn_plot_model_line(ax, wave, total, "total_model", "red", lw=1.2)
    _agn_plot_model_line(ax, wave, model.get("stellar_model"), "stellar_model", "#ff8c00", ls="-.")
    _agn_plot_model_line(ax, wave, model.get("powerlaw_model"), "powerlaw_model", "#2ca02c", ls="--")
    _agn_plot_model_line(ax, wave, model.get("feii_model"), "feii_model", "#9467bd", ls=":")
    _agn_plot_model_line(ax, wave, model.get("local_continuum_model"), "local_continuum_model", "0.45", ls=":")
    _agn_plot_model_line(ax, wave, model.get("emission_total_model"), "emission_total_model", "#1f77b4", ls="-")
    if annotate_all_lines:
        _agn_annotate_lines(ax, wave)
    ax.set_ylabel("Flux")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    _agn_scale_y_to_data(ax, flux, total)
    _agn_style_axes(ax)

    axes[1].plot(wave, residual, color="0.15", lw=0.6)
    axes[1].axhline(0.0, color="0.6", lw=0.7)
    axes[1].set_ylabel("Residual")
    _agn_style_axes(axes[1])

    axes[2].plot(wave, chi, color="#6a3d9a", lw=0.55)
    axes[2].axhline(0.0, color="0.6", lw=0.7)
    axes[2].axhline(3.0, color="0.8", lw=0.5, ls="--")
    axes[2].axhline(-3.0, color="0.8", lw=0.5, ls="--")
    axes[2].set_ylabel("Chi")
    _agn_style_axes(axes[2])

    axes[3].axis("off")
    warning_text = "\n".join(warnings[:8]) if warnings else "No sanity-check warnings."
    if len(warnings) > 8:
        warning_text += f"\n... {len(warnings) - 8} more warning(s); see log."
    axes[3].text(0.01, 0.95, "Warnings:", va="top", ha="left", fontsize=10, weight="bold")
    axes[3].text(0.13, 0.95, warning_text, va="top", ha="left", fontsize=8, wrap=True)
    axes[2].set_xlabel(payload["wave_label"])
    fig.suptitle(
        f"{header.get('OBJECTID', '')}  {header.get('SURVEY', '')}  z={_to_float(header.get('REDSHIFT', np.nan)):.4f}",
        fontsize=14,
    )
    return fig


def _agn_page_window(
    payload: dict[str, Any],
    title: str,
    xlim: tuple[float, float],
    annotate_all_lines: bool = True,
    include_feii: bool = False,
    min_component_peak: float | None = None,
    broad_first: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    wave = np.asarray(payload["plot_wave"], dtype=float)
    spec = payload["spectrum"]
    model = payload["model_total"]
    flux = np.asarray(spec.get("flux", np.full_like(wave, np.nan)), dtype=float)
    total = np.asarray(model.get("total_model", np.full_like(wave, np.nan)), dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.0), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.4], "hspace": 0.05})
    mask = (wave >= xlim[0]) & (wave <= xlim[1])
    axes[0].plot(wave[mask], flux[mask], color="0.25", lw=0.75, label="data")
    axes[0].plot(wave[mask], total[mask], color="red", lw=1.2, label="total_model")
    _agn_plot_model_line(axes[0], wave, model.get("stellar_model"), "stellar_model", "#ff8c00", ls="-.")
    _agn_plot_model_line(axes[0], wave, model.get("powerlaw_model"), "powerlaw_model", "#2ca02c", ls="--")
    if include_feii:
        for name in ("feii_total", "feii_blue", "feii_central", "feii_red"):
            _agn_plot_component_window(axes[0], wave, payload["components"].get(name), xlim, name, alpha=0.85, ls=":")
    if annotate_all_lines:
        _agn_annotate_lines(axes[0], wave, xlim=xlim)
    axes[0].set_xlim(*xlim)
    axes[0].set_ylabel("Flux")

    component_items = _agn_components_in_window(payload, xlim, min_component_peak=min_component_peak)
    if broad_first:
        component_items = sorted(component_items, key=lambda item: ("broad" not in item[0], item[0]))
    else:
        component_items = sorted(component_items, key=lambda item: _agn_rest_wave_for_name(item[0]) if np.isfinite(_agn_rest_wave_for_name(item[0])) else 1e9)
    emission_total = np.asarray(model.get("emission_total_model", np.zeros_like(wave)), dtype=float)
    if emission_total.size != wave.size:
        emission_total = np.zeros_like(wave)
    baseline = total - np.nan_to_num(emission_total, nan=0.0)
    for name, values in component_items:
        if not name.endswith("_broad_total"):
            continue
        arr = np.asarray(values, dtype=float)
        if arr.size != wave.size:
            continue
        masked = _agn_mask_component(arr)
        if masked is None:
            continue
        valid = np.isfinite(masked) & mask
        if np.count_nonzero(valid) < 2:
            continue
        axes[0].plot(
            wave[valid],
            baseline[valid] + masked[valid],
            color=_agn_component_color(abs(hash(name)) % 20),
            lw=1.2,
            ls="--",
            alpha=0.85,
            label=f"{name}+baseline",
        )
    axes[0].legend(loc="upper left", fontsize=8, ncol=3)
    _agn_scale_y_to_data(axes[0], flux[mask], total[mask])
    _agn_style_axes(axes[0])
    for name, values in component_items:
        _agn_plot_component_window(
            axes[1],
            wave,
            values,
            xlim,
            name,
            alpha=0.45 if name.endswith("_broad_total") else 0.92,
            lw=1.8 if name.endswith("_broad_total") else 1.1,
            ls="--" if name.endswith("_broad_total") else "-",
        )
    if include_feii:
        for name in ("feii_total", "feii_blue", "feii_central", "feii_red"):
            _agn_plot_component_window(axes[1], wave, payload["components"].get(name), xlim, name, alpha=0.75, ls=":")
    axes[1].axhline(0.0, color="0.6", lw=0.7)
    axes[1].set_ylabel("Component flux")
    axes[1].set_xlabel(payload["wave_label"])
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, ncol=1)
    _agn_style_axes(axes[1])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.82, 0.96))
    return fig


def _agn_page_all_components(payload: dict[str, Any], min_component_peak: float | None = None) -> Any:
    import matplotlib.pyplot as plt

    wave = np.asarray(payload["plot_wave"], dtype=float)
    components = [
        (name, values)
        for name, values in payload.get("plot_components", {}).items()
        if _agn_nonzero(values, min_component_peak=min_component_peak)
    ]
    components.sort(key=lambda item: _agn_rest_wave_for_name(item[0]) if np.isfinite(_agn_rest_wave_for_name(item[0])) else 1e9)

    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.5), sharex=True, gridspec_kw={"height_ratios": [2.1, 2.4], "hspace": 0.08})
    total = np.zeros_like(wave, dtype=float)
    support = np.zeros_like(wave, dtype=bool)
    for idx, (name, values) in enumerate(components):
        masked = _agn_mask_component(values)
        if masked is None:
            continue
        valid = np.isfinite(masked)
        total[valid] += masked[valid]
        support |= valid
        color = _agn_component_color(idx)
        axes[0].plot(wave, masked, color=color, lw=1.0, alpha=0.9, label=name)
        peak = np.nanmax(masked[valid])
        if np.isfinite(peak) and peak > 0.0:
            y = masked / peak + idx * 1.12
            axes[1].plot(wave, y, color=color, lw=0.9, alpha=0.9)
            rest = _agn_rest_wave_for_name(name)
            if np.isfinite(rest):
                axes[1].text(rest, idx * 1.12 + 0.18, name, fontsize=6, rotation=60, ha="left", va="bottom")
    if support.any():
        ytotal = np.full_like(total, np.nan)
        ytotal[support] = total[support]
        axes[0].plot(wave, ytotal, color="black", lw=1.3, label="emission_total_from_components")
    for ax in axes:
        ax.axhline(0.0, color="0.65", lw=0.6)
        _agn_style_axes(ax)
    axes[0].set_ylabel("Component flux")
    axes[1].set_ylabel("Normalized + offset")
    axes[1].set_xlabel(payload["wave_label"])
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.5, ncol=2)
    fig.suptitle("All individual emission components from COMPONENTS / LINE_PARAMS fallback", fontsize=14)
    fig.tight_layout(rect=(0, 0, 0.78, 0.96))
    return fig


def _agn_line_param_table_pages(payload: dict[str, Any], warnings: list[str]) -> list[Any]:
    import matplotlib.pyplot as plt

    rows = payload.get("line_params", [])
    columns = ["name", "flux", "flux_err", "snr", "velocity", "fwhm_kms", "component_type", "tied_group", "detected_flag", "quality_flag"]
    chunk_size = 26
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)] or [[]]
    figures = []
    for page_idx, chunk in enumerate(chunks, start=1):
        fig, ax = plt.subplots(figsize=(14.0, 8.5))
        ax.axis("off")
        table_rows = []
        for row in chunk:
            table_rows.append([_agn_table_value(row.get(col, "")) for col in columns])
        table = ax.table(cellText=table_rows, colLabels=columns, loc="upper left", cellLoc="left", colLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(6.2)
        table.scale(1.0, 1.25)
        title = "LINE_PARAMS summary"
        if len(chunks) > 1:
            title += f" ({page_idx}/{len(chunks)})"
        ax.set_title(title, loc="left", fontsize=14, pad=10)
        if page_idx == len(chunks):
            warning_text = "\n".join(warnings[:10]) if warnings else "No sanity-check warnings."
            if len(warnings) > 10:
                warning_text += f"\n... {len(warnings) - 10} more warning(s); see log."
            ax.text(0.01, -0.03, "Warnings:\n" + warning_text, transform=ax.transAxes, va="top", fontsize=7)
        figures.append(fig)
    return figures


def _agn_table_value(value: Any) -> str:
    if isinstance(value, (float, int, np.floating, np.integer)):
        numeric = _to_float(value)
        return "" if not np.isfinite(numeric) else f"{numeric:.4g}"
    return str(_decode_scalar(value))


def _agn_components_in_window(
    payload: dict[str, Any],
    xlim: tuple[float, float],
    min_component_peak: float | None = None,
) -> list[tuple[str, np.ndarray]]:
    wave = np.asarray(payload["plot_wave"], dtype=float)
    mask = (wave >= xlim[0]) & (wave <= xlim[1])
    found = []
    for name, values in payload.get("plot_components", {}).items():
        arr = np.asarray(values, dtype=float)
        if arr.size != wave.size:
            continue
        if _agn_nonzero(arr[mask], min_component_peak=min_component_peak):
            found.append((name, arr))
            continue
        rest = _agn_rest_wave_for_name(name)
        if np.isfinite(rest) and xlim[0] <= rest <= xlim[1] and _agn_nonzero(arr, min_component_peak=min_component_peak):
            found.append((name, arr))
    return found


def _agn_plot_model_line(ax: Any, wave: np.ndarray, values: Any, label: str, color: str, lw: float = 1.0, ls: str = "-") -> None:
    if values is None:
        return
    arr = np.asarray(values, dtype=float)
    if arr.size != wave.size or not _agn_nonzero(arr):
        return
    ax.plot(wave, arr, color=color, lw=lw, ls=ls, label=label)


def _agn_plot_component_window(
    ax: Any,
    wave: np.ndarray,
    values: Any,
    xlim: tuple[float, float],
    name: str,
    alpha: float = 0.9,
    lw: float = 1.0,
    ls: str = "-",
) -> None:
    if values is None:
        return
    arr = np.asarray(values, dtype=float)
    if arr.size != wave.size:
        return
    masked = _agn_mask_component(arr)
    if masked is None:
        return
    region = (wave >= xlim[0]) & (wave <= xlim[1])
    valid = np.isfinite(masked) & region
    if np.count_nonzero(valid) < 2:
        return
    ax.plot(wave[valid], masked[valid], lw=lw, ls=ls, alpha=alpha, color=_agn_component_color(abs(hash(name)) % 20), label=name)


def _agn_mask_component(values: Any) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    peak = np.nanmax(np.abs(arr[finite]))
    if not np.isfinite(peak) or peak <= 1.0e-12:
        return None
    mask = finite & (np.abs(arr) > max(1.0e-4 * peak, 1.0e-12))
    if np.count_nonzero(mask) < 2:
        return None
    out = np.full_like(arr, np.nan, dtype=float)
    out[mask] = arr[mask]
    return out


def _agn_annotate_lines(ax: Any, wave: np.ndarray, xlim: tuple[float, float] | None = None) -> None:
    lo = float(np.nanmin(wave)) if xlim is None else xlim[0]
    hi = float(np.nanmax(wave)) if xlim is None else xlim[1]
    ymin, ymax = ax.get_ylim()
    for rest, label in AGN_KNOWN_LINE_POSITIONS:
        if lo <= rest <= hi:
            ax.axvline(rest, color="0.75", lw=0.5, alpha=0.6)
            ax.text(rest, ymax, label, rotation=90, va="top", ha="right", fontsize=6, color="0.35")
    ax.set_ylim(ymin, ymax)


def _agn_scale_y_to_data(ax: Any, *arrays: Any) -> None:
    pieces = []
    for values in arrays:
        arr = np.asarray(values, dtype=float)
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                pieces.append(finite)
    if not pieces:
        return
    merged = np.concatenate(pieces)
    lo, hi = np.nanpercentile(merged, [0.5, 99.7])
    pad = 0.08 * max(hi - lo, 1.0)
    ax.set_ylim(lo - pad, hi + pad)


def _agn_style_axes(ax: Any) -> None:
    ax.tick_params(direction="in", top=True, right=True, which="both")
    ax.minorticks_on()
    ax.grid(alpha=0.12)


def _agn_component_color(index: int) -> Any:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    return cmap(index % 20)


__all__ = [
    "SavedFitResult",
    "attach_spectrum_to_result",
    "ensure_saved_fit_result",
    "load_fit_result_from_fits",
    "plot_agn_fit_from_fits",
    "plot_fit_result_pdf",
    "save_fit_result_to_fits",
]
