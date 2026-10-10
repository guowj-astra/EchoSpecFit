"""Command-line entrypoint for EchoSpecFit."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import EchoSpecFitConfig, fit_directory


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="echospecfit",
        description="EchoSpecFit: fit single spectra with pPXF host+powerlaw and Fe II."
    )
    parser.add_argument("input", help="Spectrum text file or directory.")
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Glob used when input is a directory.",
    )
    parser.add_argument("--outdir", default="echospecfit_output")
    parser.add_argument("--survey", default="DESI")
    parser.add_argument("--z", type=float, default=0.0)
    parser.add_argument("--template-library", default="emiles_basti_ch")
    parser.add_argument("--max-templates", type=int, default=None)
    parser.add_argument("--stellar-template-path", default=None)
    parser.add_argument("--all-stellar-templates", action="store_true")
    parser.add_argument("--representative-n-metallicities", type=int, default=6)
    parser.add_argument("--stellar-mask-width-kms", type=float, default=8000.0)
    parser.add_argument("--ppxf-feii-error-scale", type=float, default=1.0e6)
    parser.add_argument(
        "--no-ppxf-feii-mask",
        action="store_true",
        help="Do not downweight Fe II/Mg II pseudo-continuum windows during pPXF.",
    )
    parser.add_argument(
        "--agn-powerlaw-slopes",
        default="-3,-2,-1,0,1,2",
        help=(
            "Comma-separated power-law slopes included in the pPXF host stage."
        ),
    )
    parser.add_argument("--feii-template-path", default=None)
    parser.add_argument("--feii-scale-mode", default="single")
    parser.add_argument("--disable-staged-continuum", action="store_true")
    parser.add_argument(
        "--keep-staged-feii-for-line-fit",
        action="store_true",
        help="Do not subtract the staged optical Fe II model before line fitting.",
    )
    parser.add_argument("--no-mgii", action="store_true")
    parser.add_argument("--no-uv-feii", action="store_true")
    parser.add_argument("--uv-feii-template-path", default=None)
    parser.add_argument("--include-mgii-narrow", action="store_true")
    parser.add_argument("--mgii-broad-model", choices=["single", "double"], default="double")
    parser.add_argument("--mgii-narrow-mode", choices=["blend", "doublet"], default="blend")
    parser.add_argument(
        "--no-force-mgii-broad",
        action="store_true",
        help=(
            "Do not force broad MgII by default; keep only the EchoSpec/BIC/SNR "
            "selected broad component."
        ),
    )
    parser.add_argument(
        "--mgii-broad-snr-threshold",
        type=float,
        default=0.5,
        help=(
            "Fallback broad_snr threshold for keeping broad MgII when "
            "--no-force-mgii-broad is used and delta_BIC > 10."
        ),
    )
    parser.add_argument(
        "--low-host-threshold",
        type=float,
        default=0.30,
        help="Deprecated compatibility option; the staged fit now runs for every source.",
    )
    parser.add_argument(
        "--force-broad-hb",
        action="store_true",
        help="Keep a broad Hbeta component instead of applying the BIC/SNR gate.",
    )
    parser.add_argument("--broad-hb-model", choices=["single", "double"], default="double")
    parser.add_argument(
        "--hb-broad-max-component-fwhm-kms",
        type=float,
        default=20000.0,
        help=(
            "Remove/refit any broad component whose FWHM exceeds this value; "
            "applies to Hbeta, Halpha, and MgII broad components."
        ),
    )
    parser.add_argument("--broad-ha-model", choices=["single", "double"], default="double")
    parser.add_argument("--no-blue-balmer", action="store_true")
    parser.add_argument("--no-blue-balmer-neiii", action="store_true")
    parser.add_argument("--include-blue-balmer-hepsilon", action="store_true")
    parser.add_argument("--force-broad-blue-balmer", action="store_true")
    parser.add_argument("--no-oiii-wing", action="store_true")
    parser.add_argument(
        "--enable-joint-refit",
        action="store_true",
        help="Enable the legacy full-spectrum joint refit after local line fits.",
    )
    parser.add_argument(
        "--enable-no-local-continuum-refit",
        action="store_true",
        help="Compatibility flag; the final global component refit is enabled by default.",
    )
    parser.add_argument(
        "--disable-no-local-continuum-refit",
        action="store_true",
        help=(
            "Disable the final global continuum + emission-line component refit."
        ),
    )
    parser.add_argument(
        "--no-local-continuum-err-floor-fraction",
        type=float,
        default=0.0,
        help="Error floor fraction used in the final no-local-continuum refit.",
    )
    parser.add_argument(
        "--no-local-forced-blue-broad-min-scale",
        type=float,
        default=1.0,
        help=(
            "Minimum scale, relative to the blue-Balmer window fit, for forced "
            "broad Hγ/Hδ in the final no-local-continuum refit."
        ),
    )
    parser.add_argument(
        "--no-local-blue-broad-profile",
        choices=["filtered_hbeta", "hbeta"],
        default="filtered_hbeta",
        help=(
            "Broad Hγ/Hδ profile used in the final no-local-continuum refit. "
            "filtered_hbeta keeps only non-extreme Hβ broad components."
        ),
    )
    parser.add_argument(
        "--no-local-blue-broad-max-sigma-kms",
        type=float,
        default=3000.0,
        help="Maximum Hβ broad sigma allowed by filtered_hbeta for Hγ/Hδ.",
    )
    parser.add_argument(
        "--no-local-blue-broad-max-abs-velocity-kms",
        type=float,
        default=2000.0,
        help="Maximum absolute Hβ broad velocity allowed by filtered_hbeta for Hγ/Hδ.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-line-plots", action="store_true")
    parser.add_argument("--no-fits-products", action="store_true")
    parser.add_argument("--write-array-products", action="store_true")
    args = parser.parse_args(argv)

    cfg = EchoSpecFitConfig(
        template_library=args.template_library,
        max_templates=args.max_templates,
        representative_stellar_templates=not args.all_stellar_templates,
        stellar_template_path=args.stellar_template_path,
        representative_n_metallicities=args.representative_n_metallicities,
        stellar_mask_width_kms=args.stellar_mask_width_kms,
        ppxf_mask_feii_windows=not args.no_ppxf_feii_mask,
        ppxf_feii_error_scale=args.ppxf_feii_error_scale,
        agn_powerlaw_slopes=_parse_slopes(args.agn_powerlaw_slopes),
        staged_continuum=not args.disable_staged_continuum,
        staged_subtract_feii_before_lines=not args.keep_staged_feii_for_line_fit,
        feii_template_path=args.feii_template_path,
        feii_scale_mode=args.feii_scale_mode,
        include_mgii=not args.no_mgii,
        include_uv_feii=not args.no_uv_feii,
        uv_feii_template_path=args.uv_feii_template_path,
        include_mgii_narrow=args.include_mgii_narrow,
        force_mgii_broad=not args.no_force_mgii_broad,
        mgii_broad_model=args.mgii_broad_model,
        mgii_narrow_mode=args.mgii_narrow_mode,
        mgii_broad_snr_threshold=args.mgii_broad_snr_threshold,
        low_host_refit_threshold=args.low_host_threshold,
        include_oiii_wing=not args.no_oiii_wing,
        force_broad_hb=args.force_broad_hb,
        broad_hb_model=args.broad_hb_model,
        hb_broad_max_component_fwhm_kms=args.hb_broad_max_component_fwhm_kms,
        broad_ha_model=args.broad_ha_model,
        include_blue_balmer=not args.no_blue_balmer,
        include_blue_balmer_neiii=not args.no_blue_balmer_neiii,
        include_blue_balmer_hepsilon=args.include_blue_balmer_hepsilon,
        force_broad_blue_balmer=args.force_broad_blue_balmer,
        joint_refit=args.enable_joint_refit,
        no_local_continuum_refit=not args.disable_no_local_continuum_refit,
        no_local_continuum_err_floor_fraction=args.no_local_continuum_err_floor_fraction,
        no_local_forced_blue_broad_min_scale=args.no_local_forced_blue_broad_min_scale,
        no_local_blue_broad_profile=args.no_local_blue_broad_profile,
        no_local_blue_broad_max_sigma_kms=args.no_local_blue_broad_max_sigma_kms,
        no_local_blue_broad_max_abs_velocity_kms=args.no_local_blue_broad_max_abs_velocity_kms,
        make_plots=not args.no_plots,
        make_line_plots=not args.no_line_plots,
        write_array_products=args.write_array_products,
        write_fits_products=not args.no_fits_products,
    )
    rows = fit_directory(
        Path(args.input),
        Path(args.outdir),
        cfg,
        pattern=args.pattern,
        z=args.z,
        survey=args.survey,
    )
    print(f"Wrote {Path(args.outdir) / 'desipecfit_epoch_fit.csv'}")
    print(f"Processed {len(rows)} spectrum/s")


def _parse_slopes(value: str) -> tuple[float, ...]:
    slopes = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not slopes:
        raise ValueError("At least one AGN power-law slope is required.")
    return slopes


if __name__ == "__main__":
    main()
