# Preparing scientific templates

EchoSpecFit ships its fitting software and line recipes. Stellar population
libraries and empirical Fe II template data are supplied separately. The
software license does not grant a license to those external datasets; follow
their providers' terms and citation requirements.

## Stellar population templates

Obtain MILES or E-MILES SSP FITS models from the official IAC pages:

- [MILES SSP models](https://research.iac.es/proyecto/miles/pages/ssp-models.php)
- [E-MILES spectral energy distributions](https://research.iac.es/proyecto/miles/pages/spectral-energy-distributions-seds/e-miles.php)

The default single-spectrum workflow uses an age/metallicity-balanced subset
of E-MILES BaSTI Chabrier models, identified as `emiles_basti_ch`. It expects
the original E-MILES FITS filenames so it can parse their ages and metallicities.

Set a template root if you want to store these files outside the existing
EchoSpec cache:

```bash
export ECHOSPECFIT_TEMPLATE_DIR="$HOME/astronomy/templates"
```

Without this variable, the compatible default is `~/.echospec/templates`.
Each library has its own subdirectory, for example
`$ECHOSPECFIT_TEMPLATE_DIR/emiles_basti_ch/`.

After downloading a provider's archive, prepare it from the repository root:

```bash
python scripts/download_ppxf_templates.py \
  --library emiles_basti_ch \
  --local-archive /path/to/EMILES_BASTI_BASE_CH_FITS.tar.gz \
  --template-root "$ECHOSPECFIT_TEMPLATE_DIR"
```

The preparation script unpacks a local archive and records a source manifest;
it does not download scientific templates automatically. Omit `--template-root`
to use the configured root, or the legacy cache if the variable is unset. You can also place the FITS files
directly in the library's directory.

An explicit library path overrides the configured root:

```bash
echospecfit spectrum.txt \
  --stellar-template-path /path/to/emiles_basti_ch
```

For another compatible library or a prepared NumPy template matrix, use the
general loader rather than the default representative E-MILES selector:

```bash
echospecfit spectrum.txt \
  --template-library miles \
  --all-stellar-templates \
  --stellar-template-path /path/to/miles
```

The Python interface also accepts `template_wave` and `template_flux` directly
in `fit_h1spectrum()`. The older engine-specific environment variables
`ECHOSPEC_PPXF_TEMPLATE_CACHE` and `ECHOSPEC_PPXF_TEMPLATE_LIBRARY` remain
available to engine callers.

## Optical and ultraviolet Fe II templates

Provide a valid wavelength/flux text template with appropriate coverage for
each fitted region:

```bash
echospecfit spectrum.txt \
  --stellar-template-path /path/to/emiles_basti_ch \
  --feii-template-path /path/to/optical_feii.txt \
  --uv-feii-template-path /path/to/uv_feii.txt \
  --outdir outputs/target
```

The staged optical fit uses 4200–5600 Angstrom; the Mg II/UV Fe II window uses
2200–3090 Angstrom. Check that the chosen templates are suitable for those
regions and preserve their source and citation information. The historical
local file `fetemplate_no3` is not included in this repository because its
redistribution provenance was not recorded in the original project.

Use `--no-uv-feii` for an intentional UV fit without Fe II. Python callers can
set `EchoSpecFitConfig(include_feii=False)` to disable the optical Fe II model.

The existing fallback behavior is preserved: if no optical Fe II template is
available, the staged Fe II result is unusable with reason
`missing_feii_template`, and the H beta fit emits a warning and records a
missing-template flag while continuing without Fe II. Numerical completion
therefore does not imply that the full Fe II workflow ran. Inspect the staged
result and fit-quality flags before interpreting measurements.

## Software-only demo templates

`python scripts/download_ppxf_templates.py --library demo` prepares templates
for software checks. Use `--all-stellar-templates --template-library demo` to
load that library. Demo templates are not suitable for scientific inference.
