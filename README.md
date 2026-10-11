# EchoSpecFit

EchoSpecFit decomposes optical AGN spectra into host starlight, an AGN power law,
Fe II and emission lines. It combines the existing DESIPecFit single-spectrum
workflow and EchoSpec repeat-spectrum engine in one installable package.

The single-spectrum workflow fits the host continuum with pPXF, refits a
fixed-shape host plus power law and optical Fe II, fits the emission-line windows,
and performs a final full-spectrum component amplitude refit. It supports
[O II], the blue Balmer lines, Hβ/[O III], Hα/[N II]/[S II] and optional Mg II.
Repeat-spectrum utilities compare line fluxes, BPT ratios and [O III]/[O II].

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/guowj-astra/EchoSpecFit.git
cd EchoSpecFit
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Prepare the templates

Real stellar and empirical Fe II templates are external resources. Download and
prepare the stellar library, then supply your optical and, where relevant, UV
Fe II template files. See [template preparation](docs/templates.md).

For example, unpack a locally downloaded E-MILES archive:

```bash
python scripts/download_ppxf_templates.py \
  --library emiles_basti_ch \
  --local-archive /path/to/emiles_archive.tar.gz
```

The existing `~/.echospec/templates/` cache remains supported. Set
`ECHOSPECFIT_TEMPLATE_DIR` to use a different cache root.

## Fit a spectrum

Input text files contain wavelength, flux density and error in three columns.
Use consistent flux and error units. Supply the redshift with `--z` for an
observed-frame input; `--z 0` means the input is already in the rest frame.
Line centres use air wavelengths; prepare the wavelength convention accordingly.

```bash
echospecfit target.txt --z 0.05 \
  --stellar-template-path /path/to/emiles_basti_ch \
  --feii-template-path /path/to/optical_feii.txt \
  --uv-feii-template-path /path/to/uv_feii.txt \
  --outdir outputs/target
```

Inspect all options with `echospecfit --help`. For optical-only spectra, use
`--no-mgii`; for Mg II fitting without UV Fe II, use `--no-uv-feii`.

The Python interface exposes the same configuration:

```python
from echospecfit import (
    EchoSpecFitConfig, H1Spectrum, fit_h1spectrum, save_fit_result,
)

spec = H1Spectrum.from_txt(
    "target.txt", z=0.05, survey="DESI", object_id="target",
)
config = EchoSpecFitConfig(
    stellar_template_path="/path/to/emiles_basti_ch",
    feii_template_path="/path/to/optical_feii.txt",
    include_mgii=False,
)
result = fit_h1spectrum(spec, config)
print(result.summary["final_OIII_5007"])
save_fit_result("outputs/target", result)
```

Outputs include diagnostic figures, JSON/CSV measurements and FITS component
products with input errors, masks and residuals. Inspect fit-quality flags and
component diagnostics before using measurements for scientific inference.
Missing stellar templates raise an error. Missing Fe II templates retain the
inherited warning/flag and fallback behavior; a completed run alone does not
confirm that Fe II was fitted.

## Repeat spectra and compatibility

The bundled `echospec` engine retains its existing API:

```python
from echospec.pipeline import fit_repeat_pair

# epoch1 and epoch2 are echospec.AGNSpectrum objects.
result1, result2, comparison = fit_repeat_pair(epoch1, epoch2, do_stellar=False)
print(comparison.flux_ratio["OIII_5007"])
```

Use `do_stellar=False` only when the spectra have already had their stellar
continuum removed or when that choice is appropriate for your analysis. See
[the engine API](docs/legacy_echospec.md) for template-based fitting.

The old `desipecfit` imports and command remain compatibility aliases.
`DesiPecFitConfig` and `DesiPecFitResult` alias the new class names. Historical
measurement keys and filenames, including `desipecfit_epoch_fit.csv`, remain
unchanged so existing analysis scripts can read new outputs.

This release renames and packages the software; it does not rerun observations
or reproduce all manuscript-specific selections, preprocessing and Monte Carlo
uncertainty calculations. Those choices must be supplied by the analysis using
the package. See [source provenance](docs/provenance.md).

## License

Software is distributed under the [MIT license](LICENSE), as declared by the
original packages. External stellar and Fe II template data have their own
terms and citation requirements and are not included here.
