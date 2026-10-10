# EchoSpec

EchoSpec is designed for differential spectroscopy. It applies identical stellar-continuum and emission-line fitting recipes to repeat spectra, enabling robust measurements of narrow-line variability and diagnostic-line migration.

EchoSpec is a stellar-corrected differential emission-line fitting package for repeat optical spectra of AGNs. It is optimized for SDSS-DESI repeat spectra and for measuring narrow-line evolution, especially [O III], [O II], Hβ, Hα, [N II], [S II], BPT migration, and [O III]/[O II] differential response.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[test]"
```

## Minimal Usage

Fit one rest-frame spectrum:

```python
import numpy as np

from echospec import AGNSpectrum
from echospec.pipeline import fit_single_epoch

spec = AGNSpectrum.from_arrays(
    wave=wave_rest,
    flux=flux,
    err=err,
    z=0.0,
    survey="DESI",
    object_id="target-id",
)

epoch_result = fit_single_epoch(spec)
print(epoch_result.hb_oiii.line_flux["OIII_5007"])
```

Fit a repeat pair:

```python
from echospec.pipeline import fit_repeat_pair

result1, result2, comparison = fit_repeat_pair(sdss_spec, desi_spec)

print(comparison.flux_ratio["OIII_5007"])
print(comparison.differential_oiii_oii_index)
```

`fit_repeat_pair` also returns an object with named fields: `epoch1`, `epoch2`, and `comparison`.

## Science Design

EchoSpec is built for repeat-spectrum differential measurements, not generic AGN fitting. It applies identical YAML emission-line recipes to each epoch so that measured changes are less sensitive to fitting choices.

The package supports optional stellar-continuum subtraction with pPXF before emission-line fitting. Emission-line models are fit with lmfit, using bounded and tied parameters, including fixed atomic doublet ratios such as [O III] λ5007/λ4959 and [N II] λ6584/λ6548.

High-level comparison utilities report narrow-line flux changes, BPT migration, and the [O III]/[O II] differential response, making SDSS-DESI repeat spectroscopy workflows compact and reproducible.

When `do_stellar=True`, EchoSpec requires real pPXF stellar templates. Pass
`template_wave` and `template_flux` directly, or prepare a local MILES/E-MILES
template library:

```bash
python scripts/download_ppxf_templates.py --library miles
# or
python scripts/download_ppxf_templates.py --library emiles
```

Download MILES/E-MILES SSP templates from the official pages, then place the
FITS files under:

```text
~/.echospec/templates/miles/
```

or unpack a local archive into the cache:

```bash
python scripts/download_ppxf_templates.py \
  --library miles \
  --local-archive /path/to/downloaded_templates.zip
```

For software testing only, EchoSpec can create a small demo template library:

```bash
python scripts/download_ppxf_templates.py --library demo
```

The demo templates are only for software testing and must not be used for science.

If no templates are available, the pipeline raises `FileNotFoundError` instead
of silently disabling stellar-continuum subtraction.

## Tests

```bash
pytest
```
