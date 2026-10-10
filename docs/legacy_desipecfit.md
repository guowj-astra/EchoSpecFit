# DESIPecFit

`desipecfit` is a small standalone wrapper around the EchoSpec single-spectrum
workflow for DESI-like one-dimensional AGN spectra.

The default flow is:

1. Fit the host galaxy and featureless AGN continuum with pPXF over the full
   wavelength grid. Optical Fe II regions stay in the grid but receive a very
   small statistical weight, while emission lines are masked internally.
2. Freeze the pPXF host template shape, velocity, and dispersion. Refit only
   its non-negative amplitude together with one power law and a broadened,
   shifted optical Fe II template over 4200-5600 Angstrom. Broad emission-line
   windows are excluded and the measured input errors are used unchanged.
3. Subtract the staged host, power law, and Fe II continuum before fitting
   Mg II, [O II], the blue Balmer block, H beta + [O III], and H alpha +
   [O I] + [N II] + [S II]. Fe II is not fitted a second time in the H beta
   window.
4. Reassemble the fixed-shape continuum, Fe II, and emission-line components
   on the full rest-frame wavelength grid. Fixed-ratio doublets remain tied
   during this final amplitude refit.
5. Save local line diagnostics, a full-spectrum component diagnostic, a FITS
   continuum product containing the input errors, masks, components and
   residuals, plus JSON/CSV tables containing the emission-line parameters and
   fit-quality flags.

The staged result distinguishes numerical completion from a useful Fe II
measurement. `staged_feii_secure` requires Delta BIC > 10, positive Fe II
flux, width and shift away from their bounds, a credible host decomposition,
and reduced chi-square < 3. The full-spectrum component refit is enabled by
default; use `--disable-no-local-continuum-refit` only for a local-window-only
comparison run. The older joint nonlinear refit remains opt-in.

## Install

Install the sibling EchoSpec package first, then install this package:

```bash
pip install -e ../echospec
pip install -e .
```

## Python Usage

```python
from desipecfit import H1Spectrum, DesiPecFitConfig, fit_h1spectrum

spec = H1Spectrum.from_txt(
    "target_desi.txt",
    survey="DESI",
    object_id="target",
)

result = fit_h1spectrum(spec, DesiPecFitConfig())
print(result.summary["final_OIII_5007"])
print(result.staged_continuum["feii_secure"])
```

Save the default reports, diagnostic figure, and reusable FITS model product:

```python
from desipecfit import save_fit_result

save_fit_result("outputs/target", result)
```

By default this writes the staged host/Fe II diagnostic plus the original
Mg II, [O II], blue-Balmer, H beta/[O III], and H alpha/[N II]/[S II]
broad/narrow line diagnostics. Set `make_line_plots=False` only for a fast
continuum-only pilot run; the line fits themselves are still measured.

`H1Spectrum` is intentionally close to EchoSpec's `AGNSpectrum` but is kept in
this package as the single-spectrum input type. Use `to_agn_spectrum()` when
you need the underlying EchoSpec container.

## CLI Usage

Fit one file:

```bash
desipecfit target_desi.txt --outdir outputs/desipecfit_target
```

Keep a tentative broad H beta component even when the automatic secure-line
gate rejects it:

```bash
desipecfit target_desi.txt --force-broad-hb --outdir outputs/desipecfit_target_forced_broad
```

When broad H beta is forced, DESIPecFit also forces the broad Hγ/Hδ
blue-Balmer components and keeps them in the final no-local-continuum refit.
Use `--no-local-forced-blue-broad-min-scale` to lower the retained broad
blue-Balmer scale for comparison runs.

Mg II is enabled by default for spectra covering 2200-3090 Angstrom, and the
Mg II window includes UV Fe II by default. DESIPecFit also keeps a broad Mg II
component by default, even when EchoSpec's automatic secure-line gate rejects
it; the broad component is still removed and refit if any component exceeds the
configured 20000 km/s FWHM limit. Use `--no-force-mgii-broad` for conservative
comparison runs. If no UV Fe II template is supplied, DESIPecFit uses EchoSpec's
`templates/fetemplate_no3`; use `--no-uv-feii` to fit Mg II without Fe II, or
`--uv-feii-template-path` to provide a different template.

Use `--no-ppxf-feii-mask` only for comparison runs where pPXF should use normal
weights in the Fe II/Mg II pseudo-continuum windows. Use
`--enable-joint-refit` only when an explicit legacy nonlinear comparison is
needed.

Fit all matching text spectra in a directory:

```bash
desipecfit spectra_txt_rest --outdir outputs/desipecfit_batch
```

Input text files are expected to contain three columns: wavelength, flux, error.
Use `--z` if the file is observed-frame; internally the spectrum is moved to
rest frame before fitting.
