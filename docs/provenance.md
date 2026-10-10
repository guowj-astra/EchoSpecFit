# Source provenance

EchoSpecFit 0.1.0 combines the active DESIPecFit and EchoSpec source used by the
author's local spectral-fitting workflows, copied on 10 October 2026. The
original project metadata declared MIT licenses. Source hashes before the
rename are recorded in `source_sha256.json`.

The public single-spectrum API and command are named `echospecfit`.
Configuration and result classes are named `EchoSpecFitConfig` and
`EchoSpecFitResult`. Compatibility aliases preserve the old `desipecfit` imports,
class names and command. The internal `echospec` engine remains available for
repeat-spectrum analysis.

The release changes branding, packaging, resource discovery and documentation.
It preserves fitting algorithms, measurement keys and historical output
filenames. It includes the original synthetic/software tests, with seven
outdated test fixtures corrected to exercise the current API and valid model
bounds; scientific assertions and tolerances are retained. The original
README files are retained as historical API documentation; their separate
installation instructions are superseded by the root README.

Research spectra, fitted observational products, paper drafts and downloaded
stellar or empirical Fe II template libraries are not part of this release.
Manuscript-specific source selection, spectral preprocessing and uncertainty
analysis are separate from this general-purpose package.
