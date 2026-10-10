# Validation of the rename

The release was checked on 10 October 2026 with Python 3.11 and NumPy 1.23.5,
SciPy 1.11.2, Astropy 5.3.3, matplotlib 3.9.2, pandas 2.0.3, PyYAML 6.0.1,
lmfit 1.2.2 and pPXF 9.4.2.

The final complete suite passed: **73 passed**, with 10 warnings, in 16.60 s.

Validation includes canonical/legacy import identity, CLI compatibility,
configuration aliases, preserved output schema, source-archive contents and
an installed-wheel smoke check run outside the source directory. The wheel
check reads both [O II] recipes, recovers a synthetic line flux and verifies
template-root configuration without any external scientific template data.

The initial inherited suite had seven failures, reproduced with the original
pre-rename source in the same environment. These were outdated test conditions:

- Two broad-Hβ comparisons used old metadata with the newer [O III] wing
  branch. Tests explicitly select the core-only comparison they originally
  intended to exercise.
- Two Fe II tests expected target-width broadening while the default is extra
  broadening. They now select `feii_broadening_mode="target"` explicitly.
- Toy Fe II profiles now have flux in the required normalization interval.
- A synthetic Hα width is inside the fitter's existing bounds.
- The stellar-only pPXF test uses a wider template wavelength range and
  disables the optional AGN power-law templates.

All scientific assertions and numerical tolerances are retained. No fitting
algorithm was changed to make these tests pass. The checks validate software
behavior and packaging, not the scientific interpretation of observed targets.
