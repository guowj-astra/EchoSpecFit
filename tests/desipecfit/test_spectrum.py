import numpy as np

from desipecfit import H1Spectrum


def test_h1spectrum_sorts_and_masks_bad_pixels():
    spec = H1Spectrum.from_arrays(
        wave=[5002.0, 5000.0, 5001.0],
        flux=[3.0, 1.0, np.nan],
        err=[1.0, 1.0, 1.0],
        object_id="demo",
    )

    clean = spec.mask_bad_pixels()

    assert clean.object_id == "demo"
    assert clean.wave.tolist() == [5000.0, 5002.0]
    assert clean.flux.tolist() == [1.0, 3.0]

