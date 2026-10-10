from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("matplotlib")

from echospec.io_fits import (
    SavedFitResult,
    load_fit_result_from_fits,
    plot_fit_result_pdf,
    save_fit_result_to_fits,
)


def _mock_saved_result() -> SavedFitResult:
    wave = np.linspace(3600.0, 6900.0, 900)
    err = np.full_like(wave, 0.2)
    stellar = 4.0 + 0.0002 * (wave - 5000.0)
    powerlaw = 0.8 * (wave / 5100.0) ** -1.0
    hb = 10.0 * np.exp(-0.5 * ((wave - 4861.33) / 3.0) ** 2)
    oiii = 20.0 * np.exp(-0.5 * ((wave - 5006.84) / 2.5) ** 2)
    ha = 30.0 * np.exp(-0.5 * ((wave - 6562.8) / 3.0) ** 2)
    oii = 8.0 * np.exp(-0.5 * ((wave - 3727.5) / 2.8) ** 2)
    emission = hb + oiii + ha + oii
    total = stellar + powerlaw + emission
    flux = total + 0.01 * np.sin(wave / 30.0)
    residual = flux - total
    components = {
        "stellar_model": stellar,
        "host_stellar": stellar,
        "galaxy_host": stellar,
        "powerlaw": powerlaw,
        "Hb_narrow": hb,
        "OIII_5007_core": oiii,
        "Ha_narrow": ha,
        "OII_model": oii,
    }
    return SavedFitResult(
        object_id="mock",
        redshift=0.1,
        survey="TEST",
        strategy="window_joint",
        status="ok",
        quality="good",
        mc_n=5,
        random_seed=42,
        config={"fit_strategy": "window_joint", "mc_n": 5},
        spectrum={
            "wave": wave,
            "flux": flux,
            "err": err,
            "mask_good": np.ones_like(wave, dtype=bool),
            "rest_wave": wave,
            "original_flux": flux,
        },
        model_total={
            "wave": wave,
            "total_model": total,
            "residual": residual,
            "chi": residual / err,
            "stellar_model": stellar,
            "powerlaw_model": powerlaw,
            "feii_model": np.zeros_like(wave),
            "local_continuum_model": np.zeros_like(wave),
            "emission_total_model": emission,
        },
        components=components,
        line_params=[
            {
                "name": "Hb_narrow",
                "flux": 75.0,
                "flux_err": 5.0,
                "velocity": 0.0,
                "sigma_kms": 180.0,
                "fwhm_kms": 423.9,
                "snr": 15.0,
                "tied_group": "narrow",
                "component_type": "narrow",
                "detected_flag": True,
                "quality_flag": "",
            }
        ],
        diagnostics={"chi2": 1.0, "redchi": 1.0, "bic": 10.0, "fit_quality": "good"},
    )


def test_save_load_fit_result_fits(tmp_path):
    result = _mock_saved_result()
    fits_path = tmp_path / "mock_result.fits"
    save_fit_result_to_fits(result, fits_path)

    loaded = load_fit_result_from_fits(fits_path)
    assert loaded.object_id == "mock"
    assert np.allclose(loaded.spectrum["wave"], result.spectrum["wave"])
    assert np.allclose(loaded.components["Hb_narrow"], result.components["Hb_narrow"])
    assert loaded.line_params[0]["name"] == "Hb_narrow"


def test_replot_from_fits(tmp_path):
    result = _mock_saved_result()
    fits_path = tmp_path / "mock_result.fits"
    pdf_path = tmp_path / "mock_diagnostic.pdf"
    save_fit_result_to_fits(result, fits_path)
    plot_fit_result_pdf(fits_path, pdf_path)

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0
