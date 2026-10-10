"""Tests for repeat-epoch comparison utilities."""

from types import SimpleNamespace

import numpy as np

from echospec.compare import EpochFitSet, compare_epochs
from echospec.diagnostics import assess_epoch_comparison, assess_fit_quality


def mock_fit(
    line_flux: dict[str, float],
    line_flux_err: dict[str, float] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(line_flux=line_flux, line_flux_err=line_flux_err or {})


def test_compare_epochs_computes_flux_changes_and_significance() -> None:
    epoch1 = EpochFitSet(
        object_id="agn-1",
        survey="SDSS",
        oii=mock_fit({"OII_3727_blend": 50.0}, {"OII_3727_blend": 5.0}),
        hb_oiii=mock_fit(
            {"OIII_5007": 100.0, "Hb_narrow": 20.0},
            {"OIII_5007": 10.0, "Hb_narrow": 2.0},
        ),
        ha_nii_sii=mock_fit(
            {
                "Ha_narrow": 60.0,
                "NII_6584": 30.0,
                "SII_6716": 10.0,
                "SII_6731": 20.0,
            }
        ),
    )
    epoch2 = EpochFitSet(
        object_id="agn-1",
        survey="DESI",
        oii=mock_fit({"OII_3727_blend": 50.0}, {"OII_3727_blend": 5.0}),
        hb_oiii=mock_fit(
            {"OIII_5007": 200.0, "Hb_narrow": 40.0},
            {"OIII_5007": 10.0, "Hb_narrow": 2.0},
        ),
        ha_nii_sii=mock_fit(
            {
                "Ha_narrow": 120.0,
                "NII_6584": 120.0,
                "SII_6716": 20.0,
                "SII_6731": 40.0,
            }
        ),
    )

    result = compare_epochs(epoch1, epoch2)

    assert result.object_id == "agn-1"
    np.testing.assert_allclose(result.flux_ratio["OIII_5007"], 2.0)
    np.testing.assert_allclose(result.fractional_change["OIII_5007"], 1.0)
    np.testing.assert_allclose(
        result.change_significance["OIII_5007"],
        100.0 / np.sqrt(10.0**2 + 10.0**2),
    )


def test_compare_epochs_computes_bpt_and_delta_log_ratios() -> None:
    epoch1 = EpochFitSet(
        object_id="agn-1",
        survey="SDSS",
        oii=mock_fit({"OII_3727_blend": 50.0}),
        hb_oiii=mock_fit({"OIII_5007": 100.0, "Hb_narrow": 20.0}),
        ha_nii_sii=mock_fit(
            {
                "Ha_narrow": 60.0,
                "NII_6584": 30.0,
                "SII_6717": 10.0,
                "SII_6731": 20.0,
            }
        ),
    )
    epoch2 = EpochFitSet(
        object_id="agn-1",
        survey="DESI",
        oii=mock_fit({"OII_3727_blend": 50.0}),
        hb_oiii=mock_fit({"OIII_5007": 200.0, "Hb_narrow": 40.0}),
        ha_nii_sii=mock_fit(
            {
                "Ha_narrow": 120.0,
                "NII_6584": 120.0,
                "SII_6717": 20.0,
                "SII_6731": 40.0,
            }
        ),
    )

    result = compare_epochs(epoch1, epoch2)

    np.testing.assert_allclose(result.bpt_sdss, (np.log10(0.5), np.log10(5.0)))
    np.testing.assert_allclose(result.bpt_desi, (np.log10(1.0), np.log10(5.0)))
    np.testing.assert_allclose(result.bpt_vector, (np.log10(2.0), 0.0))
    np.testing.assert_allclose(
        result.delta_log_ratio["delta_log_NII_Ha"],
        np.log10(1.0) - np.log10(0.5),
    )
    np.testing.assert_allclose(result.delta_log_ratio["delta_log_OIII_Hb"], 0.0)
    np.testing.assert_allclose(result.delta_log_ratio["delta_log_SII_Ha"], 0.0)
    np.testing.assert_allclose(
        result.delta_log_ratio["delta_log_OIII_OII"],
        np.log10(4.0) - np.log10(2.0),
    )
    np.testing.assert_allclose(result.differential_oiii_oii_index, 2.0)


def test_assess_fit_quality_flags_low_snr_and_boundaries() -> None:
    result = SimpleNamespace(
        success=True,
        line_flux={"OIII_5007": 10.0, "Hb_narrow": 10.0},
        line_flux_err={"OIII_5007": 5.0, "Hb_narrow": 1.0},
        params={
            "narrow_sigma": 51.0,
            "narrow_velocity": 790.0,
            "delta_bic_broad": 12.0,
        },
        mc_summary={},
    )

    flags = assess_fit_quality(result)

    assert flags["flag_low_snr_line"]
    assert flags["low_snr_lines"] == ["OIII_5007"]
    assert flags["flag_boundary_sigma"]
    assert flags["flag_boundary_velocity"]
    assert flags["flag_broad_balmer_required"]


def test_assess_epoch_comparison_flags_differential_oiii_response() -> None:
    comparison = SimpleNamespace(
        flux_ratio={
            "OII_3727_blend": 1.02,
            "OIII_5007": 1.8,
            "Hb_narrow": 1.01,
            "Ha_narrow": 1.0,
            "NII_6584": 1.03,
        },
        change_significance={
            "OII_3727_blend": 0.5,
            "OIII_5007": 5.0,
        },
    )

    flags = assess_epoch_comparison(comparison)

    assert flags["flag_differential_oiii_response"]
    assert not flags["flag_possible_calibration_offset"]


def test_assess_epoch_comparison_flags_possible_calibration_offset() -> None:
    comparison = SimpleNamespace(
        flux_ratio={
            "OII_3727_blend": 1.25,
            "OIII_5007": 1.28,
            "Hb_narrow": 1.24,
            "Ha_narrow": 1.26,
            "NII_6584": 1.27,
            "SII_6716": 1.23,
        },
        change_significance={},
    )

    flags = assess_epoch_comparison(comparison)

    assert flags["flag_possible_calibration_offset"]
    assert len(flags["calibration_offset_lines"]) >= 4
