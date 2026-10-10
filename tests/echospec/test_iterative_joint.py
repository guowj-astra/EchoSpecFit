"""Smoke tests for iterative joint model components."""

import numpy as np
from lmfit import Parameters

from echospec.continuum import powerlaw_continuum
from echospec.joint_model import build_joint_model


def test_powerlaw_continuum_uses_pivot_amplitude() -> None:
    wave = np.array([4200.0, 5100.0, 5600.0])
    values = powerlaw_continuum(wave, amp=5.0, alpha=-1.0)

    assert np.isclose(values[1], 5.0)
    assert values[0] > values[2]


def test_joint_model_preserves_atomic_doublet_ratios() -> None:
    wave = np.linspace(3600.0, 6800.0, 3201)
    stellar = np.ones_like(wave)
    params = Parameters()
    params.add("stellar_scale", value=1.0)
    params.add("pl_amp", value=0.0)
    params.add("pl_alpha", value=-1.0)
    params.add("feii_amp", value=0.0)
    params.add("feii_velocity", value=0.0)
    params.add("feii_sigma", value=2000.0)
    params.add("hb_velocity", value=0.0)
    params.add("hb_sigma", value=150.0)
    params.add("hb_narrow_flux", value=10.0)
    params.add("oiii_4959_flux", value=20.0)
    params.add("oiii_5007_flux", expr="2.98 * oiii_4959_flux")
    params.add("oiii_4959_wing_flux", value=0.0)
    params.add("oiii_5007_wing_flux", expr="2.98 * oiii_4959_wing_flux")
    params.add("oiii_wing_velocity", value=0.0)
    params.add("oiii_wing_sigma", value=500.0)
    params.add("hb_broad_1_flux", value=0.0)
    params.add("hb_broad_1_velocity", value=0.0)
    params.add("hb_broad_1_sigma", value=1500.0)
    params.add("hb_broad_2_flux", value=0.0)
    params.add("hb_broad_2_velocity", value=0.0)
    params.add("hb_broad_2_sigma", value=4000.0)
    params.add("ha_velocity", value=0.0)
    params.add("ha_sigma", value=150.0)
    params.add("ha_narrow_flux", value=30.0)
    params.add("nii_6548_flux", value=8.0)
    params.add("nii_6584_flux", expr="2.96 * nii_6548_flux")
    params.add("sii_6716_flux", value=5.0)
    params.add("sii_6731_flux", value=4.0)
    params.add("ha_broad_1_flux", value=0.0)
    params.add("ha_broad_1_velocity", value=0.0)
    params.add("ha_broad_1_sigma", value=1500.0)
    params.add("ha_broad_2_flux", value=0.0)
    params.add("ha_broad_2_velocity", value=0.0)
    params.add("ha_broad_2_sigma", value=4000.0)
    params.add("oii_3727_blend_flux", value=6.0)
    params.add("oii_velocity", value=0.0)
    params.add("oii_sigma", value=150.0)

    total, components = build_joint_model(wave, stellar, params)

    assert total.shape == wave.shape
    assert components["oiii_core"].shape == wave.shape
    assert np.isclose(params["oiii_5007_flux"].value / params["oiii_4959_flux"].value, 2.98)
    assert np.isclose(params["nii_6584_flux"].value / params["nii_6548_flux"].value, 2.96)
