"""Joint stellar, continuum, Fe II, and emission-line model utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from echospec.continuum import powerlaw_continuum
from echospec.feii import feii_unit_flux_template

C_KMS = 299792.458
HBETA = 4861.33
OIII_4959 = 4958.91
OIII_5007 = 5006.84
OIII_RATIO = 2.98
HALPHA = 6562.80
NII_6548 = 6548.05
NII_6584 = 6583.45
NII_RATIO = 2.96
SII_6716 = 6716.44
SII_6731 = 6730.82
OII_BLEND = 3727.50


def build_joint_model(
    wave: np.ndarray,
    stellar_model_initial: np.ndarray,
    params: Any,
    components_config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build the iterative-joint total model and component dictionary."""

    config = dict(components_config or {})
    feii_template = config.get("feii_template")
    include_feii = bool(config.get("include_feii", feii_template is not None))
    include_broad_hb = bool(config.get("include_broad_hb", True))
    include_broad_ha = bool(config.get("include_broad_ha", True))
    include_oiii_wing = bool(config.get("include_oiii_wing", True))

    wave = np.asarray(wave, dtype=float)
    stellar_shape = np.asarray(stellar_model_initial, dtype=float)
    components: dict[str, np.ndarray] = {}
    components["stellar"] = _value(params, "stellar_scale", 1.0) * stellar_shape
    components["powerlaw"] = powerlaw_continuum(
        wave,
        _value(params, "pl_amp", 0.0),
        _value(params, "pl_alpha", -1.0),
    )
    if include_feii and feii_template is not None:
        components["feii"] = _feii_component(
            wave,
            feii_template,
            _value(params, "feii_flux_4434_4684", _value(params, "feii_amp", 0.0)),
            _value(params, "feii_shift_kms", _value(params, "feii_velocity", 0.0)),
            _value(params, "feii_extra_sigma_kms", 0.0),
            _value(params, "feii_smooth_sigma_kms", 500.0),
        )
    else:
        components["feii"] = np.zeros_like(wave)

    hb_velocity = _value(params, "hb_velocity", 0.0)
    hb_sigma = _value(params, "hb_sigma", 200.0)
    components["hbeta_narrow"] = _gaussian_integrated(
        wave, HBETA, _value(params, "hb_narrow_flux", 0.0), hb_velocity, hb_sigma
    )
    components["oiii_core"] = (
        _gaussian_integrated(
            wave,
            OIII_4959,
            _value(params, "oiii_4959_flux", 0.0),
            hb_velocity,
            hb_sigma,
        )
        + _gaussian_integrated(
            wave,
            OIII_5007,
            _value(params, "oiii_5007_flux", OIII_RATIO * _value(params, "oiii_4959_flux", 0.0)),
            hb_velocity,
            hb_sigma,
        )
    )
    components["oiii_wing"] = np.zeros_like(wave)
    if include_oiii_wing:
        components["oiii_wing"] = (
            _gaussian_integrated(
                wave,
                OIII_4959,
                _value(params, "oiii_4959_wing_flux", 0.0),
                _value(params, "oiii_wing_velocity", -200.0),
                _value(params, "oiii_wing_sigma", 600.0),
            )
            + _gaussian_integrated(
                wave,
                OIII_5007,
                _value(params, "oiii_5007_wing_flux", OIII_RATIO * _value(params, "oiii_4959_wing_flux", 0.0)),
                _value(params, "oiii_wing_velocity", -200.0),
                _value(params, "oiii_wing_sigma", 600.0),
            )
        )
    components["hbeta_broad_1"] = np.zeros_like(wave)
    components["hbeta_broad_2"] = np.zeros_like(wave)
    if include_broad_hb:
        components["hbeta_broad_1"] = _gaussian_integrated(
            wave,
            HBETA,
            _value(params, "hb_broad_1_flux", 0.0),
            _value(params, "hb_broad_1_velocity", 0.0),
            _value(params, "hb_broad_1_sigma", 1500.0),
        )
        components["hbeta_broad_2"] = _gaussian_integrated(
            wave,
            HBETA,
            _value(params, "hb_broad_2_flux", 0.0),
            _value(params, "hb_broad_2_velocity", 0.0),
            _value(params, "hb_broad_2_sigma", 4000.0),
        )
    components["hbeta_broad_total"] = components["hbeta_broad_1"] + components["hbeta_broad_2"]

    ha_velocity = _value(params, "ha_velocity", 0.0)
    ha_sigma = _value(params, "ha_sigma", 200.0)
    components["halpha_narrow"] = _gaussian_integrated(
        wave, HALPHA, _value(params, "ha_narrow_flux", 0.0), ha_velocity, ha_sigma
    )
    components["nii"] = (
        _gaussian_integrated(
            wave, NII_6548, _value(params, "nii_6548_flux", 0.0), ha_velocity, ha_sigma
        )
        + _gaussian_integrated(
            wave,
            NII_6584,
            _value(params, "nii_6584_flux", NII_RATIO * _value(params, "nii_6548_flux", 0.0)),
            ha_velocity,
            ha_sigma,
        )
    )
    components["sii"] = (
        _gaussian_integrated(
            wave, SII_6716, _value(params, "sii_6716_flux", 0.0), ha_velocity, ha_sigma
        )
        + _gaussian_integrated(
            wave, SII_6731, _value(params, "sii_6731_flux", 0.0), ha_velocity, ha_sigma
        )
    )
    components["halpha_broad_1"] = np.zeros_like(wave)
    components["halpha_broad_2"] = np.zeros_like(wave)
    if include_broad_ha:
        components["halpha_broad_1"] = _gaussian_integrated(
            wave,
            HALPHA,
            _value(params, "ha_broad_1_flux", 0.0),
            _value(params, "ha_broad_1_velocity", 0.0),
            _value(params, "ha_broad_1_sigma", 1500.0),
        )
        components["halpha_broad_2"] = _gaussian_integrated(
            wave,
            HALPHA,
            _value(params, "ha_broad_2_flux", 0.0),
            _value(params, "ha_broad_2_velocity", 0.0),
            _value(params, "ha_broad_2_sigma", 4000.0),
        )
    components["halpha_broad_total"] = components["halpha_broad_1"] + components["halpha_broad_2"]
    components["oii"] = _gaussian_integrated(
        wave,
        OII_BLEND,
        _value(params, "oii_3727_blend_flux", 0.0),
        _value(params, "oii_velocity", 0.0),
        _value(params, "oii_sigma", 180.0),
    )

    total = np.zeros_like(wave)
    for values in components.values():
        total = total + values
    return total, components


def _value(params: Any, name: str, default: float) -> float:
    if hasattr(params, "__contains__") and name in params:
        item = params[name]
        return float(getattr(item, "value", item))
    return float(default)


def _gaussian_integrated(
    wave: np.ndarray,
    rest_wave: float,
    flux: float,
    velocity: float,
    sigma_velocity: float,
) -> np.ndarray:
    if flux <= 0.0 or sigma_velocity <= 0.0:
        return np.zeros_like(wave)
    center = rest_wave * (1.0 + velocity / C_KMS)
    sigma_wave = center * sigma_velocity / C_KMS
    norm = flux / (sigma_wave * np.sqrt(2.0 * np.pi))
    return norm * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)


def _feii_component(
    wave: np.ndarray,
    template: tuple[np.ndarray, np.ndarray],
    flux_4434_4684: float,
    shift_kms: float,
    extra_sigma_kms: float,
    smooth_sigma_kms: float,
) -> np.ndarray:
    if flux_4434_4684 <= 0.0:
        return np.zeros_like(wave)
    template_wave, template_flux = template
    # The empirical Fe II template is already broadened. The joint model uses
    # the same DASpec-like normalization as the H-beta fitter: integrated Fe II
    # flux in 4434-4684 A, with only low-pass smoothing plus optional extra
    # convolution in log-lambda velocity space.
    model_sigma_kms = max(float(smooth_sigma_kms), 0.0) + max(float(extra_sigma_kms), 0.0)
    try:
        unit, _diagnostics = feii_unit_flux_template(
            wave,
            template_wave,
            template_flux,
            width_kms=model_sigma_kms,
            shift_kms=shift_kms,
            feii_broadening_mode="extra",
            return_diagnostics=True,
        )
    except Exception:
        return np.zeros_like(wave)
    return flux_4434_4684 * unit


__all__ = ["build_joint_model"]
