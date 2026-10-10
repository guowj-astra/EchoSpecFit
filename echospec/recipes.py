"""Recipe loading and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from lmfit import Parameters, minimize

from echospec.spectrum import AGNSpectrum

C_KMS = 299792.458


@dataclass
class LineSpec:
    """Emission-line definition in a fitting recipe."""

    name: str
    restwave: float
    group: str
    flux_ratio_to: str | None = None
    flux_ratio: float | None = None
    optional: bool = False


@dataclass
class KinematicGroup:
    """Shared velocity and width constraints for a set of lines."""

    name: str
    velocity_bounds: tuple[float, float]
    sigma_bounds: tuple[float, float]


@dataclass
class Recipe:
    """Emission-line fitting recipe."""

    name: str
    rest_min: float
    rest_max: float
    continuum_type: str
    kinematic_groups: dict[str, KinematicGroup]
    lines: list[LineSpec]

    @classmethod
    def from_yaml(cls, path: str | Path, name: str | None = None) -> "Recipe":
        """Load a recipe from YAML."""

        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if "recipes" in data:
            recipe_name = name or data.get("default_recipe")
            if recipe_name is None:
                recipe_name = next(iter(data["recipes"]))
            data = data["recipes"][recipe_name]
        elif name is not None and name != data.get("name"):
            raise KeyError(f"Recipe {name!r} not found in {path}.")

        groups = {
            group["name"]: KinematicGroup(
                name=group["name"],
                velocity_bounds=(
                    float(group["velocity_bounds"][0]),
                    float(group["velocity_bounds"][1]),
                ),
                sigma_bounds=(
                    float(group["sigma_bounds"][0]),
                    float(group["sigma_bounds"][1]),
                ),
            )
            for group in data["kinematic_groups"]
        }
        lines = [
            LineSpec(
                name=line["name"],
                restwave=float(line["restwave"]),
                group=line["group"],
                flux_ratio_to=line.get("flux_ratio_to"),
                flux_ratio=(
                    None if line.get("flux_ratio") is None else float(line["flux_ratio"])
                ),
                optional=bool(line.get("optional", False)),
            )
            for line in data["lines"]
        ]
        return cls(
            name=data["name"],
            rest_min=float(data["rest_min"]),
            rest_max=float(data["rest_max"]),
            continuum_type=data["continuum_type"],
            kinematic_groups=groups,
            lines=lines,
        )


class RecipeEmissionFitter:
    """Build and run an emission-line model from a :class:`Recipe`."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe
        if recipe.continuum_type != "linear":
            raise ValueError("Only linear continua are currently supported.")

    def fit(self, spec: AGNSpectrum, include_optional: bool = True):
        """Fit the recipe to a rest-frame spectrum."""

        from echospec.fitting import EmissionFitResult

        window_spec = spec.select_window(self.recipe.rest_min, self.recipe.rest_max)
        window_spec = window_spec.mask_bad_pixels()
        if len(window_spec.wave) == 0:
            raise ValueError(
                f"No valid pixels available in the {self.recipe.rest_min:g}-"
                f"{self.recipe.rest_max:g} Å fitting window."
            )

        active_lines = self._active_lines(include_optional)
        params = self._initial_parameters(window_spec.wave, window_spec.flux, active_lines)
        fit = minimize(
            self._residual,
            params,
            args=(window_spec.wave, window_spec.flux, window_spec.err, active_lines),
        )
        model_flux = self._model_flux(window_spec.wave, fit.params, active_lines)
        continuum_flux = self._continuum(window_spec.wave, fit.params)
        residual_flux = window_spec.flux - model_flux
        chi2 = float(np.sum((residual_flux / window_spec.err) ** 2))
        component_flux = self._component_flux(window_spec.wave, fit.params, active_lines)

        return EmissionFitResult(
            success=bool(fit.success),
            message=str(fit.message),
            params={name: float(param.value) for name, param in fit.params.items()},
            line_flux={
                line.name: float(fit.params[self._flux_param(line.name)].value)
                for line in active_lines
            },
            line_flux_err={},
            model_flux=model_flux,
            continuum_flux=continuum_flux,
            residual_flux=residual_flux,
            chi2=chi2,
            redchi=float(fit.redchi),
            ndata=int(fit.ndata),
            nvarys=int(fit.nvarys),
            window_name=self.recipe.name,
            component_flux=component_flux,
        )

    def _active_lines(self, include_optional: bool) -> list[LineSpec]:
        return [
            line
            for line in self.recipe.lines
            if include_optional or not line.optional
        ]

    def _initial_parameters(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        lines: list[LineSpec],
    ) -> Parameters:
        cont_intercept, cont_slope = self._estimate_continuum(wave, flux, lines)
        residual_flux = flux - (cont_intercept + cont_slope * (wave - self._pivot()))

        params = Parameters()
        params.add("cont_intercept", value=cont_intercept)
        params.add("cont_slope", value=cont_slope)
        active_groups = {line.group for line in lines}
        for group in self.recipe.kinematic_groups.values():
            if group.name not in active_groups:
                continue
            params.add(
                self._velocity_param(group.name),
                value=0.0,
                min=group.velocity_bounds[0],
                max=group.velocity_bounds[1],
            )
            sigma_value = max(group.sigma_bounds[0], min(200.0, group.sigma_bounds[1]))
            if group.sigma_bounds[0] >= 800.0:
                sigma_value = max(group.sigma_bounds[0], min(1500.0, group.sigma_bounds[1]))
            params.add(
                self._sigma_param(group.name),
                value=sigma_value,
                min=group.sigma_bounds[0],
                max=group.sigma_bounds[1],
            )

        active_names = {line.name for line in lines}
        for line in lines:
            flux_param = self._flux_param(line.name)
            if line.flux_ratio_to is not None:
                if line.flux_ratio_to not in active_names:
                    continue
                params.add(
                    flux_param,
                    expr=f"{float(line.flux_ratio)} * {self._flux_param(line.flux_ratio_to)}",
                )
                continue

            half_width = 35.0 if line.optional else 8.0
            value = max(
                self._estimate_line_flux(wave, residual_flux, line.restwave, half_width),
                1.0e-12,
            )
            params.add(flux_param, value=value, min=0.0)
        return params

    def _model_flux(
        self,
        wave: np.ndarray,
        params: Parameters,
        lines: list[LineSpec],
    ) -> np.ndarray:
        model = self._continuum(wave, params)
        for line in lines:
            flux_param = self._flux_param(line.name)
            if flux_param not in params:
                continue
            model += self._gaussian_integrated(
                wave=wave,
                restwave=line.restwave,
                flux=params[flux_param].value,
                velocity=params[self._velocity_param(line.group)].value,
                sigma_velocity=params[self._sigma_param(line.group)].value,
            )
        return model

    def _component_flux(
        self,
        wave: np.ndarray,
        params: Parameters,
        lines: list[LineSpec],
    ) -> dict[str, np.ndarray]:
        components: dict[str, np.ndarray] = {
            "continuum": self._continuum(wave, params),
        }
        for line in lines:
            flux_param = self._flux_param(line.name)
            if flux_param not in params:
                continue
            components[line.name] = self._gaussian_integrated(
                wave=wave,
                restwave=line.restwave,
                flux=params[flux_param].value,
                velocity=params[self._velocity_param(line.group)].value,
                sigma_velocity=params[self._sigma_param(line.group)].value,
            )
        return components

    def _residual(
        self,
        params: Parameters,
        wave: np.ndarray,
        flux: np.ndarray,
        err: np.ndarray,
        lines: list[LineSpec],
    ) -> np.ndarray:
        return (flux - self._model_flux(wave, params, lines)) / err

    def _continuum(self, wave: np.ndarray, params: Parameters) -> np.ndarray:
        return params["cont_intercept"].value + params["cont_slope"].value * (
            wave - self._pivot()
        )

    def _estimate_continuum(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        lines: list[LineSpec],
    ) -> tuple[float, float]:
        line_mask = np.zeros_like(wave, dtype=bool)
        for line in lines:
            line_mask |= np.abs(wave - line.restwave) < 25.0
        continuum_mask = ~line_mask
        if np.count_nonzero(continuum_mask) >= 2:
            slope, intercept_at_zero = np.polyfit(
                wave[continuum_mask], flux[continuum_mask], 1
            )
            return float(intercept_at_zero + slope * self._pivot()), float(slope)
        return float(np.nanmedian(flux)), 0.0

    @staticmethod
    def _estimate_line_flux(
        wave: np.ndarray,
        residual_flux: np.ndarray,
        restwave: float,
        half_width: float,
    ) -> float:
        mask = np.abs(wave - restwave) <= half_width
        if np.count_nonzero(mask) < 2:
            return 0.0
        positive_residual = np.clip(residual_flux[mask], 0.0, None)
        return float(np.trapz(positive_residual, wave[mask]))

    @staticmethod
    def _gaussian_integrated(
        wave: np.ndarray,
        restwave: float,
        flux: float,
        velocity: float,
        sigma_velocity: float,
    ) -> np.ndarray:
        center = restwave * (1.0 + velocity / C_KMS)
        sigma_wave = center * sigma_velocity / C_KMS
        norm = flux / (sigma_wave * np.sqrt(2.0 * np.pi))
        return norm * np.exp(-0.5 * ((wave - center) / sigma_wave) ** 2)

    def _pivot(self) -> float:
        return 0.5 * (self.recipe.rest_min + self.recipe.rest_max)

    @staticmethod
    def _velocity_param(group: str) -> str:
        return f"{group}_velocity"

    @staticmethod
    def _sigma_param(group: str) -> str:
        return f"{group}_sigma"

    @staticmethod
    def _flux_param(line_name: str) -> str:
        return f"{line_name}_flux"


__all__ = ["LineSpec", "KinematicGroup", "Recipe", "RecipeEmissionFitter"]
