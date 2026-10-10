"""EchoSpecFit single-spectrum fitting tools, powered by EchoSpec."""

from .pipeline import (
    EchoSpecFitConfig,
    EchoSpecFitResult,
    DesiPecFitConfig,
    DesiPecFitResult,
    fit_directory,
    fit_h1spectrum,
    fit_spectrum_file,
    save_fit_result,
)
from .spectrum import H1Spectrum, SingleH1Spectrum
from .staged import fit_fixed_host_feii, load_representative_stellar_templates

__version__ = "0.1.0"

__all__ = [
    "EchoSpecFitConfig",
    "EchoSpecFitResult",
    "DesiPecFitConfig",
    "DesiPecFitResult",
    "H1Spectrum",
    "SingleH1Spectrum",
    "fit_directory",
    "fit_h1spectrum",
    "fit_spectrum_file",
    "fit_fixed_host_feii",
    "load_representative_stellar_templates",
    "save_fit_result",
]
