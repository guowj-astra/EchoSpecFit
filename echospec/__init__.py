"""EchoSpec: differential emission-line fitting for repeat AGN spectra."""

from echospec.io_fits import (
    SavedFitResult,
    attach_spectrum_to_result,
    load_fit_result_from_fits,
    plot_fit_result_pdf,
    save_fit_result_to_fits,
)
from echospec.spectrum import AGNSpectrum

__version__ = "0.1.0"

__all__ = [
    "AGNSpectrum",
    "SavedFitResult",
    "__version__",
    "attach_spectrum_to_result",
    "load_fit_result_from_fits",
    "plot_fit_result_pdf",
    "save_fit_result_to_fits",
]
