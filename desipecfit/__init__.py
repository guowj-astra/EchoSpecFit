"""Compatibility imports for scripts using the former DESIPecFit name.

The pipeline, spectrum, and staged modules are aliases of their canonical
EchoSpecFit modules so that imports and monkeypatches share the same objects.
"""

from importlib import import_module
import sys

from echospecfit import *  # noqa: F401,F403
from echospecfit import __all__, __version__

for _module_name in ("pipeline", "spectrum", "staged"):
    _module = import_module(f"echospecfit.{_module_name}")
    sys.modules[f"{__name__}.{_module_name}"] = _module
    globals()[_module_name] = _module

del _module_name, _module
