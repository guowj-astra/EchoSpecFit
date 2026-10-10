"""Prepare local MILES/E-MILES stellar templates for EchoSpec pPXF fits.

The first version is intentionally semi-automatic: the MILES/E-MILES web pages
offer multiple SSP model grids and package layouts, so this script prints the
official download locations and can unpack a user-downloaded archive into the
standard EchoSpec template cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np

MILES_SSP_URL = "https://research.iac.es/proyecto/miles/pages/ssp-models.php"
EMILES_SED_URL = (
    "https://research.iac.es/proyecto/miles/pages/"
    "spectral-energy-distributions-seds/e-miles.php"
)
DEFAULT_TEMPLATE_ROOT = Path(
    os.environ.get("ECHOSPECFIT_TEMPLATE_DIR", str(Path.home() / ".echospec" / "templates"))
).expanduser()
LIBRARY_URLS = {
    "miles": MILES_SSP_URL,
    "emiles": EMILES_SED_URL,
    "emiles_padova00_ch": EMILES_SED_URL,
    "emiles_basti_ch": EMILES_SED_URL,
    "demo": "pPXF package examples or synthetic EchoSpec demo templates",
}
COLLECTED_SUFFIXES = (
    ".fits",
    ".fit",
    ".fits.gz",
    ".dat",
    ".txt",
    ".npy",
    ".npz",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        required=True,
        choices=sorted(LIBRARY_URLS),
        help="Template library to prepare.",
    )
    parser.add_argument(
        "--local-archive",
        type=Path,
        default=None,
        help="Path to a user-downloaded MILES/E-MILES zip or tar.gz archive.",
    )
    parser.add_argument(
        "--template-root",
        type=Path,
        default=DEFAULT_TEMPLATE_ROOT,
        help="Root directory for EchoSpec stellar templates.",
    )
    args = parser.parse_args()

    template_dir = args.template_root.expanduser() / args.library
    template_dir.mkdir(parents=True, exist_ok=True)

    if args.library == "demo" and args.local_archive is None:
        manifest = prepare_demo_library(template_dir)
        manifest_path = template_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest: {manifest_path}")
        print("WARNING: demo templates are for pipeline testing only, not for science.")
    elif args.local_archive is None:
        _print_manual_instructions(args.library, template_dir)
    else:
        manifest = prepare_local_archive(
            library=args.library,
            local_archive=args.local_archive,
            template_dir=template_dir,
        )
        manifest_path = template_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Wrote manifest: {manifest_path}")

    print(f"Template files prepared in {template_dir}")


def prepare_demo_library(template_dir: Path) -> dict[str, object]:
    """Prepare a demo template library for software testing only."""

    template_dir.mkdir(parents=True, exist_ok=True)
    output_path = template_dir / "demo_templates.npz"
    source = "synthetic"

    loaded = _try_load_ppxf_example_templates()
    if loaded is None:
        template_wave, template_flux = _make_synthetic_demo_templates()
    else:
        template_wave, template_flux, source = loaded

    np.savez_compressed(
        output_path,
        template_wave=template_wave,
        template_flux=template_flux,
        warning="demo templates are for pipeline testing only, not for science",
        source=source,
    )

    return {
        "library": "demo",
        "source_url": LIBRARY_URLS["demo"],
        "local_path": str(output_path),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "number_of_files": 1,
        "file_extensions": [".npz"],
        "template_count": int(template_flux.shape[1]),
        "wavelength_min": float(template_wave.min()),
        "wavelength_max": float(template_wave.max()),
        "warning": "demo templates are for pipeline testing only, not for science",
        "source": source,
    }


def prepare_local_archive(
    library: str,
    local_archive: str | Path,
    template_dir: Path,
) -> dict[str, object]:
    """Extract supported template files and write metadata for the archive."""

    archive_path = Path(local_archive).expanduser()
    if not archive_path.exists():
        raise FileNotFoundError(f"Local archive not found: {archive_path}")
    if not archive_path.is_file():
        raise ValueError(f"Local archive must be a file: {archive_path}")

    original_dir = template_dir / "original_archives"
    original_dir.mkdir(parents=True, exist_ok=True)
    preserved_archive = original_dir / archive_path.name
    if archive_path.resolve() != preserved_archive.resolve():
        shutil.copy2(archive_path, preserved_archive)

    extracted_files = _extract_supported_files(archive_path, template_dir)
    file_extensions = sorted({_template_extension(path) for path in extracted_files})

    return {
        "library": library,
        "source_url": LIBRARY_URLS[library],
        "local_path": str(preserved_archive),
        "created_time": datetime.now(timezone.utc).isoformat(),
        "number_of_files": len(extracted_files),
        "file_extensions": file_extensions,
    }


def _print_manual_instructions(library: str, template_dir: Path) -> None:
    print("Automatic MILES/E-MILES download is not enabled in this first version.")
    print()
    print("Official download pages:")
    print(f"MILES SSP models: {MILES_SSP_URL}")
    print(f"E-MILES models: {EMILES_SED_URL}")
    print()
    print("Please manually download the SSP FITS files or archive for:")
    print(f"  library: {library}")
    print()
    print("Then either place the files under:")
    print(f"  {template_dir}")
    print()
    print("or run:")
    print(
        "  python scripts/download_ppxf_templates.py "
        f"--library {library} --local-archive /path/to/downloaded_templates.zip"
    )


def _try_load_ppxf_example_templates() -> tuple[np.ndarray, np.ndarray, str] | None:
    """Try to reuse package-accessible pPXF example templates if available."""

    ppxf_spec = importlib.util.find_spec("ppxf")
    if ppxf_spec is None or ppxf_spec.origin is None:
        return None

    ppxf_dir = Path(ppxf_spec.origin).resolve().parent
    for path in sorted(ppxf_dir.rglob("*.npz")):
        try:
            with np.load(path) as data:
                wave_key = _first_existing_key(data, ("template_wave", "wave", "lam"))
                flux_key = _first_existing_key(
                    data,
                    ("template_flux", "templates", "flux"),
                )
                if wave_key is None or flux_key is None:
                    continue
                template_wave = np.asarray(data[wave_key], dtype=float)
                template_flux = np.asarray(data[flux_key], dtype=float)
        except (OSError, ValueError):
            continue

        template_flux = _as_template_matrix(template_flux, template_wave.size)
        if not _usable_demo_grid(template_wave, template_flux):
            continue
        return template_wave, template_flux, str(path)

    return None


def _make_synthetic_demo_templates(
    n_pixels: int = 4000,
    n_templates: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    wave = np.linspace(3500.0, 7500.0, n_pixels)
    centers = np.array(
        [
            3933.66,
            3968.47,
            4101.74,
            4304.0,
            4861.33,
            5175.0,
            5892.0,
            6562.80,
        ]
    )
    templates: list[np.ndarray] = []
    scaled_wave = (wave - wave.mean()) / np.ptp(wave)
    for index in range(n_templates):
        age_like = index / max(n_templates - 1, 1)
        slope = -0.55 + 1.1 * age_like
        curvature = 0.18 * np.cos(index)
        continuum = 1.0 + slope * scaled_wave + curvature * scaled_wave**2
        template = continuum.copy()
        for line_index, center in enumerate(centers):
            width = 2.0 + 0.55 * line_index + 1.6 * age_like
            depth = 0.035 + 0.13 * age_like
            if center in (4861.33, 6562.80):
                depth *= 0.45
            template -= depth * np.exp(-0.5 * ((wave - center) / width) ** 2)
        template += 0.012 * np.sin(wave / (120.0 + 12.0 * index))
        template = np.clip(template, 0.02, None)
        template /= np.nanmedian(template)
        templates.append(template)
    return wave, np.column_stack(templates)


def _first_existing_key(container, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in container:
            return key
    return None


def _as_template_matrix(template_flux: np.ndarray, n_pixels: int) -> np.ndarray:
    if template_flux.ndim == 1:
        return template_flux[:, None]
    if template_flux.ndim == 2:
        if template_flux.shape[0] == n_pixels:
            return template_flux
        if template_flux.shape[1] == n_pixels:
            return template_flux.T
    if template_flux.ndim == 3 and template_flux.shape[0] == n_pixels:
        return template_flux.reshape(n_pixels, -1)
    raise ValueError("Cannot reshape pPXF example templates into a 2D matrix.")


def _usable_demo_grid(template_wave: np.ndarray, template_flux: np.ndarray) -> bool:
    if template_wave.ndim != 1 or template_flux.ndim != 2:
        return False
    if template_wave.size != template_flux.shape[0] or template_flux.shape[1] < 4:
        return False
    if not np.all(np.isfinite(template_wave)) or not np.all(np.diff(template_wave) > 0):
        return False
    return template_wave.min() <= 3500.0 and template_wave.max() >= 7500.0


def _extract_supported_files(archive_path: Path, template_dir: Path) -> list[Path]:
    template_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        return _extract_zip_supported_files(archive_path, template_dir)
    if tarfile.is_tarfile(archive_path):
        return _extract_tar_supported_files(archive_path, template_dir)
    raise ValueError(
        "Unsupported archive format. Please provide a .zip, .tar, .tar.gz, or .tgz file."
    )


def _extract_zip_supported_files(archive_path: Path, output_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not _is_supported_template_file(member.filename):
                continue
            target = _safe_member_path(output_dir, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def _extract_tar_supported_files(archive_path: Path, output_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _is_supported_template_file(member.name):
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target = _safe_member_path(output_dir, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as dst:
                shutil.copyfileobj(source, dst)
            extracted.append(target)
    return extracted


def _safe_member_path(output_dir: Path, member_name: str) -> Path:
    target = (output_dir / member_name).resolve()
    output_root = output_dir.resolve()
    if output_root not in target.parents and target != output_root:
        raise ValueError(f"Archive member would escape output directory: {member_name}")
    return target


def _is_supported_template_file(path: str) -> bool:
    lower_path = path.lower()
    return any(lower_path.endswith(suffix) for suffix in COLLECTED_SUFFIXES)


def _template_extension(path: Path) -> str:
    lower_name = path.name.lower()
    if lower_name.endswith(".fits.gz"):
        return ".fits.gz"
    return path.suffix.lower()


if __name__ == "__main__":
    main()
