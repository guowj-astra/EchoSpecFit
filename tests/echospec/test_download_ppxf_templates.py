"""Tests for the pPXF template preparation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import zipfile


def _load_script_module():
    script_path = Path(__file__).parents[2] / "scripts" / "download_ppxf_templates.py"
    spec = importlib.util.spec_from_file_location("download_ppxf_templates", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_local_archive_collects_supported_files(tmp_path) -> None:
    module = _load_script_module()
    archive_path = tmp_path / "templates.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("grid/model_1.fits", "fits")
        archive.writestr("grid/model_2.fits.gz", "fits gz")
        archive.writestr("grid/readme.md", "ignore me")

    template_dir = tmp_path / "templates" / "miles"
    manifest = module.prepare_local_archive("miles", archive_path, template_dir)

    assert manifest["library"] == "miles"
    assert manifest["number_of_files"] == 2
    assert manifest["file_extensions"] == [".fits", ".fits.gz"]
    assert (template_dir / "grid" / "model_1.fits").exists()
    assert (template_dir / "grid" / "model_2.fits.gz").exists()
    assert (template_dir / "original_archives" / "templates.zip").exists()


def test_prepare_demo_library_writes_npz_templates(tmp_path) -> None:
    module = _load_script_module()
    template_dir = tmp_path / "templates" / "demo"

    manifest = module.prepare_demo_library(template_dir)

    output_path = template_dir / "demo_templates.npz"
    assert manifest["library"] == "demo"
    assert manifest["template_count"] >= 4
    assert output_path.exists()

    import numpy as np

    with np.load(output_path) as data:
        assert "template_wave" in data
        assert "template_flux" in data
        assert data["template_wave"].ndim == 1
        assert data["template_flux"].ndim == 2
        assert data["template_flux"].shape[0] == data["template_wave"].shape[0]
