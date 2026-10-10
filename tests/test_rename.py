"""Public-name compatibility checks independent of external template libraries."""

from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import desipecfit
import echospecfit
from echospecfit import cli


def test_configuration_and_result_aliases_are_identical():
    assert echospecfit.EchoSpecFitConfig is echospecfit.DesiPecFitConfig
    assert echospecfit.EchoSpecFitResult is echospecfit.DesiPecFitResult
    assert desipecfit.DesiPecFitConfig is echospecfit.EchoSpecFitConfig
    assert desipecfit.DesiPecFitResult is echospecfit.EchoSpecFitResult
    assert asdict(desipecfit.DesiPecFitConfig()) == asdict(
        echospecfit.EchoSpecFitConfig()
    )
    assert echospecfit.EchoSpecFitConfig.__name__ == "EchoSpecFitConfig"
    assert echospecfit.EchoSpecFitResult.__name__ == "EchoSpecFitResult"


def test_legacy_submodule_monkeypatches_share_canonical_modules(monkeypatch):
    for name in ("pipeline", "spectrum", "staged"):
        canonical = import_module(f"echospecfit.{name}")
        legacy = import_module(f"desipecfit.{name}")
        assert legacy is canonical

    canonical = import_module("echospecfit.pipeline")
    legacy = import_module("desipecfit.pipeline")
    marker = object()
    monkeypatch.setattr(legacy, "fit_h1spectrum", marker)
    assert canonical.fit_h1spectrum is marker


def test_renamed_cli_uses_canonical_defaults_and_preserves_csv_name(
    monkeypatch, capsys
):
    captured = {}

    def fake_fit_directory(input_path, outdir, config, **kwargs):
        captured.update(input=input_path, outdir=outdir, config=config, **kwargs)
        return [{}]

    monkeypatch.setattr(cli, "fit_directory", fake_fit_directory)
    cli.main(["example.txt"])

    assert captured["input"] == Path("example.txt")
    assert captured["outdir"] == Path("echospecfit_output")
    assert isinstance(captured["config"], echospecfit.EchoSpecFitConfig)
    assert captured["pattern"] == "*.txt"
    assert captured["z"] == 0.0
    assert captured["survey"] == "DESI"
    output = capsys.readouterr().out
    assert "echospecfit_output/desipecfit_epoch_fit.csv" in output
    assert "Processed 1 spectrum/s" in output


def test_legacy_cli_calls_canonical_main():
    assert import_module("desipecfit.cli").main is cli.main
