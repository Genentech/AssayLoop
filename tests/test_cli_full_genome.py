from types import SimpleNamespace

from typer.testing import CliRunner

from assayloop.cli import app


def test_open_vocab_llm_defaults_to_shared_f2_universe(monkeypatch):
    captured = {}

    def fake_load_screens(*, target_set, dataset_names=None):
        assert target_set == "public"
        assert dataset_names is None
        return [
            SimpleNamespace(genes=["A", "B"]),
            SimpleNamespace(genes=["B", "C"]),
        ]

    def fake_run_sweep(cfg, **kwargs):
        captured["cfg"] = cfg
        return SimpleNamespace(sweep_id="test", per_screen=[], aggregate={})

    monkeypatch.setattr("assayloop.tasks.load_screens", fake_load_screens)
    monkeypatch.setattr("assayloop.experiment.runner.run_sweep", fake_run_sweep)

    result = CliRunner().invoke(app, [
        "run", "--model", "null", "--acq", "llm_single", "--no-persist",
    ])

    assert result.exit_code == 0, result.output
    assert captured["cfg"].universe_genes == ["B"]


def test_screen_library_can_override_open_vocab_default(monkeypatch):
    captured = {}

    def fake_run_sweep(cfg, **kwargs):
        captured["cfg"] = cfg
        return SimpleNamespace(sweep_id="test", per_screen=[], aggregate={})

    monkeypatch.setattr("assayloop.experiment.runner.run_sweep", fake_run_sweep)
    result = CliRunner().invoke(app, [
        "run", "--model", "null", "--acq", "llm_single",
        "--screen-library", "--no-persist",
    ])

    assert result.exit_code == 0, result.output
    assert captured["cfg"].universe_genes is None
