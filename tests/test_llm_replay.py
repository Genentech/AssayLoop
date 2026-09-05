import json

import pytest

from assayloop.amortized.warmstart import load_run_traces
from assayloop.llm.replay import (
    load_logged_response_texts,
    normalise_gene_batches,
    replay_llm_steps,
)


def test_normalise_gene_batches_validates_and_deduplicates_globally():
    batches = normalise_gene_batches(
        [["tp53", "OUTSIDE", "EGFR", "TP53"], ["egfr", "MYC"]],
        ["TP53", "EGFR", "MYC"],
        batch_size=2,
    )
    assert batches == [["TP53", "EGFR"], ["MYC"]]


def test_replay_prefers_raw_response_and_falls_back_for_legacy_step():
    steps = [
        {
            "acquired_batch": ["TP53"],
            "acquisition_trace": {"response_text": "TP53, OOL_VALID, FAKE"},
        },
        {"acquired_batch": ["EGFR"], "acquisition_trace": {}},
    ]
    assert replay_llm_steps(
        steps, ["TP53", "OOL_VALID", "EGFR"]
    ) == [["TP53", "OOL_VALID"], ["EGFR"]]


def test_replay_can_override_truncated_result_trace_with_full_log_text():
    steps = [{
        "acquired_batch": ["TP53"],
        "acquisition_trace": {"response_text": "TP53...<truncated>"},
    }]
    assert replay_llm_steps(
        steps,
        ["TP53", "OOL_VALID"],
        response_texts=["TP53, OOL_VALID"],
    ) == [["TP53", "OOL_VALID"]]


def test_replay_rejects_truncated_trace_without_full_call_log():
    steps = [{
        "acquired_batch": ["TP53"],
        "acquisition_trace": {"response_text": "TP53...<truncated>"},
    }]
    with pytest.raises(ValueError, match="llm_calls.jsonl"):
        replay_llm_steps(steps, ["TP53"])


def test_replay_preserves_published_parser_semantics():
    """A parser upgrade must not retroactively change benchmark selections."""
    steps = [{"acquired_batch": [], "acquisition_trace": {}}]
    assert replay_llm_steps(
        steps,
        ["TP53", "EGFR"],
        response_texts=["- TP53, EGFR"],
    ) == [["EGFR"]]


def test_load_logged_response_texts_requires_one_success_per_step(tmp_path):
    calls = [
        {"response_text": "ignored", "error": "timeout"},
        {"response_text": "TP53, OOL_VALID", "error": None},
    ]
    (tmp_path / "llm_calls.jsonl").write_text(
        "".join(json.dumps(call) + "\n" for call in calls)
    )
    assert load_logged_response_texts(
        tmp_path, expected_steps=1
    ) == ["TP53, OOL_VALID"]
    assert load_logged_response_texts(tmp_path, expected_steps=2) is None


def test_load_run_traces_replays_raw_response_when_universe_is_given(tmp_path):
    run_dir = tmp_path / "sweep-test-00-screen_a"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({
        "task_id": "option2/screen_a",
        "steps": [{
            "acquired_batch": ["TP53"],
            "acquisition_trace": {"response_text": "TP53, OOL_VALID"},
        }],
    }))

    legacy = load_run_traces(tmp_path, "sweep-test-")
    replayed = load_run_traces(
        tmp_path, "sweep-test-", universe_genes=["TP53", "OOL_VALID"]
    )

    assert legacy == {"screen_a": [[["TP53"]]]}
    assert replayed == {"screen_a": [[["TP53", "OOL_VALID"]]]}


def test_load_run_traces_uses_full_call_log_over_truncated_result(tmp_path):
    run_dir = tmp_path / "sweep-test-00-screen_a"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({
        "task_id": "option2/screen_a",
        "steps": [{
            "acquired_batch": ["TP53"],
            "acquisition_trace": {"response_text": "TP53...<truncated>"},
        }],
    }))
    (run_dir / "llm_calls.jsonl").write_text(json.dumps({
        "response_text": "TP53, OOL_VALID",
        "error": None,
    }) + "\n")

    replayed = load_run_traces(
        tmp_path, "sweep-test-", universe_genes=["TP53", "OOL_VALID"]
    )

    assert replayed == {"screen_a": [[["TP53", "OOL_VALID"]]]}


def test_load_run_traces_does_not_replay_internal_calls_for_greedy_model(tmp_path):
    run_dir = tmp_path / "sweep-test-00-screen_a"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(json.dumps({
        "task_id": "option2/screen_a",
        "config": {"model": "llmnn", "acquisition": "greedy"},
        "steps": [{
            "acquired_batch": ["TP53"],
            "acquisition_trace": {"top_scores": {"TP53": 1.0}},
        }],
    }))
    (run_dir / "llm_calls.jsonl").write_text(json.dumps({
        "response_text": "OOL_VALID",
        "error": None,
    }) + "\n")

    replayed = load_run_traces(
        tmp_path, "sweep-test-", universe_genes=["TP53", "OOL_VALID"]
    )

    assert replayed == {"screen_a": [[["TP53"]]]}
