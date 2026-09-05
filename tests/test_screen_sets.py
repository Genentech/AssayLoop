"""Screen-set naming and backwards-compatibility tests."""

from assayloop.tasks import screen_sets


def _record_calls(monkeypatch):
    calls = []

    def fake_load_screens(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(screen_sets, "_load_screens", fake_load_screens)
    return calls


def test_paper_set_names_resolve_to_curated_manifests(monkeypatch):
    calls = _record_calls(monkeypatch)

    screen_sets.load_screens()
    screen_sets.load_screens(target_set="paper_validation")

    assert calls == [
        (("assayloop-test",), {"strict": None}),
        (("assayloop-validation",), {"strict": None}),
    ]


def test_complete_fold_names_resolve_to_splits(monkeypatch):
    calls = _record_calls(monkeypatch)

    for name in ("train", "validation", "test"):
        screen_sets.load_screens(target_set=name)

    assert calls == [
        ((), {"split_value": "train", "strict": None}),
        ((), {"split_value": "validation", "strict": None}),
        ((), {"split_value": "test", "strict": None}),
    ]


def test_legacy_public_names_remain_aliases(monkeypatch):
    calls = _record_calls(monkeypatch)

    for name in (
        "public",
        "public_validation",
        "public_train",
        "public_val",
        "public_test",
    ):
        screen_sets.load_screens(target_set=name)

    assert calls == [
        (("assayloop-test",), {"strict": None}),
        (("assayloop-validation",), {"strict": None}),
        ((), {"split_value": "train", "strict": None}),
        ((), {"split_value": "validation", "strict": None}),
        ((), {"split_value": "test", "strict": None}),
    ]


def test_explicit_names_use_the_named_sets_fold(monkeypatch):
    calls = _record_calls(monkeypatch)

    screen_sets.load_screens(
        target_set="paper_validation", dataset_names=["example"], strict=True
    )

    assert calls == [
        ((), {"dataset_names": ["example"], "split_value": "validation", "strict": True})
    ]
