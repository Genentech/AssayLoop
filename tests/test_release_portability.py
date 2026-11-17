import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('bundle', ROOT / 'scripts/build_sweep_bundle.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


def test_redaction_scans_past_200_entries_without_printing_secret():
    secret = 'fake-private-value-for-testing'
    obj = [{} for _ in range(250)] + [{'api_key': secret}]
    problems = bundle.deny_scan_tree(obj, 'fixture')
    assert problems and all(secret not in p for p in problems)
    cleaned = bundle.checked_scrub(obj, {}, {})
    assert cleaned[-1]['api_key'] == 'REDACTED'
    assert not bundle.deny_scan_tree(cleaned, 'fixture')


def test_scientific_content_and_historical_prompt_survive_redaction():
    original = {'prompt': 'Historical template. Predict TP53 and EGFR. /cv/home/tester/cache',
                'response_text': 'TP53, EGFR', 'metrics': {'ef': 4.2},
                'acquired_batch': ['TP53', 'EGFR'], 'api_key': 'not-a-real-key'}
    audit = {}
    cleaned = bundle.checked_scrub(original, {}, audit)
    assert cleaned['prompt'] == 'Historical template. Predict TP53 and EGFR. <redacted-internal-path>'
    assert cleaned['response_text'] == original['response_text']
    assert cleaned['metrics'] == original['metrics']
    assert cleaned['acquired_batch'] == original['acquired_batch']
    assert audit['protected_fields_verified'] == 2
    assert not bundle.deny_scan(json.dumps(cleaned).encode(), 'fixture')
    with pytest.raises(ValueError, match='protected field'):
        bundle.verify_redaction(original, {**cleaned, 'acquired_batch': ['MYC']}, {})
    with pytest.raises(ValueError, match='parsed response genes'):
        bundle.verify_redaction(original, {**cleaned, 'response_text': 'MYC'}, {})


def test_deny_scan_does_not_echo_secret():
    secret = 'sk-' + 'a' * 24
    problems = bundle.deny_scan(secret.encode(), 'fixture')
    assert problems and secret not in str(problems)


def archive(path, files):
    with tarfile.open(path, 'w:gz') as tf:
        for name, content in files.items():
            data = content.encode()
            info = tarfile.TarInfo('bundle/' + name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fallback', [False, True])
def test_fetch_download_verify_merge_skip_and_reject_corruption(tmp_path, fallback):
    mirror = tmp_path / 'mirror'; mirror.mkdir()
    one = mirror / 'assayloop-sweeps-v1.tar.gz'
    two = mirror / 'assayloop-llm-calls-v1.tar.gz'
    hashes = [archive(one, {'MANIFEST.json': '{}', 'sweeps/s/sweep.json': '{}', 'runs/r/result.json': '{}'}),
              archive(two, {'MANIFEST-llm-calls.json': '{}', 'runs/r/llm_calls.jsonl': '{}\n'})]
    script = (ROOT / 'scripts/fetch_sweeps.sh').read_text()
    expected = re.findall(r'"([a-f0-9]{64})"', script)
    assert len(expected) == 2
    for old, new in zip(expected, hashes):
        script = script.replace(old, new)
    target_script = tmp_path / 'fetch.sh'; target_script.write_text(script)
    commands = tmp_path / 'bin'; commands.mkdir()
    for name in ['curl', 'tar', 'gzip', 'cut', 'mktemp', 'mkdir', 'rm', 'dirname', 'find', 'wc', 'sed'] + (['shasum'] if fallback else ['sha256sum']):
        source = shutil.which(name)
        if source is None:
            pytest.skip(f'{name} unavailable')
        (commands / name).symlink_to(source)
    out = tmp_path / 'out'; scratch = tmp_path / 'scratch'; scratch.mkdir()
    env = {**os.environ, 'ASSAYLOOP_PUBLISHED': str(out), 'ASSAYLOOP_BUNDLE_URL': mirror.as_uri(),
           'TMPDIR': str(scratch), 'PATH': str(commands)}
    bash = os.environ.get('TEST_BASH', '/bin/bash')
    def run(*args):
        return subprocess.run([bash, str(target_script), *args], env=env, text=True, capture_output=True)
    first = run()
    assert first.returncode == 0, first.stderr
    assert (out / 'runs/r/result.json').exists() and (out / 'runs/r/llm_calls.jsonl').exists()
    assert (out / 'MANIFEST.json').exists() and (out / 'MANIFEST-llm-calls.json').exists()
    assert not list(scratch.iterdir())
    assert '[skip]' in run().stdout
    one.write_bytes(b'corrupt')
    bad = run('--force')
    assert bad.returncode != 0 and 'sha256 mismatch' in bad.stderr
    assert not list(scratch.iterdir())
    one.unlink()
    missing = run('--force')
    assert missing.returncode != 0 and 'could not download' in missing.stderr
    assert not list(scratch.iterdir())


def test_bundle_requires_llm_calls_and_is_reproducible(tmp_path, monkeypatch):
    from types import SimpleNamespace

    root = tmp_path / "source"
    sweep_dir = root / "sweeps" / "sweep-test"
    run_dir = root / "runs" / "run-test"
    sweep_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    sweep = {"config": {"acq": "llm_single"},
             "per_screen": [{"run_id": "run-test"}], "aggregate": {"ef": 2.0}}
    (sweep_dir / "sweep.json").write_text(json.dumps(sweep))
    (run_dir / "result.json").write_text(json.dumps({
        "final_metrics": {"ef": 2.0}, "steps": [{"acquired_batch": ["TP53"]}]}))
    seen = []
    def required(roots):
        seen.extend(roots)
        return [("sweep-test", "Test LLM")]
    monkeypatch.setattr(bundle, "required_sweep_ids", required)
    args = SimpleNamespace(results=str(root), shared=None, with_llm_calls=True,
                           out=str(tmp_path / "release"))
    with pytest.raises(SystemExit, match="missing llm_calls"):
        bundle.build(args)
    assert seen == [root]
    (run_dir / "llm_calls.jsonl").write_text(json.dumps({
        "response_text": "TP53", "api_key": "test-only-secret"}) + "\n")
    assert bundle.build(args) == 0
    archives = [Path(args.out) / bundle.SWEEP_ARCHIVE, Path(args.out) / bundle.CALLS_ARCHIVE]
    hashes = [bundle.sha256_file(p) for p in archives]
    assert bundle.build(args) == 0
    assert hashes == [bundle.sha256_file(p) for p in archives]
    for p in archives:
        with tarfile.open(p) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    assert b"test-only-secret" not in tf.extractfile(member).read()
