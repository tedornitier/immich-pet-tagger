"""Tests for inference_session refcounting and the run_scan subprocess protocol."""
import sys
import threading
import time

import pytest

import inference


class _FakeWorkers:
    """Records start/stop/wait calls in order, optionally failing on wait."""

    def __init__(self, name, calls, fail_wait=False):
        self.name = name
        self.calls = calls
        self.fail_wait = fail_wait

    def start_workers(self):
        self.calls.append(f"{self.name}.start")

    def stop_workers(self):
        self.calls.append(f"{self.name}.stop")

    def wait_for_ready(self, timeout=300):
        self.calls.append(f"{self.name}.wait")
        if self.fail_wait:
            raise RuntimeError(f"{self.name} failed to load")


@pytest.fixture
def fake_workers(monkeypatch):
    calls = []
    det = _FakeWorkers("det", calls)
    emb = _FakeWorkers("emb", calls)
    monkeypatch.setattr(inference, "det", det)
    monkeypatch.setattr(inference, "emb", emb)
    return calls, det, emb


def test_session_loads_then_unloads(fake_workers):
    calls, _, _ = fake_workers
    with inference.inference_session():
        assert "det.start" in calls and "emb.start" in calls
        assert "det.stop" not in calls
        assert inference.is_active()
    assert calls[-2:] == ["det.stop", "emb.stop"]
    assert not inference.is_active()


def test_nested_sessions_load_and_unload_once(fake_workers):
    calls, _, _ = fake_workers
    with inference.inference_session():
        with inference.inference_session():
            pass
        assert "det.stop" not in calls
    assert calls.count("det.start") == 1
    assert calls.count("det.stop") == 1


def test_failed_load_stops_workers_and_allows_retry(fake_workers):
    calls, _, emb = fake_workers
    emb.fail_wait = True
    with pytest.raises(RuntimeError, match="emb failed to load"):
        with inference.inference_session():
            pass
    assert "det.stop" in calls and "emb.stop" in calls
    assert not inference.is_active()

    emb.fail_wait = False
    calls.clear()
    with inference.inference_session():
        pass
    assert "det.start" in calls


def _run_with_stub(tmp_path, monkeypatch, stub_body, **kwargs):
    stub = tmp_path / "stub_worker.py"
    stub.write_text("import sys, json\nargs = json.loads(sys.stdin.read())\n" + stub_body)
    monkeypatch.setattr(inference, "_SCAN_WORKER", str(stub))
    return inference.run_scan(str(tmp_path), **kwargs)


def test_run_scan_returns_counts_and_relays_progress(tmp_path, monkeypatch):
    dates, counts_seen = [], []
    counts, low_conf = _run_with_stub(
        tmp_path, monkeypatch,
        'print(json.dumps({"type": "date", "date": "2026-01-01"}))\n'
        'print(json.dumps({"type": "counts", "counts": {"added": 1}}))\n'
        'print(json.dumps({"type": "done", "counts": {"added": 2}, "low_conf": [{"id": "a"}]}))\n',
        manual=True, scan_since="2026-01-01T00:00:00.000Z",
        on_date=dates.append, on_counts=counts_seen.append,
    )
    assert counts == {"added": 2}
    assert low_conf == [{"id": "a"}]
    assert dates == ["2026-01-01"]
    assert counts_seen == [{"added": 1}]


def test_run_scan_passes_args_on_stdin(tmp_path, monkeypatch):
    counts, _ = _run_with_stub(
        tmp_path, monkeypatch,
        'print(json.dumps({"type": "done", "counts": args, "low_conf": []}))\n',
        manual=True, scan_until="2026-02-01", migrate=True,
    )
    assert counts["manual"] is True
    assert counts["scan_until"] == "2026-02-01"
    assert counts["migrate"] is True
    assert counts["data_dir"] == str(tmp_path)


def test_run_scan_raises_on_worker_error(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="boom"):
        _run_with_stub(
            tmp_path, monkeypatch,
            'print(json.dumps({"type": "error", "error": "boom"}))\nsys.exit(1)\n',
        )


def test_run_scan_cancel_terminates_worker(tmp_path, monkeypatch):
    cancel = threading.Event()
    cancel.set()
    t0 = time.time()
    with pytest.raises(RuntimeError):
        _run_with_stub(
            tmp_path, monkeypatch,
            'import time\ntime.sleep(30)\n',
            cancel=cancel,
        )
    assert time.time() - t0 < 10


def test_run_scan_relays_progress_counts_unchanged(tmp_path, monkeypatch):
    """fetched/processed reach on_counts as sent, so the API can build total/processed from them."""
    counts_seen = []
    _run_with_stub(
        tmp_path, monkeypatch,
        'print(json.dumps({"type": "counts", "counts": {"added": 0, "fetched": 409, "processed": 212}}))\n'
        'print(json.dumps({"type": "done", "counts": {}, "low_conf": []}))\n',
        on_counts=counts_seen.append,
    )
    assert counts_seen == [{"added": 0, "fetched": 409, "processed": 212}]
