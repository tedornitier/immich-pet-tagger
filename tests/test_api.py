import asyncio

import pytest

import api
import immich


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body if json_body is not None else {}
        self.text = text

    def json(self):
        return self._json_body


class _FakeAsyncClient:
    """Records every call so tests can assert on which Immich endpoints were hit."""

    def __init__(self, calls, responses, *a, **kw):
        self.calls = calls
        self.responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def _dispatch(self, method, url, **kw):
        self.calls.append((method, url, kw.get("json")))
        for key, resp in self.responses.items():
            if key in url:
                return resp
        return _FakeResponse(200)

    async def delete(self, url, **kw):
        return await self._dispatch("DELETE", url, **kw)

    async def post(self, url, **kw):
        return await self._dispatch("POST", url, **kw)

    async def put(self, url, **kw):
        return await self._dispatch("PUT", url, **kw)

    async def request(self, method, url, **kw):
        return await self._dispatch(method, url, **kw)


@pytest.fixture(autouse=True)
def reset_review_tag():
    immich.REVIEW_TAG = "Review"
    immich._review_tag_id = "tag-9"
    immich._review_tag_resolved = True
    yield
    immich.REVIEW_TAG = ""
    immich._review_tag_id = None
    immich._review_tag_resolved = False


def test_reset_pet_immich_removes_review_tag(monkeypatch, tmp_path):
    config = {"fido": {"person_id": "person-1"}}
    refs = [{"asset_id": "asset-1", "crop_idx": None, "bbox": None, "face_id": "face-1"}]

    monkeypatch.setattr(api.data, "load_config", lambda data_dir: config)
    monkeypatch.setattr(api.data, "save_config", lambda cfg, data_dir: None)
    monkeypatch.setattr(api.data, "load_pet_refs", lambda person_id, data_dir: refs if person_id == "person-1" else [])
    monkeypatch.setattr(api.data, "save_pet_refs", lambda person_id, refs, data_dir: None)
    monkeypatch.setattr(api, "PETS_DIR", tmp_path)

    calls = []
    responses = {
        "/api/people/person-1": _FakeResponse(200),
        "/api/people": _FakeResponse(201, {"id": "person-2"}),
        "/api/tags/tag-9/assets": _FakeResponse(200, []),
    }
    monkeypatch.setattr(api.httpx, "AsyncClient", lambda *a, **kw: _FakeAsyncClient(calls, responses, *a, **kw))

    asyncio.run(api.reset_pet_immich("fido"))

    untag_calls = [c for c in calls if c[0] == "DELETE" and "/api/tags/tag-9/assets" in c[1]]
    assert untag_calls == [("DELETE", f"{immich.IMMICH_URL}/api/tags/tag-9/assets", {"ids": ["asset-1"]})], (
        "reset_pet_immich must strip the review tag from every ref's asset, same as delete_pet does"
    )


def test_scan_result_running_includes_progress(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(api, "ENABLE_SCAN_PROGRESS", True)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "scan_low_conf_assets", [])
    monkeypatch.setattr(state, "manual_scan_result", {
        "status": "running", "started_at": "2026-11-25T17:43:19+00:00",
        "counts": {"added": 3, "fetched": 409, "processed": 212},
    })
    r = asyncio.run(api.get_scan_result())
    assert r["total"] == 409
    assert r["processed"] == 212
    assert r["counts"]["added"] == 3


def test_scan_result_running_before_fetch_has_null_total(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(api, "ENABLE_SCAN_PROGRESS", True)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "scan_low_conf_assets", [])
    monkeypatch.setattr(state, "manual_scan_result", {"status": "running", "started_at": "x", "counts": {}})
    r = asyncio.run(api.get_scan_result())
    assert r["total"] is None
    assert r["processed"] == 0


def test_scan_result_finished_has_no_progress_fields(tmp_path, monkeypatch):
    import state
    monkeypatch.setattr(api, "ENABLE_SCAN_PROGRESS", True)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "scan_low_conf_assets", [])
    monkeypatch.setattr(state, "manual_scan_result", {"status": "idle", "ran_at": "x", "counts": {"fetched": 5, "processed": 5}})
    r = asyncio.run(api.get_scan_result())
    assert "total" not in r and "processed" not in r
    assert "processed" not in r["counts"]


def test_scan_result_progress_disabled_by_default(tmp_path, monkeypatch):
    """With ENABLE_SCAN_PROGRESS off, a running scan's response matches the pre-flag shape."""
    import state
    monkeypatch.setattr(api, "ENABLE_SCAN_PROGRESS", False)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(state, "scan_low_conf_assets", [])
    monkeypatch.setattr(state, "manual_scan_result", {
        "status": "running", "started_at": "x", "counts": {"added": 1, "fetched": 10, "processed": 4},
    })
    r = asyncio.run(api.get_scan_result())
    assert "total" not in r and "processed" not in r
    assert "processed" not in r["counts"]
    assert r["counts"]["added"] == 1


def test_manual_scan_progress_flows_from_on_counts_to_scan_result(tmp_path, monkeypatch):
    """Counts relayed mid-scan show up as total/processed, and a later snapshot never
    leaves the result without a total (no clear-then-update gap)."""
    import inference
    import state
    monkeypatch.setattr(api, "ENABLE_SCAN_PROGRESS", True)
    monkeypatch.setattr(api, "DATA_DIR", tmp_path)
    seen = []

    def fake_run_scan(data_dir, *, on_counts, **kw):
        on_counts({"added": 0, "fetched": 10, "processed": 3})
        seen.append(asyncio.run(api.get_scan_result()))
        on_counts({"added": 1, "fetched": 10, "processed": 7})
        seen.append(asyncio.run(api.get_scan_result()))
        return {}, []

    monkeypatch.setattr(inference, "run_scan", fake_run_scan)

    async def go():
        monkeypatch.setattr(state, "scan_lock", asyncio.Lock())
        state.scan_generation += 1
        await api._run_manual_scan(state.scan_generation, "2026-01-01")

    asyncio.run(go())
    assert [(r["total"], r["processed"]) for r in seen] == [(10, 3), (10, 7)]
    assert all("processed" not in r["counts"] for r in seen)
