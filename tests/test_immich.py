import types
import unittest.mock as mock

import pytest
import immich


@pytest.fixture(autouse=True)
def reset_owner_id():
    """Reset the module-level cache between tests."""
    immich._owner_id = None
    yield
    immich._owner_id = None


@pytest.fixture(autouse=True)
def reset_review_tag():
    """Reset the review tag config and its cached resolution between tests."""
    def clear():
        immich.REVIEW_TAG = ""
        immich._review_tag_id = None
        immich._review_tag_resolved = False

    clear()
    yield
    clear()


def _cache_review_tag(tag_id: str) -> None:
    """Pretend resolution already happened, so a test can exercise tagging alone."""
    immich._review_tag_id = tag_id
    immich._review_tag_resolved = True


# ---------------------------------------------------------------------------
# get_owner_id
# ---------------------------------------------------------------------------

def _ok_response(user_id: str):
    r = mock.MagicMock()
    r.status_code = 200
    r.json.return_value = {"id": user_id, "email": "a@b.com"}
    return r


def _error_response(status: int = 401):
    r = mock.MagicMock()
    r.status_code = status
    return r


def test_get_owner_id_returns_id(monkeypatch):
    monkeypatch.setattr(immich.requests, "get", lambda *a, **kw: _ok_response("user-123"))
    assert immich.get_owner_id() == "user-123"


def test_get_owner_id_caches(monkeypatch):
    calls = []

    def fake_get(*a, **kw):
        calls.append(1)
        return _ok_response("user-abc")

    monkeypatch.setattr(immich.requests, "get", fake_get)
    immich.get_owner_id()
    immich.get_owner_id()
    assert len(calls) == 1


def test_get_owner_id_returns_none_on_error(monkeypatch):
    monkeypatch.setattr(immich.requests, "get", lambda *a, **kw: _error_response(401))
    assert immich.get_owner_id() is None


def test_get_owner_id_returns_none_on_exception(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("unreachable")

    monkeypatch.setattr(immich.requests, "get", boom)
    assert immich.get_owner_id() is None


# ---------------------------------------------------------------------------
# _fetch_assets owner filtering
# ---------------------------------------------------------------------------

def _search_response(items: list[dict]):
    r = mock.MagicMock()
    r.status_code = 200
    r.json.return_value = {"assets": {"items": items}}
    return r


def _make_asset(asset_id: str, owner_id: str, ts: str = "2024-01-01T00:00:00") -> dict:
    return {"id": asset_id, "ownerId": owner_id, "fileCreatedAt": ts, "createdAt": ts}


def test_fetch_assets_filters_out_other_owner(monkeypatch):
    immich._owner_id = "owner-A"
    assets = [
        _make_asset("asset-1", "owner-A"),
        _make_asset("asset-2", "owner-B"),
        _make_asset("asset-3", "owner-A"),
    ]
    monkeypatch.setattr(immich.requests, "post", lambda *a, **kw: _search_response(assets))
    result = immich._fetch_assets({"takenAfter": "2024-01-01"}, ts_field="fileCreatedAt", label="test")
    ids = [r[0] for r in result]
    assert ids == ["asset-1", "asset-3"]
    assert "asset-2" not in ids


def test_fetch_assets_keeps_all_when_owner_id_unknown(monkeypatch):
    immich._owner_id = None
    monkeypatch.setattr(immich.requests, "get", lambda *a, **kw: _error_response())
    assets = [
        _make_asset("asset-1", "owner-A"),
        _make_asset("asset-2", "owner-B"),
    ]
    monkeypatch.setattr(immich.requests, "post", lambda *a, **kw: _search_response(assets))
    result = immich._fetch_assets({"takenAfter": "2024-01-01"}, ts_field="fileCreatedAt", label="test")
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Review tag
# ---------------------------------------------------------------------------

def _json_response(payload, status: int = 200):
    r = mock.MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def _fake_put(calls: list, responses: dict):
    """Record PUT calls and answer from a {url_suffix: response} map."""
    def put(url, **kw):
        calls.append((url, kw.get("json")))
        for suffix, resp in responses.items():
            if url.endswith(suffix):
                return resp
        raise AssertionError(f"unexpected PUT {url}")
    return put


def test_resolve_review_tag_id_upserts_and_caches(monkeypatch):
    immich.REVIEW_TAG = "Pets/Review"
    calls = []
    monkeypatch.setattr(immich.requests, "put", _fake_put(calls, {
        "/api/tags": _json_response([{"id": "tag-1", "value": "Pets/Review"}]),
    }))
    assert immich.resolve_review_tag_id_sync() == "tag-1"
    assert immich.resolve_review_tag_id_sync() == "tag-1"
    assert len(calls) == 1
    assert calls[0][1] == {"tags": ["Pets/Review"]}


def test_resolve_review_tag_id_disabled_when_unset(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not call Immich when the tag is unset")

    monkeypatch.setattr(immich.requests, "put", boom)
    assert immich.resolve_review_tag_id_sync() is None


def test_resolve_review_tag_id_none_on_error(monkeypatch):
    immich.REVIEW_TAG = "Review"
    monkeypatch.setattr(immich.requests, "put", lambda *a, **kw: _json_response(None, status=403))
    assert immich.resolve_review_tag_id_sync() is None


def test_resolve_review_tag_id_gives_up_after_one_failure(monkeypatch):
    """A key without tag.create must not make every face write retry the upsert:
    each retry holds the lock across a 15s-timeout call and stalls all scan threads."""
    immich.REVIEW_TAG = "Review"
    calls = []

    def failing_put(*a, **kw):
        calls.append(1)
        return _json_response(None, status=403)

    monkeypatch.setattr(immich.requests, "put", failing_put)
    for _ in range(5):
        immich.apply_review_tag_sync("asset-1")
    assert len(calls) == 1


def test_apply_review_tag_puts_asset_on_tag(monkeypatch):
    immich.REVIEW_TAG = "Review"
    _cache_review_tag("tag-9")
    calls = []
    monkeypatch.setattr(immich.requests, "put", _fake_put(calls, {
        "/api/tags/tag-9/assets": _json_response([{"id": "asset-1", "success": True}]),
    }))
    immich.apply_review_tag_sync("asset-1")
    assert calls == [(f"{immich.IMMICH_URL}/api/tags/tag-9/assets", {"ids": ["asset-1"]})]


def test_apply_review_tag_swallows_errors(monkeypatch):
    immich.REVIEW_TAG = "Review"
    _cache_review_tag("tag-9")

    def boom(*a, **kw):
        raise ConnectionError("unreachable")

    monkeypatch.setattr(immich.requests, "put", boom)
    immich.apply_review_tag_sync("asset-1")  # must not raise


def test_post_face_sync_applies_review_tag(monkeypatch):
    immich.REVIEW_TAG = "Review"
    _cache_review_tag("tag-9")
    put_calls = []
    monkeypatch.setattr(immich.requests, "put", _fake_put(put_calls, {
        "/api/tags/tag-9/assets": _json_response([{"id": "asset-1", "success": True}]),
    }))
    monkeypatch.setattr(immich.requests, "post", lambda *a, **kw: _json_response(None, status=201))
    monkeypatch.setattr(immich.requests, "get",
                        lambda *a, **kw: _json_response([{"id": "face-1", "person": {"id": "person-1"}}]))

    assert immich.post_face_sync("asset-1", "person-1") == "face-1"
    assert put_calls == [(f"{immich.IMMICH_URL}/api/tags/tag-9/assets", {"ids": ["asset-1"]})]


def test_post_face_sync_does_not_tag_when_face_fails(monkeypatch):
    immich.REVIEW_TAG = "Review"
    _cache_review_tag("tag-9")

    def boom(*a, **kw):
        raise AssertionError("must not tag when face creation failed")

    monkeypatch.setattr(immich.requests, "put", boom)
    monkeypatch.setattr(immich.requests, "post", lambda *a, **kw: _json_response(None, status=400))
    assert immich.post_face_sync("asset-1", "person-1") is None


# ---------------------------------------------------------------------------
# Benchmark analysis helpers
# ---------------------------------------------------------------------------

def test_fetch_assets_in_range_keeps_full_items(monkeypatch):
    items = [{"id": "a0", "type": "IMAGE", "fileCreatedAt": "2024-01-01"},
             {"id": "a1", "type": "VIDEO", "fileCreatedAt": "2024-01-02"}]
    calls = []

    def post(url, json, **kw):
        calls.append(json)
        return _search_response(items)

    monkeypatch.setattr(immich.requests, "post", post)
    result = immich.fetch_assets_in_range("2024-01-01T00:00:00.000Z", "2024-01-31T23:59:59.999Z")
    assert result == items
    assert calls[0]["takenAfter"] == "2024-01-01T00:00:00.000Z"
    assert calls[0]["takenBefore"] == "2024-01-31T23:59:59.999Z"


def test_fetch_assets_in_range_drops_other_owner(monkeypatch):
    # Partner photos can never carry this user's pet faces, so the benchmark would count
    # every correct match on them as a false positive.
    immich._owner_id = "owner-A"
    assets = [_make_asset("a0", "owner-A"), _make_asset("a1", "owner-B")]
    monkeypatch.setattr(immich.requests, "post", lambda *a, **kw: _search_response(assets))
    result = immich.fetch_assets_in_range("2024-01-01T00:00:00.000Z")
    assert [a["id"] for a in result] == ["a0"]


def test_drop_foreign_assets_keeps_all_when_owner_unknown(monkeypatch):
    immich._owner_id = None
    monkeypatch.setattr(immich.requests, "get", lambda *a, **kw: _error_response())
    assets = [_make_asset("a0", "owner-A"), _make_asset("a1", "owner-B")]
    assert immich.drop_foreign_assets(assets, "test") == assets
