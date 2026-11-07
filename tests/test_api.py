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


def test_ref_analysis_flags_duplicate_and_scores(monkeypatch):
    import contextlib

    import numpy as np

    import state

    config = {"cat": {"person_id": "p-cat"}, "dog": {"person_id": "p-dog"}}
    refs = {
        "p-cat": [{"asset_id": "c1"}, {"asset_id": "c2"}, {"asset_id": "c3"}],
        "p-dog": [{"asset_id": "d1"}, {"asset_id": "d2"}],
    }
    # c1/c2 are near-identical cat embeddings (a duplicate pair); c3 is a distinct cat.
    vecs = {
        "c1": np.array([1.0, 0.02, 0.0, 0.0]),
        "c2": np.array([1.0, 0.00, 0.0, 0.0]),
        "c3": np.array([0.6, 0.6, 0.0, 0.0]),
        "d1": np.array([0.0, 0.0, 1.0, 0.1]),
        "d2": np.array([0.0, 0.0, 0.2, 1.0]),
    }

    monkeypatch.setattr(api.data, "load_config", lambda data_dir: config)
    monkeypatch.setattr(api.data, "load_pet_refs", lambda folder_key, data_dir: refs.get(folder_key, []))
    monkeypatch.setattr(api.data, "load_negatives", lambda data_dir: [])
    monkeypatch.setattr(api.emb, "embed_asset_crops", lambda aid, require_animal=False: [vecs[aid]])
    monkeypatch.setattr(api, "inference_session", lambda: contextlib.nullcontext())

    state.refanalysis_cancel.clear()
    asyncio.run(api._run_ref_analysis(1))

    result = state.refanalysis_result
    assert result["config"]["n_pets"] == 2
    assert result["config"]["n_refs"] == 5
    assert result["config"]["n_ref_vecs"] == 5
    assert result["config"]["partial"] is False

    cat = {r["asset_id"]: r for r in result["pets"]["cat"]["refs"]}
    # The duplicate pair should each report the other as their nearest, at very high similarity.
    assert cat["c1"]["nearest_asset_id"] == "c2"
    assert cat["c2"]["nearest_asset_id"] == "c1"
    assert cat["c1"]["redundancy"] > 0.99
    # The distinct cat ref is less redundant than a member of the duplicate pair.
    assert cat["c3"]["redundancy"] < cat["c1"]["redundancy"]
    # Leave-one-out scores are populated for every embedded ref (each pet keeps >=1 ref
    # when one is dropped, so its class never vanishes).
    for r in result["pets"]["cat"]["refs"] + result["pets"]["dog"]["refs"]:
        assert r["embedded"] is True
        assert r["self_score"] is not None
        assert r["influence"] is not None


def test_ref_analysis_catches_bad_crop(monkeypatch):
    """A cat ref whose crop actually landed on the dog should stand out: low held-out
    self score, and the most-negative influence (removing it helps the other cat refs).
    Guards the log-odds influence measure against silently collapsing to ~0 for every ref."""
    import contextlib

    import numpy as np

    import state

    config = {"cat": {"person_id": "p-cat"}, "dog": {"person_id": "p-dog"}}
    refs = {
        "p-cat": [{"asset_id": "c1"}, {"asset_id": "c2"}, {"asset_id": "c3"}, {"asset_id": "cbad"}],
        "p-dog": [{"asset_id": "d1"}, {"asset_id": "d2"}, {"asset_id": "d3"}],
    }
    rng = np.random.RandomState(0)
    cat_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dog_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    vecs = {
        "c1": cat_c + 0.03 * rng.randn(6),
        "c2": cat_c + 0.03 * rng.randn(6),
        "c3": cat_c + 0.03 * rng.randn(6),
        "cbad": dog_c + 0.03 * rng.randn(6),   # labeled cat, but sits in dog territory
        "d1": dog_c + 0.03 * rng.randn(6),
        "d2": dog_c + 0.03 * rng.randn(6),
        "d3": dog_c + 0.03 * rng.randn(6),
    }

    monkeypatch.setattr(api.data, "load_config", lambda data_dir: config)
    monkeypatch.setattr(api.data, "load_pet_refs", lambda folder_key, data_dir: refs.get(folder_key, []))
    monkeypatch.setattr(api.data, "load_negatives", lambda data_dir: [])
    monkeypatch.setattr(api.emb, "embed_asset_crops", lambda aid, require_animal=False: [vecs[aid]])
    monkeypatch.setattr(api, "inference_session", lambda: contextlib.nullcontext())

    state.refanalysis_cancel.clear()
    asyncio.run(api._run_ref_analysis(1))

    cat_refs = state.refanalysis_result["pets"]["cat"]["refs"]
    cat = {r["asset_id"]: r for r in cat_refs}
    good = [cat["c1"], cat["c2"], cat["c3"]]
    bad = cat["cbad"]
    # self_score reliably flags the mislabeled crop: held out, the rest don't recognize it.
    assert bad["self_score"] < 0.5
    assert bad["self_score"] < min(r["self_score"] for r in good)
    # Influence is measured in log-odds specifically so it does not collapse to ~0 on
    # well-separated refs the way probability-space influence did (which rounded to 0.0%
    # for every ref); there must be real magnitude to rank by.
    assert max(abs(r["influence"]) for r in cat_refs) > 0.005


def test_neg_analysis_ranks_hard_negatives(monkeypatch):
    """The 'not a pet' analysis: a negative sitting in a pet's cluster should read as more
    pet-like (and be confused with that pet) than one far from every pet, and near-duplicate
    negatives should be flagged redundant."""
    import contextlib

    import numpy as np

    import state

    config = {"cat": {"person_id": "p-cat"}, "dog": {"person_id": "p-dog"}}
    refs = {
        "p-cat": [{"asset_id": "c1"}, {"asset_id": "c2"}, {"asset_id": "c3"}],
        "p-dog": [{"asset_id": "d1"}, {"asset_id": "d2"}, {"asset_id": "d3"}],
    }
    # Easy negatives form their own cluster so the model learns that region is "not a pet"
    # (a lone far-away point would just be extrapolated to a pet by the linear boundary).
    neg_ids = ["nhard", "neasy1", "neasy2", "neasy3", "ndup1", "ndup2"]
    rng = np.random.RandomState(0)
    cat_c = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dog_c = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    far = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    dup = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    vecs = {
        "c1": cat_c + 0.03 * rng.randn(6), "c2": cat_c + 0.03 * rng.randn(6), "c3": cat_c + 0.03 * rng.randn(6),
        "d1": dog_c + 0.03 * rng.randn(6), "d2": dog_c + 0.03 * rng.randn(6), "d3": dog_c + 0.03 * rng.randn(6),
        "nhard": cat_c + 0.03 * rng.randn(6),    # a "not a pet" photo that looks like the cat
        "neasy1": far + 0.03 * rng.randn(6),     # a cluster of clearly-not-a-pet photos
        "neasy2": far + 0.03 * rng.randn(6),
        "neasy3": far + 0.03 * rng.randn(6),
        "ndup1": dup + 0.001 * rng.randn(6),     # near-duplicate pair
        "ndup2": dup + 0.001 * rng.randn(6),
    }

    monkeypatch.setattr(api.data, "load_config", lambda data_dir: config)
    monkeypatch.setattr(api.data, "load_pet_refs", lambda folder_key, data_dir: refs.get(folder_key, []))
    monkeypatch.setattr(api.data, "load_negatives", lambda data_dir: [{"asset_id": a} for a in neg_ids])
    monkeypatch.setattr(api.emb, "embed_asset_crops", lambda aid, require_animal=False: [vecs[aid]])
    monkeypatch.setattr(api.emb, "embed_crop_by_bbox", lambda aid, bbox: vecs[aid])
    monkeypatch.setattr(api, "inference_session", lambda: contextlib.nullcontext())

    state.neganalysis_cancel.clear()
    asyncio.run(api._run_neg_analysis(1))

    result = state.neganalysis_result
    assert result["config"]["n_negatives"] == 6
    assert result["config"]["n_negatives_analyzed"] == 6
    assert result["config"]["neg_subsampled"] is False

    negs = {n["asset_id"]: n for n in result["negatives"]["items"]}
    # The cat-lookalike negative is the valuable hard negative: most pet-like, confused w/ cat.
    assert negs["nhard"]["confused_pet"] == "cat"
    assert negs["nhard"]["pet_likeness"] > negs["neasy1"]["pet_likeness"]
    # Model-free cosine similarity to the confused pet: high for the cat-lookalike, and higher
    # than for a photo that looks like nothing any pet.
    assert negs["nhard"]["pet_similarity"] > 0.9
    assert negs["nhard"]["pet_similarity"] > negs["neasy1"]["pet_similarity"]
    # The duplicate pair flags each other as near-identical.
    assert negs["ndup1"]["redundancy"] > 0.99
    assert negs["ndup1"]["nearest_asset_id"] == "ndup2"
