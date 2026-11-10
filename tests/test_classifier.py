import numpy as np
import pytest
from unittest.mock import patch
import classifier as clf_mod


def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# Two clearly separated clusters: seeds 0-9 for cat, 100-109 for dog, 200-209 for unknown.
CAT_VECS  = [_unit_vec(i) for i in range(10)]
DOG_VECS  = [_unit_vec(i + 100) for i in range(10)]
NEG_VECS  = [_unit_vec(i + 200) for i in range(20)]

CAT_REFS  = [{"asset_id": f"cat_{i}"} for i in range(10)]
DOG_REFS  = [{"asset_id": f"dog_{i}"} for i in range(10)]
NEG_IDS   = [{"asset_id": f"neg_{i}", "bbox": None} for i in range(20)]


def _mock_embed(asset_id, require_animal=False):
    if asset_id.startswith("cat_"):
        return [CAT_VECS[int(asset_id.split("_")[1])]]
    if asset_id.startswith("dog_"):
        return [DOG_VECS[int(asset_id.split("_")[1])]]
    if asset_id.startswith("neg_"):
        return [NEG_VECS[int(asset_id.split("_")[1])]]
    return []


@pytest.fixture
def two_pet_classifier():
    refs = {"cat": CAT_REFS, "dog": DOG_REFS}
    with patch("classifier.emb.embed_asset_crops", side_effect=_mock_embed), \
         patch("classifier.emb.embed_crop_by_bbox", return_value=None):
        result = clf_mod.build_classifier(["cat", "dog"], refs, NEG_IDS)
    assert result is not None
    return result


# ---------------------------------------------------------------------------
# build_classifier
# ---------------------------------------------------------------------------

def test_build_classifier_returns_names_clf_scaler(two_pet_classifier):
    names, clf, scaler = two_pet_classifier
    assert "cat" in names
    assert "dog" in names
    assert "unknown" in names


def test_build_classifier_no_refs_returns_none():
    with patch("classifier.emb.embed_asset_crops", return_value=[]), \
         patch("classifier.emb.embed_crop_by_bbox", return_value=None):
        result = clf_mod.build_classifier(["cat"], {"cat": [{"asset_id": "x"}]}, [])
    assert result is None


def test_build_classifier_no_negatives_still_trains():
    refs = {"cat": CAT_REFS}
    with patch("classifier.emb.embed_asset_crops", side_effect=_mock_embed), \
         patch("classifier.emb.embed_crop_by_bbox", return_value=None):
        result = clf_mod.build_classifier(["cat"], refs, [])
    assert result is not None
    names, clf, scaler = result
    assert "unknown" in names


def test_build_classifier_skips_missing_embeddings():
    refs = {"cat": [{"asset_id": "missing"}]}
    with patch("classifier.emb.embed_asset_crops", return_value=[]), \
         patch("classifier.emb.embed_crop_by_bbox", return_value=None):
        result = clf_mod.build_classifier(["cat"], refs, [])
    assert result is None


# ---------------------------------------------------------------------------
# classify
# ---------------------------------------------------------------------------

def test_classify_cat(two_pet_classifier):
    names, clf, scaler = two_pet_classifier
    label, prob = clf_mod.classify(CAT_VECS[0], names, clf, scaler)
    assert label == "cat"
    assert prob > 0.5


def test_classify_dog(two_pet_classifier):
    names, clf, scaler = two_pet_classifier
    label, prob = clf_mod.classify(DOG_VECS[0], names, clf, scaler)
    assert label == "dog"
    assert prob > 0.5


def test_classify_unknown(two_pet_classifier):
    names, clf, scaler = two_pet_classifier
    label, prob = clf_mod.classify(NEG_VECS[0], names, clf, scaler)
    assert label == "unknown"


def test_classify_returns_float_prob(two_pet_classifier):
    names, clf, scaler = two_pet_classifier
    _, prob = clf_mod.classify(CAT_VECS[0], names, clf, scaler)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# poller helpers (pure functions, no mocking needed)
# ---------------------------------------------------------------------------

from poller import asset_in_range, parse_date


def test_parse_date_valid():
    assert parse_date("2024-03-15T12:00:00Z") is not None


def test_parse_date_none():
    assert parse_date(None) is None


def test_parse_date_invalid():
    assert parse_date("not-a-date") is None


def test_asset_in_range_no_bounds():
    assert asset_in_range("2024-06-01T00:00:00Z", None, None) is True


def test_asset_in_range_within():
    assert asset_in_range("2024-06-01T00:00:00Z", "2024-01-01", "2024-12-31") is True


def test_asset_in_range_before_since():
    assert asset_in_range("2023-12-31T00:00:00Z", "2024-01-01", None) is False


def test_asset_in_range_after_until():
    assert asset_in_range("2025-01-01T00:00:00Z", None, "2024-12-31") is False


def test_build_classifier_crop_negative_embeds_only_that_crop():
    """A crop negative goes through embed_crop_by_bbox, never the whole-photo path, so the
    other crops of the photo (e.g. the user's pet) are not trained as unknown."""
    refs = {"cat": CAT_REFS}
    negs = [{"asset_id": "mixed", "bbox": [0.1, 0.1, 0.4, 0.4]}]
    with patch("classifier.emb.embed_asset_crops", side_effect=_mock_embed) as whole, \
         patch("classifier.emb.embed_crop_by_bbox", return_value=NEG_VECS[0]) as crop:
        result = clf_mod.build_classifier(["cat"], refs, negs)
    assert result is not None
    crop.assert_called_once_with("mixed", [0.1, 0.1, 0.4, 0.4])
    assert all(c.args[0] != "mixed" for c in whole.call_args_list)


def test_build_classifier_subsampling_is_deterministic_with_entries():
    refs = {"cat": CAT_REFS[:2]}  # target 6 negatives out of 20
    seen = []
    for order in (NEG_IDS, list(reversed(NEG_IDS))):
        with patch("classifier.emb.embed_asset_crops", side_effect=_mock_embed) as whole, \
             patch("classifier.emb.embed_crop_by_bbox", return_value=None):
            clf_mod.build_classifier(["cat"], refs, order)
        seen.append(sorted(c.args[0] for c in whole.call_args_list if c.args[0].startswith("neg_")))
    assert len(seen[0]) == 6
    assert seen[0] == seen[1]
