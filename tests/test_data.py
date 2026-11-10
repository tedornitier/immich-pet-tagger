import json
import pytest
from pathlib import Path
import data


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_missing(tmp_path):
    assert data.load_config(tmp_path) == {}


def test_load_config_valid(tmp_path):
    cfg = {"cat": {"person_id": "abc"}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert data.load_config(tmp_path) == cfg


def test_load_config_corrupted(tmp_path):
    (tmp_path / "config.json").write_text("{bad json")
    assert data.load_config(tmp_path) == {}


def test_load_config_roundtrip(tmp_path):
    cfg = {"dog": {"person_id": "xyz", "since": "2024-01-01"}}
    data.save_config(cfg, tmp_path)
    assert data.load_config(tmp_path) == cfg


# ---------------------------------------------------------------------------
# load_pet_refs
# ---------------------------------------------------------------------------

def test_load_pet_refs_missing(tmp_path):
    assert data.load_pet_refs("cat", tmp_path) == []


def test_load_pet_refs_valid(tmp_path):
    refs = [{"asset_id": "a1", "face_id": "f1"}, {"asset_id": "a2", "face_id": None}]
    data.save_pet_refs("cat", refs, tmp_path)
    assert data.load_pet_refs("cat", tmp_path) == refs


def test_load_pet_refs_corrupted(tmp_path):
    ref_file = tmp_path / "pets" / "cat" / "refs.json"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_text("{broken")
    assert data.load_pet_refs("cat", tmp_path) == []


def test_load_pet_refs_legacy_strings(tmp_path):
    ref_file = tmp_path / "pets" / "cat" / "refs.json"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_text(json.dumps(["id1", "id2"]))
    result = data.load_pet_refs("cat", tmp_path)
    assert result == [{"asset_id": "id1", "face_id": None}, {"asset_id": "id2", "face_id": None}]


def test_load_pet_refs_empty_file(tmp_path):
    ref_file = tmp_path / "pets" / "cat" / "refs.json"
    ref_file.parent.mkdir(parents=True)
    ref_file.write_text("[]")
    assert data.load_pet_refs("cat", tmp_path) == []


# ---------------------------------------------------------------------------
# load_negatives / crop entries
# ---------------------------------------------------------------------------

def test_load_negatives_missing(tmp_path):
    assert data.load_negatives(tmp_path) == []


def test_load_negatives_roundtrip(tmp_path):
    entries = [{"asset_id": "id1", "bbox": None}, {"asset_id": "id2", "bbox": [0.1, 0.2, 0.5, 0.6]}]
    data.save_negatives(entries, tmp_path)
    assert data.load_negatives(tmp_path) == entries


def test_load_negatives_legacy_strings_are_whole_photo(tmp_path):
    (tmp_path / "negatives.json").write_text('["id1", "id2"]')
    assert data.load_negatives(tmp_path) == [{"asset_id": "id1", "bbox": None}, {"asset_id": "id2", "bbox": None}]
    # Loading does not rewrite the file.
    assert (tmp_path / "negatives.json").read_text() == '["id1", "id2"]'


def test_load_negatives_corrupted(tmp_path):
    (tmp_path / "negatives.json").write_text("not json at all")
    assert data.load_negatives(tmp_path) == []


def test_merge_crop_entries_dedupes_on_rounded_bbox():
    existing = data.normalize_crop_entries(["a", {"asset_id": "b", "bbox": [0.1, 0.1, 0.5, 0.5]}])
    new = data.normalize_crop_entries(["a", {"asset_id": "b", "bbox": [0.100001, 0.1, 0.5, 0.5]},
                                       {"asset_id": "b", "bbox": None}])
    merged = data.merge_crop_entries(existing, new)
    assert merged == existing + [{"asset_id": "b", "bbox": None}]


def test_crop_entry_matches():
    idx = data.index_crop_entries(data.normalize_crop_entries(
        ["whole", {"asset_id": "crop", "bbox": [0.1, 0.1, 0.5, 0.5]}]))
    assert data.crop_entry_matches(idx, "whole", [0.6, 0.6, 0.9, 0.9])
    assert data.crop_entry_matches(idx, "whole", None)
    assert data.crop_entry_matches(idx, "crop", [0.11, 0.09, 0.51, 0.5])
    assert not data.crop_entry_matches(idx, "crop", [0.6, 0.6, 0.9, 0.9])
    assert not data.crop_entry_matches(idx, "crop", None)
    assert not data.crop_entry_matches(idx, "other", [0.1, 0.1, 0.5, 0.5])
    assert data.whole_photo_ids(idx) == {"whole"}


def test_crop_entry_token_whole_photo_is_bare_id():
    assert data.crop_entry_token({"asset_id": "a", "bbox": None}) == "a"
    assert data.crop_entry_token({"asset_id": "a", "bbox": [0.1, 0.2, 0.3, 0.4]}) == "a@0.1000,0.2000,0.3000,0.4000"


# ---------------------------------------------------------------------------
# load_skipped_ids
# ---------------------------------------------------------------------------

def test_load_skipped_ids_missing(tmp_path):
    assert data.load_skipped_ids(tmp_path) == []


def test_load_skipped_ids_valid(tmp_path):
    ids = ["s1", "s2"]
    data.save_skipped_ids(ids, tmp_path)
    assert data.load_skipped_ids(tmp_path) == ids


def test_load_skipped_ids_corrupted(tmp_path):
    (tmp_path / "skipped.json").write_text("[unclosed")
    assert data.load_skipped_ids(tmp_path) == []


# ---------------------------------------------------------------------------
# load_poll_status
# ---------------------------------------------------------------------------

def test_load_poll_status_missing(tmp_path):
    assert data.load_poll_status(tmp_path) == {"status": "never"}


def test_load_poll_status_valid(tmp_path):
    payload = {"status": "idle", "ran_at": "2024-01-01T00:00:00Z"}
    (tmp_path / "last_poll_status.json").write_text(json.dumps(payload))
    assert data.load_poll_status(tmp_path) == payload


def test_load_poll_status_corrupted(tmp_path):
    (tmp_path / "last_poll_status.json").write_text("???")
    assert data.load_poll_status(tmp_path) == {"status": "never"}


# ---------------------------------------------------------------------------
# load_pet_asset_ids (dedup)
# ---------------------------------------------------------------------------

def test_load_pet_asset_ids_deduplicates(tmp_path):
    refs = [
        {"asset_id": "a1", "face_id": "f1"},
        {"asset_id": "a1", "face_id": "f2"},
        {"asset_id": "a2", "face_id": "f3"},
    ]
    data.save_pet_refs("cat", refs, tmp_path)
    result = data.load_pet_asset_ids("cat", tmp_path)
    assert result == ["a1", "a2"]
