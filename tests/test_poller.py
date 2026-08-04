"""Tests for poller helpers, in particular the poll cursor advance logic."""
import types

import pytest

import poller


def test_advance_ms_increments_millisecond():
    assert poller._advance_ms("2026-07-25T14:35:03.549Z") == "2026-07-25T14:35:03.550Z"


def test_advance_ms_rolls_over_second():
    assert poller._advance_ms("2026-07-25T14:35:03.999Z") == "2026-07-25T14:35:04.000Z"


def test_advance_ms_rolls_over_day():
    assert poller._advance_ms("2026-07-25T23:59:59.999Z") == "2026-07-26T00:00:00.000Z"


def test_advance_ms_result_excludes_source_asset():
    """The whole point: an inclusive createdAfter/takenAfter filter using the
    advanced cursor must not match an asset with the original timestamp."""
    original = "2026-07-25T14:35:03.549Z"
    advanced = poller._advance_ms(original)
    assert advanced > original


# ---------------------------------------------------------------------------
# process_asset: whole-image (no YOLO detection) handling
#
# process_asset is a closure inside _run_poll_cycle, so these drive a whole
# one-asset poll cycle with everything around it stubbed out.
# ---------------------------------------------------------------------------

class _Recorder:
    """Collects the side effects a poll cycle would have on Immich and the cache."""

    def __init__(self):
        self.faces = []
        self.stored_crops = []
        self.embedded = 0


@pytest.fixture
def cycle(monkeypatch, tmp_path):
    """Run a single-asset poll cycle. Caller supplies what YOLO detected."""
    rec = _Recorder()

    def run(detected, prob=0.99, low_conf_out=None):
        monkeypatch.setattr(poller.data, "load_config", lambda dd: {"Rex": {"person_id": "person-1"}})
        monkeypatch.setattr(poller.data, "load_pet_refs", lambda key, dd: [{"asset_id": "ref-1"}])
        monkeypatch.setattr(poller.data, "load_negative_ids", lambda dd: set())
        monkeypatch.setattr(poller.data, "load_last_timestamp", lambda dd: "2026-07-25T00:00:00Z")
        monkeypatch.setattr(poller.data, "save_last_timestamp", lambda ts, dd: None)
        monkeypatch.setattr(poller.data, "write_poll_status", lambda dd, s: None)

        monkeypatch.setattr(poller.clf_mod, "build_classifier",
                            lambda names, refs, negs: (["Rex", "unknown"], object(), object()))
        monkeypatch.setattr(poller.clf_mod, "classify", lambda vec, names, clf, scaler: ("Rex", prob))

        monkeypatch.setattr(poller.imm, "fetch_assets_created_after",
                            lambda ts: [("asset-1", "2026-07-26T10:00:00Z")])
        monkeypatch.setattr(poller.imm, "fetch_asset_face_person_ids", lambda aid: set())

        def fake_post_face(aid, person_id, bbox_norm=None, img_size=None):
            rec.faces.append((aid, person_id, bbox_norm, img_size))
            return "face-1"

        monkeypatch.setattr(poller.imm, "post_face_sync", fake_post_face)

        img = types.SimpleNamespace(size=(100, 100))
        monkeypatch.setattr(poller.emb, "fetch_thumbnail", lambda aid: img)
        monkeypatch.setattr(poller.emb, "crop_animals", lambda i: detected)

        def fake_embed(crop):
            rec.embedded += 1
            return [0.1, 0.2]

        monkeypatch.setattr(poller.emb, "embed_image", fake_embed)
        monkeypatch.setattr(poller.emb, "store_crops", lambda aid, pairs: rec.stored_crops.append((aid, pairs)))
        monkeypatch.setattr(poller.emb, "save_embed_cache", lambda: None)
        monkeypatch.setattr(poller.emb, "reset_batch_stats", lambda: None)
        monkeypatch.setattr(poller.emb, "get_avg_batch_size", lambda: 0)
        monkeypatch.setattr(poller.emb, "SCAN_WORKERS", 1)

        counts = {}
        poller.run_poll_cycle(str(tmp_path), live_counts=counts, low_conf_out=low_conf_out)
        return counts, rec

    return run


def test_whole_image_match_is_never_auto_tagged(monkeypatch, cycle):
    """The core rule: with no YOLO detection the embedding describes the whole
    scene, so however confidently it scores it goes to review instead of being
    tagged. Previously a 0.99 here was written straight to Immich as a face."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    counts, rec = cycle(detected=[], prob=0.99, low_conf_out=low_conf)
    assert rec.faces == []
    assert counts["added"] == 0
    assert counts["full_image"] == 1
    assert counts["low_confidence"] == 0


def test_whole_image_match_is_queued_for_review_with_a_flag(monkeypatch, cycle):
    """It must still reach the queue -- the recall these matches provide is the
    whole point of keeping the fallback -- carrying the flag the UI needs to warn
    that accepting it as a reference trains the classifier on the background."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    cycle(detected=[], prob=0.99, low_conf_out=low_conf)
    assert low_conf == [{"asset_id": "asset-1", "pet_name": "Rex", "prob": 0.99,
                         "date": "2026-07-26", "bbox": None, "full_image": True}]


def test_whole_image_match_below_threshold_counts_as_low_confidence(monkeypatch, cycle):
    """full_image is reserved for matches that would otherwise have been tagged,
    so the stat reads as 'this many auto-tags were withheld'."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    counts, _ = cycle(detected=[], prob=0.5, low_conf_out=low_conf)
    assert counts["low_confidence"] == 1
    assert counts["full_image"] == 0
    assert low_conf[0]["full_image"] is True


def test_whole_image_crop_is_not_cached(cycle):
    """Only real animal crops go in the crop cache; an empty list marks the asset
    as having no detectable animal so later passes skip it."""
    _, rec = cycle(detected=[], prob=0.99)
    assert rec.stored_crops == [("asset-1", [])]


def test_detected_animal_is_still_tagged(cycle):
    """A real detection is unaffected: it still gets a face, with the detected
    bbox and the real image size."""
    counts, rec = cycle(detected=[((0.1, 0.2, 0.5, 0.6), object())])
    assert counts["added"] == 1
    assert counts["full_image"] == 0
    assert rec.faces == [("asset-1", "person-1", (0.1, 0.2, 0.5, 0.6), (100, 100))]


def test_detected_low_confidence_entry_keeps_bbox(monkeypatch, cycle):
    """A detection-backed review entry carries its bbox and is not flagged, so the
    review UI crops to the animal instead of showing the whole photo."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    counts, _ = cycle(detected=[((0.1, 0.2, 0.5, 0.6), object())], prob=0.5, low_conf_out=low_conf)
    assert counts["low_confidence"] == 1
    assert low_conf == [{"asset_id": "asset-1", "pet_name": "Rex", "prob": 0.5,
                         "date": "2026-07-26", "bbox": [0.1, 0.2, 0.5, 0.6],
                         "full_image": False}]
