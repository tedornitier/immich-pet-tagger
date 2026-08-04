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
# process_asset: no-animal handling
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


def test_no_yolo_detection_is_not_classified(cycle):
    """A photo with no detected animal must not be classified at all: the classifier
    always returns a best-matching class, so on an animal-free photo that match is
    noise, and it used to be tagged as a real hit."""
    counts, rec = cycle(detected=[])
    assert rec.faces == []
    assert rec.embedded == 0
    assert counts["no_animal"] == 1
    assert counts["added"] == 0


def test_no_yolo_detection_caches_empty_result(cycle):
    """Skipping must still record 'no animal here' so later passes don't re-run YOLO."""
    _, rec = cycle(detected=[])
    assert rec.stored_crops == [("asset-1", [])]


def test_detected_animal_is_still_tagged(cycle):
    """The skip must not touch the normal path: a real detection still gets a face,
    with the detected bbox and the real image size."""
    counts, rec = cycle(detected=[((0.1, 0.2, 0.5, 0.6), object())])
    assert counts["added"] == 1
    assert counts["no_animal"] == 0
    assert rec.faces == [("asset-1", "person-1", (0.1, 0.2, 0.5, 0.6), (100, 100))]


def test_low_confidence_entry_keeps_bbox(monkeypatch, cycle):
    """Low-confidence review entries always carry the detected bbox now that a
    bbox-less (whole-image) crop can no longer reach the classifier."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    counts, _ = cycle(detected=[((0.1, 0.2, 0.5, 0.6), object())], prob=0.5, low_conf_out=low_conf)
    assert counts["low_confidence"] == 1
    assert low_conf == [{"asset_id": "asset-1", "pet_name": "Rex", "prob": 0.5,
                         "date": "2026-07-26", "bbox": [0.1, 0.2, 0.5, 0.6]}]


def test_no_yolo_detection_produces_no_low_confidence_entry(monkeypatch, cycle):
    """The old whole-image fallback also polluted the review queue with animal-free
    photos, which got accepted as whole-image refs and fed the next round of
    false positives. Nothing without a detection should reach the queue now."""
    monkeypatch.setattr(poller, "THRESHOLD", 0.8)
    low_conf = []
    counts, _ = cycle(detected=[], prob=0.5, low_conf_out=low_conf)
    assert low_conf == []
    assert counts["low_confidence"] == 0
    assert counts["no_animal"] == 1
