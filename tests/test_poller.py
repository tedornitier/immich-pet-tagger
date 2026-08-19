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
# classify_outcome
# ---------------------------------------------------------------------------

def test_classify_outcome_unknown():
    assert poller.classify_outcome("unknown", 0.99, "2026-06-01T00:00:00Z", {}) == "unknown"


def test_classify_outcome_out_of_range_beats_low_confidence():
    """A low-confidence guess for a pet who was not even in range on that date must
    be dropped as out_of_range, not surfaced as a low-confidence review candidate."""
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "out_of_range"


def test_classify_outcome_out_of_range_beats_confident():
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "out_of_range"


def test_classify_outcome_low_confidence_when_in_range():
    cfg = {"since": "2025-01-01", "until": "2027-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "low_confidence"


def test_classify_outcome_confident_when_in_range():
    cfg = {"since": "2025-01-01", "until": "2027-12-31"}
    outcome = poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", cfg, threshold=0.8)
    assert outcome == "confident"


def test_classify_outcome_no_date_bounds_never_out_of_range():
    outcome = poller.classify_outcome("Dobby", 0.67, "2026-06-01T00:00:00Z", {}, threshold=0.8)
    assert outcome == "low_confidence"


# ---------------------------------------------------------------------------
# classify_outcome: WHOLE_IMAGE_MATCH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("tag", "tag"), ("review", "review"), ("ignore", "ignore"),
    ("REVIEW", "review"), ("  ignore  ", "ignore"),
])
def test_parse_whole_image_match_accepts_valid_values(value, expected):
    assert poller.parse_whole_image_match(value) == expected


@pytest.mark.parametrize("value", [None, "", "   ", "reveiw", "off", "true", "1"])
def test_parse_whole_image_match_falls_back_to_tag(value):
    """A typo must not silently stop the tagger tagging, so anything unrecognized keeps
    prior behavior rather than quietly selecting a stricter mode."""
    assert poller.parse_whole_image_match(value) == "tag"


def outcome_for(mode: str, prob: float, full_image: bool = True) -> str:
    return poller.classify_outcome("Dobby", prob, "2026-06-01T00:00:00Z", {}, threshold=0.8,
                                   full_image=full_image, whole_image_match=mode)


def test_whole_image_tag_mode_is_prior_behaviour():
    """The default must leave a whole-image match indistinguishable from a
    detection-backed one, so upgrading without setting the var changes nothing."""
    assert outcome_for("tag", 0.95) == "confident"
    assert outcome_for("tag", 0.67) == "low_confidence"


def test_whole_image_review_mode_withholds_confident_match():
    """The point of review mode: however confident, it goes to the queue, not to Immich."""
    assert outcome_for("review", 0.95) == "whole_image_review"
    assert outcome_for("review", 0.999) == "whole_image_review"


def test_whole_image_review_mode_leaves_low_confidence_alone():
    """Below the threshold it was already heading for the queue; it stays a plain
    low-confidence entry so the withheld-auto-tags stat counts only real withholds."""
    assert outcome_for("review", 0.67) == "low_confidence"


def test_whole_image_ignore_mode_drops_at_any_confidence():
    """Ignore means gone, not queued: neither confident nor low-confidence whole-image
    matches should reach the review queue."""
    assert outcome_for("ignore", 0.95) == "whole_image_ignored"
    assert outcome_for("ignore", 0.67) == "whole_image_ignored"


def test_whole_image_modes_never_affect_detection_backed_crops():
    """A real detection is untouched by the setting, in every mode."""
    for mode in poller.WHOLE_IMAGE_MODES:
        assert outcome_for(mode, 0.95, full_image=False) == "confident"
        assert outcome_for(mode, 0.67, full_image=False) == "low_confidence"


def test_whole_image_out_of_range_still_wins():
    """Date range is still checked first: an out-of-range guess is dropped as such
    rather than being reported as a withheld whole-image match."""
    cfg = {"since": "2025-01-01", "until": "2025-12-31"}
    for mode in poller.WHOLE_IMAGE_MODES:
        outcome = poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", cfg,
                                          threshold=0.8, full_image=True, whole_image_match=mode)
        assert outcome == "out_of_range"


def test_whole_image_unknown_still_wins():
    for mode in poller.WHOLE_IMAGE_MODES:
        assert poller.classify_outcome("unknown", 0.95, "2026-06-01T00:00:00Z", {},
                                       full_image=True, whole_image_match=mode) == "unknown"


def test_whole_image_match_defaults_to_module_setting(monkeypatch):
    """Omitting the argument falls back to the env-configured module value."""
    monkeypatch.setattr(poller, "WHOLE_IMAGE_MATCH", "review")
    assert poller.classify_outcome("Dobby", 0.95, "2026-06-01T00:00:00Z", {},
                                   threshold=0.8, full_image=True) == "whole_image_review"


# ---------------------------------------------------------------------------
# WHOLE_IMAGE_MATCH through a real poll cycle
#
# process_asset is a closure inside _run_poll_cycle, so these drive a whole
# one-asset cycle with Immich, YOLO and CLIP stubbed out.
# ---------------------------------------------------------------------------

class _Recorder:
    """Collects the side effects a poll cycle would have on Immich and the caches."""

    def __init__(self):
        self.faces = []
        self.stored_crops = []
        self.embedded = 0


@pytest.fixture
def cycle(monkeypatch, tmp_path):
    """Run a single-asset poll cycle. Caller supplies what YOLO detected."""
    rec = _Recorder()

    def run(mode, detected, prob=0.99, low_conf_out=None):
        monkeypatch.setattr(poller, "WHOLE_IMAGE_MATCH", mode)
        monkeypatch.setattr(poller, "THRESHOLD", 0.8)
        monkeypatch.setattr(poller, "THRESHOLD_FALLBACK", 0.8)

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
                            lambda ts: [("asset-1", "2026-07-26T10:00:00Z", "2026-07-26T10:00:00Z")])
        monkeypatch.setattr(poller.imm, "fetch_asset_face_person_ids", lambda aid: set())

        def fake_post_face(aid, person_id, bbox_norm=None, img_size=None):
            rec.faces.append((aid, person_id, bbox_norm, img_size))
            return "face-1"

        monkeypatch.setattr(poller.imm, "post_face_sync", fake_post_face)

        img = types.SimpleNamespace(size=(100, 100))
        monkeypatch.setattr(poller.emb, "fetch_thumbnail", lambda aid: img)
        monkeypatch.setattr(poller.emb, "crop_animals", lambda i, conf=None: detected)

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


def test_cycle_tag_mode_tags_whole_image_match(cycle):
    """Default behaviour, unchanged: a confident whole-image match is written to Immich,
    with no bbox and no image size, exactly as before this setting existed."""
    counts, rec = cycle("tag", detected=[])
    assert counts["added"] == 1
    assert rec.faces == [("asset-1", "person-1", None, None)]


def test_cycle_review_mode_queues_instead_of_tagging(cycle):
    low_conf = []
    counts, rec = cycle("review", detected=[], low_conf_out=low_conf)
    assert rec.faces == []
    assert counts["added"] == 0
    assert counts["whole_image_review"] == 1
    assert counts["low_confidence"] == 0
    assert low_conf == [{"asset_id": "asset-1", "pet_name": "Rex", "prob": 0.99,
                         "date": "2026-07-26", "bbox": None, "outcome": "whole_image_review"}]


def test_cycle_ignore_mode_drops_without_embedding(cycle):
    """Ignore should not just discard the result, it should never pay for it: no CLIP
    embedding is computed for an asset with no detection."""
    low_conf = []
    counts, rec = cycle("ignore", detected=[], low_conf_out=low_conf)
    assert rec.faces == []
    assert rec.embedded == 0
    assert low_conf == []
    assert counts["whole_image_ignored"] == 1
    assert counts["added"] == 0


def test_cycle_ignore_mode_still_caches_empty_detection(cycle):
    """Skipping early must still record 'no animal here' so later passes and the
    suggestion/borderline tools don't re-run YOLO on this asset."""
    _, rec = cycle("ignore", detected=[])
    assert rec.stored_crops == [("asset-1", [])]


@pytest.mark.parametrize("mode", poller.WHOLE_IMAGE_MODES)
def test_cycle_detected_animal_is_tagged_in_every_mode(cycle, mode):
    """The setting only governs the fallback: a real detection tags in all three modes."""
    counts, rec = cycle(mode, detected=[((0.1, 0.2, 0.5, 0.6), object())])
    assert counts["added"] == 1
    assert counts["whole_image_review"] == 0
    assert counts["whole_image_ignored"] == 0
    assert rec.faces == [("asset-1", "person-1", (0.1, 0.2, 0.5, 0.6), (100, 100))]
