"""Tests for poller helpers, in particular the poll cursor advance logic."""
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


def _stub_scan(monkeypatch, assets, thumb):
    import classifier as clf_mod
    import data
    import embedder as emb
    import immich as imm
    import inference

    monkeypatch.setattr(data, "load_config", lambda dd: {"Rex": {"person_id": "p1"}})
    monkeypatch.setattr(data, "load_pet_refs", lambda name, dd: ["ref"])
    monkeypatch.setattr(data, "load_negatives", lambda dd: [])
    monkeypatch.setattr(clf_mod, "build_classifier", lambda *a: (["Rex"], None, None))
    monkeypatch.setattr(imm, "fetch_assets_taken_after", lambda since, before: assets)
    monkeypatch.setattr(emb, "fetch_thumbnail", thumb)
    monkeypatch.setattr(emb, "save_embed_cache", lambda: None)
    monkeypatch.setattr(inference, "describe_model_failure", lambda before: None)


def test_scan_counts_processed_for_every_outcome(tmp_path, monkeypatch):
    """processed counts each finished asset, whether it had no thumbnail or raised."""
    assets = [(f"a{i}", "2026-01-02T00:00:00.000Z") for i in range(5)]

    def thumb(aid):
        if aid == "a0":
            raise RuntimeError("boom")
        return None

    _stub_scan(monkeypatch, assets, thumb)
    counts: dict = {}
    poller.run_poll_cycle(str(tmp_path), live_counts=counts, manual=True, scan_since="2026-01-01T00:00:00.000Z")

    assert counts["fetched"] == 5
    assert counts["processed"] == 5
    assert counts["no_thumb"] == 4
    assert counts["failed"] == 1


def test_scan_processed_is_live_during_scan(tmp_path, monkeypatch):
    """The shared counts dict exposes progress while assets are still being processed."""
    assets = [(f"a{i}", "2026-01-02T00:00:00.000Z") for i in range(3)]
    counts: dict = {}
    seen = []

    def thumb(aid):
        seen.append((counts.get("fetched"), counts.get("processed")))
        return None

    import embedder as emb
    monkeypatch.setattr(emb, "SCAN_WORKERS", 1)
    _stub_scan(monkeypatch, assets, thumb)
    poller.run_poll_cycle(str(tmp_path), live_counts=counts, manual=True, scan_since="2026-01-01T00:00:00.000Z")

    assert all(total == 3 for total, _ in seen)
    assert seen[0][1] == 0
    assert counts["processed"] == 3


def test_processed_is_not_persisted_to_poll_status(tmp_path, monkeypatch):
    import data
    assets = [("a0", "2026-01-02T00:00:00.000Z")]
    _stub_scan(monkeypatch, assets, lambda aid: None)
    poller.run_poll_cycle(str(tmp_path), live_counts={}, manual=True, scan_since="2026-01-01T00:00:00.000Z")
    status = data.load_poll_status(tmp_path)
    assert status["counts"]["fetched"] == 1
    assert "processed" not in status["counts"]
