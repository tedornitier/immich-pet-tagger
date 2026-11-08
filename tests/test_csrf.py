import pytest
from fastapi.testclient import TestClient

import main

# TestClient sends "Host: testserver", so this is the UI's own origin.
OWN = "http://testserver"
EVIL = "http://evil.example"


@pytest.fixture
def client():
    # No "with": the lifespan (poller, embed cache) is not needed to test request filtering.
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def no_trusted_origins(monkeypatch):
    monkeypatch.setattr(main, "TRUSTED_ORIGINS", set(), raising=False)


def test_same_origin_post_passes(client):
    assert client.post("/api/scan/stop", headers={"Origin": OWN}).status_code == 200


def test_post_without_origin_passes(client):
    # curl and scripts send no Origin header and must keep working.
    assert client.post("/api/scan/stop").status_code == 200


def test_foreign_origin_post_blocked(client):
    r = client.post("/api/scan/stop", headers={"Origin": EVIL})
    assert r.status_code == 403
    assert r.json() == {"detail": "Cross-site request blocked."}


@pytest.mark.parametrize("origin", ["null", "http://testserver:8080", "http://testserver.evil.example",
                                    "http://testserver@evil.example"])
def test_lookalike_origins_blocked(client, origin):
    assert client.post("/api/scan/stop", headers={"Origin": origin}).status_code == 403


def test_tls_terminating_proxy_keeping_host_passes(client):
    # Browser on https, proxy keeps Host and talks plain http to the app: only host and port
    # are compared, never the scheme, or every request behind such a proxy would be blocked.
    r = client.post("/api/scan/stop", headers={"Host": "pets.example.com", "Origin": "https://pets.example.com"})
    assert r.status_code == 200


def test_origin_compared_case_insensitive(client):
    assert client.post("/api/scan/stop", headers={"Origin": "HTTP://TestServer/"}).status_code == 200


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def test_every_non_get_method_checked(client, method):
    r = client.request(method, "/api/pets/Lisa", headers={"Origin": EVIL})
    assert r.status_code == 403


def test_malformed_origin_blocked_not_crashing(client):
    r = client.post("/api/scan/stop", headers={"Origin": "http://[::1"})
    assert r.status_code == 403


def test_json_post_without_content_type_is_parsed(client):
    # The case from the issue: FastAPI 0.115 parses a body without Content-Type as JSON, so a
    # cross-site no-cors fetch can reach JSON endpoints. An invalid date shows the body was
    # parsed (400 from the handler, not 422) without starting a scan.
    r = client.post("/api/scan", content=b'{"scan_since": "not-a-date"}')
    assert r.status_code == 400


def test_json_post_without_content_type_blocked_cross_site(client):
    r = client.post("/api/scan", content=b'{"scan_since": "not-a-date"}', headers={"Origin": EVIL})
    assert r.status_code == 403


def test_reads_from_foreign_origin_pass(client):
    # GET does not change state; blocking it would break links and bookmarks.
    assert client.get("/api/version", headers={"Origin": EVIL, "Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_sec_fetch_site_cross_site_blocked_even_without_origin(client):
    assert client.post("/api/scan/stop", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_sec_fetch_site_same_site_still_checks_origin(client):
    # same-site = another origin on the same site (other port or subdomain).
    r = client.post("/api/scan/stop", headers={"Sec-Fetch-Site": "same-site", "Origin": "http://immich.testserver"})
    assert r.status_code == 403


def test_same_origin_form_with_null_origin_passes(client):
    # Browsers send "Origin: null" for same-origin form posts under Referrer-Policy: no-referrer.
    r = client.post("/api/scan/stop", headers={"Sec-Fetch-Site": "same-origin", "Origin": "null"})
    assert r.status_code == 200


# Reverse proxy that rewrites Host (e.g. nginx without "proxy_set_header Host $host"):
# the browser is on https://pets.example.com, the app sees Host: immich-pet-tagger:8000.
PROXY = {"Host": "immich-pet-tagger:8000", "Origin": "https://pets.example.com"}


def test_rewritten_host_modern_browser_passes(client):
    r = client.post("/api/scan/stop", headers={**PROXY, "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 200


def test_rewritten_host_without_sec_fetch_site_blocked_by_default(client):
    assert client.post("/api/scan/stop", headers=PROXY).status_code == 403


def test_rewritten_host_trusted_origin_passes(client, monkeypatch):
    monkeypatch.setattr(main, "TRUSTED_ORIGINS", {"https://pets.example.com"})
    assert client.post("/api/scan/stop", headers=PROXY).status_code == 200


def test_trusted_origin_does_not_open_other_origins(client, monkeypatch):
    monkeypatch.setattr(main, "TRUSTED_ORIGINS", {"https://pets.example.com"})
    headers = {**PROXY, "Origin": EVIL}
    assert client.post("/api/scan/stop", headers=headers).status_code == 403


def test_rewritten_host_cross_site_still_blocked(client, monkeypatch):
    monkeypatch.setattr(main, "TRUSTED_ORIGINS", {"https://pets.example.com"})
    headers = {**PROXY, "Origin": EVIL, "Sec-Fetch-Site": "cross-site"}
    assert client.post("/api/scan/stop", headers=headers).status_code == 403
