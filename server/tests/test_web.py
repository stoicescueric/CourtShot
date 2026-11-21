"""The dashboard files and the security headers on every response."""

import pytest
from pydantic import ValidationError

from app.settings import Settings
from app.web import NOT_BUILT, SECURITY_HEADERS


def assert_secure(response) -> None:
    for name, value in SECURITY_HEADERS.items():
        assert response.headers[name] == value


def test_dashboard_not_built(anonymous):
    response = anonymous.get("/")
    assert response.status_code == 200 and response.text == NOT_BUILT
    assert response.headers["content-type"].startswith("text/plain")
    assert_secure(response)


def test_dashboard_files_and_fallback(anonymous, settings, tmp_path):
    dist = settings.web_dist
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>CourtShot</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("secret")

    asset = anonymous.get("/assets/app.js")
    assert asset.text == "console.log(1)" and "javascript" in asset.headers["content-type"]
    assert_secure(asset)
    for path in ("/", "/matches/3", "/login", "/index.html"):
        page = anonymous.get(path)
        assert page.status_code == 200 and "CourtShot" in page.text
        assert page.headers["cache-control"] == "no-cache"
        assert_secure(page)
    for trick in ("/..%2Fsecret.txt", "/assets/..%2F..%2Fsecret.txt", "/%2e%2e/secret.txt"):
        assert "secret" not in anonymous.get(trick).text
    for unusable in ("/a%00b", "/assets/" + "a" * 300):  # a NUL byte, a name the file system refuses
        page = anonymous.get(unusable)
        assert page.status_code == 200 and "CourtShot" in page.text
        assert_secure(page)


def test_unknown_api_paths_are_json_404(anonymous):
    for path in ("/api", "/api/nothing", "/api/matches/1/nothing"):
        response = anonymous.get(path)
        assert response.status_code in (401, 404)
        assert response.headers["content-type"] == "application/json"
        assert_secure(response)
    assert anonymous.get("/api/nothing").json() == {"detail": "Not Found"}


def test_security_headers_on_api_errors_and_files(browser, anonymous):
    assert_secure(anonymous.get("/api/auth/me"))
    assert_secure(browser.get("/api/config"))
    assert "unsafe" not in SECURITY_HEADERS["Content-Security-Policy"]  # design 8.4: the site itself only


def test_config(browser):
    assert browser.get("/api/config").json() == {
        "min_video_seconds": 10,
        "max_video_seconds": 900,
        "max_upload_bytes": 8 * 1024**3,
        "chunk_bytes": 8 * 1024**2,
        "calibration_max_error_m": 0.25,
        "min_calibration_points": 6,
        "extensions": ["mp4", "mov", "m4v", "webm"],
        "worker_online_seconds": 30,
    }


def test_errors_carry_a_string_detail(browser):
    response = browser.post("/api/matches", json={"title": 3})
    assert response.status_code == 422 and isinstance(response.json()["detail"], str)


def test_settings_refuse_values_that_would_silently_break_checks(tmp_path):
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, public_origin="https://courtshot.example.org/")
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, trusted_proxy="localhost")
    with pytest.raises(ValidationError):
        Settings(data_dir=tmp_path, chunk_bytes=32 * 1024**2)
    assert Settings(data_dir=tmp_path, trusted_proxy="127.0.0.1").trusted_proxy == "127.0.0.1"
