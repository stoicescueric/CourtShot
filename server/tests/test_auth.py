"""Sign-in: hashing, throttling before hashing, cookies, sessions, origin check (design 13)."""

import shutil
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update

import app.auth as auth
import app.routes.auth as auth_routes
from app.main import create_app
from app.models import Session, User, Worker
from app.settings import Settings
from conftest import ORIGIN, PASSWORD, USERNAME, add_user, create_match, queued_match


def login(client: TestClient, password: str = PASSWORD, username: str = USERNAME, **headers: str):
    return client.post("/api/auth/login", json={"username": username, "password": password}, headers=headers)


def test_password_is_stored_as_argon2id(db, anonymous):
    add_user(db)
    with db.read() as s:
        assert s.scalar(select(User.password_hash)).startswith("$argon2id$")
    assert login(anonymous).json() == {"username": USERNAME}


def test_wrong_password_and_unknown_user_are_refused(db, anonymous):
    add_user(db)
    assert login(anonymous, password="wrong password!").status_code == 401
    assert login(anonymous, username="nobody").status_code == 401
    assert anonymous.get("/api/auth/me").status_code == 401


def test_unknown_user_costs_one_verification(monkeypatch):
    calls = []
    real = auth._hasher
    spy = SimpleNamespace(verify=lambda h, p: calls.append(h) or real.verify(h, p))
    monkeypatch.setattr(auth, "_hasher", spy)
    assert auth.verify_password(None, PASSWORD) is False
    assert calls == [auth._DUMMY_HASH]


def test_secure_session_cookie(db, anonymous, settings):
    add_user(db)
    cookie = login(anonymous).headers["set-cookie"]
    assert cookie.startswith("__Host-courtshot_session=")
    for flag in ("HttpOnly", "Secure", "SameSite=strict", "Path=/"):
        assert flag.lower() in cookie.lower()
    assert "domain" not in cookie.lower()
    assert anonymous.get("/api/auth/me").json() == {"username": USERNAME}


def test_local_development_cookie_without_secure(tmp_path, migrated_db):
    data = tmp_path / "data"
    data.mkdir()
    shutil.copy(migrated_db, data / "app.db")
    settings = Settings(data_dir=data, public_origin="http://testserver", secure_cookies=False)
    app = create_app(settings)
    add_user(app.state.db)
    client = TestClient(app, headers={"Origin": "http://testserver"})
    cookie = login(client).headers["set-cookie"]
    assert cookie.startswith("courtshot_session=")
    assert "secure" not in cookie.lower()
    assert client.get("/api/auth/me").status_code == 200


def test_only_the_token_hash_is_stored(db, anonymous):
    add_user(db)
    login(anonymous)
    token = anonymous.cookies.get("__Host-courtshot_session")
    with db.read() as s:
        stored = s.scalar(select(Session.token_hash))
    assert stored == auth.token_hash(token) and token not in stored


def test_logout_deletes_the_session(browser, db):
    assert browser.post("/api/auth/logout").status_code == 204
    with db.read() as s:
        assert s.scalar(select(Session)) is None
    assert browser.get("/api/auth/me").status_code == 401


def _age_session(db, **ages: timedelta) -> None:
    now = auth.utcnow()
    with db.write() as s:
        s.execute(update(Session).values(**{column: now - age for column, age in ages.items()}))


def test_session_expires_after_seven_idle_days(browser, db):
    _age_session(db, last_seen_at=timedelta(days=7, seconds=1))
    assert browser.get("/api/auth/me").status_code == 401


def test_session_expires_after_thirty_days(browser, db):
    _age_session(db, expires_at=timedelta(seconds=1))
    assert browser.get("/api/auth/me").status_code == 401


def test_last_seen_is_written_at_most_once_a_minute(browser, db):
    _age_session(db, last_seen_at=timedelta(seconds=30))
    with db.read() as s:
        before = s.scalar(select(Session.last_seen_at))
    browser.get("/api/auth/me")
    with db.read() as s:
        assert s.scalar(select(Session.last_seen_at)) == before
    _age_session(db, last_seen_at=timedelta(minutes=2))
    browser.get("/api/auth/me")
    with db.read() as s:
        assert auth.utcnow() - s.scalar(select(Session.last_seen_at)) < timedelta(seconds=5)


def test_throttle_is_checked_before_hashing(db, anonymous, monkeypatch):
    add_user(db)
    verified = []
    real = auth_routes.verify_password
    monkeypatch.setattr(auth_routes, "verify_password", lambda h, p: verified.append(1) or real(h, p))
    for _ in range(5):
        assert login(anonymous, password="wrong password!").status_code == 401
    throttled = login(anonymous)  # even the right password
    assert throttled.status_code == 429
    assert int(throttled.headers["retry-after"]) > 0
    assert len(verified) == 5


def test_throttle_is_per_address_behind_the_trusted_proxy(app, db, anonymous):
    add_user(db)
    app.state.settings = app.state.settings.model_copy(update={"trusted_proxy": "testclient"})
    for _ in range(5):
        login(anonymous, password="wrong password!", **{"X-Forwarded-For": "203.0.113.9"})
    assert login(anonymous, **{"X-Forwarded-For": "203.0.113.9"}).status_code == 429
    assert login(anonymous, **{"X-Forwarded-For": "198.51.100.7"}).status_code == 200


def test_success_forgives_its_attempt():
    throttle = auth.LoginThrottle(per_address=2)
    for _ in range(5):
        throttle.forgive("a", throttle.admit("a"))
    throttle.admit("a")
    throttle.admit("a")
    with pytest.raises(auth.HTTPException) as refused:
        throttle.admit("a")
    assert refused.value.status_code == 429


def test_global_budget_spaces_out_sign_ins_without_locking(monkeypatch):
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock.now)
    throttle = auth.LoginThrottle(per_address=100, global_budget=3, spacing_s=3)
    for address in ("a", "b", "c"):
        throttle.admit(address)
    with pytest.raises(auth.HTTPException) as refused:
        throttle.admit("d")  # spent budget, too soon after the last attempt
    assert refused.value.status_code == 429
    clock.now += 3
    throttle.admit("d")  # spaced out, not locked
    clock.now += 901
    throttle.admit("e")
    throttle.admit("f")  # the window has passed: no spacing


def test_at_most_two_hash_operations_at_once(monkeypatch):
    running, peak = [0], [0]
    lock = threading.Lock()

    def slow_verify(_hash: str, _password: str) -> bool:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return True

    monkeypatch.setattr(auth, "_hasher", SimpleNamespace(verify=slow_verify))
    threads = [threading.Thread(target=auth.verify_password, args=("$argon2id$x", "p")) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak[0] == 2


def _request(peer: str, forwarded: str | None = None):
    headers = {"x-forwarded-for": forwarded} if forwarded else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


def test_client_address_from_the_trusted_proxy_only():
    assert auth.client_address(_request("127.0.0.1", "1.2.3.4, 10.0.0.1"), "127.0.0.1") == "10.0.0.1"
    assert auth.client_address(_request("127.0.0.1", "garbage"), "127.0.0.1") == "127.0.0.1"
    assert auth.client_address(_request("9.9.9.9", "10.0.0.1"), "127.0.0.1") == "9.9.9.9"
    assert auth.client_address(_request("9.9.9.9", "10.0.0.1"), "") == "9.9.9.9"


def test_origin_is_required_on_cookie_routes_that_change_state(browser):
    assert browser.get("/api/matches", headers={"Origin": ""}).status_code == 200
    browser.headers.pop("Origin")
    assert browser.post("/api/matches", json={}).status_code == 403
    assert browser.post("/api/matches", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    assert browser.post("/api/auth/logout").status_code == 403
    browser.headers["Origin"] = ORIGIN
    create_match(browser)


def test_worker_routes_ignore_origin_and_cookies(browser, worker):
    queued_match(browser)
    # A session cookie is not a worker credential.
    assert browser.post("/api/worker/claim", json={"info": {}}).status_code == 401
    claim = worker.http.post("/api/worker/claim", json={"info": {}}, headers={"Origin": "https://evil.example"})
    assert claim.status_code == 200
    assert worker.heartbeat(claim.json()["job_id"], 1).status_code == 200


def test_revoked_or_unknown_worker_token_is_refused(app, db, worker):
    with db.write() as s:
        s.execute(update(Worker).values(revoked_at=auth.utcnow()))
    assert worker.claim().status_code == 401
    bogus = TestClient(app, base_url=ORIGIN, headers={"Authorization": "Bearer not-a-token"})
    assert bogus.post("/api/worker/claim", json={"info": {}}).status_code == 401
