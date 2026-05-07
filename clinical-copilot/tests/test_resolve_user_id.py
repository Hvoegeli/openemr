"""Isolated tests for the username → users.id dynamic resolver.

Two layers under test:

- `OpenEMRWriter.resolve_user_id_by_username` — the HTTP-side helper that
  hits OpenEMR's standard-API `/api/user?username=…`, surfaces the
  integer `id`, and returns None on no-match / inactive-only matches.
  HTTP is mocked with a tiny stand-in for `httpx.AsyncClient` so the
  test runs without OpenEMR.

- `access_control.resolve_user_id` — the small async helper that
  (a) consults the static `USERNAME_TO_USER_ID` override, then falls
  back to (b) the writer's lookup, with a process-lifetime cache.

The cache lives in module-level state so each test resets it via the
`autouse` fixture below — otherwise tests that run after a hit would
see stale cache entries and produce false greens.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app import access_control
from app.fhir.writer import OpenEMRWriteError, OpenEMRWriter


@pytest.fixture(autouse=True)
def _reset_resolver_cache() -> None:
    """Drop both ACL caches before/after each test so resolution state
    can't leak across cases. The dict-clear matches what
    `invalidate_panel(None)` does in production."""
    access_control._RESOLVED_USER_ID.clear()
    yield
    access_control._RESOLVED_USER_ID.clear()


# ─── Writer-side: HTTP behavior ───────────────────────────────────────


class _FakeHttpResponse:
    def __init__(self, *, status_code: int, body: Any, raw_text: str | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.text = raw_text if raw_text is not None else json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeHttpClient:
    """Captures the GET call so the test can assert on URL + params,
    and replays a single canned response."""

    def __init__(self, response: _FakeHttpResponse) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def get(
        self, url: str, *, headers: dict | None = None, params: dict | None = None,
    ) -> _FakeHttpResponse:
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self._response

    async def post(self, *args: Any, **kwargs: Any) -> _FakeHttpResponse:
        raise AssertionError("user-id resolver should never POST")

    async def aclose(self) -> None:
        pass


def _wired_writer(http: _FakeHttpClient) -> OpenEMRWriter:
    """Return an OpenEMRWriter with `_http` swapped for the fake and
    `_ensure_token` short-circuited so we don't try a real OAuth round."""
    w = OpenEMRWriter()
    w._http = http  # type: ignore[assignment]

    async def _fake_token() -> str:
        return "fake-token-xyz"

    w._ensure_token = _fake_token  # type: ignore[assignment]
    return w


class TestWriterResolveUserId:
    """Covers the HTTP transport. We assert the URL the writer hits and
    the row-walk that picks the active match."""

    @pytest.mark.asyncio
    async def test_returns_int_id_for_active_match(self) -> None:
        http = _FakeHttpClient(
            _FakeHttpResponse(
                status_code=200,
                body={"data": [{"id": 5, "username": "Smith", "active": 1}]},
            ),
        )
        w = _wired_writer(http)
        try:
            assert await w.resolve_user_id_by_username("Smith") == 5
        finally:
            await w.aclose()
        # Hits /api/user with the right query param and a Bearer header.
        assert len(http.calls) == 1
        call = http.calls[0]
        assert call["url"].endswith("/api/user")
        assert call["params"] == {"username": "Smith"}
        assert call["headers"]["Authorization"] == "Bearer fake-token-xyz"

    @pytest.mark.asyncio
    async def test_empty_data_returns_none(self) -> None:
        http = _FakeHttpClient(
            _FakeHttpResponse(status_code=200, body={"data": []}),
        )
        w = _wired_writer(http)
        try:
            assert await w.resolve_user_id_by_username("nobody") is None
        finally:
            await w.aclose()

    @pytest.mark.asyncio
    async def test_inactive_only_match_returns_none(self) -> None:
        # `active=0` integer (one of the falsy forms OpenEMR can emit
        # depending on which serializer is in front of the route).
        http = _FakeHttpClient(
            _FakeHttpResponse(
                status_code=200,
                body={"data": [{"id": 9, "username": "ghost", "active": 0}]},
            ),
        )
        w = _wired_writer(http)
        try:
            assert await w.resolve_user_id_by_username("ghost") is None
        finally:
            await w.aclose()

    @pytest.mark.asyncio
    async def test_inactive_then_active_picks_active(self) -> None:
        # If a search ever returns multiple rows (former-username
        # collision), pick the active one.
        http = _FakeHttpClient(
            _FakeHttpResponse(
                status_code=200,
                body={"data": [
                    {"id": 99, "username": "smith", "active": False},
                    {"id": 5, "username": "Smith", "active": True},
                ]},
            ),
        )
        w = _wired_writer(http)
        try:
            assert await w.resolve_user_id_by_username("Smith") == 5
        finally:
            await w.aclose()

    @pytest.mark.asyncio
    async def test_non_200_raises(self) -> None:
        http = _FakeHttpClient(
            _FakeHttpResponse(
                status_code=500, body={}, raw_text="boom",
            ),
        )
        w = _wired_writer(http)
        try:
            with pytest.raises(OpenEMRWriteError, match="user lookup failed"):
                await w.resolve_user_id_by_username("Smith")
        finally:
            await w.aclose()

    @pytest.mark.asyncio
    async def test_non_json_body_raises(self) -> None:
        bad = _FakeHttpResponse(
            status_code=200,
            body=ValueError("not json"),
            raw_text="<html>error</html>",
        )
        http = _FakeHttpClient(bad)
        w = _wired_writer(http)
        try:
            with pytest.raises(OpenEMRWriteError, match="non-JSON"):
                await w.resolve_user_id_by_username("Smith")
        finally:
            await w.aclose()

    @pytest.mark.asyncio
    async def test_empty_username_returns_none_without_http_call(self) -> None:
        http = _FakeHttpClient(
            _FakeHttpResponse(status_code=200, body={"data": []}),
        )
        w = _wired_writer(http)
        try:
            assert await w.resolve_user_id_by_username("") is None
        finally:
            await w.aclose()
        assert http.calls == []


# ─── access_control side: static + cache + writer fallback ────────────


class _CountingWriter:
    """Just enough of an OpenEMRWriter shape to satisfy
    `access_control.resolve_user_id` — counts calls and returns
    canned values so we can exercise the cache-vs-fetch logic."""

    def __init__(self, return_value: int | None | Exception) -> None:
        self._return = return_value
        self.calls: list[str] = []

    async def resolve_user_id_by_username(self, username: str) -> int | None:
        self.calls.append(username)
        if isinstance(self._return, Exception):
            raise self._return
        return self._return


class TestAccessControlResolveUserId:
    @pytest.mark.asyncio
    async def test_static_override_short_circuits_writer(self) -> None:
        # `admin` is in USERNAME_TO_USER_ID with id=1; the writer must
        # not be consulted for it.
        w = _CountingWriter(return_value=99)
        out = await access_control.resolve_user_id(w, "admin")  # type: ignore[arg-type]
        assert out == 1
        assert w.calls == []

    @pytest.mark.asyncio
    async def test_dynamic_hit_caches_result(self) -> None:
        w = _CountingWriter(return_value=7)
        # First call hits the writer.
        assert await access_control.resolve_user_id(w, "Cohen") == 7  # type: ignore[arg-type]
        # Second call uses the cache.
        assert await access_control.resolve_user_id(w, "Cohen") == 7  # type: ignore[arg-type]
        assert w.calls == ["Cohen"]

    @pytest.mark.asyncio
    async def test_dynamic_miss_caches_none(self) -> None:
        w = _CountingWriter(return_value=None)
        assert await access_control.resolve_user_id(w, "ghost") is None  # type: ignore[arg-type]
        assert await access_control.resolve_user_id(w, "ghost") is None  # type: ignore[arg-type]
        # Cached miss should NOT re-hit the writer.
        assert w.calls == ["ghost"]

    @pytest.mark.asyncio
    async def test_writer_failure_returns_none_but_does_not_cache(self) -> None:
        # Transient transport failure must not pin the user as
        # unmapped — caller (typically an HTTP handler) gets a clean
        # None back, but a follow-up call gets to retry.
        boom = _CountingWriter(return_value=OpenEMRWriteError("HTTP 500"))
        assert await access_control.resolve_user_id(boom, "Hale") is None  # type: ignore[arg-type]
        assert "hale" not in access_control._RESOLVED_USER_ID
        # Now flip the writer to a hit; the next call should retry and succeed.
        good = _CountingWriter(return_value=12)
        assert await access_control.resolve_user_id(good, "Hale") == 12  # type: ignore[arg-type]
        assert good.calls == ["Hale"]

    @pytest.mark.asyncio
    async def test_empty_username_returns_none(self) -> None:
        w = _CountingWriter(return_value=42)
        assert await access_control.resolve_user_id(w, None) is None  # type: ignore[arg-type]
        assert await access_control.resolve_user_id(w, "") is None  # type: ignore[arg-type]
        assert w.calls == []

    @pytest.mark.asyncio
    async def test_invalidate_panel_clears_user_id_cache(self) -> None:
        w = _CountingWriter(return_value=33)
        assert await access_control.resolve_user_id(w, "Cohen") == 33  # type: ignore[arg-type]
        # Pre-condition: cache populated.
        assert access_control._RESOLVED_USER_ID["cohen"] == 33

        access_control.invalidate_panel("Cohen")
        assert "cohen" not in access_control._RESOLVED_USER_ID

        # After invalidation we re-fetch via the writer.
        w2 = _CountingWriter(return_value=44)
        assert await access_control.resolve_user_id(w2, "Cohen") == 44  # type: ignore[arg-type]
        assert w2.calls == ["Cohen"]
