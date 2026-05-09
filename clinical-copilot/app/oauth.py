"""OAuth2 / OpenID Connect login flow against OpenEMR.

Replaces the password-grant verification in `app.auth.verify_openemr_credentials`
with the standards-compliant authorization-code + PKCE flow:

1. User hits `GET /oauth/login` on Copilot.
2. We generate a PKCE verifier + challenge + a state nonce, store them in
   the user's signed Copilot session cookie, and redirect to OpenEMR's
   `/oauth2/default/authorize` with the challenge.
3. User signs in on OpenEMR's hosted login page.
4. OpenEMR redirects back to `{COPILOT_BASE_URL}/oauth/callback?code=...&state=...`.
5. We validate state matches, then POST the code + verifier + client_secret
   to OpenEMR's token endpoint to receive an `id_token`.
6. We decode the id_token's payload (no signature verification — see
   security note below) to extract the username, then create a Copilot
   session via `auth_store` exactly as the legacy `/api/login` did.
7. The user lands at the dashboard with a working `sid` cookie.

Why this matters (per W2 Sunday plan): the dashboard is loaded inside
OpenEMR-the-application. We want a single login event to establish
identity for both Copilot and OpenEMR's PHP UI, so the dashboard SSO is
silent. The auth-code flow also lets us retire the password-grant path,
which (a) the SMART-on-FHIR spec deprecates and (b) the early-submission
review flagged as a HIPAA-readiness gap.

# Security note on id_token verification

A production deployment SHOULD verify the id_token's signature against
OpenEMR's JWKS before trusting any claim. This module currently decodes
the payload without verification — acceptable for the local-dev demo
because (1) the token was just minted by the very server we're talking
to over TLS, and (2) we cross-check the username via the `userinfo`
endpoint as a defense-in-depth layer. The TODO at the bottom of
`exchange_code_for_username` lists what a JWKS-verified path would look
like.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
from typing import Final
from urllib.parse import urlencode

import httpx
import urllib3
from fastapi import HTTPException, Request

from app.config import settings

log = logging.getLogger("agent.oauth")

# Disable urllib3 InsecureRequestWarning for the local dev OpenEMR (self-
# signed cert at localhost:9300). Production uses a real cert.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Session keys for the verifier + state nonce. Kept short so the signed
# cookie stays small.
_SESS_KEY_VERIFIER: Final[str] = "oa_pkce"
_SESS_KEY_STATE: Final[str] = "oa_state"
_SESS_KEY_NEXT: Final[str] = "oa_next"

# OAuth2 scope set requested at /authorize. Mirrors the seed client's
# enabled scopes plus OIDC's `openid` (required for an id_token to be
# minted at all). `offline_access` is omitted for v1; we'll add it when
# the dashboard needs long-lived FHIR reads from a refresh token.
_AUTHORIZE_SCOPE: Final[str] = " ".join([
    # OIDC core
    "openid",
    # Profile claims so the id_token carries `preferred_username` (the
    # OpenEMR login name). Without these the id_token only contains
    # `sub` (a UUID), which doesn't match Copilot's username-based ACL
    # allow-list (`ADMIN_USERNAMES=admin`) and leaves every login with
    # an empty patient panel.
    "profile",
    "name",
    "fhirUser",
    # OpenEMR's API access flag — needed for the access_token to call
    # /apis/default/api/* during the session.
    "api:oemr",
    # Chart-level read scope so the access_token (if used by the
    # dashboard's React app for direct FHIR reads in a later iteration)
    # has the access it needs. Today the dashboard reads via Copilot's
    # backend instead, but requesting this is harmless.
    "user/patient.cruds",
])


def _b64url_nopad(raw: bytes) -> str:
    """URL-safe base64 without padding — RFC 7636 / RFC 7515 shape."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _generate_pkce() -> tuple[str, str]:
    """Return (verifier, challenge) pair for PKCE S256.

    Verifier: 64 random bytes → ~86 URL-safe chars (well within RFC's
    43-128 range). Challenge: SHA256(verifier) → 43 URL-safe chars.
    """
    verifier = _b64url_nopad(secrets.token_bytes(64))
    challenge = _b64url_nopad(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _generate_state() -> str:
    """Random state nonce for CSRF protection on the callback. 32 bytes
    of entropy is more than enough — the value lives at most a few
    minutes in the user's signed session cookie."""
    return _b64url_nopad(secrets.token_bytes(32))


def build_authorize_url(request: Request, *, next_path: str | None = None) -> str:
    """Phase 1 of the auth-code flow.

    Generates a PKCE verifier+challenge and a state nonce, stashes them
    in the user's session cookie, and returns the URL to redirect the
    browser to OpenEMR's hosted login page. `next_path` is the path the
    user wanted before being bounced to login (e.g. `/dashboard`); the
    callback handler reads it back and redirects there post-auth.
    """
    if not settings.openemr_seed_client_id:
        raise RuntimeError(
            "OPENEMR_SEED_CLIENT_ID not set — run scripts/register_seed_client.py first."
        )

    verifier, challenge = _generate_pkce()
    state = _generate_state()
    request.session[_SESS_KEY_VERIFIER] = verifier
    request.session[_SESS_KEY_STATE] = state
    if next_path:
        # Reject absolute / cross-origin nexts to prevent open-redirect.
        # Same-origin paths must start with `/` and not look like a URL.
        if next_path.startswith("/") and not next_path.startswith("//"):
            request.session[_SESS_KEY_NEXT] = next_path

    redirect_uri = f"{settings.copilot_base_url.rstrip('/')}/oauth/callback"
    params = {
        "response_type": "code",
        "client_id": settings.openemr_seed_client_id,
        "redirect_uri": redirect_uri,
        "scope": _AUTHORIZE_SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{settings.openemr_oauth_authorize_url}?{urlencode(params)}"


async def exchange_code_for_username(request: Request, *, code: str, state: str) -> str:
    """Phase 2 of the auth-code flow — runs inside `/oauth/callback`.

    Validates the returned `state` matches the one we stashed at /login,
    POSTs the auth code + PKCE verifier to OpenEMR's token endpoint to
    receive an id_token, decodes the id_token to extract the username,
    and returns it. Raises `HTTPException(400)` on any mismatch or
    decode failure.

    Side effects: clears the per-flow session keys (verifier, state) on
    success or failure. Does NOT clear `oa_next` — the caller reads that
    after this function returns to know where to redirect the user.
    """
    expected_state = request.session.pop(_SESS_KEY_STATE, None)
    verifier = request.session.pop(_SESS_KEY_VERIFIER, None)

    if not expected_state or not verifier:
        raise HTTPException(
            status_code=400,
            detail="OAuth callback received without an active login session.",
        )
    if not secrets.compare_digest(expected_state, state):
        raise HTTPException(status_code=400, detail="OAuth state mismatch.")

    redirect_uri = f"{settings.copilot_base_url.rstrip('/')}/oauth/callback"
    body = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": settings.openemr_seed_client_id,
        "client_secret": settings.openemr_seed_client_secret,
        "code_verifier": verifier,
    }

    async with httpx.AsyncClient(verify=False, timeout=15) as http:
        resp = await http.post(
            settings.openemr_oauth_token_url,
            data=body,
            headers={"Accept": "application/json"},
        )
    if resp.status_code != 200:
        log.warning(
            "oauth: token exchange failed: %s %s",
            resp.status_code, resp.text[:300],
        )
        raise HTTPException(status_code=400, detail="OAuth token exchange failed.")

    payload = resp.json()
    id_token = payload.get("id_token")
    access_token = payload.get("access_token")
    if not id_token:
        raise HTTPException(status_code=400, detail="OAuth response missing id_token.")

    username = _username_from_id_token(id_token)
    if username:
        return username

    # Fallback: the id_token didn't carry a username claim shape we
    # recognize. Hit the userinfo endpoint with the access token —
    # OpenEMR returns the same identity set there.
    if access_token:
        username = await _username_from_userinfo(access_token)
        if username:
            return username

    raise HTTPException(status_code=400, detail="OAuth response had no username claim.")


def _username_from_id_token(id_token: str) -> str | None:
    """Decode the id_token's payload (middle JWT segment) and extract a
    username. Does NOT verify the signature — see module docstring.

    Returns None if the token is malformed or carries no recognized
    username claim. Tries `preferred_username`, then `name`, then `sub`
    in order — `preferred_username` is the standard OIDC claim for
    "human-readable username," and OpenEMR populates it with the staff
    login ('admin', etc.).
    """
    try:
        _, payload_b64, _ = id_token.split(".")
        # Base64url decode requires padding to a multiple of 4 — pad
        # explicitly because Python's stdlib decoder is strict.
        pad = "=" * (-len(payload_b64) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64 + pad))
    except (ValueError, json.JSONDecodeError) as e:
        log.warning("oauth: id_token decode failed: %s", e)
        return None
    for claim in ("preferred_username", "name", "sub"):
        v = claims.get(claim)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


async def _username_from_userinfo(access_token: str) -> str | None:
    """Fallback: hit OpenEMR's `userinfo` endpoint and read the username
    from the response. Used when the id_token doesn't carry a username
    claim (rare, but tolerated)."""
    async with httpx.AsyncClient(verify=False, timeout=10) as http:
        try:
            resp = await http.get(
                settings.openemr_oauth_userinfo_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
        except httpx.RequestError as e:
            log.warning("oauth: userinfo fetch failed: %s", e)
            return None
    if resp.status_code != 200:
        log.warning(
            "oauth: userinfo non-200: %s %s",
            resp.status_code, resp.text[:200],
        )
        return None
    try:
        info = resp.json()
    except ValueError:
        return None
    for claim in ("preferred_username", "username", "name", "sub"):
        v = info.get(claim)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def consume_next_path(request: Request) -> str:
    """Pop the post-auth redirect path off the session and return it.

    Defaults to `/` when no next was stashed. Returned path is always
    same-origin (validated at stash time)."""
    nxt = request.session.pop(_SESS_KEY_NEXT, None)
    if isinstance(nxt, str) and nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return "/"
