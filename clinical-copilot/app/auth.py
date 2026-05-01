"""Co-pilot login uses the same credentials as OpenEMR.

We don't run our own user store. To validate a username/password pair we ask
OpenEMR to: POST /oauth2/default/token with `grant_type=password` against our
existing seed client. If OpenEMR mints a token, the credentials are good. We
discard the token — the agent's own FHIR client uses its own private_key_jwt
flow for chart reads. The cookie session that the co-pilot sets afterwards
is opaque to OpenEMR and stores only the sid (an opaque pointer into the
durable session store at `app.state.auth_store`); the username, idle/absolute
timeouts, and revocation state live there, not in the cookie.
"""

import httpx
import urllib3
from fastapi import HTTPException, Request

from app.auth_db import Session
from app.config import settings

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Same scope set the seed client was registered with — we only need the grant
# to *succeed*, but OpenEMR rejects unknown scopes outright.
_VERIFY_SCOPE = " ".join([
    "openid", "api:oemr",
    "user/patient.cruds",
])


async def verify_openemr_credentials(username: str, password: str) -> bool:
    """Return True iff OpenEMR accepts this user/pass pair via password grant."""
    if not settings.openemr_seed_client_id or not settings.openemr_seed_client_secret:
        raise RuntimeError(
            "OPENEMR_SEED_CLIENT_ID/SECRET not set — run "
            "scripts/register_seed_client.py first."
        )

    async with httpx.AsyncClient(verify=False, timeout=15) as http:
        resp = await http.post(
            settings.openemr_oauth_token_url,
            data={
                "grant_type": "password",
                "client_id": settings.openemr_seed_client_id,
                "client_secret": settings.openemr_seed_client_secret,
                "username": username,
                "password": password,
                "scope": _VERIFY_SCOPE,
                "user_role": "users",
            },
            headers={"Accept": "application/json"},
        )
    if resp.status_code == 200 and resp.json().get("access_token"):
        return True
    return False


def current_session(request: Request) -> Session | None:
    """Resolve the logged-in session, or return None.

    Page handlers prefer this — they want to redirect to /login on a missing
    or expired session, not raise a 401. API endpoints use `current_user`
    below, which raises instead. A revoked / idle-timed-out / absolute-
    timed-out session also clears the stale cookie so the user gets a clean
    redirect chain rather than bouncing forever on a dead sid.
    """
    sid = request.session.get("sid")
    if not sid:
        return None
    session = request.app.state.auth_store.get_active_session(sid)
    if session is None:
        request.session.clear()
        return None
    return session


def current_user(request: Request) -> str:
    """FastAPI dependency: return the logged-in username, or 401.

    The username comes out of the durable session store at
    `app.state.auth_store`, not the cookie — server-side state is the
    source of truth for whether a session is still valid.
    """
    session = current_session(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Not logged in")
    return session.username


def _admin_set() -> set[str]:
    """Parse `settings.admin_usernames` into a set of lowercased usernames.

    Comma-separated list, whitespace-tolerant. Empty entries dropped.
    Lowercased for case-insensitive matching against the resolved session
    username — OpenEMR is case-insensitive on username via its login flow,
    and we mirror that here so `Admin` and `admin` don't drift apart."""
    raw = settings.admin_usernames or ""
    return {u.strip().lower() for u in raw.split(",") if u.strip()}


def is_admin(username: str | None) -> bool:
    """True if `username` is in the configured admin allow-list."""
    if not username:
        return False
    return username.strip().lower() in _admin_set()


def require_admin(request: Request) -> str:
    """Dependency: return the admin's username, or 403 if not an admin.
    Use as `Depends(require_admin)` on admin-only endpoints. 401 still
    fires first if the user isn't logged in at all."""
    username = current_user(request)
    if not is_admin(username):
        raise HTTPException(status_code=403, detail="Admin access required")
    return username
