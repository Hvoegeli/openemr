"""Co-pilot login uses the same credentials as OpenEMR.

We don't run our own user store. To validate a username/password pair we ask
OpenEMR to: POST /oauth2/default/token with `grant_type=password` against our
existing seed client. If OpenEMR mints a token, the credentials are good. We
discard the token — the agent's own FHIR client uses its own private_key_jwt
flow for chart reads. The cookie session that the co-pilot sets afterwards
is opaque to OpenEMR.
"""

import httpx
import urllib3
from fastapi import HTTPException, Request

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


def current_user(request: Request) -> str:
    """Dependency: return the logged-in username, or 401."""
    username = request.session.get("username")
    if not username:
        raise HTTPException(status_code=401, detail="Not logged in")
    return username
