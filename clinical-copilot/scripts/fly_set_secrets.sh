#!/usr/bin/env bash
# Push runtime secrets from .env into Fly.io.
#
# Run AFTER `fly launch --no-deploy --copy-config` (which creates the app)
# and BEFORE `fly deploy`. Re-run any time you need to rotate a secret or
# point at a different OpenEMR (e.g. a new cloudflared tunnel URL).
#
# Usage:
#   scripts/fly_set_secrets.sh
#   OPENEMR_TUNNEL_URL=https://abc.trycloudflare.com scripts/fly_set_secrets.sh
#
# When OPENEMR_TUNNEL_URL is set, FHIR/OAuth URLs are derived from it instead
# of read from .env — the agent_forge .env points at localhost which is
# unreachable from Fly.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "no .env in $(pwd) — aborting" >&2
  exit 1
fi

# Load .env into the shell without echoing any of it
set -a
# shellcheck disable=SC1091
source .env
set +a

REQUIRED=(
  ANTHROPIC_API_KEY
  OPENEMR_CLIENT_ID
  OPENEMR_KID
)
for var in "${REQUIRED[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "missing $var in .env" >&2
    exit 1
  fi
done

if [[ -n "${OPENEMR_TUNNEL_URL:-}" ]]; then
  FHIR="${OPENEMR_TUNNEL_URL%/}/apis/default/fhir"
  OAUTH="${OPENEMR_TUNNEL_URL%/}/oauth2/default/token"
else
  FHIR="${OPENEMR_FHIR_BASE_URL:-https://localhost:9300/apis/default/fhir}"
  OAUTH="${OPENEMR_OAUTH_TOKEN_URL:-https://localhost:9300/oauth2/default/token}"
fi

echo "Pushing secrets to Fly.io..."
echo "  FHIR base   = $FHIR"
echo "  OAuth token = $OAUTH"

fly secrets set \
  ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  OPENEMR_CLIENT_ID="$OPENEMR_CLIENT_ID" \
  OPENEMR_KID="$OPENEMR_KID" \
  OPENEMR_FHIR_BASE_URL="$FHIR" \
  OPENEMR_OAUTH_TOKEN_URL="$OAUTH" \
  ${OPENEMR_CLIENT_SECRET:+OPENEMR_CLIENT_SECRET="$OPENEMR_CLIENT_SECRET"} \
  ${LANGSMITH_API_KEY:+LANGSMITH_API_KEY="$LANGSMITH_API_KEY"} \
  ${LANGSMITH_TRACING:+LANGSMITH_TRACING="$LANGSMITH_TRACING"} \
  ${LANGSMITH_PROJECT:+LANGSMITH_PROJECT="$LANGSMITH_PROJECT"}

echo "✓ secrets set"
