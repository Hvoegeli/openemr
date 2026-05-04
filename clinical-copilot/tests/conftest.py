"""Shared pytest configuration for clinical-copilot unit tests.

Puts `clinical-copilot/` on `sys.path` so tests can `from app.extraction
import schemas` exactly the way the running app does, without relying on
the user setting `PYTHONPATH=.` before invoking pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
