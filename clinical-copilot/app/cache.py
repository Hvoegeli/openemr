"""Simple in-memory TTL cache + pre-warmer for the dashboard endpoints.

The FHIR layer in OpenEMR's dev stack is slow (~3s per call). The dashboard
hits 3+ endpoints on every login. We cache responses in-process so that:
  - The first call after startup hits FHIR and populates the cache.
  - Every subsequent call within `ttl_seconds` returns from memory.
  - A background pre-warm during app startup means even the *first* call
    after login is instant.

This is fine for MVP (in-process, no PHI in disk). Production would use
Redis or memcached, and would invalidate on chart writes from OpenEMR.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

log = logging.getLogger("agent.cache")


class TTLCache:
    """Tiny dict-backed cache. Single-process; no eviction beyond TTL."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, key: str) -> Any | None:
        item = self._store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self._ttl:
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.time(), value)

    async def get_or_compute(
        self, key: str, compute: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Cache-aside with per-key locking so concurrent misses share work."""
        cached = self.get(key)
        if cached is not None:
            return cached
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            cached = self.get(key)
            if cached is not None:
                return cached
            value = await compute()
            self.set(key, value)
            return value
