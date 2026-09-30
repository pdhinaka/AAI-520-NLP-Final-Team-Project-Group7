"""Tiny JSON disk cache so we don't burn free-tier API limits.

Each entry is a JSON file under data/cache/<namespace>/ keyed by a hash of
the call's parameters.
"""

import hashlib
import json
import time
from typing import Any, Callable

from finagent import config


class CacheMissError(RuntimeError):
    """Raised in offline mode when nothing is cached for a call."""


def _path(namespace: str, params: dict):
    key = json.dumps(params, sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return config.CACHE_DIR / namespace / f"{digest}.json"


def cached(
    namespace: str,
    params: dict,
    fetch: Callable[[], Any],
    ttl_hours: float = 24,
    cache_empty: bool = True,
) -> Any:
    """Return a cached result if fresh, otherwise call fetch() and save it.

    In offline mode (CACHE_MODE=offline) stale entries are still returned
    and a miss raises CacheMissError instead of touching the network.
    With cache_empty=False an empty result ([] or {}) is returned but not
    saved, so a temporary outage isn't cached as "no data".
    """
    path = _path(namespace, params)
    if path.exists():
        entry = json.loads(path.read_text())
        if not cache_empty and entry["data"] in ([], {}):
            entry = None  # saved before cache_empty existed; retry
    else:
        entry = None
    if entry is not None:
        age_h = (time.time() - entry["saved_at"]) / 3600
        if config.CACHE_MODE == "offline" or age_h < ttl_hours:
            return entry["data"]

    if config.CACHE_MODE == "offline":
        raise CacheMissError(f"No cached {namespace} data for {params}")

    data = fetch()
    if not cache_empty and data in ([], {}):
        return data
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"saved_at": time.time(), "params": params, "data": data}
    path.write_text(json.dumps(entry, default=str))
    return data
