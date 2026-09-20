"""The single source of Redis key construction.

Every key in the system is built here. Nothing else in the codebase contains a
string starting with ``mercury:`` -- there is a test that enforces it.

That rule exists because cache keys are a schema. Once key strings are
scattered across modules, changing the format means finding every place it was
formatted, and the ones that are missed do not fail: they simply read and
write a different key space, so the cache quietly stops working and the only
symptom is a hit rate that fell.

Two structural choices carry most of the operational weight:

**The model version is part of the key.** Promoting a model therefore
invalidates its entire cache atomically and instantly, with no scan and no
delete storm; the old entries are simply never read again and expire on their
own TTL. This is the single most useful cache-invalidation decision available
here, and it costs nothing.

**Context is hashed into a bucketed fingerprint** rather than embedded whole.
Using the raw hour would multiply the key space by 24; six buckets match the
granularity at which behaviour actually changes.
"""

from __future__ import annotations

import hashlib
from typing import Final

#: Bumped only when a key's *meaning* changes in a way that would make old
#: entries wrong rather than merely stale.
KEY_VERSION: Final = "v1"

PREFIX: Final = f"mercury:{KEY_VERSION}"


def context_fingerprint(
    *,
    hour_bucket: int | None = None,
    weekday: int | None = None,
    region_id: int | None = None,
    vertical: int | None = None,
    device: int | None = None,
    session_signature: str | None = None,
) -> str:
    """Short, stable digest of the request context.

    blake2b at 6 bytes: 12 hex characters is ample to avoid collisions across
    a key space this size, and short enough to keep keys readable in
    ``redis-cli`` during an incident.
    """
    parts = [
        str(hour_bucket) if hour_bucket is not None else "-",
        str(weekday) if weekday is not None else "-",
        str(region_id) if region_id is not None else "-",
        str(vertical) if vertical is not None else "-",
        str(device) if device is not None else "-",
        session_signature or "-",
    ]
    return hashlib.blake2b("|".join(parts).encode(), digest_size=6).hexdigest()


def recommendations(model_version: str, user_id: str, context_hash: str, k: int) -> str:
    """Cached recommendation payload for one (user, context, k)."""
    return f"{PREFIX}:rec:{model_version}:{user_id}:{context_hash}:{k}"


def recommendation_index(user_id: str) -> str:
    """Set of this user's live recommendation keys.

    Invalidation on a new event is O(that user's cached keys) -- typically one
    to three -- rather than a ``SCAN`` of the whole keyspace. **There is no
    ``KEYS`` or ``SCAN`` anywhere in the hot path**, which is the difference
    between an invalidation that is free and one that periodically stalls the
    server.
    """
    return f"{PREFIX}:recidx:{user_id}"


def user_features(schema_version: int, user_id: str) -> str:
    return f"{PREFIX}:feat:u:{schema_version}:{user_id}"


def item_features(schema_version: int, item_id: int) -> str:
    return f"{PREFIX}:feat:i:{schema_version}:{item_id}"


def popularity(region_id: int | None, vertical: int | None, hour_bucket: int | None) -> str:
    region = region_id if region_id is not None else "all"
    vert = vertical if vertical is not None else "all"
    bucket = hour_bucket if hour_bucket is not None else "all"
    return f"{PREFIX}:pop:{region}:{vert}:{bucket}"


def user_embedding(model_version: str, user_id: str) -> str:
    return f"{PREFIX}:emb:u:{model_version}:{user_id}"


def similar_items(model_version: str, item_id: int) -> str:
    return f"{PREFIX}:sim:{model_version}:{item_id}"


def session(session_id: str) -> str:
    return f"{PREFIX}:sess:{session_id}"


def refresh_lock(user_id: str) -> str:
    """Lock guarding a background refresh.

    Held with ``SET NX EX`` so exactly one request refreshes a stale entry
    while the rest keep serving it. Without it, a popular user's expiry
    triggers a stampede in which every concurrent request recomputes the same
    recommendation.
    """
    return f"{PREFIX}:lock:refresh:{user_id}"


def production_models() -> str:
    """Hash of model type -> version currently serving."""
    return f"{PREFIX}:model:production"


__all__ = [
    "KEY_VERSION",
    "PREFIX",
    "context_fingerprint",
    "item_features",
    "popularity",
    "production_models",
    "recommendation_index",
    "recommendations",
    "refresh_lock",
    "session",
    "similar_items",
    "user_embedding",
    "user_features",
]
