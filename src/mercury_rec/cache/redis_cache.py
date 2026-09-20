"""Redis-backed recommendation cache.

Three behaviours here are the ones that matter operationally.

**Stale-while-revalidate.** Every payload carries a soft expiry earlier than
its hard TTL. Past the soft expiry the cached value is still served
immediately and a refresh is triggered in the background under a lock. A
plain TTL instead means every popular user's expiry lands as a latency spike
on whichever unlucky request arrives first, and concurrent requests all
recompute the same answer.

**Jittered TTLs.** Entries written together would otherwise expire together.
After a deploy warms the cache, that produces a synchronised mass expiry a few
minutes later - a thundering herd that looks exactly like a traffic spike and
is entirely self-inflicted. A +/-15% spread removes it.

**Fail-open.** Every operation is wrapped so a Redis outage degrades latency
rather than availability. A cache is an optimisation; a recommendation service
that returns 500 because its cache is down has converted an optimisation into
a dependency. Failures are counted and surfaced on ``/ready``, so degradation
is visible rather than silent.

Payloads are serialised with orjson: it is several times faster than the
standard library on the nested dicts this service returns, and the request
path serialises on every miss.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import orjson

from mercury_rec.cache import keys
from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CacheStats:
    """Counters behind the reported cache hit rate.

    Real counters rather than an estimate: the dashboard's hit-rate figure is
    computed from these, which is why it is allowed to be shown at all.
    """

    hits: int = 0
    misses: int = 0
    stale_hits: int = 0
    errors: int = 0
    writes: int = 0
    invalidations: int = 0

    @property
    def total_reads(self) -> int:
        return self.hits + self.misses + self.stale_hits

    @property
    def hit_rate(self) -> float:
        return (self.hits + self.stale_hits) / self.total_reads if self.total_reads else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stale_hits": self.stale_hits,
            "errors": self.errors,
            "writes": self.writes,
            "invalidations": self.invalidations,
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass(slots=True)
class CachedPayload:
    """A cache read result."""

    value: dict[str, Any]
    is_stale: bool
    age_seconds: float


@dataclass(slots=True)
class RecommendationCache:
    """Recommendation cache with stale-while-revalidate and fail-open reads.

    Args:
        client: A ``redis.Redis`` instance, or None to disable caching
            entirely. None is a supported mode, not a broken one: the service
            runs correctly without Redis, just slower.
        ttl_seconds: Hard expiry.
        soft_ttl_seconds: Age past which a value is served but refreshed.
        jitter_ratio: Fractional TTL spread.
    """

    client: Any = None
    ttl_seconds: int = 300
    soft_ttl_seconds: int = 120
    jitter_ratio: float = 0.15
    stats: CacheStats = field(default_factory=CacheStats)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def _jittered_ttl(self) -> int:
        spread = self.ttl_seconds * self.jitter_ratio
        return max(1, int(self.ttl_seconds + random.uniform(-spread, spread)))  # noqa: S311

    def get(self, key: str) -> CachedPayload | None:
        """Read a payload. Returns None on miss, on error, or when disabled."""
        if not self.enabled:
            return None
        try:
            raw = self.client.get(key)
        except Exception as exc:
            self.stats.errors += 1
            logger.warning("cache.read_failed", key=key, error=str(exc)[:120])
            return None

        if raw is None:
            self.stats.misses += 1
            return None

        try:
            payload = orjson.loads(raw)
        except orjson.JSONDecodeError:
            # A corrupt entry must not poison the request. Treat it as a miss
            # and let the write path replace it.
            self.stats.errors += 1
            logger.warning("cache.corrupt_entry", key=key)
            return None

        written_at = float(payload.get("_cached_at", 0.0))
        age = time.time() - written_at
        is_stale = age > self.soft_ttl_seconds

        if is_stale:
            self.stats.stale_hits += 1
        else:
            self.stats.hits += 1

        return CachedPayload(value=payload, is_stale=is_stale, age_seconds=age)

    def set(self, key: str, value: dict[str, Any], *, user_id: str | None = None) -> None:
        """Write a payload, registering it in the user's invalidation index."""
        if not self.enabled:
            return
        payload = {**value, "_cached_at": time.time()}
        ttl = self._jittered_ttl()

        try:
            pipeline = self.client.pipeline()
            pipeline.set(key, orjson.dumps(payload), ex=ttl)
            if user_id is not None:
                index = keys.recommendation_index(user_id)
                pipeline.sadd(index, key)
                # The index must outlive the entries it points at, or an
                # invalidation could miss a key that is still live.
                pipeline.expire(index, ttl + 60)
            pipeline.execute()
            self.stats.writes += 1
        except Exception as exc:
            self.stats.errors += 1
            logger.warning("cache.write_failed", key=key, error=str(exc)[:120])

    def invalidate_user(self, user_id: str) -> int:
        """Drop every cached recommendation for one user.

        Driven by the per-user index, so this is O(that user's keys) with no
        keyspace scan. ``UNLINK`` rather than ``DEL``: reclamation happens on
        a background thread instead of blocking the server.
        """
        if not self.enabled:
            return 0
        index = keys.recommendation_index(user_id)
        try:
            members = self.client.smembers(index)
            if not members:
                return 0
            pipeline = self.client.pipeline()
            pipeline.unlink(*members)
            pipeline.unlink(index)
            pipeline.execute()
            self.stats.invalidations += len(members)
            logger.debug("cache.invalidated", user_id=user_id, keys=len(members))
            return len(members)
        except Exception as exc:
            self.stats.errors += 1
            logger.warning("cache.invalidate_failed", user_id=user_id, error=str(exc)[:120])
            return 0

    def acquire_refresh_lock(self, user_id: str, *, ttl_seconds: int = 10) -> bool:
        """Try to become the single refresher for a stale entry.

        ``SET NX EX`` so the lock is self-expiring: a process that dies
        mid-refresh cannot wedge the key permanently.
        """
        if not self.enabled:
            return True
        try:
            acquired = self.client.set(keys.refresh_lock(user_id), b"1", nx=True, ex=ttl_seconds)
            return bool(acquired)
        except Exception:
            self.stats.errors += 1
            # Fail-open: without the lock the worst case is a duplicated
            # refresh, which is far better than never refreshing.
            return True

    def ping(self) -> tuple[bool, str | None]:
        """Liveness probe for ``/ready``."""
        if not self.enabled:
            return False, "cache disabled"
        try:
            self.client.ping()
            return True, None
        except Exception as exc:
            return False, str(exc)[:120]


def build_client(dsn: str | None) -> Any:
    """Create a Redis client, or None when no DSN is configured.

    Returning None rather than raising keeps "no Redis" a supported mode:
    ``mercury serve`` runs without it, and the tests exercise that path.
    """
    if not dsn:
        logger.info("cache.disabled", reason="no MERCURY_REDIS_DSN configured")
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            dsn,
            decode_responses=False,  # payloads are orjson bytes
            socket_timeout=0.25,
            socket_connect_timeout=0.25,
            health_check_interval=30,
        )
        client.ping()
        logger.info("cache.connected")
        return client
    except Exception as exc:
        # A cache that will not connect must not stop the service starting.
        logger.warning("cache.connect_failed", error=str(exc)[:160])
        return None


__all__ = ["CacheStats", "CachedPayload", "RecommendationCache", "build_client"]
