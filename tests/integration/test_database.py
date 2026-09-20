"""Integration tests against live PostgreSQL and Redis.

Marked ``integration`` and skipped when the services are absent, so the
default ``pytest`` run stays fast and hermetic while CI (which starts both as
service containers) exercises the real thing.

They deliberately test what a unit test with a fake cannot: that the
constraints are enforced by the database rather than by application
convention, and that the cache behaves correctly against a real Redis with
real TTLs.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from mercury_rec.cache import keys
from mercury_rec.cache.redis_cache import RecommendationCache, build_client
from mercury_rec.db.models import Base, Experiment, Merchant, ModelMetadata
from mercury_rec.db.session import create_db_engine, healthcheck

pytestmark = pytest.mark.integration

POSTGRES_DSN = os.environ.get(
    "MERCURY_POSTGRES_DSN",
    "postgresql+psycopg://mercury:mercury@127.0.0.1:5432/mercury_rec",
)
REDIS_DSN = os.environ.get("MERCURY_REDIS_DSN", "redis://127.0.0.1:6379/15")


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    db = create_db_engine(POSTGRES_DSN)
    ok, detail = healthcheck(db)
    if not ok:
        pytest.skip(f"PostgreSQL unavailable: {detail}")
    Base.metadata.create_all(db)
    yield db
    db.dispose()


@pytest.fixture
def clean_db(engine: Engine) -> Iterator[Engine]:
    """Truncate between tests so ordering cannot matter."""
    with engine.begin() as connection:
        connection.execute(
            text(
                "TRUNCATE model_metadata, experiments, experiment_results, "
                "merchants RESTART IDENTITY CASCADE"
            )
        )
    yield engine


@pytest.fixture
def cache() -> Iterator[RecommendationCache]:
    """A cache on a dedicated Redis database, flushed around each test.

    Database 15 rather than 0: a test that flushes must not be able to wipe a
    developer's working cache.
    """
    client = build_client(REDIS_DSN)
    if client is None:
        pytest.skip("Redis unavailable")
    client.flushdb()
    yield RecommendationCache(client=client, ttl_seconds=5, soft_ttl_seconds=1)
    client.flushdb()
    client.close()


class TestSchemaInvariants:
    def test_at_most_one_production_model_per_type(self, clean_db: Engine) -> None:
        """The invariant must live in the database, not in application code.

        Expressed only in Python it is a convention that any manual UPDATE can
        break, leaving two models claiming to serve production with nothing to
        say which one actually does.
        """
        from sqlalchemy.orm import Session

        with Session(clean_db) as session:
            session.add(
                ModelMetadata(
                    model_version="v1",
                    model_type="two_tower",
                    trained_at=datetime.now(UTC),
                    feature_schema_version=1,
                    status="production",
                )
            )
            session.commit()

            session.add(
                ModelMetadata(
                    model_version="v2",
                    model_type="two_tower",
                    trained_at=datetime.now(UTC),
                    feature_schema_version=1,
                    status="production",
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()

    def test_multiple_non_production_versions_are_allowed(self, clean_db: Engine) -> None:
        """The constraint must be partial: only production is restricted."""
        from sqlalchemy.orm import Session

        with Session(clean_db) as session:
            for version in ("c1", "c2", "c3"):
                session.add(
                    ModelMetadata(
                        model_version=version,
                        model_type="two_tower",
                        trained_at=datetime.now(UTC),
                        feature_schema_version=1,
                        status="candidate",
                    )
                )
            session.commit()
            assert session.query(ModelMetadata).count() == 3

    def test_experiments_default_to_simulated(self, clean_db: Engine) -> None:
        """The honesty requirement is encoded in the schema.

        A row asserting a live experiment has to say so explicitly rather than
        by leaving a column unset.
        """
        from sqlalchemy.orm import Session

        with Session(clean_db) as session:
            session.add(
                Experiment(
                    experiment_id="e1",
                    name="ranker-v2",
                    control_model_version="v1",
                    treatment_model_version="v2",
                )
            )
            session.commit()
            assert session.query(Experiment).one().is_simulated is True

    def test_invalid_model_status_is_rejected(self, clean_db: Engine) -> None:
        with clean_db.begin() as connection, pytest.raises(Exception, match="ck_model_status"):
            connection.execute(
                text(
                    "INSERT INTO model_metadata "
                    "(model_version, model_type, trained_at, feature_schema_version, status) "
                    "VALUES ('x', 'two_tower', now(), 1, 'not-a-status')"
                )
            )

    def test_negative_price_is_rejected(self, clean_db: Engine) -> None:
        """A zero or negative price would break every price-ratio feature."""
        with clean_db.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO merchants "
                    "(merchant_id, vertical, region_id, rating, delivery_radius_km, "
                    "avg_delivery_minutes) VALUES (1, 0, 0, 4.0, 5.0, 30)"
                )
            )
        with clean_db.begin() as connection, pytest.raises(Exception, match="ck_item_price"):
            connection.execute(
                text(
                    "INSERT INTO items (item_id, source_item_id, merchant_id, vertical, "
                    "price, available_from, is_cold) "
                    "VALUES (1, 100, 1, 0, -5.0, now(), false)"
                )
            )

    def test_merchant_rating_range_is_enforced(self, clean_db: Engine) -> None:
        from sqlalchemy.orm import Session

        with Session(clean_db) as session:
            session.add(
                Merchant(
                    merchant_id=2,
                    vertical=0,
                    region_id=0,
                    rating=9.9,  # out of the 0-5 range
                    delivery_radius_km=5.0,
                    avg_delivery_minutes=30,
                )
            )
            with pytest.raises(IntegrityError):
                session.commit()


class TestCacheAgainstRealRedis:
    def test_round_trips_a_payload(self, cache: RecommendationCache) -> None:
        key = keys.recommendations("v1", "u_1", "ctx", 10)
        cache.set(key, {"items": [{"item_id": 5}]}, user_id="u_1")

        cached = cache.get(key)
        assert cached is not None
        assert cached.value["items"][0]["item_id"] == 5
        assert cached.is_stale is False

    def test_miss_returns_none_and_counts(self, cache: RecommendationCache) -> None:
        assert cache.get(keys.recommendations("v1", "nobody", "ctx", 10)) is None
        assert cache.stats.misses == 1

    def test_entry_becomes_stale_then_expires(self, cache: RecommendationCache) -> None:
        """Stale-while-revalidate: still served after the soft TTL.

        A plain TTL makes every popular key's expiry a latency spike on
        whichever request arrives first.
        """
        key = keys.recommendations("v1", "u_2", "ctx", 10)
        cache.set(key, {"items": []}, user_id="u_2")

        time.sleep(1.5)  # past the 1s soft TTL, inside the 5s hard TTL
        cached = cache.get(key)
        assert cached is not None, "entry should still be served while stale"
        assert cached.is_stale is True
        assert cache.stats.stale_hits == 1

    def test_invalidation_clears_every_key_for_a_user(self, cache: RecommendationCache) -> None:
        """Driven by the per-user index, with no keyspace scan."""
        for k in (5, 10, 20):
            cache.set(keys.recommendations("v1", "u_3", "ctx", k), {"items": []}, user_id="u_3")

        removed = cache.invalidate_user("u_3")
        assert removed == 3
        for k in (5, 10, 20):
            assert cache.get(keys.recommendations("v1", "u_3", "ctx", k)) is None

    def test_invalidation_does_not_touch_other_users(self, cache: RecommendationCache) -> None:
        cache.set(keys.recommendations("v1", "u_4", "ctx", 10), {"items": []}, user_id="u_4")
        cache.set(keys.recommendations("v1", "u_5", "ctx", 10), {"items": []}, user_id="u_5")

        cache.invalidate_user("u_4")
        assert cache.get(keys.recommendations("v1", "u_5", "ctx", 10)) is not None

    def test_model_version_in_key_isolates_versions(self, cache: RecommendationCache) -> None:
        """Promotion invalidates atomically because the version is in the key."""
        cache.set(keys.recommendations("v1", "u_6", "ctx", 10), {"items": [1]}, user_id="u_6")

        # A new model version reads a different key space entirely.
        assert cache.get(keys.recommendations("v2", "u_6", "ctx", 10)) is None
        assert cache.get(keys.recommendations("v1", "u_6", "ctx", 10)) is not None

    def test_refresh_lock_admits_exactly_one_holder(self, cache: RecommendationCache) -> None:
        """Without this a popular key's expiry causes a refresh stampede."""
        assert cache.acquire_refresh_lock("u_7") is True
        assert cache.acquire_refresh_lock("u_7") is False

    def test_hit_rate_reflects_real_counters(self, cache: RecommendationCache) -> None:
        """The dashboard's hit rate is computed from these, so they must be real."""
        key = keys.recommendations("v1", "u_8", "ctx", 10)
        cache.set(key, {"items": []}, user_id="u_8")

        cache.get(key)
        cache.get(keys.recommendations("v1", "absent", "ctx", 10))

        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == pytest.approx(0.5)

    def test_ping_reports_connected(self, cache: RecommendationCache) -> None:
        ready, detail = cache.ping()
        assert ready is True
        assert detail is None
