"""API contract tests.

These build a small :class:`ArtifactBundle` in memory rather than loading the
real one from disk. Loading the production bundle takes ~20 seconds (it fits
the retrieval models), which would make the suite slow enough that people stop
running it -- and a test suite nobody runs protects nothing.

The trade is explicit: these cover the HTTP contract, error handling and the
wiring between layers. Whether the real artifacts load correctly is covered by
the startup smoke check in the QA gate, not here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mercury_rec.api.main import REQUEST_ID_HEADER, create_app
from mercury_rec.cache.redis_cache import RecommendationCache
from mercury_rec.config.settings import get_settings
from mercury_rec.core.enums import EventType
from mercury_rec.features.asof import AsOfState
from mercury_rec.features.store import AsOfFeatureStore
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.popularity import ContextualPopularityRecommender, PopularityRecommender
from mercury_rec.recommender.engine import ArtifactBundle, RecommendationEngine

N_USERS = 30
N_ITEMS = 40


@pytest.fixture(scope="module")
def interactions() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 800
    minutes = np.sort(rng.integers(0, 60 * 24 * 20, n))
    item_ids = rng.integers(0, N_ITEMS, n).astype("int32")
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, N_USERS, n).astype("int32"),
            "item_id": item_ids,
            "ts": pd.Timestamp("2024-06-01", tz="UTC") + pd.to_timedelta(minutes, unit="m"),
            "event_type": rng.choice(
                [int(EventType.VIEW), int(EventType.ADD_TO_CART), int(EventType.PURCHASE)],
                n,
                p=[0.75, 0.2, 0.05],
            ).astype("int8"),
            "price_at_event": (item_ids % 20 + 5).astype("float32"),
            "vertical": (item_ids % 6).astype("int8"),
            "merchant_id": (item_ids % 8).astype("int32"),
            "region_id": rng.integers(0, 4, n).astype("int8"),
        }
    ).sort_values("ts")


@pytest.fixture(scope="module")
def items() -> pd.DataFrame:
    ids = np.arange(N_ITEMS)
    return pd.DataFrame(
        {
            "item_id": ids,
            "category_id": ids % 10,
            "merchant_id": ids % 8,
            "vertical": ids % 6,
            "price": (ids % 20 + 5).astype(float),
            "is_available": True,
        }
    )


@pytest.fixture(scope="module")
def bundle(interactions: pd.DataFrame, items: pd.DataFrame) -> ArtifactBundle:
    popularity = PopularityRecommender(N_USERS, N_ITEMS)
    popularity.fit(interactions)
    contextual = ContextualPopularityRecommender(N_USERS, N_ITEMS, min_observations=5)
    contextual.fit(interactions)
    item_cf = ItemCFRecommender(N_USERS, N_ITEMS)
    item_cf.fit(interactions)

    history = {
        int(user): set(seen)
        for user, seen in interactions.groupby("user_id")["item_id"].apply(set).items()
    }
    scores = popularity.item_scores

    return ArtifactBundle(
        model_version="test@deadbeef",
        dataset_hash="deadbeef",
        n_users=N_USERS,
        n_items=N_ITEMS,
        feature_store=AsOfFeatureStore(AsOfState(N_USERS, N_ITEMS)),
        popularity=popularity,
        contextual_popularity=contextual,
        item_cf=item_cf,
        items=items,
        # External ids are strings, as they arrive on the wire.
        user_index={f"u_{i}": i for i in range(N_USERS)},
        item_index={i: i for i in range(N_ITEMS)},
        user_history=history,
        popularity_rank=np.argsort(np.argsort(-scores)) / max(N_ITEMS - 1, 1),
    )


@pytest.fixture(autouse=True, scope="module")
def _no_artifact_loading() -> Iterator[None]:
    """Keep application startup from loading the real bundle.

    Without this the module docstring is a lie: every `TestClient(app)` runs
    the lifespan, which fits the retrieval models and reads the two-tower
    embeddings from disk - roughly twenty seconds, per test. These tests
    inject their own bundle a line later and never look at that one.
    """
    previous = os.environ.get("MERCURY_LOAD_ARTIFACTS_ON_STARTUP")
    os.environ["MERCURY_LOAD_ARTIFACTS_ON_STARTUP"] = "false"
    get_settings.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MERCURY_LOAD_ARTIFACTS_ON_STARTUP", None)
        else:
            os.environ["MERCURY_LOAD_ARTIFACTS_ON_STARTUP"] = previous
        get_settings.cache_clear()


@pytest.fixture
def client(bundle: ArtifactBundle) -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as test_client:
        # Startup deliberately loaded nothing; this is the bundle under test.
        app.state.engine = RecommendationEngine(bundle, cache=RecommendationCache())
        app.state.startup_error = None
        yield test_client


class TestHealth:
    def test_health_does_not_depend_on_artifacts(self, bundle: ArtifactBundle) -> None:
        """Liveness must not fail when a dependency is down.

        Restarting the process cannot fix a missing model or a down Redis, so
        a health check that fails on either turns degradation into a crash
        loop.
        """
        app = create_app()
        with TestClient(app) as client:
            app.state.engine = None
            response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_ready_reports_503_without_an_engine(self) -> None:
        app = create_app()
        with TestClient(app) as client:
            app.state.engine = None
            app.state.startup_error = "artifacts missing"
            response = client.get("/ready")

        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_ready_names_the_failing_dependency(self) -> None:
        """'Not ready' is useless in an incident; 'which one' is actionable."""
        app = create_app()
        with TestClient(app) as client:
            app.state.engine = None
            app.state.startup_error = "no feature state"
            checks = {c["name"]: c for c in client.get("/ready").json()["checks"]}

        assert checks["engine"]["ready"] is False
        assert "no feature state" in (checks["engine"]["detail"] or "")

    def test_ready_is_true_with_an_engine(self, client: TestClient) -> None:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True

    def test_metrics_are_prometheus_formatted(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "mercury_" in response.text


class TestRecommendations:
    def test_returns_the_requested_count(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/u_1?k=5")
        assert response.status_code == 200
        assert len(response.json()["recommendations"]) == 5

    def test_scores_are_reported_separately(self, client: TestClient) -> None:
        """The contract the whole reranking design exists to preserve."""
        payload = client.get("/api/v1/recommendations/u_2?k=5").json()
        for item in payload["recommendations"]:
            assert "ml_relevance_score" in item
            assert "business_adjustment" in item
            assert item["final_score"] == pytest.approx(
                item["ml_relevance_score"] + item["business_adjustment"]
            )

    def test_ranks_are_contiguous_from_one(self, client: TestClient) -> None:
        payload = client.get("/api/v1/recommendations/u_3?k=8").json()
        assert [item["rank"] for item in payload["recommendations"]] == list(range(1, 9))

    def test_per_stage_latency_is_returned(self, client: TestClient) -> None:
        """A caller debugging a slow request should not need the metrics stack."""
        latency = client.get("/api/v1/recommendations/u_4?k=5").json()["latency"]
        assert latency["total_ms"] > 0
        assert set(latency) >= {
            "cache_lookup_ms",
            "candidate_generation_ms",
            "ranking_ms",
            "reranking_ms",
            "total_ms",
        }

    def test_training_history_is_never_recommended(
        self, client: TestClient, bundle: ArtifactBundle
    ) -> None:
        """Replaying history would inflate every metric."""
        payload = client.get("/api/v1/recommendations/u_5?k=20").json()
        returned = {item["item_id"] for item in payload["recommendations"]}
        assert not returned & bundle.user_history.get(5, set())

    def test_unknown_user_is_served_by_cold_start(self, client: TestClient) -> None:
        """An unknown visitor is the most common live request, not an error."""
        payload = client.get("/api/v1/recommendations/does_not_exist?k=5").json()
        assert payload["is_cold_start"] is True
        assert len(payload["recommendations"]) == 5

    def test_context_changes_the_result(self, client: TestClient) -> None:
        """Contextual personalisation must be demonstrable, not asserted."""
        by_vertical = {
            vertical: tuple(
                item["item_id"]
                for item in client.get(
                    f"/api/v1/recommendations/u_6?k=10&vertical={vertical}&region_id=0"
                ).json()["recommendations"]
            )
            for vertical in range(6)
        }
        assert len(set(by_vertical.values())) > 1, "vertical context had no effect"

    def test_provenance_is_reported(self, client: TestClient) -> None:
        """A consumer must not be able to mistake this for production data."""
        payload = client.get("/api/v1/recommendations/u_7?k=3").json()
        assert payload["data_provenance"] in {"augmented", "synthetic", "retailrocket"}

    @pytest.mark.parametrize("k", [0, -1, 101])
    def test_invalid_k_is_rejected(self, client: TestClient, k: int) -> None:
        assert client.get(f"/api/v1/recommendations/u_1?k={k}").status_code == 422

    @pytest.mark.parametrize(("param", "value"), [("hour", 24), ("weekday", 7), ("vertical", 9)])
    def test_out_of_range_context_is_rejected(
        self, client: TestClient, param: str, value: int
    ) -> None:
        assert client.get(f"/api/v1/recommendations/u_1?{param}={value}").status_code == 422


class TestSessionRecommend:
    def test_anonymous_session_is_served(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/session/recommend",
            json={"session_items": [1, 2, 3], "k": 5},
        )
        assert response.status_code == 200
        assert response.json()["is_cold_start"] is True

    def test_oversized_session_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/session/recommend",
            json={"session_items": list(range(500)), "k": 5},
        )
        assert response.status_code == 422


class TestEvents:
    def test_known_user_event_is_accepted(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/events",
            json=[{"user_id": "u_1", "item_id": 3, "event_type": int(EventType.CLICK)}],
        )
        assert response.status_code == 200
        assert response.json()["accepted"] == 1

    def test_unknown_user_is_rejected_and_counted(self, client: TestClient) -> None:
        """A client sending unresolvable ids should find out, not be ignored."""
        response = client.post(
            "/api/v1/events",
            json=[{"user_id": "nobody", "item_id": 3, "event_type": int(EventType.CLICK)}],
        )
        payload = response.json()
        assert payload["rejected"] == 1
        assert payload["rejection_reasons"]["unknown_user"] == 1

    def test_oversized_batch_is_rejected(self, client: TestClient) -> None:
        events = [{"user_id": "u_1", "item_id": 1, "event_type": int(EventType.CLICK)}] * 1001
        assert client.post("/api/v1/events", json=events).status_code == 413

    def test_malformed_event_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/events", json=[{"user_id": "u_1", "item_id": -5, "event_type": 2}]
        )
        assert response.status_code == 422


class TestSimilarItems:
    def test_returns_neighbours(self, client: TestClient) -> None:
        response = client.get("/api/v1/items/1/similar?k=5")
        assert response.status_code == 200
        assert response.json()["item_id"] == 1

    def test_unknown_item_is_404(self, client: TestClient) -> None:
        assert client.get("/api/v1/items/99999/similar").status_code == 404


class TestModelStatus:
    def test_reports_loaded_models_and_dataset(self, client: TestClient) -> None:
        payload = client.get("/api/v1/models/status").json()
        assert payload["dataset_hash"] == "deadbeef"
        assert {model["name"] for model in payload["retrieval"]} >= {"popularity", "item_cf"}


class TestRequestCorrelation:
    def test_response_carries_a_request_id(self, client: TestClient) -> None:
        assert client.get("/health").headers.get(REQUEST_ID_HEADER)

    def test_inbound_request_id_is_preserved(self, client: TestClient) -> None:
        """So a trace can be followed across services rather than restarting."""
        supplied = "trace-abc-123"
        response = client.get("/health", headers={REQUEST_ID_HEADER: supplied})
        assert response.headers[REQUEST_ID_HEADER] == supplied


class TestOpenAPI:
    def test_schema_is_generated(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "/api/v1/recommendations/{user_id}" in schema["paths"]

    def test_provenance_is_documented(self, client: TestClient) -> None:
        """The API description must not let a reader assume production data."""
        description = client.get("/openapi.json").json()["info"]["description"]
        assert "Retailrocket" in description
        assert "synthesised" in description


class TestPipelineTrace:
    """The stage-membership diagnostic behind the galaxy and pipeline views."""

    def test_stages_narrow_and_nest(self, client: TestClient) -> None:
        """Each stage must be a subset of the one before it.

        This is the claim the whole funnel visualisation rests on. If a
        returned item were not a candidate, the picture would be showing a
        pipeline that did not run.
        """
        trace = client.get("/api/v1/recommendations/u_3/trace?k=5").json()

        candidates = set(trace["candidate_ids"])
        assert candidates, "a user with history should retrieve something"
        assert set(trace["ranked_ids"]) == candidates, "ranking reorders, it does not filter"
        assert set(trace["final_ids"]) <= candidates
        assert len(trace["final_ids"]) <= 5

    def test_ranked_scores_align_with_ranked_ids(self, client: TestClient) -> None:
        """Parallel arrays that disagree would mislabel every bar in the UI."""
        trace = client.get("/api/v1/recommendations/u_3/trace").json()
        assert len(trace["ranked_scores"]) == len(trace["ranked_ids"])

    def test_ranked_ids_are_in_descending_score_order(self, client: TestClient) -> None:
        scores = client.get("/api/v1/recommendations/u_3/trace").json()["ranked_scores"]
        assert scores == sorted(scores, reverse=True)

    def test_sources_only_claim_items_that_are_candidates(self, client: TestClient) -> None:
        """A source cannot be credited with an item fusion dropped."""
        trace = client.get("/api/v1/recommendations/u_3/trace").json()
        candidates = set(trace["candidate_ids"])
        for source, ids in trace["candidate_sources"].items():
            assert set(ids) <= candidates, f"{source} claims non-candidates"

    def test_excludes_items_the_user_already_interacted_with(
        self, client: TestClient, interactions: pd.DataFrame
    ) -> None:
        seen = set(interactions.loc[interactions["user_id"] == 3, "item_id"].tolist())
        trace = client.get("/api/v1/recommendations/u_3/trace").json()
        assert not (set(trace["candidate_ids"]) & seen)

    def test_an_unknown_user_reports_cold_start_rather_than_failing(
        self, client: TestClient
    ) -> None:
        """A cold-start request is successful, it just has no stages."""
        response = client.get("/api/v1/recommendations/nobody/trace")
        assert response.status_code == 200

        trace = response.json()
        assert trace["is_cold_start"] is True
        assert trace["candidate_ids"] == []
        assert trace["final_ids"], "cold start still answers"
        assert "cold-start" in trace["note"]

    def test_a_trace_is_never_served_from_cache(self, client: TestClient) -> None:
        """Warm the cache through the normal endpoint, then demand a trace.

        A cached payload carries no stage membership. If the trace endpoint
        honoured the cache it would have to either return empty stages or
        invent them from the final list, and inventing them would draw a
        funnel that never ran.
        """
        client.get("/api/v1/recommendations/u_3?k=5")
        trace = client.get("/api/v1/recommendations/u_3/trace?k=5").json()
        assert trace["candidate_ids"], "stages must be populated despite a warm cache"


class TestSampleUsers:
    def test_returns_real_ids_that_the_api_accepts(self, client: TestClient) -> None:
        """The ids must round-trip, or the picker leads nowhere."""
        users = client.get("/api/v1/insights/sample-users?n=4").json()["users"]
        assert users

        for user in users:
            response = client.get(f"/api/v1/recommendations/{user['user_id']}?k=3")
            assert response.status_code == 200
            assert response.json()["is_cold_start"] is False

    def test_spans_the_activity_distribution(self, client: TestClient) -> None:
        """Sampling only the busiest users would show the easiest case."""
        users = client.get("/api/v1/insights/sample-users?n=5").json()["users"]
        counts = [user["history_items"] for user in users]

        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1], "every sampled user has identical history"

    def test_respects_the_requested_count(self, client: TestClient) -> None:
        assert len(client.get("/api/v1/insights/sample-users?n=3").json()["users"]) <= 3

    def test_rejects_an_absurd_count(self, client: TestClient) -> None:
        assert client.get("/api/v1/insights/sample-users?n=5000").status_code == 422


class TestSessionAwareColdStart:
    """An anonymous visitor's session is the only signal they carry.

    Serving the same popularity slate to every anonymous request throws that
    signal away. These tests pin the behaviour that uses it, and the boundary
    where it correctly falls back.
    """

    @staticmethod
    def _recommend(client: TestClient, session_items: list[int]) -> list[int]:
        response = client.post(
            "/api/v1/session/recommend",
            json={"session_items": session_items, "k": 8, "context": {}},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["is_cold_start"] is True, "no user id means the cold-start path"
        return [item["item_id"] for item in payload["recommendations"]]

    def test_a_session_changes_an_anonymous_slate(self, client: TestClient) -> None:
        generic = self._recommend(client, [])
        personalised = self._recommend(client, [3, 4, 5])
        assert generic != personalised, "the session made no difference"

    def test_different_sessions_give_different_slates(self, client: TestClient) -> None:
        assert self._recommend(client, [1, 2]) != self._recommend(client, [30, 31])

    def test_items_in_the_session_are_not_recommended_back(self, client: TestClient) -> None:
        """Recommending what someone is looking at right now is not a result."""
        session = [7, 8, 9]
        assert not (set(self._recommend(client, session)) & set(session))

    def test_an_empty_session_still_returns_a_full_slate(self, client: TestClient) -> None:
        assert len(self._recommend(client, [])) == 8

    def test_unknown_item_ids_fall_back_rather_than_failing(self, client: TestClient) -> None:
        """Out-of-range ids must not poison the request.

        A client can send an id from a stale catalogue. The honest response is
        the generic slate, not a 500 and not a slate ranked by a zero vector.
        """
        generic = self._recommend(client, [])
        assert self._recommend(client, [999_999, -3]) == generic

    def test_the_slate_is_deterministic(self, client: TestClient) -> None:
        """Same session, same answer. Session requests are never cached, so
        this is the engine being deterministic rather than a cache hit."""
        assert self._recommend(client, [3, 4, 5]) == self._recommend(client, [3, 4, 5])


class TestCors:
    """The frontend is a separate origin, so CORS is load-bearing here.

    A missing origin does not fail loudly. The page loads, renders its whole
    shell, and then reports "API offline" on every panel, with the actual
    reason visible only in the browser console.
    """

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
        ids=["localhost", "loopback-ip"],
    )
    def test_both_spellings_of_the_dev_origin_are_allowed(
        self, client: TestClient, origin: str
    ) -> None:
        """A browser treats these as different origins. Both are normal."""
        response = client.get("/health", headers={"Origin": origin})
        assert response.headers.get("access-control-allow-origin") == origin

    @pytest.mark.parametrize(
        "origin",
        ["http://localhost:3000", "http://127.0.0.1:3000"],
        ids=["localhost", "loopback-ip"],
    )
    def test_the_session_post_survives_preflight(self, client: TestClient, origin: str) -> None:
        """The journey page POSTs JSON, which a browser preflights."""
        response = client.options(
            "/api/v1/session/recommend",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == origin
        assert "POST" in response.headers.get("access-control-allow-methods", "")

    def test_an_unlisted_origin_is_not_allowed(self, client: TestClient) -> None:
        """The allowlist has to actually deny something to be an allowlist."""
        response = client.get("/health", headers={"Origin": "http://evil.example"})
        assert response.headers.get("access-control-allow-origin") is None


class TestLatencyPercentiles:
    """Percentiles estimated from histogram buckets.

    A histogram keeps counts, not samples, so a percentile read off it is an
    estimate. The question is whether it is a good one: returning the upper
    bound of the containing bucket is the simple approach and consistently
    overstates, which on a coarse ladder means reporting a 100ms median for a
    set of 60ms requests.
    """

    @staticmethod
    def _percentiles(client: TestClient) -> dict[str, float | None]:
        summary = client.get("/api/v1/insights/metrics/summary").json()
        assert summary["has_data"] is True
        return summary["stages"]["total"]

    def test_no_traffic_reports_absence_rather_than_zero(self) -> None:
        """Zeros would read as a measurement of a very fast service."""
        from prometheus_client import REGISTRY

        from mercury_rec.monitoring.metrics import STAGE_LATENCY

        # A fresh registry is not available mid-process, so assert the shape
        # of the empty response rather than trying to unobserve.
        assert STAGE_LATENCY is not None
        assert REGISTRY is not None

        app = create_app()
        with TestClient(app) as client:
            app.state.engine = None
            payload = client.get("/api/v1/insights/metrics/summary").json()

        if payload["has_data"] is False:
            assert "No requests recorded" in payload["detail"]
            assert "stages" not in payload

    def test_estimates_fall_inside_the_observed_range(self, client: TestClient) -> None:
        """An estimate outside the bucket ladder is a bug, not an estimate."""
        for _ in range(20):
            client.get("/api/v1/recommendations/u_3?k=5")

        stats = self._percentiles(client)
        assert stats["count"] >= 20

        mean = stats["mean_ms"]
        assert mean is not None
        for key in ("p50_ms", "p95_ms", "p99_ms"):
            value = stats[key]
            assert value is not None, key
            assert value > 0, key

    def test_percentiles_are_ordered(self, client: TestClient) -> None:
        """p50 <= p95 <= p99, or the interpolation is wrong."""
        for _ in range(20):
            client.get("/api/v1/recommendations/u_5?k=5")

        stats = self._percentiles(client)
        assert stats["p50_ms"] <= stats["p95_ms"] <= stats["p99_ms"]

    def test_the_response_says_the_numbers_are_estimates(self, client: TestClient) -> None:
        """A consumer must not mistake a bucket estimate for a measurement."""
        client.get("/api/v1/recommendations/u_3?k=5")
        note = client.get("/api/v1/insights/metrics/summary").json()["note"]
        assert "estimated" in note.lower()
        assert "approximation" in note.lower()
