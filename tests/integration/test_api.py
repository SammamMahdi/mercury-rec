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

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mercury_rec.api.main import REQUEST_ID_HEADER, create_app
from mercury_rec.cache.redis_cache import RecommendationCache
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


@pytest.fixture
def client(bundle: ArtifactBundle) -> Iterator[TestClient]:
    app = create_app()
    with TestClient(app) as test_client:
        # Replace whatever startup loaded with the in-memory bundle.
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
