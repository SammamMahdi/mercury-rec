"""Behavioural tests for the retrieval models.

These check properties that must hold for any correct implementation --
exclusion is honoured, output is deterministic, training actually learns --
rather than pinning specific metric values. A test asserting "recall@10 ==
0.0099" would fail on every legitimate improvement while catching none of the
bugs that matter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mercury_rec.core.enums import EventType
from mercury_rec.models.base import RecommendationContext, Recommender
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.mf import BPRRecommender
from mercury_rec.models.popularity import (
    ContextualPopularityRecommender,
    PopularityRecommender,
    TrendingRecommender,
    hour_bucket,
)

N_USERS = 60
N_ITEMS = 40


@pytest.fixture
def interactions() -> pd.DataFrame:
    """Synthetic interactions with a deliberately skewed popularity profile.

    Item 0 is by far the most popular, so popularity-based models have an
    unambiguous correct answer to check against.
    """
    rng = np.random.default_rng(3)
    rows = []
    base = pd.Timestamp("2024-05-01", tz="UTC")

    for user in range(N_USERS):
        for _ in range(rng.integers(6, 15)):
            # Zipf-ish: low ids far more likely.
            item = int(min(rng.zipf(1.6), N_ITEMS))
            item = min(item - 1, N_ITEMS - 1)
            rows.append(
                {
                    "user_id": user,
                    "item_id": item,
                    "ts": base + pd.to_timedelta(int(rng.integers(0, 60 * 24 * 30)), unit="m"),
                    "event_type": int(
                        rng.choice(
                            [EventType.VIEW, EventType.ADD_TO_CART, EventType.PURCHASE],
                            p=[0.7, 0.2, 0.1],
                        )
                    ),
                    "price_at_event": float(rng.uniform(5, 50)),
                    "vertical": int(item % 6),
                    "merchant_id": int(item % 10),
                    "region_id": int(user % 4),
                }
            )
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


ALL_MODELS = [
    PopularityRecommender,
    ContextualPopularityRecommender,
    TrendingRecommender,
    ItemCFRecommender,
]


@pytest.mark.parametrize("model_class", ALL_MODELS)
class TestCommonContract:
    """Properties every recommender must satisfy, checked identically."""

    def test_recommending_before_fit_is_an_error(self, model_class: type[Recommender]) -> None:
        model = model_class(N_USERS, N_ITEMS)
        with pytest.raises(RuntimeError, match="must be fitted"):
            model.recommend(0)

    def test_returns_requested_count(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        model = model_class(N_USERS, N_ITEMS)
        model.fit(interactions)
        assert len(model.recommend(0, k=10)) == 10

    def test_excluded_items_never_appear(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        """The single most important guard.

        Re-recommending training history inflates every metric, because those
        are exactly the items the model has the most evidence for.
        """
        model = model_class(N_USERS, N_ITEMS)
        model.fit(interactions)
        exclude = {0, 1, 2, 3, 4}
        assert not set(model.recommend(0, k=20, exclude=exclude)) & exclude

    def test_output_is_deterministic(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        model = model_class(N_USERS, N_ITEMS)
        model.fit(interactions)
        assert model.recommend(5, k=10) == model.recommend(5, k=10)

    def test_no_duplicates_in_output(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        model = model_class(N_USERS, N_ITEMS)
        model.fit(interactions)
        result = model.recommend(7, k=20)
        assert len(result) == len(set(result))

    def test_respects_candidate_restriction(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        """The ranking stage scores only what retrieval proposed."""
        model = model_class(N_USERS, N_ITEMS)
        model.fit(interactions)
        candidates = np.array([3, 7, 11, 19])
        assert set(model.recommend(0, k=4, candidates=candidates)) <= set(candidates.tolist())

    def test_fit_result_records_training_time(
        self, model_class: type[Recommender], interactions: pd.DataFrame
    ) -> None:
        model = model_class(N_USERS, N_ITEMS)
        result = model.fit(interactions)
        assert result.train_seconds >= 0
        assert result.n_interactions == len(interactions)
        assert result.model == model.name


class TestPopularity:
    def test_ranks_the_most_popular_item_first(self, interactions: pd.DataFrame) -> None:
        """With a Zipf profile the top item must lead an unfiltered list."""
        model = PopularityRecommender(N_USERS, N_ITEMS, half_life_days=None)
        model.fit(interactions)

        weighted = (
            interactions.groupby("item_id")["event_type"]
            .apply(lambda s: sum(model._weights.get(int(e), 1.0) for e in s))
            .sort_values(ascending=False)
        )
        assert model.recommend(0, k=1)[0] == int(weighted.index[0])

    def test_is_identical_for_every_user(self, interactions: pd.DataFrame) -> None:
        """Global popularity is by definition not personalised."""
        model = PopularityRecommender(N_USERS, N_ITEMS)
        model.fit(interactions)
        assert model.recommend(0, k=10) == model.recommend(42, k=10)

    def test_recency_decay_changes_the_ranking(self, interactions: pd.DataFrame) -> None:
        """If decay never changed anything the parameter would be decorative."""
        decayed = PopularityRecommender(N_USERS, N_ITEMS, half_life_days=1.0)
        flat = PopularityRecommender(N_USERS, N_ITEMS, half_life_days=None)
        decayed.fit(interactions)
        flat.fit(interactions)
        assert decayed.recommend(0, k=20) != flat.recommend(0, k=20)

    def test_trending_uses_only_the_recent_window(self, interactions: pd.DataFrame) -> None:
        model = TrendingRecommender(N_USERS, N_ITEMS, window_days=3.0)
        result = model.fit(interactions)
        assert result.extra["window_events"] < len(interactions)


class TestContextualPopularity:
    def test_context_changes_recommendations(self, interactions: pd.DataFrame) -> None:
        """Contextual personalisation must actually do something.

        If every context produced the same list, the contextual demo would be
        a hard-coded illusion - which the spec explicitly forbids.
        """
        model = ContextualPopularityRecommender(N_USERS, N_ITEMS, min_observations=1)
        model.fit(interactions)

        by_vertical = {
            vertical: model.recommend(
                0, k=10, context=RecommendationContext(vertical=vertical, region_id=0)
            )
            for vertical in range(6)
        }
        distinct = {tuple(items) for items in by_vertical.values()}
        assert len(distinct) > 1, "vertical context had no effect on the ranking"

    def test_falls_back_when_a_slice_is_too_thin(self, interactions: pd.DataFrame) -> None:
        """An unseen context must still return something sensible."""
        model = ContextualPopularityRecommender(N_USERS, N_ITEMS, min_observations=10_000)
        model.fit(interactions)
        assert len(model.recommend(0, k=5, context=RecommendationContext(region_id=99))) == 5


@pytest.mark.parametrize(
    ("hour", "expected"), [(0, 0), (5, 0), (8, 1), (12, 2), (15, 3), (19, 4), (23, 5)]
)
def test_hour_bucketing(hour: int, expected: int) -> None:
    assert hour_bucket(hour) == expected


class TestItemCF:
    def test_similar_items_are_symmetric_in_spirit(self, interactions: pd.DataFrame) -> None:
        """A co-occurring pair should find each other as neighbours."""
        model = ItemCFRecommender(N_USERS, N_ITEMS, popularity_damping=0.0)
        model.fit(interactions)
        neighbours = model.similar_items(0, k=5)
        assert neighbours
        assert all(item != 0 for item, _ in neighbours), "an item is its own neighbour"
        assert all(score > 0 for _, score in neighbours)

    def test_is_personalised(self, interactions: pd.DataFrame) -> None:
        """Unlike popularity, CF must differ between users with different history."""
        model = ItemCFRecommender(N_USERS, N_ITEMS)
        model.fit(interactions)
        lists = {tuple(model.recommend(user, k=10)) for user in range(10)}
        assert len(lists) > 1

    def test_user_without_history_scores_zero(self) -> None:
        """No neighbourhood signal must not become an invented ranking."""
        frame = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "item_id": [0, 1, 0, 1],
                "ts": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
                "event_type": [int(EventType.PURCHASE)] * 4,
            }
        )
        model = ItemCFRecommender(10, 5)
        model.fit(frame)
        scores = model._score(0, np.arange(5), RecommendationContext())
        assert np.allclose(scores, 0.0)


class TestBPR:
    def test_training_reduces_the_loss(self, interactions: pd.DataFrame) -> None:
        """A model whose loss does not fall is not learning."""
        model = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=12, device="cpu")
        result = model.fit(interactions)
        assert result.extra["final_loss"] < result.extra["first_loss"]

    def test_is_personalised(self, interactions: pd.DataFrame) -> None:
        model = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=10, device="cpu")
        model.fit(interactions)
        lists = {tuple(model.recommend(user, k=10)) for user in range(10)}
        assert len(lists) > 1

    def test_same_seed_reproduces_the_model(self, interactions: pd.DataFrame) -> None:
        """Reproducibility is a stated requirement, so it is asserted."""
        first = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=5, seed=7, device="cpu")
        second = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=5, seed=7, device="cpu")
        first.fit(interactions)
        second.fit(interactions)
        assert first.recommend(3, k=10) == second.recommend(3, k=10)

    def test_embeddings_contain_no_nans(self, interactions: pd.DataFrame) -> None:
        """NaNs would propagate silently into the ANN index and the 3D view."""
        model = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=8, device="cpu")
        model.fit(interactions)
        assert np.isfinite(model.item_factors).all()

    def test_round_trips_through_disk(self, interactions: pd.DataFrame, tmp_path: Path) -> None:
        """Serving loads a trained model, so persistence must be exact."""
        model = BPRRecommender(N_USERS, N_ITEMS, dim=16, epochs=5, device="cpu")
        model.fit(interactions)
        before = model.recommend(4, k=10)

        path = tmp_path / "bpr.pt"
        model.save(path)
        restored = BPRRecommender.load(path)

        assert restored.recommend(4, k=10) == before
        np.testing.assert_allclose(restored.item_factors, model.item_factors)
