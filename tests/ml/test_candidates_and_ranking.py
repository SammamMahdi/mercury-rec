"""Tests for candidate fusion and the learning-to-rank dataset builder.

Two failure modes here are silent and expensive:

- Fusion dropping or duplicating candidates changes what the ranker can ever
  return, without raising anything.
- A LightGBM group array that does not line up with the labels optimises a
  ranking of the wrong rows, trains happily, and produces a model that is
  wrong in a way no metric on the training run reveals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.core.enums import RetrievalSource
from mercury_rec.models.ranking.dataset import (
    RANKING_FEATURES,
    RankingDataset,
    build_ranking_dataset,
)
from mercury_rec.retrieval.candidates import (
    SourceResult,
    reciprocal_rank_fusion,
    retrieval_recall,
    retrieve_from_scores,
)


def _source(source: RetrievalSource, items: list[int], scores: list[float]) -> SourceResult:
    return SourceResult(
        source=source,
        item_ids=np.array(items, dtype=np.int64),
        scores=np.array(scores, dtype=np.float32),
        elapsed_ms=0.1,
    )


class TestFusion:
    def test_union_of_sources_is_retained(self) -> None:
        """A candidate missed by fusion can never be recommended."""
        results = [
            _source(RetrievalSource.POPULARITY, [1, 2, 3], [9.0, 8.0, 7.0]),
            _source(RetrievalSource.ITEM_CF, [3, 4, 5], [0.9, 0.8, 0.7]),
        ]
        fused = reciprocal_rank_fusion(results)
        assert set(fused.item_ids.tolist()) == {1, 2, 3, 4, 5}

    def test_items_in_several_sources_rank_higher(self) -> None:
        """Agreement between independent sources is the core RRF signal."""
        results = [
            _source(RetrievalSource.POPULARITY, [1, 2, 3], [9.0, 8.0, 7.0]),
            _source(RetrievalSource.ITEM_CF, [3, 2, 1], [0.9, 0.8, 0.7]),
            _source(RetrievalSource.TWO_TOWER, [2, 9, 8], [0.5, 0.4, 0.3]),
        ]
        fused = reciprocal_rank_fusion(results)
        # Item 2 appears in all three sources, near the top of each.
        assert fused.item_ids[0] == 2
        assert fused.sources_per_item[0] == 3

    def test_no_duplicates_in_the_pool(self) -> None:
        results = [
            _source(RetrievalSource.POPULARITY, [1, 2, 3], [3.0, 2.0, 1.0]),
            _source(RetrievalSource.ITEM_CF, [1, 2, 3], [0.3, 0.2, 0.1]),
        ]
        ids = reciprocal_rank_fusion(results).item_ids.tolist()
        assert len(ids) == len(set(ids))

    def test_excluded_items_are_dropped(self) -> None:
        results = [_source(RetrievalSource.POPULARITY, [1, 2, 3], [3.0, 2.0, 1.0])]
        fused = reciprocal_rank_fusion(results, exclude={1, 3})
        assert fused.item_ids.tolist() == [2]

    def test_faiss_padding_is_ignored(self) -> None:
        """FAISS pads short results with -1, which are not item ids."""
        results = [_source(RetrievalSource.TWO_TOWER, [5, -1, -1], [0.9, 0.0, 0.0])]
        assert reciprocal_rank_fusion(results).item_ids.tolist() == [5]

    def test_pool_is_capped(self) -> None:
        """The cap is the retrieval/ranking budget dial."""
        results = [
            _source(
                RetrievalSource.POPULARITY,
                list(range(100)),
                [float(100 - i) for i in range(100)],
            )
        ]
        assert len(reciprocal_rank_fusion(results, max_candidates=20)) == 20

    def test_per_source_scores_preserve_provenance(self) -> None:
        """A candidate's origin must stay visible for features and the UI."""
        results = [
            _source(RetrievalSource.POPULARITY, [1, 2], [9.0, 8.0]),
            _source(RetrievalSource.ITEM_CF, [2, 3], [0.9, 0.8]),
        ]
        fused = reciprocal_rank_fusion(results)
        position = fused.item_ids.tolist().index(1)

        assert not np.isnan(fused.per_source_scores["popularity"][position])
        # Item 1 was never returned by item_cf, so that score is absent rather
        # than zero - zero would be a claim the source scored it badly.
        assert np.isnan(fused.per_source_scores["item_cf"][position])

    def test_empty_sources_produce_an_empty_pool(self) -> None:
        fused = reciprocal_rank_fusion([])
        assert len(fused) == 0

    def test_retrieve_from_scores_returns_the_top_k_in_order(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.7], dtype=np.float32)
        result = retrieve_from_scores(scores, source=RetrievalSource.POPULARITY, k=3)
        assert result.item_ids.tolist() == [1, 3, 2]


class TestRetrievalRecall:
    def test_measures_the_ceiling(self) -> None:
        """Retrieval recall bounds what any downstream stage can achieve."""
        assert retrieval_recall(np.array([1, 2, 3]), {2, 3}) == pytest.approx(1.0)
        assert retrieval_recall(np.array([1, 2, 3]), {2, 99}) == pytest.approx(0.5)
        assert retrieval_recall(np.array([1, 2, 3]), {98, 99}) == pytest.approx(0.0)

    def test_empty_relevant_set_scores_zero(self) -> None:
        assert retrieval_recall(np.array([1, 2]), set()) == 0.0


def _candidate_frame(n_users: int = 5, n_per_user: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for user in range(n_users):
        for position in range(n_per_user):
            row: dict[str, float | int] = {
                "user_id": user,
                "item_id": user * 100 + position,
                # Exactly one positive per user, at a varying position.
                "label": int(position == user % n_per_user),
            }
            for name in RANKING_FEATURES:
                row[name] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


class TestRankingDataset:
    def test_groups_align_with_labels(self) -> None:
        """A mis-aligned group array trains happily on the wrong rows."""
        dataset = build_ranking_dataset(_candidate_frame())
        assert dataset.groups.sum() == len(dataset.labels)
        assert dataset.n_groups == 5

    def test_mismatched_groups_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Group sizes sum to"):
            RankingDataset(
                features=np.zeros((10, len(RANKING_FEATURES)), dtype=np.float32),
                labels=np.zeros(10, dtype=np.int32),
                groups=np.array([3, 3]),  # sums to 6, not 10
                user_ids=np.zeros(10, dtype=np.int64),
                item_ids=np.zeros(10, dtype=np.int64),
            )

    def test_groups_without_positives_are_dropped(self) -> None:
        """LambdaRank's gradient is identically zero on them."""
        frame = _candidate_frame()
        frame.loc[frame["user_id"] == 0, "label"] = 0

        dataset = build_ranking_dataset(frame, drop_groups_without_positives=True)
        assert dataset.n_groups == 4
        assert 0 not in set(dataset.user_ids.tolist())

    def test_groups_can_be_kept_when_requested(self) -> None:
        frame = _candidate_frame()
        frame.loc[frame["user_id"] == 0, "label"] = 0
        dataset = build_ranking_dataset(frame, drop_groups_without_positives=False)
        assert dataset.n_groups == 5

    def test_feature_column_order_is_the_contract(self) -> None:
        """SHAP and the API both index by position, so order is load-bearing."""
        dataset = build_ranking_dataset(_candidate_frame())
        assert dataset.feature_names == RANKING_FEATURES
        assert dataset.features.shape[1] == len(RANKING_FEATURES)

    def test_missing_features_are_reported_by_name(self) -> None:
        frame = _candidate_frame().drop(columns=["retrieval_rank_fused"])
        with pytest.raises(ValueError, match="retrieval_rank_fused"):
            build_ranking_dataset(frame)

    def test_missing_label_column_is_reported(self) -> None:
        frame = _candidate_frame().drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            build_ranking_dataset(frame)

    def test_all_negative_input_raises_rather_than_training_on_nothing(self) -> None:
        frame = _candidate_frame()
        frame["label"] = 0
        with pytest.raises(ValueError, match="No ranking groups survived"):
            build_ranking_dataset(frame)

    def test_summary_reports_the_shape(self) -> None:
        summary = build_ranking_dataset(_candidate_frame()).summary()
        assert summary["groups"] == 5
        assert summary["positives"] == 5
        assert summary["n_features"] == len(RANKING_FEATURES)
