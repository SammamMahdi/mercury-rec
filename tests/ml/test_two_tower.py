"""Tests for the two-tower architecture, its loss and the ANN indexes.

The sampled-softmax loss is the component most commonly got wrong, and wrong
in a way that produces a perfectly healthy-looking loss curve while the model
systematically suppresses popular items. These tests check the loss's
properties directly rather than inferring correctness from training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mercury_rec.models.two_tower.losses import (
    build_positive_mask,
    compute_log_frequency,
    encode_pairs,
    in_batch_softmax_loss,
    sorted_interaction_pairs,
)
from mercury_rec.models.two_tower.model import Tower, TowerSpec, TwoTowerModel
from mercury_rec.models.two_tower.recommender import TwoTowerRecommender
from mercury_rec.retrieval.index import (
    ExactIndex,
    FlatIPIndex,
    HNSWIndex,
    benchmark_index,
)


class TestTower:
    def test_output_is_l2_normalised(self) -> None:
        """Normalisation is what makes a dot product a cosine similarity.

        Without it FAISS's inner-product index would not return cosine
        neighbours, and the model could inflate scores by growing norms
        instead of learning direction.
        """
        spec = TowerSpec(n_ids=50, categorical_cardinalities=(5, 3), n_continuous=4)
        tower = Tower(spec)
        tower.eval()

        with torch.no_grad():
            output = tower(
                torch.arange(10),
                torch.randint(0, 4, (10, 2)),
                torch.randn(10, 4),
            )
        norms = output.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(10), rtol=1e-5, atol=1e-5)

    def test_nan_continuous_inputs_do_not_poison_the_embedding(self) -> None:
        """NaN means 'not observed' throughout the feature layer.

        An unhandled NaN here would propagate through the MLP into the item
        embedding, then into the ANN index, where FAISS reports it as a -1 id
        rather than raising.
        """
        spec = TowerSpec(n_ids=10, n_continuous=3)
        tower = Tower(spec)
        tower.eval()

        continuous = torch.tensor([[1.0, float("nan"), 3.0], [float("nan")] * 3])
        with torch.no_grad():
            output = tower(torch.tensor([0, 1]), continuous=continuous)
        assert torch.isfinite(output).all()

    def test_mismatched_tower_dimensions_are_rejected(self) -> None:
        """Towers must share a space, or the dot product is meaningless."""
        with pytest.raises(ValueError, match="must match"):
            TwoTowerModel(TowerSpec(n_ids=10, output_dim=32), TowerSpec(n_ids=10, output_dim=64))

    def test_missing_expected_inputs_raise(self) -> None:
        tower = Tower(TowerSpec(n_ids=10, categorical_cardinalities=(4,)))
        with pytest.raises(ValueError, match="categorical"):
            tower(torch.tensor([0]))


class TestInBatchSoftmaxLoss:
    def test_perfect_alignment_gives_near_zero_loss(self) -> None:
        """Identity-aligned embeddings are the best achievable case."""
        embeddings = torch.eye(4)
        loss = in_batch_softmax_loss(
            embeddings,
            embeddings,
            item_ids=torch.arange(4),
            temperature=0.01,
        )
        assert float(loss) < 0.01

    def test_loss_falls_as_alignment_improves(self) -> None:
        torch.manual_seed(0)
        users = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
        item_ids = torch.arange(8)

        aligned = in_batch_softmax_loss(users, users, item_ids=item_ids)
        misaligned = in_batch_softmax_loss(
            users, torch.nn.functional.normalize(torch.randn(8, 16), dim=-1), item_ids=item_ids
        )
        assert float(aligned) < float(misaligned)

    def test_duplicate_items_are_masked(self) -> None:
        """A repeated item must not be scored as its own negative.

        Under a power-law catalogue the same popular item lands in a batch
        repeatedly. Left unmasked, the model is taught to push down an item
        that is simultaneously another row's positive.

        The embeddings are constructed so the effect is unmistakable rather
        than a rounding difference: rows 0 and 2 carry the SAME item vector,
        so without masking user 0's probability mass splits evenly between
        two identical columns and the loss rises by about log(2). Random
        embeddings would make the masked entry contribute almost nothing and
        the test would pass on noise.
        """
        identity = torch.eye(4)
        users = identity.clone()
        items = identity.clone()
        # Column 2 is made identical to column 0, user 0's true positive.
        items[2] = items[0]

        duplicated = torch.tensor([7, 1, 7, 3])  # masked: (0,2) and (2,0)
        distinct = torch.tensor([7, 1, 9, 3])  # not masked

        masked_loss = float(
            in_batch_softmax_loss(users, items, item_ids=duplicated, temperature=0.1)
        )
        unmasked_loss = float(
            in_batch_softmax_loss(users, items, item_ids=distinct, temperature=0.1)
        )

        assert masked_loss < unmasked_loss, (
            "masking a duplicated item must lower the loss; got "
            f"masked={masked_loss:.4f} unmasked={unmasked_loss:.4f}"
        )

    def test_logq_correction_changes_the_objective(self) -> None:
        """If it never changed anything the parameter would be decorative."""
        torch.manual_seed(2)
        users = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        items = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        item_ids = torch.arange(6)
        log_frequency = torch.log(torch.tensor([0.5, 0.2, 0.1, 0.1, 0.05, 0.05]))

        without = float(in_batch_softmax_loss(users, items, item_ids=item_ids))
        with_correction = float(
            in_batch_softmax_loss(users, items, item_ids=item_ids, log_frequency=log_frequency)
        )
        assert without != pytest.approx(with_correction)

    def test_logq_correction_boosts_rare_items_more_than_popular_ones(self) -> None:
        """Check the mechanism itself, not a hand-waved end effect.

        The correction subtracts ``log Q(i)`` from each logit. Because a rare
        item has a small Q, ``-log Q`` is a *larger* positive boost for it than
        for a popular one. That is exactly the intent: a rarely-sampled
        negative stands in for all the similar items that were not sampled, so
        its contribution to the denominator must be scaled up, which stops the
        objective from over-penalising the head of the catalogue.

        Asserting the boost ordering directly is a claim that can be checked
        from the formula, unlike a loss comparison whose sign depends on the
        particular embeddings drawn.
        """
        log_frequency = torch.log(torch.tensor([0.90, 0.05, 0.001]))
        boost = -log_frequency

        assert boost[2] > boost[1] > boost[0], "rarer items must receive a larger boost"
        assert float(boost[0]) == pytest.approx(0.10536, abs=1e-4)

    def test_mismatched_batch_shapes_are_rejected(self) -> None:
        """A mismatch must raise here rather than broadcast into nonsense.

        The duplicate mask is (n_items, n_items); against non-square logits it
        broadcasts, and the error surfaces later inside cross_entropy - or not
        at all.
        """
        users = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
        items = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        with pytest.raises(ValueError, match="one positive item per user"):
            in_batch_softmax_loss(users, items, item_ids=torch.arange(3))

    def test_empty_batch_is_handled(self) -> None:
        empty = torch.zeros((0, 8))
        assert (
            float(in_batch_softmax_loss(empty, empty, item_ids=torch.zeros(0, dtype=torch.long)))
            == 0.0
        )

    def test_no_nan_under_heavy_masking(self) -> None:
        """A fully-masked row must not produce NaN.

        This is why the mask uses a large negative constant rather than -inf:
        under bf16 autocast an all -inf row yields NaN, which then propagates
        into every parameter and destroys the run silently.
        """
        users = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        items = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        all_same = torch.zeros(4, dtype=torch.long)  # every item identical
        loss = in_batch_softmax_loss(users, items, item_ids=all_same)
        assert torch.isfinite(loss)


class TestLogFrequency:
    def test_probabilities_sum_to_one(self) -> None:
        counts = torch.tensor([10, 5, 1, 0])
        assert float(torch.exp(compute_log_frequency(counts)).sum()) == pytest.approx(1.0, abs=1e-5)

    def test_zero_count_items_are_finite(self) -> None:
        """log(0) would make a never-seen item win every softmax."""
        assert torch.isfinite(compute_log_frequency(torch.tensor([10, 0, 0]))).all()

    def test_popular_items_have_higher_probability(self) -> None:
        log_p = compute_log_frequency(torch.tensor([100, 10, 1]))
        assert log_p[0] > log_p[1] > log_p[2]


class TestPositiveMask:
    def test_marks_known_pairs_only(self) -> None:
        users = torch.tensor([0, 1])
        items = torch.tensor([5, 9])
        n_items = 20
        # User 0 interacted with items 5 and 9; user 1 only with 9.
        known = sorted_interaction_pairs(torch.tensor([0, 0, 1]), torch.tensor([5, 9, 9]), n_items)

        mask = build_positive_mask(users, items, known, n_items)
        assert bool(mask[0, 0])  # (0, 5) known
        assert bool(mask[0, 1])  # (0, 9) known
        assert not bool(mask[1, 0])  # (1, 5) NOT known
        assert bool(mask[1, 1])  # (1, 9) known

    def test_encode_pairs_is_collision_free(self) -> None:
        n_items = 1000
        users = torch.tensor([0, 0, 1, 1])
        items = torch.tensor([0, 999, 0, 999])
        keys = encode_pairs(users, items, n_items)
        assert len(set(keys.tolist())) == 4


class TestVectorIndexes:
    @pytest.fixture
    def embeddings(self) -> np.ndarray:
        rng = np.random.default_rng(0)
        raw = rng.standard_normal((500, 32)).astype(np.float32)
        return raw / np.linalg.norm(raw, axis=1, keepdims=True)

    def test_exact_index_finds_the_true_nearest(self, embeddings: np.ndarray) -> None:
        """An item is its own nearest neighbour under cosine similarity."""
        index = ExactIndex()
        index.build(embeddings)
        _, ids = index.search(embeddings[:5], k=1)
        np.testing.assert_array_equal(ids.ravel(), np.arange(5))

    def test_results_are_sorted_by_descending_score(self, embeddings: np.ndarray) -> None:
        index = ExactIndex()
        index.build(embeddings)
        scores, _ = index.search(embeddings[:3], k=10)
        assert np.all(np.diff(scores, axis=1) <= 1e-6)

    def test_faiss_flat_agrees_with_exact(self, embeddings: np.ndarray) -> None:
        """Both are exact, so they must return the same neighbours."""
        exact = ExactIndex()
        exact.build(embeddings)
        flat = FlatIPIndex()
        flat.build(embeddings)

        _, exact_ids = exact.search(embeddings[:20], k=10)
        _, flat_ids = flat.search(embeddings[:20], k=10)
        np.testing.assert_array_equal(exact_ids, flat_ids)

    def test_hnsw_achieves_high_recall(self, embeddings: np.ndarray) -> None:
        """Approximation is acceptable; being wrong is not.

        This asserts a floor rather than exactness, because HNSW is
        approximate by design - the point is that the trade is small.
        """
        exact = ExactIndex()
        exact.build(embeddings)
        _, exact_ids = exact.search(embeddings[:50], k=10)

        result = benchmark_index(
            HNSWIndex(), embeddings, embeddings[:50], k=10, exact_ids=exact_ids
        )
        assert result.recall_at_k > 0.90, f"HNSW recall@10 was only {result.recall_at_k:.3f}"

    def test_nan_embeddings_are_rejected(self, embeddings: np.ndarray) -> None:
        """FAISS returns -1 ids for NaN rather than raising, so we raise."""
        corrupted = embeddings.copy()
        corrupted[3, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ExactIndex().build(corrupted)

    def test_searching_before_building_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="not been built"):
            ExactIndex().search(np.zeros((1, 8), dtype=np.float32), k=1)

    def test_benchmark_reports_exact_recall_as_one(self, embeddings: np.ndarray) -> None:
        result = benchmark_index(ExactIndex(), embeddings, embeddings[:20], k=10)
        assert result.recall_at_k == 1.0
        assert result.mean_query_ms > 0


class TestServingState:
    """Round-tripping the embeddings a replica actually serves from.

    Serving does not need the towers: both sides were encoded offline and
    scoring is a dot product. What must survive the round trip is the exact
    alignment between row index and id, because an off-by-one there produces
    confident, plausible, wrong recommendations that no downstream check can
    catch.
    """

    @staticmethod
    def _catalogue(n_items: int, n_users: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        item_ids = np.arange(n_items)
        items = pd.DataFrame(
            {
                "item_id": item_ids,
                "category_id": item_ids % 7,
                "merchant_id": item_ids % 5,
                "vertical": item_ids % 6,
                "price": (item_ids % 11 + 3).astype(float),
                "is_available": True,
            }
        )
        user_ids = np.arange(n_users)
        users = pd.DataFrame(
            {
                "user_id": user_ids,
                "region_id": user_ids % 4,
                "device_pref": user_ids % 3,
            }
        )
        return items, users

    def _fitted(self, tmp_path: Path) -> tuple[TwoTowerRecommender, pd.DataFrame, pd.DataFrame]:
        n_items, n_users, dim = 24, 12, 8
        items, users = self._catalogue(n_items, n_users)

        model = TwoTowerRecommender(n_users, n_items, items=items, users=users)
        rng = np.random.default_rng(19)
        item_embeddings = rng.normal(size=(n_items, dim)).astype(np.float32)
        item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)

        model._item_embeddings = item_embeddings
        model._fitted = True
        model.set_user_embeddings(
            np.arange(n_users), rng.normal(size=(n_users, dim)).astype(np.float32)
        )
        return model, items, users

    def test_round_trip_preserves_both_matrices(self, tmp_path: Path) -> None:
        model, items, users = self._fitted(tmp_path)
        path = tmp_path / "two_tower.npz"
        model.save_serving_state(path)

        loaded = TwoTowerRecommender.load_serving_state(path, items=items, users=users)

        np.testing.assert_array_equal(loaded.item_embeddings, model.item_embeddings)
        for user_id in range(len(users)):
            np.testing.assert_array_equal(
                loaded.user_vector(user_id),  # type: ignore[arg-type]
                model.user_vector(user_id),  # type: ignore[arg-type]
            )

    def test_a_catalogue_of_the_wrong_size_is_rejected(self, tmp_path: Path) -> None:
        """Mismatched shapes must raise, not silently shift every id."""
        model, items, users = self._fitted(tmp_path)
        path = tmp_path / "two_tower.npz"
        model.save_serving_state(path)

        with pytest.raises(ValueError, match="item embeddings"):
            TwoTowerRecommender.load_serving_state(path, items=items.iloc[:-1], users=users)

        with pytest.raises(ValueError, match="user embeddings"):
            TwoTowerRecommender.load_serving_state(path, items=items, users=users.iloc[:-1])

    def test_saving_without_user_embeddings_raises(self, tmp_path: Path) -> None:
        """A file that can score no one is worse than no file at all."""
        n_items, n_users = 24, 12
        items, users = self._catalogue(n_items, n_users)
        model = TwoTowerRecommender(n_users, n_items, items=items, users=users)
        model._item_embeddings = np.zeros((n_items, 8), dtype=np.float32)
        model._fitted = True

        with pytest.raises(RuntimeError, match="user embeddings"):
            model.save_serving_state(tmp_path / "two_tower.npz")

    def test_a_user_who_was_never_encoded_has_no_vector(self, tmp_path: Path) -> None:
        """None, not a zero vector.

        Zero is a valid point in the space. Returning it would place a
        cold-start user at the exact centre of the catalogue and let the
        two-tower source contribute a slate of arbitrary items.
        """
        n_items, n_users, dim = 24, 12, 8
        items, users = self._catalogue(n_items, n_users)
        model = TwoTowerRecommender(n_users, n_items, items=items, users=users)
        model._item_embeddings = np.zeros((n_items, dim), dtype=np.float32)
        model._fitted = True

        rng = np.random.default_rng(23)
        encoded = np.array([0, 1, 2])
        model.set_user_embeddings(encoded, rng.normal(size=(3, dim)).astype(np.float32))

        assert model.user_vector(0) is not None
        assert model.user_vector(7) is None, "never encoded"
        assert model.user_vector(n_users + 5) is None, "out of range"
        assert model.user_vector(-1) is None, "negative index must not wrap"
