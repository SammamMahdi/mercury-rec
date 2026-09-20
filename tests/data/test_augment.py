"""Tests for the synthesised-attribute layer.

Two properties matter most here and are both non-obvious:

1. **Determinism by id.** An entity's generated attributes must depend only on
   its id and the seed - never on ordering, batching, or how many entities were
   processed first. That is what makes the ``demo`` preset a genuine subset of
   ``full`` rather than a separately-generated dataset.
2. **The realised distribution matches the configured one.** The vertical
   assignment in particular is easy to get subtly wrong, and did go wrong:
   assigning roots independently produced an item distribution nothing like
   the configured weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from mercury_rec.core.enums import Vertical
from mercury_rec.data.augment import (
    AugmentationConfig,
    assign_item_ratings,
    assign_items_to_merchants,
    assign_prices,
    assign_user_regions,
    assign_verticals_to_roots,
    build_merchants,
    zipf_weights,
)

_WEIGHTS = {
    "food_delivery": 0.22,
    "restaurant": 0.18,
    "grocery": 0.25,
    "pharmacy": 0.10,
    "commerce": 0.15,
    "lifestyle": 0.10,
}


@pytest.fixture
def config() -> AugmentationConfig:
    return AugmentationConfig(
        seed=42,
        vertical_weights=dict(_WEIGHTS),
        region_count=12,
        region_zipf_s=1.1,
        merchants_per_vertical=20,
        merchant_rating_beta=(8.0, 2.0),
        delivery_radius_km=(2.0, 15.0),
        delivery_minutes=(15, 75),
        price_lognormal={
            "food_delivery": {"mu": 2.5, "sigma": 0.5},
            "restaurant": {"mu": 3.0, "sigma": 0.6},
            "grocery": {"mu": 1.6, "sigma": 0.7},
            "pharmacy": {"mu": 2.2, "sigma": 0.6},
            "commerce": {"mu": 3.4, "sigma": 0.9},
            "lifestyle": {"mu": 3.1, "sigma": 1.0},
        },
        min_price=0.99,
    )


# --- determinism -----------------------------------------------------------


def test_region_assignment_depends_only_on_id(config: AugmentationConfig) -> None:
    """A user's region must not change with batch composition or ordering."""
    ids = np.array([5, 10, 99, 1000, 77], dtype=np.int64)

    full = assign_user_regions(ids, config)
    shuffled = assign_user_regions(ids[::-1], config)
    subset = assign_user_regions(ids[2:], config)

    np.testing.assert_array_equal(full, shuffled[::-1])
    np.testing.assert_array_equal(full[2:], subset)


def test_prices_depend_only_on_id(config: AugmentationConfig) -> None:
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    verticals = np.full(5, int(Vertical.GROCERY), dtype=np.int8)

    first = assign_prices(ids, verticals, config)
    subset = assign_prices(ids[1:3], verticals[1:3], config)

    np.testing.assert_allclose(first[1:3], subset)


def test_different_salts_decorrelate_attributes(config: AugmentationConfig) -> None:
    """Region and device must not be driven by the same underlying draw.

    Sharing a salt would make every user in a region prefer the same device,
    inventing a correlation the data never contained.
    """
    ids = np.arange(5000, dtype=np.int64)
    from mercury_rec.data.augment import assign_user_devices

    regions = assign_user_regions(ids, config).astype(float)
    devices = assign_user_devices(ids, config).astype(float)

    correlation = float(np.corrcoef(regions, devices)[0, 1])
    assert abs(correlation) < 0.05, f"region/device correlation {correlation:.3f} is too high"


# --- distributions ---------------------------------------------------------


#: Root-category item shares measured from the real Retailrocket tree
#: (25 roots, largest 17.9%). Used instead of a synthetic power law because
#: the constraint that matters - whether any single root exceeds a vertical's
#: target - depends entirely on how skewed the real distribution actually is.
REAL_ROOT_SHARES = [
    0.1794,
    0.1325,
    0.0886,
    0.0863,
    0.0810,
    0.0764,
    0.0594,
    0.0533,
    0.0526,
    0.0399,
    0.0300,
    0.0240,
    0.0190,
    0.0150,
    0.0120,
    0.0095,
    0.0075,
    0.0060,
    0.0048,
    0.0038,
    0.0030,
    0.0024,
    0.0019,
    0.0015,
    0.0012,
]


def test_vertical_item_share_tracks_configured_weights(config: AugmentationConfig) -> None:
    """The realised ITEM share must match the configured weights.

    This is the regression guard for a real bug: with only ~25 root
    categories of very uneven size, drawing each root's vertical independently
    gave one vertical 0.0% of items and another 29%, against targets of
    25% and 15%.

    The fixture uses the measured real root distribution rather than a
    synthetic power law, because whether the target is achievable at all
    depends on how skewed that distribution is (see the test below).
    """
    root_ids = np.arange(len(REAL_ROOT_SHARES), dtype=np.int64)
    counts = (np.array(REAL_ROOT_SHARES) * 400_000).astype(np.int64)

    mapping = assign_verticals_to_roots(root_ids, counts, config)

    total = counts.sum()
    for name, target in _WEIGHTS.items():
        vertical = int(Vertical[name.upper()])
        got = counts[mapping["vertical"].to_numpy() == vertical].sum() / total
        assert abs(got - target) < 0.06, f"{name}: target {target:.0%}, realised {got:.0%}"


def test_dominant_root_degrades_predictably(config: AugmentationConfig) -> None:
    """Document the algorithm's real limitation rather than hiding it.

    Roots are indivisible: every item under a root gets that root's vertical.
    So if a single root holds more items than a vertical's target share, NO
    root-level assignment can hit that target - it is arithmetically
    impossible, not a tuning problem.

    The real tree's largest root holds ~18% of items, which already exceeds
    the smallest vertical target (10%). That is exactly why realised shares
    land a couple of points off target rather than exactly on it. This test
    pins the behaviour for the pathological case: the assignment must remain
    total and valid, just skewed.
    """
    root_ids = np.arange(5, dtype=np.int64)
    counts = np.array([700_000, 10_000, 10_000, 10_000, 10_000], dtype=np.int64)

    mapping = assign_verticals_to_roots(root_ids, counts, config)

    # Still a complete, valid assignment - no crash, no dropped roots.
    assert len(mapping) == 5
    assert mapping["vertical"].isin([int(v) for v in Vertical]).all()

    # And the dominant root lands in the vertical with the largest target,
    # which is the least-bad placement available.
    dominant_vertical = int(mapping.loc[mapping["root_category_id"] == 0, "vertical"].iloc[0])
    largest_target = max(_WEIGHTS, key=lambda k: _WEIGHTS[k])
    assert dominant_vertical == int(Vertical[largest_target.upper()])


def test_every_vertical_receives_items(config: AugmentationConfig) -> None:
    """No vertical may end up empty - an empty vertical breaks its features."""
    rng = np.random.default_rng(1)
    root_ids = np.arange(25, dtype=np.int64)
    counts = (rng.pareto(1.2, 25) * 500 + 10).astype(np.int64)

    mapping = assign_verticals_to_roots(root_ids, counts, config)
    assert mapping["vertical"].nunique() == len(Vertical)


def test_vertical_assignment_is_deterministic(config: AugmentationConfig) -> None:
    root_ids = np.arange(25, dtype=np.int64)
    counts = np.arange(25, dtype=np.int64) * 7 + 3
    first = assign_verticals_to_roots(root_ids, counts, config)
    second = assign_verticals_to_roots(root_ids, counts, config)
    np.testing.assert_array_equal(first["vertical"], second["vertical"])


def test_prices_match_the_configured_lognormal(config: AugmentationConfig) -> None:
    """Median price must land near exp(mu) for each vertical."""
    ids = np.arange(20_000, dtype=np.int64)
    for name, params in config.price_lognormal.items():
        vertical = int(Vertical[name.upper()])
        prices = assign_prices(ids, np.full(len(ids), vertical, dtype=np.int8), config)
        expected = float(np.exp(params["mu"]))
        median = float(np.median(prices))
        assert median == pytest.approx(expected, rel=0.06), (
            f"{name}: expected median ~{expected:.2f}, got {median:.2f}"
        )


def test_prices_are_positive_and_respect_the_floor(config: AugmentationConfig) -> None:
    ids = np.arange(5000, dtype=np.int64)
    prices = assign_prices(ids, np.full(5000, int(Vertical.GROCERY), dtype=np.int8), config)
    assert prices.min() >= config.min_price
    assert np.isfinite(prices).all()


def test_regions_follow_a_zipf_distribution(config: AugmentationConfig) -> None:
    """Demand must concentrate, or regional popularity is a useless signal."""
    regions = assign_user_regions(np.arange(50_000, dtype=np.int64), config)
    shares = np.bincount(regions, minlength=config.region_count) / len(regions)
    ordered = np.sort(shares)[::-1]

    assert ordered[0] > ordered[-1] * 5, "region populations are too uniform"
    assert np.all(np.diff(ordered) <= 1e-9), "shares should be monotonically decreasing"


def test_zipf_weights_sum_to_one() -> None:
    weights = zipf_weights(12, 1.1)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) < 0)


def test_item_ratings_are_within_range(config: AugmentationConfig) -> None:
    ratings = assign_item_ratings(np.arange(5000, dtype=np.int64), config)
    assert ratings.min() >= 0.0
    assert ratings.max() <= 5.0
    assert 3.5 < float(ratings.mean()) < 4.5, "Beta(7,2)*5 should centre near 3.9"


# --- merchants -------------------------------------------------------------


def test_merchants_are_generated_for_every_vertical(config: AugmentationConfig) -> None:
    merchants = build_merchants(config)
    assert len(merchants) == len(Vertical) * config.merchants_per_vertical
    assert merchants["vertical"].nunique() == len(Vertical)
    assert merchants["merchant_id"].is_unique


def test_merchant_ratings_stay_within_bounds(config: AugmentationConfig) -> None:
    merchants = build_merchants(config)
    assert merchants["rating"].between(0, 5).all()
    assert 3.5 < merchants["rating"].mean() < 4.5


def test_items_are_assigned_to_a_merchant_in_their_own_vertical(
    config: AugmentationConfig,
) -> None:
    """Cross-vertical assignment would make merchant diversity meaningless."""
    merchants = build_merchants(config)
    rng = np.random.default_rng(3)
    item_ids = np.arange(2000, dtype=np.int64)
    item_verticals = rng.integers(0, len(Vertical), 2000).astype(np.int8)

    assigned = assign_items_to_merchants(item_ids, item_verticals, merchants)
    lookup = merchants.set_index("merchant_id")["vertical"]

    np.testing.assert_array_equal(lookup.loc[assigned].to_numpy(), item_verticals)
