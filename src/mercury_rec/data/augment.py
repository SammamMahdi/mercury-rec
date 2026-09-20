"""Synthesise the platform attributes Retailrocket does not contain.

Retailrocket supplies real users, items, categories, timestamps, event types
and availability. It does **not** contain merchants, regions, geography,
delivery characteristics, usable prices, or the multi-vertical structure the
modelled platform is built around: its price property is hashed and its
category ids are opaque tokens.

This module generates those attributes. Three rules keep that honest:

1. **Nothing about behaviour is invented.** Every interaction, and its timing,
   comes from the real log. Only *entity attributes* are generated.
2. **Everything is a deterministic function of (real id, seed).** The same
   item receives the same merchant, vertical and price on every machine and
   every run, so results are reproducible from the config plus the raw
   download. This uses a per-entity hash rather than a sequential RNG draw, so
   an item's attributes do not depend on how many items preceded it.
3. **Provenance is a column, not a footnote.** Generated attributes are marked
   so the API and frontend can label them, and the data card lists them
   individually.

Why the distributions are the shapes they are is documented at each function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from mercury_rec.core.enums import Device, Vertical
from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

#: Namespace salts. Distinct salts mean an entity's vertical draw is
#: statistically independent of its region draw, rather than both being
#: driven by the same underlying id.
_SALT_VERTICAL: Final = 0x5EED_0001
_SALT_REGION: Final = 0x5EED_0002
_SALT_MERCHANT: Final = 0x5EED_0003
_SALT_PRICE: Final = 0x5EED_0004
_SALT_DEVICE: Final = 0x5EED_0005
_SALT_RATING: Final = 0x5EED_0006


def _hash_uniform(ids: np.ndarray, seed: int, salt: int) -> np.ndarray:
    """Map ids to reproducible uniforms in [0, 1).

    A hash rather than ``rng.random(len(ids))`` so a given id always yields the
    same value regardless of ordering, batching, or which subset is being
    processed. That is what makes ``--demo`` a genuine subset of ``full``
    rather than a differently-generated dataset.

    SplitMix64-style finalizer: cheap, vectorised, and well-distributed in the
    low bits, which matters because those bits drive the modulo below.
    """
    x = (ids.astype(np.uint64) + np.uint64(seed & 0xFFFF_FFFF) + np.uint64(salt)) & np.uint64(
        0xFFFF_FFFF_FFFF_FFFF
    )
    x ^= x >> np.uint64(30)
    x = (x * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    x ^= x >> np.uint64(27)
    x = (x * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFF_FFFF_FFFF_FFFF)
    x ^= x >> np.uint64(31)
    # 53 bits -> exactly representable as float64, uniform in [0, 1).
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _categorical_from_uniform(u: np.ndarray, weights: dict[str, float]) -> np.ndarray:
    """Assign categories by inverse-CDF sampling of a uniform."""
    labels = list(weights)
    probabilities = np.array([weights[label] for label in labels], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    edges = np.cumsum(probabilities)
    return np.searchsorted(edges, u, side="right").clip(0, len(labels) - 1)


def zipf_weights(n: int, s: float) -> np.ndarray:
    """Normalised Zipf weights over ``n`` ranks with exponent ``s``.

    Used for region population. Real marketplaces concentrate demand in a
    handful of metros; a uniform split would make regional popularity useless
    as a cold-start signal, which is precisely the behaviour being modelled.
    """
    ranks = np.arange(1, n + 1, dtype=np.float64)
    weights = 1.0 / np.power(ranks, s)
    return weights / weights.sum()


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Resolved augmentation parameters (see ``configs/data.yaml``)."""

    seed: int
    vertical_weights: dict[str, float]
    region_count: int
    region_zipf_s: float
    merchants_per_vertical: int
    merchant_rating_beta: tuple[float, float]
    delivery_radius_km: tuple[float, float]
    delivery_minutes: tuple[int, int]
    price_lognormal: dict[str, dict[str, float]]
    min_price: float


def assign_verticals_to_roots(
    root_ids: np.ndarray,
    root_item_counts: np.ndarray,
    config: AugmentationConfig,
) -> pd.DataFrame:
    """Assign each ROOT category to a vertical, balanced by item count.

    Assignment happens at the root so every descendant category inherits the
    same vertical. Assigning per-category instead would scatter sibling
    categories across verticals, making vertical features noisy and
    merchant-diversity constraints incoherent.

    The subtlety is that Retailrocket has only ~25 root categories, and their
    item counts are wildly uneven. Drawing each root's vertical independently
    from the configured weights therefore produces an *item* distribution
    nothing like those weights - in one run a vertical received 0.0% of items
    while another received 29%, because the weights govern how many roots are
    picked, not how many items they carry.

    So roots are assigned greedily instead: process them largest-first and
    give each to whichever vertical is furthest below its item-count target.
    This is the standard largest-first heuristic for balanced partitioning.
    It keeps whole subtrees intact while making the realised item
    distribution track the configured weights closely.

    Deterministic: the ordering is by (item count, root id), so the result
    depends only on the data and the config, never on iteration order.

    Limitation, stated plainly: roots are indivisible, so if a single root
    holds more items than a vertical's target share, that target cannot be
    met by any root-level assignment. On the real tree the largest root is
    ~18% of items, which is why realised shares land within a few points of
    target rather than exactly on it. A more skewed catalogue would deviate
    further, and the logged ``realised_item_share`` is what to check.

    The mapping asserts nothing about what a category contains - Retailrocket's
    category ids are opaque hashes. It imposes a *consistent* vertical
    structure, which is what the modelled platform requires.
    """
    names = list(config.vertical_weights)
    weights = np.array([config.vertical_weights[n] for n in names], dtype=np.float64)
    weights = weights / weights.sum()

    total_items = float(root_item_counts.sum())
    targets = weights * total_items
    assigned_items = np.zeros(len(names), dtype=np.float64)

    # Largest root first; ties broken by id so the result is reproducible.
    order = np.lexsort((root_ids, -root_item_counts))

    chosen = np.zeros(len(root_ids), dtype=np.int64)
    for position in order:
        deficit = targets - assigned_items
        pick = int(np.argmax(deficit))
        chosen[position] = pick
        assigned_items[pick] += float(root_item_counts[position])

    values = np.array([int(Vertical[names[i].upper()]) for i in chosen], dtype=np.int8)
    out = pd.DataFrame({"root_category_id": root_ids.astype("int32"), "vertical": values})

    realised = {
        names[i]: round(float(assigned_items[i] / total_items), 4) if total_items else 0.0
        for i in range(len(names))
    }
    logger.info(
        "augment.verticals",
        roots=len(out),
        target_share={names[i]: round(float(weights[i]), 4) for i in range(len(names))},
        realised_item_share=realised,
    )
    return out


def assign_user_regions(user_ids: np.ndarray, config: AugmentationConfig) -> np.ndarray:
    """Assign users to regions with a Zipf-distributed population."""
    u = _hash_uniform(user_ids, config.seed, _SALT_REGION)
    edges = np.cumsum(zipf_weights(config.region_count, config.region_zipf_s))
    return np.searchsorted(edges, u, side="right").clip(0, config.region_count - 1).astype(np.int8)


def assign_user_devices(user_ids: np.ndarray, config: AugmentationConfig) -> np.ndarray:
    """Assign a preferred device.

    Mobile-app dominant, matching on-demand commerce where the large majority
    of orders are placed in-app.
    """
    u = _hash_uniform(user_ids, config.seed, _SALT_DEVICE)
    weights = {"MOBILE_APP": 0.58, "MOBILE_WEB": 0.18, "DESKTOP_WEB": 0.17, "TABLET": 0.07}
    indices = _categorical_from_uniform(u, weights)
    names = list(weights)
    return np.array([int(Device[names[i]]) for i in indices], dtype=np.int8)


def build_merchants(config: AugmentationConfig) -> pd.DataFrame:
    """Generate the merchant catalogue: ``merchants_per_vertical`` per vertical.

    Merchants are generated from a seeded RNG rather than a per-id hash,
    because unlike users and items they have no pre-existing real id to hash —
    they are created here, and their ids are assigned sequentially.
    """
    rng = np.random.default_rng(config.seed + _SALT_MERCHANT)
    verticals = list(Vertical)
    total = len(verticals) * config.merchants_per_vertical

    vertical_column = np.repeat(
        np.array([int(v) for v in verticals], dtype=np.int8), config.merchants_per_vertical
    )

    # Merchants follow the same Zipf region distribution as users, so dense
    # regions have proportionally more supply. Uniform placement would make
    # delivery-distance reranking trivial in sparse regions.
    region_probabilities = zipf_weights(config.region_count, config.region_zipf_s)
    regions = rng.choice(config.region_count, size=total, p=region_probabilities).astype(np.int8)

    # Beta(8,2) scaled to [0,5]: mean ~4.0 with a left tail. Marketplace
    # ratings cluster high and are bounded above, which a Beta captures and a
    # Gaussian does not (it would produce impossible ratings above 5).
    alpha, beta = config.merchant_rating_beta
    ratings = (rng.beta(alpha, beta, size=total) * 5.0).round(2)

    radius_low, radius_high = config.delivery_radius_km
    minutes_low, minutes_high = config.delivery_minutes

    merchants = pd.DataFrame(
        {
            "merchant_id": np.arange(total, dtype="int32"),
            "vertical": vertical_column,
            "region_id": regions,
            "rating": ratings.astype("float32"),
            "delivery_radius_km": rng.uniform(radius_low, radius_high, total)
            .round(1)
            .astype("float32"),
            "avg_delivery_minutes": rng.integers(minutes_low, minutes_high, total).astype("int16"),
        }
    )
    logger.info(
        "augment.merchants", count=len(merchants), per_vertical=config.merchants_per_vertical
    )
    return merchants


def assign_items_to_merchants(
    item_ids: np.ndarray, item_verticals: np.ndarray, merchants: pd.DataFrame
) -> np.ndarray:
    """Assign each item to a merchant **within its own vertical**.

    Keeping the assignment vertical-consistent is what makes merchant
    diversity meaningful: "show fewer items from the same merchant" only makes
    sense if a merchant sells a coherent set of things.
    """
    assigned = np.zeros(len(item_ids), dtype=np.int32)
    for vertical in np.unique(item_verticals):
        candidates = merchants.loc[merchants["vertical"] == vertical, "merchant_id"].to_numpy()
        if candidates.size == 0:
            raise ValueError(f"No merchants generated for vertical {vertical}.")
        mask = item_verticals == vertical
        u = _hash_uniform(item_ids[mask], 0, _SALT_MERCHANT + int(vertical))
        assigned[mask] = candidates[(u * candidates.size).astype(np.int64) % candidates.size]
    return assigned


def assign_prices(
    item_ids: np.ndarray, item_verticals: np.ndarray, config: AugmentationConfig
) -> np.ndarray:
    """Draw a price per item from a per-vertical log-normal.

    Log-normal because prices are strictly positive, right-skewed and
    multiplicative in nature — a normal would generate negative prices and
    understate the long upper tail.

    The parameters are plausible magnitudes per vertical, NOT recovered from
    data: Retailrocket's price property is hashed and unrecoverable. This is
    stated explicitly in the data card, and no claim is made anywhere that
    these prices are real.
    """
    prices = np.zeros(len(item_ids), dtype=np.float64)

    for name, params in config.price_lognormal.items():
        vertical = int(Vertical[name.upper()])
        mask = item_verticals == vertical
        if not mask.any():
            continue
        # Inverse-CDF of the log-normal applied to the per-id uniform, so the
        # price stays a deterministic function of the item id.
        u = _hash_uniform(item_ids[mask], config.seed, _SALT_PRICE + vertical)
        u = np.clip(u, 1e-9, 1 - 1e-9)
        from scipy.special import ndtri

        prices[mask] = np.exp(params["mu"] + params["sigma"] * ndtri(u))

    return np.maximum(prices, config.min_price).round(2).astype(np.float32)


def assign_item_ratings(item_ids: np.ndarray, config: AugmentationConfig) -> np.ndarray:
    """Item ratings, Beta(7,2) scaled to [0,5] — same rationale as merchants."""
    u = np.clip(_hash_uniform(item_ids, config.seed, _SALT_RATING), 1e-9, 1 - 1e-9)
    from scipy.stats import beta as beta_dist

    ratings: np.ndarray = (beta_dist.ppf(u, 7.0, 2.0) * 5.0).round(2).astype(np.float32)
    return ratings


__all__ = [
    "AugmentationConfig",
    "assign_item_ratings",
    "assign_items_to_merchants",
    "assign_prices",
    "assign_user_devices",
    "assign_user_regions",
    "assign_verticals_to_roots",
    "build_merchants",
    "zipf_weights",
]
