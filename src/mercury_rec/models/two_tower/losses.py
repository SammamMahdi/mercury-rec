"""Sampled-softmax loss with in-batch negatives and a logQ correction.

The training signal for two-tower retrieval is a softmax over one positive
item and a set of negatives. Drawing those negatives uniformly from a
36,044-item catalogue is cheap but nearly useless: a randomly chosen item is
almost always so obviously irrelevant that separating it from the positive
teaches the model very little.

**In-batch negatives** use the other positives in the same batch instead. They
are free (already embedded), and they are sampled *proportionally to
popularity*, which makes them genuinely hard - the model must distinguish a
user's item from other items real users chose at the same moment.

That sampling is also the catch, and it is the part most often got wrong.

The logQ correction
-------------------
Because a batch-mate is drawn with probability proportional to its frequency,
popular items appear as negatives far more often than uniform sampling would
produce. Optimising the naive in-batch softmax therefore does not estimate
``P(item | user)``; it estimates something systematically biased against
popular items, and the model learns to suppress exactly the items most users
want.

The standard fix (Yi et al., 2019, *Sampled Softmax with Random Sampled
Logits*) subtracts the log sampling probability from each logit::

    corrected_logit(u, i) = score(u, i) / T  -  log Q(i)

which recovers an unbiased estimate of the full-softmax gradient. Without it a
two-tower model trained on skewed data reliably under-serves head items, and
the failure is invisible in the loss curve - it looks like healthy training.

False negatives
---------------
Two separate cases, both of which corrupt the gradient by pushing down an item
the user actually likes:

1. The same item appears more than once in a batch (common under a power-law
   catalogue), so a user's own positive appears as another row's negative.
2. A batch-mate's item is one the user genuinely interacted with.

Both are masked to ``-inf`` before the softmax. Case 1 is always masked. Case
2 requires the caller to pass the interaction lookup, and is skipped when it
is not available.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

#: Additive mask for positions excluded from the softmax. A large negative
#: constant rather than ``-inf``: under bf16 autocast, ``-inf`` in a row that
#: ends up fully masked produces NaN, which then propagates into every
#: parameter and silently destroys the run.
_MASK_VALUE: Final = -1e9


def in_batch_softmax_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    *,
    item_ids: torch.Tensor,
    temperature: float = 0.05,
    log_frequency: torch.Tensor | None = None,
    positive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over in-batch negatives, with an optional logQ correction.

    Args:
        user_embeddings: ``(batch, dim)``, L2-normalised.
        item_embeddings: ``(batch, dim)``, L2-normalised. Row *i* is the
            positive item for user *i*.
        item_ids: ``(batch,)`` item ids, used to find duplicates.
        temperature: Softmax temperature. With normalised embeddings the raw
            logits lie in [-1, 1], which is far too flat without scaling.
        log_frequency: ``(batch,)`` log P(item) over the training set. When
            given, applies the logQ correction.
        positive_mask: ``(batch, batch)`` boolean, True where column *j*'s
            item is a genuine positive for row *i*'s user. Those entries are
            masked out of the negatives.

    Returns:
        Scalar mean cross-entropy.
    """
    batch_size = user_embeddings.shape[0]
    if batch_size == 0:
        return user_embeddings.sum() * 0.0

    # In-batch negatives require row i of each matrix to be a matched pair.
    # Without this check a mismatch does not raise: the (n_items, n_items)
    # duplicate mask broadcasts the (n_users, n_items) logits up to a square,
    # and the failure surfaces much later as a confusing size error inside
    # cross_entropy - or, worse, as a silently wrong loss.
    if item_embeddings.shape[0] != batch_size or item_ids.shape[0] != batch_size:
        raise ValueError(
            "in_batch_softmax_loss expects one positive item per user: got "
            f"{batch_size} users, {item_embeddings.shape[0]} item embeddings, "
            f"{item_ids.shape[0]} item ids."
        )

    # (batch, batch): every user scored against every item in the batch.
    logits = (user_embeddings @ item_embeddings.T) / temperature

    if log_frequency is not None:
        # Broadcast over columns: the correction is a property of the
        # candidate item, not of the user scoring it.
        logits = logits - log_frequency.unsqueeze(0)

    targets = torch.arange(batch_size, device=logits.device)

    # Case 1: the same item id appearing twice in the batch. The diagonal is
    # restored afterwards because it is the target, not a negative.
    duplicates = item_ids.unsqueeze(0) == item_ids.unsqueeze(1)
    duplicates.fill_diagonal_(False)

    # Case 2: a batch-mate's item that this user genuinely interacted with.
    if positive_mask is not None:
        extra = positive_mask.clone()
        extra.fill_diagonal_(False)
        duplicates = duplicates | extra

    logits = logits.masked_fill(duplicates, _MASK_VALUE)

    return nn.functional.cross_entropy(logits, targets)


def compute_log_frequency(item_counts: torch.Tensor, *, smoothing: float = 1.0) -> torch.Tensor:
    """Log sampling probability per item, for the logQ correction.

    Under in-batch sampling an item's probability of appearing is proportional
    to its share of training interactions, so that empirical frequency is the
    estimate of ``Q``.

    Add-one smoothing keeps items with zero training interactions from
    producing ``log(0) = -inf``, which would make their corrected logit
    ``+inf`` and let a never-seen item win every softmax.
    """
    smoothed = item_counts.to(torch.float64) + smoothing
    probabilities = smoothed / smoothed.sum()
    return torch.log(probabilities).to(torch.float32)


def encode_pairs(user_ids: torch.Tensor, item_ids: torch.Tensor, n_items: int) -> torch.Tensor:
    """Pack (user, item) into one int64 key. Exact, because ids are dense."""
    return user_ids.to(torch.int64) * n_items + item_ids.to(torch.int64)


def build_positive_mask(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    known_pairs_sorted: torch.Tensor,
    n_items: int,
) -> torch.Tensor:
    """Mark (row user, column item) cells that are real interactions.

    Fully vectorised, and on whichever device the batch lives on.

    The obvious implementation -- a Python double loop over the batch doing a
    set lookup -- is O(batch^2) *in Python*. At a batch size of 4,096 that is
    16.7 million interpreter iterations per step, which measured at ~29 s per
    epoch on 92k rows and would have made full-dataset training take hours.

    Instead both sides are packed into a single int64 key and located in the
    sorted array of known pairs by binary search: O(batch^2 log n) but entirely
    inside torch, and on the GPU when the batch is there.

    Args:
        user_ids: ``(batch,)``.
        item_ids: ``(batch,)``.
        known_pairs_sorted: Ascending int64 keys of every training pair, from
            :func:`sorted_interaction_pairs`.
        n_items: Catalogue size, the packing stride.

    Returns:
        ``(batch, batch)`` boolean mask.
    """
    pairs = encode_pairs(user_ids.unsqueeze(1), item_ids.unsqueeze(0), n_items)
    position = torch.searchsorted(known_pairs_sorted, pairs)
    position = position.clamp(max=known_pairs_sorted.numel() - 1)
    return known_pairs_sorted[position] == pairs


def sorted_interaction_pairs(
    user_ids: torch.Tensor, item_ids: torch.Tensor, n_items: int
) -> torch.Tensor:
    """Sorted, de-duplicated int64 keys of every observed pair.

    Built once per fit; ``build_positive_mask`` binary-searches it per batch.
    """
    unique: torch.Tensor = torch.unique(encode_pairs(user_ids, item_ids, n_items))
    return unique


__all__ = [
    "build_positive_mask",
    "compute_log_frequency",
    "encode_pairs",
    "in_batch_softmax_loss",
    "sorted_interaction_pairs",
]
