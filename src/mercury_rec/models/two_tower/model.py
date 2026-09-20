"""Two-tower retrieval architecture.

A user tower and an item tower each map their side into a shared embedding
space, and relevance is the dot product::

    score(u, i) = f_user(u) . f_item(i)

The value of this factorisation is entirely operational. Because the item
tower does not depend on the user, every item embedding can be computed once,
offline, and loaded into an ANN index. Serving then costs one user-tower
forward pass plus an approximate nearest-neighbour lookup, independent of
catalogue size. A model that scored (user, item) *jointly* would be more
expressive and completely unservable at retrieval scale: it would need 36,044
forward passes per request.

That constraint is what the architecture is for, and it also explains why
richer cross-features are deliberately absent here: they belong in the ranking
stage, which only ever sees a few hundred candidates.

Feature handling
----------------
Both towers mix a learned id embedding with categorical side features and
continuous aggregates. The id embedding carries collaborative signal; the side
features carry content signal, which is what lets the item tower produce a
usable embedding for an item with almost no interaction history -- the
cold-start path.

Continuous inputs are passed through a ``LayerNorm`` rather than standardised
against training statistics. This avoids shipping a separate scaler artifact
that could drift out of sync with the model, which is a real class of
serving bug.

Output embeddings are **L2-normalised**, so the dot product is a cosine
similarity bounded in [-1, 1]. Two reasons: it stops the model inflating
scores by growing embedding norms instead of learning direction, and it lets
FAISS's inner-product index return exact cosine neighbours with no extra work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class TowerSpec:
    """Shape of one tower's inputs.

    Args:
        n_ids: Cardinality of the primary id embedding.
        categorical_cardinalities: Cardinality per categorical side feature.
        n_continuous: Number of continuous inputs.
        id_dim: Width of the id embedding.
        categorical_dim: Width of each categorical embedding.
        hidden: MLP hidden widths.
        output_dim: Final embedding width. Must match the other tower.
        dropout: Dropout applied between hidden layers.
    """

    n_ids: int
    categorical_cardinalities: tuple[int, ...] = ()
    n_continuous: int = 0
    id_dim: int = 64
    categorical_dim: int = 16
    hidden: tuple[int, ...] = (128, 64)
    output_dim: int = 64
    dropout: float = 0.1

    @property
    def input_width(self) -> int:
        return (
            self.id_dim
            + self.categorical_dim * len(self.categorical_cardinalities)
            + self.n_continuous
        )


class Tower(nn.Module):
    """One side of the two-tower model."""

    def __init__(self, spec: TowerSpec) -> None:
        super().__init__()
        self.spec = spec

        self.id_embedding = nn.Embedding(spec.n_ids, spec.id_dim)
        nn.init.normal_(self.id_embedding.weight, std=0.05)

        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality + 1, spec.categorical_dim)
                # +1 for an explicit unknown bucket at index 0. A missing
                # category must map somewhere deliberate; folding it into a
                # real category would teach the model a relationship that does
                # not exist.
                for cardinality in spec.categorical_cardinalities
            ]
        )
        for embedding in self.categorical_embeddings:
            # ModuleList yields Module; these are all Embedding by construction.
            nn.init.normal_(cast(nn.Embedding, embedding).weight, std=0.05)

        self.continuous_norm = nn.LayerNorm(spec.n_continuous) if spec.n_continuous > 0 else None

        layers: list[nn.Module] = []
        width = spec.input_width
        for hidden_width in spec.hidden:
            layers.extend(
                [
                    nn.Linear(width, hidden_width),
                    nn.LayerNorm(hidden_width),
                    nn.GELU(),
                    nn.Dropout(spec.dropout),
                ]
            )
            width = hidden_width
        layers.append(nn.Linear(width, spec.output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        ids: torch.Tensor,
        categorical: torch.Tensor | None = None,
        continuous: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map inputs to an L2-normalised embedding.

        Args:
            ids: ``(batch,)`` primary ids.
            categorical: ``(batch, n_categorical)``, already offset by 1 so
                index 0 means unknown.
            continuous: ``(batch, n_continuous)``.
        """
        parts = [self.id_embedding(ids)]

        if self.categorical_embeddings:
            if categorical is None:
                raise ValueError("Tower expects categorical inputs but none were given.")
            parts.extend(
                embedding(categorical[:, position])
                for position, embedding in enumerate(self.categorical_embeddings)
            )

        if self.continuous_norm is not None:
            if continuous is None:
                raise ValueError("Tower expects continuous inputs but none were given.")
            # NaN means "not observed" throughout the feature layer; zero after
            # normalisation is the neutral value, so an absent feature neither
            # pushes the embedding nor poisons it with NaN.
            cleaned = torch.nan_to_num(continuous, nan=0.0, posinf=0.0, neginf=0.0)
            parts.append(self.continuous_norm(cleaned))

        embedded = self.mlp(torch.cat(parts, dim=-1))
        return nn.functional.normalize(embedded, p=2.0, dim=-1)


@dataclass(slots=True)
class TwoTowerConfig:
    """Training and architecture configuration."""

    embedding_dim: int = 64
    user_hidden: tuple[int, ...] = (128, 64)
    item_hidden: tuple[int, ...] = (128, 64)
    dropout: float = 0.1
    temperature: float = 0.05
    """Softmax temperature. Lower sharpens the distribution.

    0.05 is the value used across the contrastive-retrieval literature; it
    matters because with L2-normalised embeddings logits are confined to
    [-1, 1], and an unscaled softmax over that range is far too flat to
    produce a useful gradient.
    """
    logq_correction: bool = True
    epochs: int = 25
    """Selected on the validation split, not by watching the loss.

    Training to 60 epochs lowers the loss from 1.60 to 1.10 while validation
    Recall@10 falls from 0.0404 to 0.0357. The extra capacity goes into
    memorising training interactions, and the loss curve alone would have
    picked the worse model.
    """
    batch_size: int = 4096
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 42


class TwoTowerModel(nn.Module):
    """User and item towers projecting into one shared space."""

    def __init__(self, user_spec: TowerSpec, item_spec: TowerSpec) -> None:
        super().__init__()
        if user_spec.output_dim != item_spec.output_dim:
            raise ValueError(
                f"Tower output dimensions must match to share a space: "
                f"user={user_spec.output_dim}, item={item_spec.output_dim}."
            )
        self.user_tower = Tower(user_spec)
        self.item_tower = Tower(item_spec)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_categorical: torch.Tensor | None = None,
        user_continuous: torch.Tensor | None = None,
        item_categorical: torch.Tensor | None = None,
        item_continuous: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.user_tower(user_ids, user_categorical, user_continuous),
            self.item_tower(item_ids, item_categorical, item_continuous),
        )


@dataclass(slots=True)
class BatchTensors:
    """One training batch, already on the target device."""

    user_ids: torch.Tensor
    item_ids: torch.Tensor
    user_categorical: torch.Tensor
    user_continuous: torch.Tensor
    item_categorical: torch.Tensor
    item_continuous: torch.Tensor
    item_log_frequency: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    """log P(item) over the training set, for the logQ correction."""


__all__ = ["BatchTensors", "Tower", "TowerSpec", "TwoTowerConfig", "TwoTowerModel"]
