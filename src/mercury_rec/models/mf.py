"""Bayesian Personalised Ranking matrix factorisation, in PyTorch.

BPR (Rendle et al., 2009) optimises a *ranking* objective rather than a
reconstruction one, which is the right choice for implicit feedback. The
alternative -- regressing observed interactions toward 1 and everything else
toward 0 -- treats "not yet seen" as "disliked", which for a 36,044-item
catalogue a user has touched 8 times is almost entirely wrong.

The objective maximises, over sampled triples (user, positive, negative)::

    log sigmoid( score(u, i+) - score(u, i-) )

so the model only ever learns that an observed item should outrank an
unobserved one. It never asserts an absolute preference the data cannot
support.

Implementation notes specific to this machine:

- **No DataLoader.** On Windows every worker is a fresh ``spawn`` that
  re-imports the module and re-pickles the dataset, and a dataset holding a
  CUDA tensor either fails to pickle or silently hangs. The training set here
  is three int32 arrays that fit comfortably in VRAM, so batching is a slice
  of a pre-shuffled permutation -- faster than a DataLoader at this scale and
  with none of the failure modes.
- **bf16 autocast, never fp16.** Blackwell supports bf16 natively, and bf16
  carries fp32's exponent range, so there is no ``GradScaler`` and no
  loss-scaling or inf/nan babysitting.
- **Rejection-sampled negatives.** A uniformly drawn "negative" that the user
  actually interacted with is a false negative, and the gradient then actively
  pushes a true preference down. Resampling collisions costs little because
  the interaction matrix is ~0.03% dense.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from mercury_rec.core.logging import get_logger
from mercury_rec.models.base import RecommendationContext, Recommender

logger = get_logger(__name__)


def resolve_device(requested: str | None = None) -> torch.device:
    """Pick a compute device, preferring CUDA when it is actually usable."""
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class _BPRModel(nn.Module):
    """User and item embedding tables plus per-item biases."""

    def __init__(self, n_users: int, n_items: int, dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim)
        # Item bias captures raw popularity, freeing the latent dimensions to
        # model taste rather than spending capacity re-learning that some
        # items are simply more popular than others.
        self.item_bias = nn.Embedding(n_items, 1)

        # Small-variance init: large initial dot products saturate the sigmoid
        # in the BPR loss and the gradient vanishes before training starts.
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_embedding(users)
        item_vectors = self.item_embedding(items)
        interaction = (user_vectors * item_vectors).sum(dim=-1)
        scores: torch.Tensor = interaction + self.item_bias(items).squeeze(-1)
        return scores


class BPRRecommender(Recommender):
    """Matrix factorisation trained with the BPR pairwise ranking loss.

    Args:
        n_users: Catalogue user count.
        n_items: Catalogue item count.
        dim: Latent dimensionality.
        epochs: Passes over the interaction set.
        batch_size: Triples per step.
        learning_rate: AdamW learning rate.
        weight_decay: L2 regularisation. Essential here: with 65k users and
            8 interactions each, unregularised embeddings memorise.
        n_negatives: Negatives sampled per positive.
        seed: Seeded for reproducibility.
        device: Override device selection.
    """

    name = "bpr_mf"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        dim: int = 64,
        epochs: int = 20,
        batch_size: int = 8192,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        n_negatives: int = 1,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(n_users, n_items)
        self.dim = dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_negatives = n_negatives
        self.seed = seed
        self.device = resolve_device(device)

        self._model: _BPRModel | None = None
        self._user_items: dict[int, set[int]] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None
        self._item_bias: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

    def params(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_negatives": self.n_negatives,
            "seed": self.seed,
            "device": str(self.device),
        }

    def _sample_negatives(
        self, users: torch.Tensor, positives: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        """Draw negatives, rejecting any the user actually interacted with.

        A false negative is actively harmful: the gradient pushes down an item
        the user genuinely liked. Two rejection rounds clear virtually all
        collisions at this density (~0.03%); a full check every round would
        cost more than the residual error is worth.
        """
        negatives = torch.randint(
            0, self.n_items, positives.shape, device=self.device, generator=generator
        )
        for _ in range(2):
            collision = negatives == positives
            if not bool(collision.any()):
                break
            negatives = torch.where(
                collision,
                torch.randint(
                    0, self.n_items, positives.shape, device=self.device, generator=generator
                ),
                negatives,
            )
        return negatives

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        users = torch.as_tensor(
            interactions["user_id"].to_numpy(dtype=np.int64), device=self.device
        )
        items = torch.as_tensor(
            interactions["item_id"].to_numpy(dtype=np.int64), device=self.device
        )
        n_samples = users.numel()

        grouped = interactions.groupby("user_id")["item_id"].apply(set)
        self._user_items = {int(cast("int", key)): set(value) for key, value in grouped.items()}

        model = _BPRModel(self.n_users, self.n_items, self.dim).to(self.device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)

        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        model.train()

        for epoch in range(1, self.epochs + 1):
            permutation = torch.randperm(n_samples, device=self.device, generator=generator)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                index = permutation[start : start + self.batch_size]
                batch_users = users[index]
                batch_positive = items[index]

                if self.n_negatives > 1:
                    batch_users = batch_users.repeat(self.n_negatives)
                    batch_positive = batch_positive.repeat(self.n_negatives)
                batch_negative = self._sample_negatives(batch_users, batch_positive, generator)

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    positive_score = model(batch_users, batch_positive)
                    negative_score = model(batch_users, batch_negative)
                    # softplus(-(x)) == -log(sigmoid(x)), computed stably.
                    loss = nn.functional.softplus(-(positive_score - negative_score)).mean()

                optimiser.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimiser.step()

                epoch_loss += float(loss.detach())
                n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            self.history.append({"epoch": epoch, "loss": mean_loss})
            if epoch == 1 or epoch % 5 == 0 or epoch == self.epochs:
                logger.info("model.bpr.epoch", epoch=epoch, loss=round(mean_loss, 5))

        model.eval()
        self._model = model
        # Cache factors as numpy: scoring then needs no GPU round trip, which
        # matters because the API process may have no GPU at all.
        with torch.no_grad():
            self._user_factors = model.user_embedding.weight.detach().cpu().numpy()
            self._item_factors = model.item_embedding.weight.detach().cpu().numpy()
            self._item_bias = model.item_bias.weight.detach().cpu().numpy().ravel()

        return {
            "final_loss": round(self.history[-1]["loss"], 5),
            "first_loss": round(self.history[0]["loss"], 5),
            "bf16_autocast": use_amp,
        }

    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        assert self._user_factors is not None
        assert self._item_factors is not None
        assert self._item_bias is not None
        user_vector = self._user_factors[user_id]
        scores: np.ndarray = (
            self._item_factors[candidates] @ user_vector + self._item_bias[candidates]
        ).astype(np.float32)
        return scores

    @property
    def item_factors(self) -> np.ndarray:
        """Item embeddings, reused by the ANN index and the 3D projection."""
        self._require_fitted()
        assert self._item_factors is not None
        return self._item_factors

    def save(self, path: Path) -> None:
        self._require_fitted()
        assert self._model is not None
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "params": self.params(),
                "n_users": self.n_users,
                "n_items": self.n_items,
                "history": self.history,
            },
            path,
        )
        logger.info("model.bpr.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> BPRRecommender:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = payload["params"]
        model = cls(
            payload["n_users"],
            payload["n_items"],
            dim=params["dim"],
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            weight_decay=params["weight_decay"],
            n_negatives=params["n_negatives"],
            seed=params["seed"],
            device="cpu",
        )
        inner = _BPRModel(payload["n_users"], payload["n_items"], params["dim"])
        inner.load_state_dict(payload["state_dict"])
        inner.eval()
        model._model = inner
        model._user_factors = inner.user_embedding.weight.detach().numpy()
        model._item_factors = inner.item_embedding.weight.detach().numpy()
        model._item_bias = inner.item_bias.weight.detach().numpy().ravel()
        model.history = payload.get("history", [])
        model._fitted = True
        return model


__all__ = ["BPRRecommender", "resolve_device"]
