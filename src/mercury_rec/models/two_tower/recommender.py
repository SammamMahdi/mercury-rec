"""Two-tower retrieval model: training, encoding and scoring.

Training examples are the **as-of feature rows** produced by
``mercury features build``, not raw interactions. That matters: each row
carries the user and item aggregates as they stood immediately before that
event, so the model is trained on exactly the inputs serving will hand it, and
no example can see its own outcome. Training on end-of-window aggregates
instead -- the convenient shortcut -- would let a January example carry
knowledge of September, which is precisely the leakage the feature layer
exists to prevent.

The item tower is evaluated once after training to produce the embedding
matrix that backs the ANN index. Item embeddings are computed against the
final training-window state, which is the state serving starts from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from mercury_rec.core.logging import get_logger
from mercury_rec.models.base import RecommendationContext, Recommender
from mercury_rec.models.mf import resolve_device
from mercury_rec.models.two_tower.losses import (
    build_positive_mask,
    compute_log_frequency,
    in_batch_softmax_loss,
    sorted_interaction_pairs,
)
from mercury_rec.models.two_tower.model import TowerSpec, TwoTowerConfig, TwoTowerModel

logger = get_logger(__name__)

#: User-side as-of features consumed by the user tower.
USER_CONTINUOUS: tuple[str, ...] = (
    "user_event_count",
    "user_purchase_count",
    "user_conversion_rate",
    "user_distinct_items",
    "user_days_since_last",
    "user_tenure_days",
    "user_avg_interaction_price",
    "user_avg_order_value",
)

#: Item-side as-of features consumed by the item tower.
ITEM_CONTINUOUS: tuple[str, ...] = (
    "item_event_count",
    "item_purchase_count",
    "item_conversion_rate",
    "item_distinct_users",
    "item_days_since_last",
    "item_age_days",
)

#: Count-like columns are log1p-compressed before the tower sees them. These
#: are power-law distributed over four orders of magnitude, and LayerNorm on
#: the raw scale is dominated by a handful of head items.
_LOG_COMPRESS: frozenset[str] = frozenset(
    {
        "user_event_count",
        "user_purchase_count",
        "user_distinct_items",
        "item_event_count",
        "item_purchase_count",
        "item_distinct_users",
    }
)


def _prepare_continuous(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Extract and compress continuous features into a float32 matrix."""
    out = np.empty((len(frame), len(columns)), dtype=np.float32)
    for position, name in enumerate(columns):
        values = frame[name].to_numpy(dtype=np.float32)
        if name in _LOG_COMPRESS:
            values = np.log1p(np.nan_to_num(values, nan=0.0))
        out[:, position] = values
    return out


class TwoTowerRecommender(Recommender):
    """Neural retrieval with separate user and item towers.

    Args:
        n_users: Catalogue user count.
        n_items: Catalogue item count.
        items: Item dimension table, supplying the item tower's categorical
            side features and price.
        users: User dimension table, supplying region and device.
        config: Architecture and training configuration.
        device: Override device selection.
    """

    name = "two_tower"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        items: pd.DataFrame,
        users: pd.DataFrame,
        config: TwoTowerConfig | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(n_users, n_items)
        self.config = config or TwoTowerConfig()
        self.device = resolve_device(device)

        self._items = items.sort_values("item_id").reset_index(drop=True)
        self._users = users.sort_values("user_id").reset_index(drop=True)

        self._model: TwoTowerModel | None = None
        self._item_embeddings: np.ndarray | None = None
        self._user_embeddings: np.ndarray | None = None
        self.history: list[dict[str, float]] = []

        self._item_categorical = self._build_item_categorical()
        self._user_categorical = self._build_user_categorical()

    # --- feature assembly -------------------------------------------------

    def _build_item_categorical(self) -> np.ndarray:
        """Per-item categorical codes, densified and offset by 1.

        Index 0 is reserved for "unknown" in every column, so a missing
        category lands somewhere deliberate rather than being folded into a
        real one.
        """
        frame = self._items
        columns = []
        for name in ("category_id", "merchant_id", "vertical"):
            codes = pd.factorize(frame[name].fillna(-1))[0] + 1
            columns.append(codes.astype(np.int64))

        # Price band rather than raw price: the tower already receives price
        # as a continuous input, and a coarse band lets it learn a
        # non-monotonic preference (budget vs premium) that a single scalar
        # cannot express.
        prices = frame["price"].to_numpy(dtype=np.float64)
        bands = np.digitize(prices, np.quantile(prices[prices > 0], [0.2, 0.4, 0.6, 0.8])) + 1
        columns.append(bands.astype(np.int64))

        return np.stack(columns, axis=1)

    def _build_user_categorical(self) -> np.ndarray:
        frame = self._users
        columns = [
            (pd.factorize(frame["region_id"].fillna(-1))[0] + 1).astype(np.int64),
            (pd.factorize(frame["device_pref"].fillna(-1))[0] + 1).astype(np.int64),
        ]
        return np.stack(columns, axis=1)

    def _item_static_continuous(self) -> np.ndarray:
        """Log price, the only item continuous not derived from as-of state."""
        prices = self._items["price"].to_numpy(dtype=np.float32)
        return np.log1p(prices).reshape(-1, 1)

    def params(self) -> dict[str, Any]:
        config = self.config
        return {
            "embedding_dim": config.embedding_dim,
            "temperature": config.temperature,
            "logq_correction": config.logq_correction,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "seed": config.seed,
            "device": str(self.device),
        }

    # --- training ---------------------------------------------------------

    def fit_features(self, features: pd.DataFrame) -> Any:
        """Train from an as-of feature frame (the supported entry point)."""
        return self.fit(features)

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        missing = [
            name
            for name in (*USER_CONTINUOUS, *ITEM_CONTINUOUS)
            if name not in interactions.columns
        ]
        if missing:
            raise ValueError(
                "TwoTowerRecommender trains on as-of feature rows, not raw "
                f"interactions. Missing columns: {missing}. "
                "Run `mercury features build` and pass features_train.parquet."
            )

        config = self.config
        torch.manual_seed(config.seed)

        user_ids = torch.as_tensor(
            interactions["user_id"].to_numpy(dtype=np.int64), device=self.device
        )
        item_ids = torch.as_tensor(
            interactions["item_id"].to_numpy(dtype=np.int64), device=self.device
        )
        user_continuous = torch.as_tensor(
            _prepare_continuous(interactions, USER_CONTINUOUS), device=self.device
        )
        item_asof = _prepare_continuous(interactions, ITEM_CONTINUOUS)
        item_price = np.log1p(self._items["price"].to_numpy(dtype=np.float32))[
            interactions["item_id"].to_numpy(dtype=np.int64)
        ].reshape(-1, 1)
        item_continuous = torch.as_tensor(np.hstack([item_asof, item_price]), device=self.device)

        user_categorical = torch.as_tensor(self._user_categorical, device=self.device)
        item_categorical = torch.as_tensor(self._item_categorical, device=self.device)

        user_spec = TowerSpec(
            n_ids=self.n_users,
            categorical_cardinalities=tuple(
                int(self._user_categorical[:, i].max()) + 1
                for i in range(self._user_categorical.shape[1])
            ),
            n_continuous=len(USER_CONTINUOUS),
            id_dim=config.embedding_dim,
            hidden=config.user_hidden,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )
        item_spec = TowerSpec(
            n_ids=self.n_items,
            categorical_cardinalities=tuple(
                int(self._item_categorical[:, i].max()) + 1
                for i in range(self._item_categorical.shape[1])
            ),
            n_continuous=len(ITEM_CONTINUOUS) + 1,
            id_dim=config.embedding_dim,
            hidden=config.item_hidden,
            output_dim=config.embedding_dim,
            dropout=config.dropout,
        )

        model = TwoTowerModel(user_spec, item_spec).to(self.device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        counts = np.bincount(
            interactions["item_id"].to_numpy(dtype=np.int64), minlength=self.n_items
        )
        log_frequency = compute_log_frequency(torch.as_tensor(counts)).to(self.device)

        # Sorted pair keys, searched per batch on the GPU. A Python-level
        # set lookup here is O(batch^2) in the interpreter and dominated the
        # entire training loop.
        known_pairs = sorted_interaction_pairs(user_ids, item_ids, self.n_items)

        generator = torch.Generator(device=self.device)
        generator.manual_seed(config.seed)
        n_samples = int(user_ids.numel())
        use_amp = self.device.type == "cuda" and torch.cuda.is_bf16_supported()

        model.train()
        for epoch in range(1, config.epochs + 1):
            permutation = torch.randperm(n_samples, device=self.device, generator=generator)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, config.batch_size):
                index = permutation[start : start + config.batch_size]
                if index.numel() < 2:
                    # In-batch negatives need at least one other row.
                    continue

                batch_users = user_ids[index]
                batch_items = item_ids[index]

                positive_mask = build_positive_mask(
                    batch_users, batch_items, known_pairs, self.n_items
                )

                with torch.autocast(
                    device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    user_vectors, item_vectors = model(
                        batch_users,
                        batch_items,
                        user_categorical=user_categorical[batch_users],
                        user_continuous=user_continuous[index],
                        item_categorical=item_categorical[batch_items],
                        item_continuous=item_continuous[index],
                    )
                    loss = in_batch_softmax_loss(
                        user_vectors.float(),
                        item_vectors.float(),
                        item_ids=batch_items,
                        temperature=config.temperature,
                        log_frequency=(
                            log_frequency[batch_items] if config.logq_correction else None
                        ),
                        positive_mask=positive_mask,
                    )

                optimiser.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()

                epoch_loss += float(loss.detach())
                n_batches += 1

            mean_loss = epoch_loss / max(n_batches, 1)
            self.history.append({"epoch": epoch, "loss": mean_loss})
            if epoch == 1 or epoch % 5 == 0 or epoch == config.epochs:
                logger.info("model.two_tower.epoch", epoch=epoch, loss=round(mean_loss, 5))

        model.eval()
        self._model = model
        self._encode_items(interactions)

        return {
            "first_loss": round(self.history[0]["loss"], 5),
            "final_loss": round(self.history[-1]["loss"], 5),
            "bf16_autocast": use_amp,
            "embedding_dim": config.embedding_dim,
        }

    @torch.no_grad()
    def _encode_items(self, interactions: pd.DataFrame) -> None:
        """Embed the whole catalogue once, for the ANN index.

        Item aggregates are taken as of the END of the training window, which
        is the state a freshly-deployed service starts from. The tower is
        evaluated in eval mode so dropout is off - leaving it on would make
        the index non-deterministic.
        """
        assert self._model is not None

        latest = (
            interactions.sort_values("ts")
            .drop_duplicates("item_id", keep="last")
            .set_index("item_id")
        )
        asof = np.zeros((self.n_items, len(ITEM_CONTINUOUS)), dtype=np.float32)
        known = latest.index.to_numpy(dtype=np.int64)
        valid = (known >= 0) & (known < self.n_items)
        asof[known[valid]] = _prepare_continuous(latest, ITEM_CONTINUOUS)[valid]

        continuous = torch.as_tensor(
            np.hstack([asof, self._item_static_continuous()]), device=self.device
        )
        categorical = torch.as_tensor(self._item_categorical, device=self.device)
        ids = torch.arange(self.n_items, device=self.device)

        chunks: list[np.ndarray] = []
        for start in range(0, self.n_items, 8192):
            end = min(start + 8192, self.n_items)
            embedded = self._model.item_tower(
                ids[start:end], categorical[start:end], continuous[start:end]
            )
            chunks.append(embedded.float().cpu().numpy())

        self._item_embeddings = np.vstack(chunks).astype(np.float32)
        logger.info("model.two_tower.encoded_items", shape=self._item_embeddings.shape)

    @torch.no_grad()
    def encode_users(self, features: pd.DataFrame) -> np.ndarray:
        """Embed users from as-of feature rows (one row per user)."""
        self._require_fitted()
        assert self._model is not None

        ids = torch.as_tensor(features["user_id"].to_numpy(dtype=np.int64), device=self.device)
        continuous = torch.as_tensor(
            _prepare_continuous(features, USER_CONTINUOUS), device=self.device
        )
        categorical = torch.as_tensor(self._user_categorical, device=self.device)[ids]
        embedded = self._model.user_tower(ids, categorical, continuous)
        vectors: np.ndarray = embedded.float().cpu().numpy().astype(np.float32)
        return vectors

    def set_user_embeddings(self, user_ids: np.ndarray, embeddings: np.ndarray) -> None:
        """Cache user embeddings so ``recommend`` can score without a forward pass."""
        cache = np.zeros((self.n_users, embeddings.shape[1]), dtype=np.float32)
        cache[user_ids] = embeddings
        self._user_embeddings = cache

    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        assert self._item_embeddings is not None
        if self._user_embeddings is None:
            raise RuntimeError(
                "User embeddings are not cached. Call encode_users() and "
                "set_user_embeddings() before recommending."
            )
        user_vector = self._user_embeddings[user_id]
        scores: np.ndarray = self._item_embeddings[candidates] @ user_vector
        return scores.astype(np.float32)

    def save_serving_state(self, path: Path) -> None:
        """Persist the two embedding matrices the serving path needs.

        Serving a two-tower model does not require the towers. Both sides were
        encoded offline and scoring is a dot product against the item matrix,
        so a deployed replica loads two float arrays instead of rebuilding a
        torch graph and a CUDA context. That asymmetry - expensive encoding
        offline, cheap lookup online - is the reason the architecture exists.

        The towers themselves are checkpointed separately by :meth:`save`,
        because retraining and re-encoding a user whose features have moved
        both need the real model.
        """
        self._require_fitted()
        assert self._item_embeddings is not None
        if self._user_embeddings is None:
            raise RuntimeError(
                "No user embeddings cached. Call set_user_embeddings() before saving "
                "serving state, or the loaded model can score no one."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            item_embeddings=self._item_embeddings,
            user_embeddings=self._user_embeddings,
            embedding_dim=np.int32(self._item_embeddings.shape[1]),
        )
        logger.info(
            "model.two_tower.serving_state_saved",
            path=str(path),
            users=int(self._user_embeddings.shape[0]),
            items=int(self._item_embeddings.shape[0]),
        )

    @classmethod
    def load_serving_state(
        cls,
        path: Path,
        *,
        items: pd.DataFrame,
        users: pd.DataFrame,
    ) -> TwoTowerRecommender:
        """Rebuild a scoring-only recommender from persisted embeddings.

        Raises:
            ValueError: If the stored matrices do not match the catalogue this
                process is serving. Shapes that disagree by one item would
                otherwise shift every id by one and produce recommendations
                that are wrong in a way nothing downstream can detect.
        """
        with np.load(path) as archive:
            item_embeddings = archive["item_embeddings"].astype(np.float32)
            user_embeddings = archive["user_embeddings"].astype(np.float32)

        model = cls(
            n_users=len(users),
            n_items=len(items),
            items=items,
            users=users,
        )
        if item_embeddings.shape[0] != model.n_items:
            raise ValueError(
                f"{path} holds {item_embeddings.shape[0]} item embeddings but the "
                f"catalogue has {model.n_items} items."
            )
        if user_embeddings.shape[0] != model.n_users:
            raise ValueError(
                f"{path} holds {user_embeddings.shape[0]} user embeddings but the "
                f"dataset has {model.n_users} users."
            )

        model._item_embeddings = item_embeddings
        model._user_embeddings = user_embeddings
        model._fitted = True
        logger.info("model.two_tower.serving_state_loaded", path=str(path))
        return model

    def user_vector(self, user_id: int) -> np.ndarray | None:
        """Return one cached user embedding, or None if there is not one.

        None rather than a zero vector for an unknown or never-encoded user.
        A zero vector is a valid point in the space and would place a
        cold-start user at its exact centre, which reads as a measurement
        instead of an absence.
        """
        cache = self._user_embeddings
        if cache is None or not 0 <= user_id < cache.shape[0]:
            return None
        vector: np.ndarray = cache[user_id]
        if not np.any(vector):
            return None
        return vector.astype(np.float32)

    @property
    def item_embeddings(self) -> np.ndarray:
        """L2-normalised item embeddings backing the ANN index."""
        self._require_fitted()
        assert self._item_embeddings is not None
        return self._item_embeddings

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
                "item_embeddings": self._item_embeddings,
                "history": self.history,
            },
            path,
        )
        logger.info("model.two_tower.saved", path=str(path))


__all__ = ["ITEM_CONTINUOUS", "USER_CONTINUOUS", "TwoTowerRecommender"]
