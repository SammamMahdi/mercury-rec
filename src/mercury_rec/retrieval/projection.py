"""Place a point in an existing low-dimensional projection.

These two functions are the serving half of the projection pipeline: given a
user embedding and a projection that was already fitted offline, say where
that user belongs in it. They live in ``retrieval`` rather than beside the
pipeline that fits the projection because the API needs them on a request
path, and a serving layer must not import from a pipeline layer.

The split matters beyond layering. Fitting is minutes of UMAP; placing is a
matmul or a weighted average over 32 neighbours.
"""

from __future__ import annotations

import numpy as np


def project_user_vector(
    user_embedding: np.ndarray,
    *,
    components: np.ndarray,
    mean: np.ndarray,
    centre: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Project a user vector into the displayed PCA space. Exact.

    Both halves of the transform the item coordinates went through: the PCA
    basis, then the centre/radius normalisation that put the scene inside a
    unit sphere. Applying only the first puts the star in the right direction
    at the wrong scale, which looks like a rendering bug and is a real one.
    """
    projected = (user_embedding - mean) @ components.T
    scaled = (projected - centre) / (radius if radius > 0 else 1.0)
    return np.asarray(scaled, dtype=np.float32)


def barycentric_user_position(
    user_embedding: np.ndarray,
    item_embeddings: np.ndarray,
    item_coords: np.ndarray,
    *,
    top_n: int = 32,
    temperature: float = 0.05,
) -> np.ndarray:
    """Place a user in a non-linear projection, by similarity-weighted average.

    UMAP has no exact transform for a new point, so the user star is placed at
    a softmax-weighted average of its nearest projected items. This is O(32),
    stable, and - importantly for the demo - **continuous in the context**:
    changing the hour moves the user's embedding slightly, which slides the
    star smoothly rather than teleporting it.

    It is an approximation, and both the API response and the UI say so.
    """
    similarities = item_embeddings @ user_embedding
    top = np.argpartition(-similarities, min(top_n, len(similarities) - 1))[:top_n]

    weights = np.exp(similarities[top] / temperature)
    weights = weights / weights.sum()
    position: np.ndarray = (item_coords[top] * weights[:, None]).sum(axis=0).astype(np.float32)
    return position


__all__ = ["barycentric_user_position", "project_user_vector"]
