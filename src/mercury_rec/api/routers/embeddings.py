"""Embedding projection endpoints backing the 3D Recommendation Galaxy.

Coordinates are sent as **base64-encoded Float32Array buffers**, not JSON
numbers. For 6,000 points that is 72 KB of binary against roughly 450 KB of
JSON text, and the browser can hand the decoded buffer straight to a
``BufferAttribute`` with no per-point object allocation. Parsing 18,000 JSON
floats into JavaScript numbers and then copying them into a typed array is the
single most expensive thing this page could do on the main thread.

The projection itself is precomputed offline (see
``pipelines/project_embeddings.py``): UMAP has no cheap transform for a new
point and takes tens of seconds to fit, so it cannot live in a request.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi import Path as PathParam

from mercury_rec.api.deps import EngineDep
from mercury_rec.core.logging import get_logger
from mercury_rec.retrieval.projection import barycentric_user_position, project_user_vector

logger = get_logger(__name__)

router = APIRouter()


def _encode(array: np.ndarray) -> str:
    """Base64-encode an array's raw little-endian bytes.

    Endianness is forced so the browser's Float32Array, which is
    little-endian on every platform that matters, reads the same numbers a
    big-endian server wrote.
    """
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


@lru_cache(maxsize=4)
def _load_projection(preset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the projection artifact once and keep it.

    Cached because it is a static artifact that only changes when a model is
    retrained, and re-reading 6,000 rows of parquet per request would be pure
    waste on the hot path of an interactive page.
    """
    directory = Path("artifacts") / "projections" / preset
    parquet = directory / "items_3d.parquet"
    manifest = directory / "manifest.json"

    if not parquet.is_file():
        raise FileNotFoundError(
            f"No projection at {parquet}. Run `mercury viz project` to build it."
        )

    frame = pd.read_parquet(parquet)
    meta = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    logger.info("embeddings.projection_loaded", preset=preset, points=len(frame))
    return frame, meta


@lru_cache(maxsize=4)
def _load_pca_basis(preset: str) -> dict[str, Any]:
    """Load the persisted PCA basis and the unit-sphere transform beside it."""
    path = Path("artifacts") / "projections" / preset / "pca_basis.npz"
    if not path.is_file():
        raise FileNotFoundError(f"No PCA basis at {path}. Run `mercury viz project`.")
    with np.load(path) as archive:
        keys = set(archive.files)
        if not {"centre", "radius"} <= keys:
            raise FileNotFoundError(
                f"{path} predates the stored unit-sphere transform. "
                "Re-run `mercury viz project` to place users exactly."
            )
        return {
            "components": archive["components"],
            "mean": archive["mean"],
            "centre": archive["centre"],
            "radius": float(archive["radius"]),
        }


@router.get("/projection", summary="3D item embedding projection")
def projection(
    method: Annotated[str, Query(pattern="^(umap|pca)$")] = "umap",
    max_items: Annotated[int, Query(ge=100, le=20000)] = 6000,
    vertical: Annotated[int | None, Query(ge=0, le=5)] = None,
    preset: Annotated[str, Query(pattern="^(demo|full)$")] = "full",
) -> dict[str, Any]:
    """Return item coordinates for the galaxy.

    Args:
        method: ``umap`` (structure-preserving, approximate for new points) or
            ``pca`` (linear, so a user vector projects in exactly).
        max_items: Point budget. The scene stops being *readable* before it
            stops being fast, so this is a legibility limit as much as a
            performance one.
        vertical: Restrict to one vertical.
        preset: Dataset preset.
    """
    try:
        frame, meta = _load_projection(preset)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No embedding projection available. Run `mercury viz project` "
                "after training the two-tower model."
            ),
        ) from exc

    if vertical is not None:
        frame = frame[frame["vertical"] == vertical]
    if len(frame) > max_items:
        # Deterministic subsample, so repeated requests return the same stars
        # rather than reshuffling the galaxy on every reload.
        frame = frame.iloc[:: max(1, len(frame) // max_items)].head(max_items)

    prefix = "u" if method == "umap" else "p"
    return {
        "projection": f"{method}3",
        "count": len(frame),
        "dataset_hash": meta.get("dataset_hash"),
        "encoding": "base64-float32-le",
        "note": (
            "Coordinates are base64 Float32Array buffers rather than JSON "
            "numbers: 72KB versus ~450KB for 6000 points, and the browser can "
            "use the decoded buffer directly."
        ),
        "x": _encode(frame[f"{prefix}x"].to_numpy(dtype="<f4")),
        "y": _encode(frame[f"{prefix}y"].to_numpy(dtype="<f4")),
        "z": _encode(frame[f"{prefix}z"].to_numpy(dtype="<f4")),
        "item_id": _encode(frame["item_id"].to_numpy(dtype="<i4")),
        "category_id": _encode(frame["category_id"].to_numpy(dtype="<i4")),
        "vertical": _encode(frame["vertical"].to_numpy(dtype="<i4")),
        "price": _encode(frame["price"].to_numpy(dtype="<f4")),
    }


@router.get("/user/{user_id}", summary="Place one user in the projection")
def user_position(
    engine: EngineDep,
    user_id: Annotated[str, PathParam(min_length=1, max_length=64)],
    method: Annotated[str, Query(pattern="^(umap|pca)$")] = "umap",
    preset: Annotated[str, Query(pattern="^(demo|full)$")] = "full",
) -> dict[str, Any]:
    """Return where a user sits among the projected items.

    Two genuinely different placements, and the response says which one it
    gave:

    - **PCA** is a linear map, so the user's embedding projects in *exactly*,
      through the same basis and the same unit-sphere normalisation the items
      went through. ``is_exact`` is true.
    - **UMAP** is non-parametric and has no transform for an unseen point. The
      star is placed at a softmax-weighted barycentre of the user's nearest
      projected items, which is an approximation. ``is_exact`` is false.

    Collapsing the two into one unlabelled number would present an estimate
    with the same authority as a measurement.
    """
    bundle = engine.bundle
    two_tower = bundle.two_tower
    if two_tower is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The two-tower model is not loaded, so there is no user embedding space.",
        )

    internal = bundle.external_user(user_id)
    if internal is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown user {user_id!r}.",
        )

    user_embedding = two_tower.user_vector(internal)
    if user_embedding is None:
        # A real serving state, not a failure: a user with no encoded
        # embedding is served by the cold-start path and has no position.
        return {
            "user_id": user_id,
            "position": None,
            "method": method,
            "is_exact": False,
            "note": (
                "This user has no two-tower embedding, so they have no position "
                "in the item space. They are served by the cold-start path."
            ),
        }

    try:
        frame, meta = _load_projection(preset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if method == "pca":
        try:
            basis = _load_pca_basis(preset)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        position = project_user_vector(
            user_embedding,
            components=basis["components"],
            mean=basis["mean"],
            centre=basis["centre"],
            radius=basis["radius"],
        )
        note = "Exact linear projection through the same basis as the items."
        is_exact = True
    else:
        item_ids = frame["item_id"].to_numpy(dtype=np.int64)
        coords = frame[["ux", "uy", "uz"]].to_numpy(dtype=np.float32)
        position = barycentric_user_position(
            user_embedding,
            two_tower.item_embeddings[item_ids],
            coords,
        )
        note = (
            "Approximate. UMAP has no transform for an unseen point, so the "
            "star is a similarity-weighted barycentre of the user's 32 nearest "
            "projected items."
        )
        is_exact = False

    return {
        "user_id": user_id,
        "position": [float(value) for value in position],
        "method": method,
        "is_exact": is_exact,
        "dataset_hash": meta.get("dataset_hash"),
        "note": note,
    }


@router.get("/projection/meta", summary="Projection manifest")
def projection_meta(
    preset: Annotated[str, Query(pattern="^(demo|full)$")] = "full",
) -> dict[str, Any]:
    """Describe the available projections and their limitations.

    Served separately so the UI can label which method is active and, for
    UMAP, state plainly that the user star is a similarity-weighted barycentre
    rather than an exact projection.
    """
    try:
        _, meta = _load_projection(preset)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No projection manifest. Run `mercury viz project`.",
        ) from exc
    return meta


__all__ = ["router"]
