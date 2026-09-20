"""Project item embeddings to 3D for the Recommendation Galaxy.

The projection is computed **offline, once per model version**, and served as
a static artifact. Three reasons, and the first is decisive:

1. **UMAP is non-parametric.** There is no cheap, stable ``transform`` for a
   new point - the standard workaround refits or runs an optimisation per
   query. It also takes tens of seconds for a few thousand points. None of
   that belongs in a request.
2. It must be **reproducible**. A seeded offline run gives every viewer the
   same galaxy; computing per session would move the stars between reloads.
3. The payload is small enough to cache and serve directly.

PCA is computed alongside and persisted as its 64x3 component matrix. That
matters for a specific reason: PCA is linear, so a **user** vector can be
projected into the same space exactly, with one matrix multiply. UMAP cannot
do that, so the user star under UMAP is placed by a similarity-weighted
barycentre of its nearest projected items - which is a different thing, and
the API labels it as such rather than implying an exact projection.

Sampling is stratified by category and always includes every item that
appears in a highlighted set, so the points the UI needs to emphasise are
guaranteed present rather than probably present.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

#: Rendering budget. Above roughly this many points the scene stops being
#: readable before it stops being fast - the structure disappears into a
#: uniform cloud, which defeats the purpose of showing it.
DEFAULT_MAX_POINTS = 6000


@dataclass(slots=True)
class ProjectionResult:
    output_path: Path
    n_points: int
    method: str
    elapsed_seconds: float


def _stratified_sample(
    items: pd.DataFrame,
    *,
    max_points: int,
    must_include: set[int],
    seed: int = 42,
) -> np.ndarray:
    """Sample item ids, keeping each category's share of the catalogue.

    Uniform sampling would under-represent small categories to the point of
    erasing them, and the category clusters are the main thing the galaxy is
    meant to show.
    """
    if len(items) <= max_points:
        return items["item_id"].to_numpy(dtype=np.int64)

    rng = np.random.default_rng(seed)
    budget = max_points - len(must_include)
    chosen: list[int] = list(must_include)

    candidates = items[~items["item_id"].isin(must_include)]
    groups = candidates.groupby("category_id", sort=True)
    total = len(candidates)

    for _, group in groups:
        share = len(group) / total
        take = max(1, round(share * budget))
        take = min(take, len(group))
        picked = rng.choice(group["item_id"].to_numpy(), size=take, replace=False)
        chosen.extend(int(i) for i in picked)

    unique = np.unique(np.array(chosen, dtype=np.int64))
    if len(unique) > max_points:
        # Rounding per group can overshoot; trim the extras, never the
        # must-include set.
        optional = np.setdiff1d(unique, np.fromiter(must_include, dtype=np.int64))
        keep = rng.choice(optional, size=max_points - len(must_include), replace=False)
        unique = np.unique(np.concatenate([keep, np.fromiter(must_include, dtype=np.int64)]))
    return unique


def _fit_pca(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 3-component PCA. Returns (coords, components, mean).

    The components and mean are persisted so a user vector can be projected
    into the identical space later without refitting - which is what makes the
    PCA view's user star exact rather than approximate.
    """
    from sklearn.decomposition import PCA

    model = PCA(n_components=3, random_state=42)
    coords = model.fit_transform(embeddings)
    return (
        coords.astype(np.float32),
        model.components_.astype(np.float32),
        model.mean_.astype(np.float32),
    )


def _fit_umap(embeddings: np.ndarray, *, seed: int = 42) -> np.ndarray | None:
    """Fit a 3D UMAP, or return None if it is unavailable.

    ``umap-learn`` is an optional extra and JIT-compiles through numba on
    first call. Returning None rather than raising keeps the galaxy working
    on a PCA projection when the extra is not installed.
    """
    try:
        import umap
    except ImportError:
        logger.warning("projection.umap_unavailable", hint="install the 'viz' extra")
        return None

    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=25,
            min_dist=0.08,
            # Cosine, because the item tower L2-normalises its output, so
            # cosine is the metric the embedding space was trained under.
            metric="cosine",
            random_state=seed,
        )
        return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("projection.umap_failed", error=str(exc)[:160])
        return None


def _scale_to_unit_sphere(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre and scale coordinates into a unit sphere.

    The frontend camera is fixed, so the projection must land in a predictable
    volume regardless of the raw scale a reducer happens to produce.

    The centre and radius are returned rather than only applied. Placing a new
    point - a user vector - into the same space later needs the same affine
    transform, and it cannot be recovered from coordinates that have already
    been through it.
    """
    centre = coords.mean(axis=0)
    centred = coords - centre
    radius = float(np.linalg.norm(centred, axis=1).max())
    if radius > 0:
        centred = centred / radius
    return centred.astype(np.float32), centre.astype(np.float32), radius


def project_embeddings(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
    must_include: set[int] | None = None,
    seed: int = 42,
) -> ProjectionResult:
    """Compute and persist the 3D projection for the galaxy view."""
    started = time.perf_counter()
    root = data_dir or Path("data")
    artifacts = artifacts_dir or Path("artifacts")
    processed_dir = root / "processed" / preset
    evaluation_dir = artifacts / "evaluation" / preset
    output_dir = artifacts / "projections" / preset
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = evaluation_dir / "item_embeddings.npy"
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"No item embeddings at {embeddings_path}. Run `mercury train baselines` "
            "so the two-tower model writes them."
        )

    embeddings = np.load(embeddings_path)
    items = pd.read_parquet(processed_dir / "items.parquet").sort_values("item_id")
    metadata = json.loads((processed_dir / "METADATA.json").read_text(encoding="utf-8"))

    sampled = _stratified_sample(
        items, max_points=max_points, must_include=must_include or set(), seed=seed
    )
    subset = embeddings[sampled]
    logger.info("projection.sampled", n_points=len(sampled), of=len(items))

    pca_coords, components, mean = _fit_pca(subset)
    pca_coords, pca_centre, pca_radius = _scale_to_unit_sphere(pca_coords)

    umap_coords = _fit_umap(subset, seed=seed)
    if umap_coords is not None:
        umap_coords, _, _ = _scale_to_unit_sphere(umap_coords)

    item_rows = items.set_index("item_id").loc[sampled]
    frame = pd.DataFrame(
        {
            "item_id": sampled,
            "px": pca_coords[:, 0],
            "py": pca_coords[:, 1],
            "pz": pca_coords[:, 2],
            "ux": umap_coords[:, 0] if umap_coords is not None else pca_coords[:, 0],
            "uy": umap_coords[:, 1] if umap_coords is not None else pca_coords[:, 1],
            "uz": umap_coords[:, 2] if umap_coords is not None else pca_coords[:, 2],
            "category_id": item_rows["category_id"].to_numpy(),
            "vertical": item_rows["vertical"].to_numpy(),
            "merchant_id": item_rows["merchant_id"].to_numpy(),
            "price": item_rows["price"].to_numpy(),
        }
    )

    output_path = output_dir / "items_3d.parquet"
    frame.to_parquet(output_path, compression="zstd", index=False)

    # The PCA basis, so a user vector can be projected into the same space
    # exactly rather than approximated.
    np.savez(
        output_dir / "pca_basis.npz",
        components=components,
        mean=mean,
        centre=pca_centre,
        radius=np.float32(pca_radius),
    )

    manifest: dict[str, Any] = {
        "preset": preset,
        "dataset_hash": metadata.get("dataset_hash"),
        "n_points": len(sampled),
        "n_catalog_items": len(items),
        "embedding_dim": int(embeddings.shape[1]),
        "methods": ["pca"] + (["umap"] if umap_coords is not None else []),
        "umap_available": umap_coords is not None,
        "seed": seed,
        "notes": {
            "pca": "Exact linear projection; a user vector maps in with one matmul.",
            "umap": (
                "Non-parametric. A user has no exact projection, so the UI places "
                "the user star at a similarity-weighted barycentre of its nearest "
                "projected items and labels it as such."
            ),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - started
    logger.info("projection.complete", seconds=round(elapsed, 1), points=len(sampled))
    return ProjectionResult(
        output_path=output_path,
        n_points=len(sampled),
        method="umap" if umap_coords is not None else "pca",
        elapsed_seconds=elapsed,
    )


__all__ = [
    "DEFAULT_MAX_POINTS",
    "ProjectionResult",
    "project_embeddings",
]
