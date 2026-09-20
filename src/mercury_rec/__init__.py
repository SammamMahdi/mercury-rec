"""MercuryRec — multi-stage recommendation and personalization platform.

Layering contract (dependencies point strictly downward)::

    api -> services -> recommender -> {models, retrieval, reranking, features}
        -> {data, db, cache} -> {config, core}

A module must not import from a layer above its own. This is enforced by
``tests/unit/test_layering.py``.

Importing this package pins OpenMP/BLAS thread counts as a side effect. That
has to happen before PyTorch, FAISS or LightGBM load, so it lives here rather
than in application code — see :mod:`mercury_rec.core._runtime`.
"""

from __future__ import annotations

# ruff: isort: off
# MUST be first: sets OMP_NUM_THREADS et al. before any native library loads.
from mercury_rec.core._runtime import NUM_THREADS

# ruff: isort: on

__version__ = "0.1.0"

__all__ = ["NUM_THREADS", "__version__"]
