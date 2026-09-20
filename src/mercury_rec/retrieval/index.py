"""Vector indexes for two-tower candidate retrieval.

Three implementations behind one protocol, because the interesting question is
not "which is fastest" but "at what catalogue size does approximation start to
pay for itself" -- and that has to be measured, not assumed.

- :class:`ExactIndex` -- brute-force numpy. For 36,044 items at 64 dimensions
  a query is a 36,044x64 matrix-vector product, about 2.3 MFLOP, which BLAS
  finishes in roughly a millisecond. It is exact by construction, so it also
  serves as the ground truth the approximate indexes are measured against.
- :class:`FlatIPIndex` -- FAISS exact inner product. Same answers as
  ``ExactIndex``, with FAISS's SIMD kernels.
- :class:`HNSWIndex` -- FAISS approximate graph search. Sub-linear, and it is
  the only one whose cost stays flat as the catalogue grows.

**At this catalogue size exact search is genuinely competitive**, and the
benchmark in ``docs/benchmarks.md`` says so rather than implying ANN was
necessary. The reason to build the HNSW path anyway is that it is the
component that would have to exist at 10 million items, and the interesting
engineering content is the crossover point and the recall it costs.

All embeddings are L2-normalised by the item tower, so inner product *is*
cosine similarity and no separate metric handling is needed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class VectorIndex(Protocol):
    """Nearest-neighbour search over item embeddings."""

    name: str

    def build(self, embeddings: np.ndarray) -> None:
        """Index the given ``(n_items, dim)`` embedding matrix."""
        ...

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, item_ids)``, each ``(n_queries, k)``, best first."""
        ...


def _validate(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D, got shape {embeddings.shape}.")
    if not np.isfinite(embeddings).all():
        # NaNs here would propagate into every recommendation silently: FAISS
        # returns them as -1 ids rather than raising.
        raise ValueError("Embeddings contain NaN or infinity; refusing to build an index.")
    return np.ascontiguousarray(embeddings, dtype=np.float32)


class ExactIndex:
    """Brute-force search. Exact by construction, and the reference answer."""

    name = "exact"

    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None

    def build(self, embeddings: np.ndarray) -> None:
        self._embeddings = _validate(embeddings)
        logger.info("index.built", index=self.name, items=len(self._embeddings))

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._embeddings is None:
            raise RuntimeError("Index has not been built.")
        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)

        scores = query_matrix @ self._embeddings.T
        top_k = min(k, scores.shape[1])
        # argpartition is O(n) for the top-k, then only those k are sorted.
        partitioned = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
        ordered = np.take_along_axis(
            partitioned,
            np.argsort(-np.take_along_axis(scores, partitioned, axis=1), axis=1),
            axis=1,
        )
        return np.take_along_axis(scores, ordered, axis=1), ordered


class FlatIPIndex:
    """FAISS exact inner-product index."""

    name = "faiss_flat_ip"

    def __init__(self) -> None:
        self._index: object | None = None

    def build(self, embeddings: np.ndarray) -> None:
        import faiss

        matrix = _validate(embeddings)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index
        logger.info("index.built", index=self.name, items=index.ntotal)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        scores, ids = self._index.search(query_matrix, k)  # type: ignore[attr-defined]
        return scores, ids


class HNSWIndex:
    """FAISS HNSW approximate index.

    Args:
        m: Graph connectivity. Higher gives better recall and a larger index.
            32 is the common default and a reasonable accuracy/memory balance.
        ef_construction: Build-time search breadth. Higher builds a better
            graph, more slowly. Paid once.
        ef_search: Query-time search breadth. **The accuracy/latency dial**,
            adjustable after the index is built, which is why it is the knob
            the benchmark sweeps.
    """

    name = "faiss_hnsw"

    def __init__(self, *, m: int = 32, ef_construction: int = 200, ef_search: int = 128) -> None:
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index: object | None = None

    def build(self, embeddings: np.ndarray) -> None:
        import faiss

        matrix = _validate(embeddings)
        index = faiss.IndexHNSWFlat(matrix.shape[1], self.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        started = time.perf_counter()
        index.add(matrix)
        index.hnsw.efSearch = self.ef_search
        self._index = index
        logger.info(
            "index.built",
            index=self.name,
            items=index.ntotal,
            build_seconds=round(time.perf_counter() - started, 2),
            m=self.m,
            ef_construction=self.ef_construction,
        )

    def set_ef_search(self, ef_search: int) -> None:
        """Retune accuracy vs latency without rebuilding."""
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        self.ef_search = ef_search
        self._index.hnsw.efSearch = ef_search  # type: ignore[attr-defined]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is None:
            raise RuntimeError("Index has not been built.")
        query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
        if query_matrix.ndim == 1:
            query_matrix = query_matrix.reshape(1, -1)
        scores, ids = self._index.search(query_matrix, k)  # type: ignore[attr-defined]
        return scores, ids


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class IndexBenchmark:
    """One index's measured cost and accuracy."""

    index: str
    n_items: int
    dim: int
    build_seconds: float
    mean_query_ms: float
    p95_query_ms: float
    recall_at_k: float
    """Overlap with exact search. 1.0 for an exact index, by definition."""

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "index": self.index,
            "n_items": self.n_items,
            "dim": self.dim,
            "build_seconds": round(self.build_seconds, 4),
            "mean_query_ms": round(self.mean_query_ms, 4),
            "p95_query_ms": round(self.p95_query_ms, 4),
            "recall_at_k": round(self.recall_at_k, 4),
        }


def benchmark_index(
    index: VectorIndex,
    embeddings: np.ndarray,
    queries: np.ndarray,
    *,
    k: int = 100,
    exact_ids: np.ndarray | None = None,
    warmup: int = 5,
) -> IndexBenchmark:
    """Measure build time, per-query latency and recall against exact search.

    Latency is measured one query at a time, because that is what serving
    does. Batched throughput would look far better and would not describe the
    request path.

    Args:
        index: The index to measure.
        embeddings: Item matrix to index.
        queries: Query vectors.
        k: Neighbours per query.
        exact_ids: Ground-truth neighbour ids for recall. When omitted,
            recall is reported as 1.0 and is only meaningful for exact indexes.
        warmup: Untimed queries first, so page faults and any lazy
            initialisation are not charged to the measurement.
    """
    started = time.perf_counter()
    index.build(embeddings)
    build_seconds = time.perf_counter() - started

    for position in range(min(warmup, len(queries))):
        index.search(queries[position : position + 1], k)

    timings: list[float] = []
    retrieved = np.empty((len(queries), k), dtype=np.int64)
    for position in range(len(queries)):
        query_start = time.perf_counter()
        _, ids = index.search(queries[position : position + 1], k)
        timings.append((time.perf_counter() - query_start) * 1000.0)
        retrieved[position] = ids[0][:k]

    if exact_ids is None:
        recall = 1.0
    else:
        overlaps = [
            len(set(retrieved[i].tolist()) & set(exact_ids[i][:k].tolist())) / k
            for i in range(len(queries))
        ]
        recall = float(np.mean(overlaps))

    durations = np.array(timings)
    return IndexBenchmark(
        index=index.name,
        n_items=len(embeddings),
        dim=embeddings.shape[1],
        build_seconds=build_seconds,
        mean_query_ms=float(durations.mean()),
        p95_query_ms=float(np.percentile(durations, 95)),
        recall_at_k=recall,
    )


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    logger.info("index.embeddings_saved", path=str(path), shape=embeddings.shape)


def load_embeddings(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"No embeddings at {path}. Train the two-tower model first.")
    loaded: np.ndarray = np.load(path)
    return loaded


__all__ = [
    "ExactIndex",
    "FlatIPIndex",
    "HNSWIndex",
    "IndexBenchmark",
    "VectorIndex",
    "benchmark_index",
    "load_embeddings",
    "save_embeddings",
]
