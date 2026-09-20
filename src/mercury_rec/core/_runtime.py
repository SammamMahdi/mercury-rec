"""Process-wide numeric runtime configuration. **Import this before torch.**

Three libraries in this project each bundle their own OpenMP runtime:
PyTorch (``libiomp5md.dll`` via MKL), LightGBM (``lib_lightgbm.dll``) and
FAISS. Loading more than one into a single process has two failure modes:

1. A hard abort — ``OMP: Error #15: Initializing libiomp5md.dll, but found
   libiomp5md.dll already initialized``.
2. Far more expensive because it looks fine: **thread oversubscription**.
   Each library defaults to one thread per core, so on an 8-core box three
   libraries spawn 24 threads that fight over 8 cores. Throughput drops
   several-fold while every process looks healthy.

Thread-count environment variables are read by OpenMP at load time, so they
must be set *before* the offending libraries are imported. This module is
imported at the top of ``mercury_rec/__init__.py`` for exactly that reason.

The documented escape hatch ``KMP_DUPLICATE_LIB_OK=TRUE`` is deliberately not
set here: it suppresses the abort without resolving the underlying ABI
conflict, and can yield silently wrong results.
"""

from __future__ import annotations

import os

#: Default intra-op thread budget. Matches the 8 physical cores of the
#: development machine; override with ``MERCURY_NUM_THREADS``.
_DEFAULT_THREADS = 8

_OMP_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_configured = False


def _resolve_thread_budget() -> int:
    raw = os.environ.get("MERCURY_NUM_THREADS")
    if raw is None:
        return _DEFAULT_THREADS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_THREADS
    return max(1, value)


def configure_numeric_runtime() -> int:
    """Pin OpenMP/BLAS thread counts. Idempotent; returns the budget applied.

    Respects any thread variable the caller already exported, so a deployment
    can still tune threads per process without editing code.
    """
    global _configured
    threads = _resolve_thread_budget()
    if _configured:
        return threads

    for var in _OMP_VARS:
        os.environ.setdefault(var, str(threads))

    _configured = True
    return threads


#: Applied at import time — before torch, faiss or lightgbm can be loaded.
NUM_THREADS: int = configure_numeric_runtime()
