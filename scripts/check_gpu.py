"""Verify the CUDA stack actually works on this GPU before training starts.

Why this exists: the development machine has an RTX 5060 Ti, which is
Blackwell — compute capability ``sm_120``. Default PyPI ``torch`` wheels are
not built with ``sm_120`` kernels. Installing one produces either

    CUDA error: no kernel image is available for execution on the device

at the first CUDA op, or — far worse — appears to work while silently
producing wrong numbers. A model trained against a broken stack yields
metrics that cannot be reproduced, which the project's honesty requirements
forbid publishing.

So this is a gate, not a diagnostic: ``mercury train`` runs it first and
refuses to proceed if it fails.

Run directly::

    uv run python scripts/check_gpu.py
"""

from __future__ import annotations

import sys

import torch

# Import the package first so OpenMP thread counts are pinned before torch
# loads its bundled MKL/OpenMP runtime.
import mercury_rec  # noqa: F401

#: Blackwell consumer parts (RTX 50-series) report compute capability 12.0.
_BLACKWELL = (12, 0)

#: Tolerance for the CPU/GPU matmul agreement check. fp32 matmul on tensor
#: cores may use TF32 internally, which is materially less precise than
#: strict IEEE fp32, so this is deliberately loose.
_MATMUL_ATOL = 1e-3


def _fail(message: str, remedy: str) -> None:
    print(f"FAIL  {message}", file=sys.stderr)
    print(f"      {remedy}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    print(f"torch          : {torch.__version__}")
    print(f"compiled CUDA  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        _fail(
            "torch.cuda.is_available() is False — no usable CUDA device.",
            "Install the cu128 build: `uv sync` (pyproject pins the "
            "pytorch-cu128 index). Check `nvidia-smi` works.",
        )

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device         : {name}")
    print(f"capability     : sm_{capability[0]}{capability[1]}")
    print(f"memory         : {total_gb:.1f} GiB")

    # The architecture list baked into this wheel. If the device's capability
    # is absent, CUDA ops fail at runtime rather than at import.
    arch_list = torch.cuda.get_arch_list()
    print(f"built archs    : {', '.join(arch_list)}")

    arch_tag = f"sm_{capability[0]}{capability[1]}"
    if capability >= _BLACKWELL and arch_tag not in arch_list:
        _fail(
            f"This torch build has no {arch_tag} kernels (device is {name}).",
            "Blackwell needs a CUDA 12.8+ build. Confirm the installed wheel "
            "is a '+cu128' variant, not a default PyPI wheel.",
        )

    # Prove a real kernel launches and returns correct numbers. An
    # architecture mismatch that slipped past the checks above surfaces here.
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    expected = a @ b
    actual = (a.cuda() @ b.cuda()).cpu()

    if not torch.allclose(expected, actual, atol=_MATMUL_ATOL):
        max_diff = (expected - actual).abs().max().item()
        _fail(
            f"GPU matmul disagrees with CPU reference (max diff {max_diff:.2e}).",
            "The CUDA stack is producing incorrect results. Do not train "
            "against it — any metric produced would be unreproducible.",
        )

    # bf16 is native on Blackwell and is what training uses. fp16 + GradScaler
    # is deliberately avoided; bf16 has the same exponent range as fp32, so
    # there is no loss-scaling machinery and no inf/nan babysitting.
    if torch.cuda.is_bf16_supported():
        print("bf16           : supported (used for autocast during training)")
    else:
        print("bf16           : NOT supported — training falls back to fp32")

    print("\nPASS  CUDA stack verified: kernels launch and results are correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
