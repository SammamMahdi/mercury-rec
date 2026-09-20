"""Assert the development environment is wired correctly.

Each check here corresponds to a failure that has a misleading symptom — the
kind that costs an afternoon because the error message points somewhere other
than the cause.

Run::

    uv run python scripts/check_env.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mercury_rec  # noqa: F401  (pins OpenMP threads before native imports)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_REMOTE_SUFFIX = "mercury-rec"

_failures: list[str] = []
_warnings: list[str] = []


def _ok(label: str, detail: str) -> None:
    print(f"  ok    {label:<22} {detail}")


def _fail(label: str, detail: str, remedy: str) -> None:
    print(f"  FAIL  {label:<22} {detail}")
    _failures.append(f"{label}: {detail}\n        -> {remedy}")


def _warn(label: str, detail: str) -> None:
    print(f"  warn  {label:<22} {detail}")
    _warnings.append(f"{label}: {detail}")


def _git(*args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def check_git_repository() -> None:
    """Guard against committing into the wrong repository.

    ``E:/Projects`` is itself a git repository for an unrelated project. Before
    this project was given its own ``.git``, ``git rev-parse --show-toplevel``
    from here resolved to that parent repo — so a ``git add -A && git commit``
    would have published MercuryRec into the wrong remote *and* committed a
    mass deletion of the other project's tracked files.
    """
    toplevel = _git("rev-parse", "--show-toplevel")
    if not toplevel:
        _fail("git repo", "not inside a git repository", "run `git init` in the project root")
        return

    if Path(toplevel).resolve() != PROJECT_ROOT:
        _fail(
            "git repo",
            f"toplevel is {toplevel}, expected {PROJECT_ROOT}",
            "this directory is owned by a DIFFERENT repository; do not commit",
        )
        return
    _ok("git repo", toplevel)

    remote = _git("config", "--get", "remote.origin.url")
    if not remote:
        _warn("git remote", "no origin configured yet")
    elif not remote.rstrip("/").removesuffix(".git").endswith(EXPECTED_REMOTE_SUFFIX):
        _fail(
            "git remote",
            f"origin is {remote}",
            f"expected a remote ending in '{EXPECTED_REMOTE_SUFFIX}'",
        )
    else:
        _ok("git remote", remote)


def check_git_identity() -> None:
    """The repo-local identity must be the repository owner.

    The machine's global identity is a placeholder. Commits made under it are
    not attributed by GitHub, and correcting them after the fact requires the
    history rewrite this project is meant to avoid.
    """
    name = _git("config", "--local", "user.name")
    email = _git("config", "--local", "user.email")
    if not name or not email:
        _fail(
            "git identity",
            "no repo-local user.name/user.email",
            'git config --local user.name "..." && git config --local user.email "..."',
        )
        return
    if "test@test" in email.lower() or name.strip().lower() == "test":
        _fail(
            "git identity",
            f"placeholder identity in use: {name} <{email}>",
            "set the repo-local identity to the repository owner",
        )
        return
    _ok("git identity", f"{name} <{email}>")


def check_python() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) != (3, 12):
        _warn("python", f"{major}.{minor} (project targets 3.12)")
    else:
        _ok("python", f"{sys.version.split()[0]} at {sys.executable}")

    # The `py` launcher on this machine points at a nonexistent C:\Python12.
    # Any `py -m ...` in a Makefile, script or README silently fails for the
    # next person, so every documented command uses `uv run` instead.
    if os.name == "nt":
        which = subprocess.run(
            ["py", "-0p"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
        if which.returncode != 0 or r"Python12/python.exe" in which.stdout:
            _warn("py launcher", "broken/misconfigured — use `uv run`, never `py`")


def check_native_libraries() -> None:
    """Import the three OpenMP-bundling libraries in a fixed, safe order.

    LightGBM's Windows wheel bundles ``lib_lightgbm.dll`` but depends on the
    MSVC 2015-2022 x64 redistributable. Without it the import raises
    ``OSError: [WinError 126] The specified module could not be found`` — which
    names a DLL that *is* present, while the actually-missing dependency goes
    unnamed. That message sends people down the wrong path, so it is caught
    and translated here.
    """
    import torch

    _ok("torch", f"{torch.__version__} (cuda {torch.version.cuda})")

    try:
        import faiss

        _ok("faiss", f"{faiss.__version__}")
    except (ImportError, OSError) as exc:
        _fail("faiss", str(exc)[:70], "reinstall faiss-cpu; see docs/troubleshooting.md")

    try:
        import lightgbm

        _ok("lightgbm", lightgbm.__version__)
    except OSError as exc:
        remedy = "install MSVC 2015-2022 x64 redistributable"
        if "126" in str(exc):
            remedy += ": https://aka.ms/vs/17/release/vc_redist.x64.exe"
        _fail("lightgbm", "WinError 126 / DLL load failed", remedy)
    except ImportError as exc:
        _fail("lightgbm", str(exc)[:70], "uv sync")


def check_threads() -> None:
    _ok("omp threads", f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    if os.environ.get("KMP_DUPLICATE_LIB_OK", "").upper() == "TRUE":
        _warn(
            "KMP_DUPLICATE_LIB_OK",
            "set to TRUE — suppresses the OpenMP abort without fixing the ABI conflict",
        )


def main() -> int:
    print(f"MercuryRec environment check\n  root: {PROJECT_ROOT}\n")
    check_git_repository()
    check_git_identity()
    check_python()
    check_threads()
    check_native_libraries()

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:\n")
        for item in _failures:
            print(f"  - {item}")
        return 1
    if _warnings:
        print(f"PASS with {len(_warnings)} warning(s).")
    else:
        print("PASS — environment is correctly configured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
