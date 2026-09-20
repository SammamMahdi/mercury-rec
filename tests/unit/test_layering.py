"""Enforce the package's layering contract.

``mercury_rec/__init__.py`` documents that dependencies point strictly
downward::

    api -> services -> recommender -> {models, retrieval, reranking, features}
        -> {data, db, cache} -> {config, core}

A documented architecture that nothing checks decays within weeks: someone
imports the API's settings object into a model for convenience, and the model
layer silently acquires a dependency on the web framework. This test parses
every module's imports and fails on the first upward edge, naming both ends.

It is deliberately a static AST check rather than an import-time check, so it
costs milliseconds and cannot be defeated by a function-local import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "mercury_rec"

#: Lower number == lower layer. A module may import from its own layer or any
#: layer below it, never above.
LAYERS: dict[str, int] = {
    "core": 0,
    "config": 0,
    "data": 1,
    "db": 1,
    "cache": 1,
    "features": 2,
    "models": 3,
    "retrieval": 3,
    "reranking": 3,
    "evaluation": 3,
    "monitoring": 3,
    "experimentation": 4,
    "recommender": 5,
    "services": 6,
    "api": 7,
    "pipelines": 8,
    "cli": 9,
}


def _layer_of(module_parts: tuple[str, ...]) -> int | None:
    """Return the layer rank for a dotted module path, or None if unranked."""
    if not module_parts:
        return None
    return LAYERS.get(module_parts[0])


def _iter_modules() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name != "__init__.py" or path.parent != PACKAGE_ROOT
    )


def _first_party_imports(tree: ast.AST) -> list[str]:
    """Collect every ``mercury_rec.*`` module imported by this file."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if alias.name.startswith("mercury_rec"))
        # Relative imports would need resolving against the package; this
        # codebase uses absolute imports throughout, so matching on the
        # dotted name is sufficient.
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("mercury_rec")
        ):
            found.append(node.module)
    return found


@pytest.mark.parametrize("module_path", _iter_modules(), ids=lambda p: str(p.name))
def test_module_does_not_import_upward(module_path: Path) -> None:
    """No module may import from a layer above its own."""
    relative = module_path.relative_to(PACKAGE_ROOT)
    own_parts = relative.with_suffix("").parts
    own_layer = _layer_of(own_parts)
    if own_layer is None:
        pytest.skip(f"{relative} is not in a ranked layer")

    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))

    violations: list[str] = []
    for imported in _first_party_imports(tree):
        target_parts = tuple(imported.split("."))[1:]  # drop the 'mercury_rec' prefix
        target_layer = _layer_of(target_parts)
        if target_layer is None:
            continue
        if target_layer > own_layer:
            violations.append(
                f"{'.'.join(own_parts)} (layer {own_layer}) imports "
                f"{imported} (layer {target_layer})"
            )

    assert not violations, "Upward import(s) break the layering contract:\n  " + "\n  ".join(
        violations
    )


def test_every_subpackage_has_a_declared_layer() -> None:
    """A new subpackage must be assigned a layer, not silently unchecked.

    Without this, adding ``mercury_rec/streaming/`` would exempt it from the
    contract entirely and the guard above would quietly stop covering it.
    """
    subpackages = {
        path.name
        for path in PACKAGE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "__init__.py").exists()
    }
    undeclared = subpackages - set(LAYERS)
    assert not undeclared, (
        f"Subpackage(s) {sorted(undeclared)} have no layer assigned. "
        "Add them to LAYERS in this test so the contract keeps covering them."
    )
