"""Write the API's OpenAPI schema to a file the frontend can build against.

Imports the application rather than querying a running server, so this works
in CI with nothing started, and cannot report the schema of a stale process
that happens to be listening on port 8000.

The output is committed. CI regenerates it and fails on a diff, which turns
"the backend changed shape and the frontend still compiles" into a failing
build instead of a runtime surprise.

    uv run python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "apps" / "web" / "openapi.json"


def export(output: Path = DEFAULT_OUTPUT) -> Path:
    """Generate the schema and write it, returning where it landed."""
    # The schema is a property of the routes and models, not of whether any
    # artifact happens to exist on this machine. Loading the bundle here would
    # make schema generation fail on a clean checkout for no reason.
    os.environ.setdefault("MERCURY_LOAD_ARTIFACTS_ON_STARTUP", "false")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from mercury_rec.api.main import create_app

    schema = create_app().openapi()

    output.parent.mkdir(parents=True, exist_ok=True)
    # Sorted keys and a trailing newline: without both, the file reorders
    # itself between runs and every regeneration looks like a change.
    output.write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


if __name__ == "__main__":
    destination = export(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT)
    print(f"Wrote {destination.relative_to(REPO_ROOT)}")
