"""Reproducible acquisition of the Retailrocket ecommerce dataset.

The dataset is **not** vendored into this repository. It is licensed
CC BY-NC-SA 4.0 (Attribution / NonCommercial / ShareAlike), which encumbers
derived artifacts as well as the raw files, so both raw and processed data are
git-ignored and fetched on demand instead. See ``docs/data-card.md``.

This talks to the Kaggle REST API directly with ``httpx`` rather than taking a
dependency on the ``kaggle`` package, for three reasons: the official client
calls ``sys.exit()`` on auth failure (unusable as a library), it insists on
specific file permissions that do not exist on Windows, and it would add a
dependency used for exactly one call.

Credentials are read from ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME``
/ ``KAGGLE_KEY`` environment variables. They are never logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

KAGGLE_API_BASE: Final = "https://www.kaggle.com/api/v1"

#: Files this project consumes, with their published names.
DATASET_FILES: Final = (
    "events.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
    "category_tree.csv",
)

_CHUNK_BYTES: Final = 1 << 20  # 1 MiB
_DOWNLOAD_TIMEOUT: Final = httpx.Timeout(30.0, read=300.0)


class KaggleAuthError(RuntimeError):
    """Raised when Kaggle credentials are absent or rejected."""


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset could not be retrieved or is malformed."""


@dataclass(frozen=True, slots=True)
class KaggleCredentials:
    """A Kaggle API username/key pair."""

    username: str
    key: str

    def __repr__(self) -> str:
        """Never render the key, so credentials cannot leak via a traceback."""
        return f"KaggleCredentials(username={self.username!r}, key='***')"


def load_credentials() -> KaggleCredentials:
    """Resolve Kaggle credentials from the environment or ``kaggle.json``.

    Environment variables win, so CI can inject secrets without writing a file.
    """
    env_user = os.environ.get("KAGGLE_USERNAME")
    env_key = os.environ.get("KAGGLE_KEY")
    if env_user and env_key:
        logger.debug("kaggle.credentials.source", source="environment")
        return KaggleCredentials(username=env_user, key=env_key)

    config_path = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"

    if not config_path.is_file():
        raise KaggleAuthError(
            "No Kaggle credentials found.\n"
            f"  Looked for: {config_path}\n"
            "  Fix: create an API token at https://www.kaggle.com/settings "
            "(Account -> API -> Create New Token) and save it there, or export "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        credentials = KaggleCredentials(username=payload["username"], key=payload["key"])
    except (json.JSONDecodeError, KeyError) as exc:
        raise KaggleAuthError(
            f"{config_path} is not a valid Kaggle token file "
            '(expected JSON with "username" and "key").'
        ) from exc

    logger.debug("kaggle.credentials.source", source=str(config_path))
    return credentials


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(
    client: httpx.Client,
    dataset: str,
    destination: Path,
) -> Path:
    """Stream the dataset archive to disk.

    Streamed in chunks rather than held in memory: the archive is ~290 MB and
    this project runs on a memory-constrained machine where an avoidable
    300 MB allocation is a real cost.
    """
    url = f"{KAGGLE_API_BASE}/datasets/download/{dataset}"
    archive = destination / "retailrocket.zip"
    partial = archive.with_suffix(".zip.part")

    logger.info("dataset.download.start", dataset=dataset)

    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code in (401, 403):
                raise KaggleAuthError(
                    "Kaggle rejected the credentials (HTTP "
                    f"{response.status_code}). Confirm the token is current and "
                    "that you have accepted the dataset's terms at "
                    f"https://www.kaggle.com/datasets/{dataset}"
                )
            response.raise_for_status()

            total = int(response.headers.get("content-length", 0))
            written = 0
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    handle.write(chunk)
                    written += len(chunk)
            logger.info("dataset.download.complete", bytes=written, expected=total or None)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DatasetDownloadError(f"Failed to download {dataset}: {exc}") from exc

    partial.replace(archive)
    return archive


def _extract(archive: Path, destination: Path) -> list[Path]:
    """Extract expected members, refusing any path that escapes the target.

    A zip entry may contain ``../`` or an absolute path; extracting one blindly
    writes outside the destination directory. The archive here is trusted, but
    validating is cheap and this code path handles a remote file.
    """
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = set(bundle.namelist())
            missing = [name for name in DATASET_FILES if name not in members]
            if missing:
                raise DatasetDownloadError(
                    f"Archive is missing expected files: {missing}. "
                    f"It contains: {sorted(members)}. The upstream dataset layout "
                    "may have changed."
                )

            for name in DATASET_FILES:
                target = (destination / name).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise DatasetDownloadError(f"Refusing unsafe archive path: {name}")
                with bundle.open(name) as source, target.open("wb") as sink:
                    while chunk := source.read(_CHUNK_BYTES):
                        sink.write(chunk)
                extracted.append(target)
                logger.info(
                    "dataset.extract",
                    file=name,
                    size_mb=round(target.stat().st_size / 1024**2, 1),
                )
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(
            f"{archive} is not a valid zip archive. Delete it and retry."
        ) from exc

    return extracted


def download_retailrocket(
    raw_dir: Path,
    *,
    dataset: str = "retailrocket/ecommerce-dataset",
    force: bool = False,
) -> dict[str, Path]:
    """Ensure the Retailrocket CSVs are present under ``raw_dir``.

    Idempotent: if every expected file already exists this returns immediately
    without touching the network, so re-running the pipeline is cheap.

    Args:
        raw_dir: Directory to populate (typically ``data/raw``).
        dataset: Kaggle dataset slug.
        force: Re-download even when the files are already present.

    Returns:
        Mapping of file name to its path on disk.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    present = {name: raw_dir / name for name in DATASET_FILES}

    if not force and all(path.is_file() for path in present.values()):
        logger.info("dataset.cached", directory=str(raw_dir))
        return present

    credentials = load_credentials()
    with httpx.Client(
        auth=(credentials.username, credentials.key),
        timeout=_DOWNLOAD_TIMEOUT,
        headers={"User-Agent": "mercury-rec/0.1"},
    ) as client:
        archive = _download_archive(client, dataset, raw_dir)

    _extract(archive, raw_dir)
    archive.unlink(missing_ok=True)

    # A checksum manifest makes "which data produced this metric?" answerable
    # later, and is what the dataset hash logged to MLflow is derived from.
    manifest = {
        "dataset": dataset,
        "license": "CC BY-NC-SA 4.0",
        "source_url": f"https://www.kaggle.com/datasets/{dataset}",
        "files": {
            name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in present.items()
        },
    }
    (raw_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("dataset.ready", directory=str(raw_dir), files=len(present))
    return present


__all__ = [
    "DATASET_FILES",
    "DatasetDownloadError",
    "KaggleAuthError",
    "KaggleCredentials",
    "download_retailrocket",
    "load_credentials",
]
