"""``mercury`` - the command-line entry point for MercuryRec.

This is the canonical interface. A ``Makefile`` wraps it for Linux, CI and
container use, but ``make`` is not available on Windows, so anything the
documentation promises must work through this CLI first.

Commands are grouped by lifecycle stage::

    mercury env check              verify the development environment
    mercury data download          fetch the Retailrocket dataset
    mercury data build             raw -> validated, split parquet
    mercury train <model>          train a single model
    mercury evaluate               offline evaluation across models
    mercury serve                  run the API

Every command that consumes configuration reads it from ``configs/*.yaml``;
none require editing source to change a hyperparameter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from mercury_rec import __version__
from mercury_rec.config.settings import get_settings
from mercury_rec.core.logging import configure_logging

console = Console()

app = typer.Typer(
    name="mercury",
    help="MercuryRec - multi-stage recommendation and personalization platform.",
    no_args_is_help=True,
    add_completion=False,
)

env_app = typer.Typer(name="env", help="Environment diagnostics.", no_args_is_help=True)
data_app = typer.Typer(
    name="data", help="Dataset acquisition and preparation.", no_args_is_help=True
)
features_app = typer.Typer(name="features", help="Feature engineering.", no_args_is_help=True)
train_app = typer.Typer(name="train", help="Model training and evaluation.", no_args_is_help=True)

app.add_typer(env_app)
app.add_typer(data_app)
app.add_typer(features_app)
app.add_typer(train_app)

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"mercury-rec {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug-level logging.")] = False,
    json_logs: Annotated[bool, typer.Option("--json-logs", help="Emit JSON log lines.")] = False,
) -> None:
    """Configure logging before any subcommand runs."""
    configure_logging(level="DEBUG" if verbose else "INFO", json_output=json_logs)


def _run_script(name: str) -> None:
    """Execute a checked script in-process-adjacent and propagate its exit code.

    Run as a subprocess rather than imported so that a hard failure (a CUDA
    abort, a DLL load error) cannot take down the CLI itself, and so the
    script stays independently runnable.
    """
    script = _SCRIPTS / name
    if not script.is_file():
        console.print(f"[red]Missing script:[/red] {script}")
        raise typer.Exit(code=1)
    result = subprocess.run([sys.executable, str(script)], check=False)  # noqa: S603
    raise typer.Exit(code=result.returncode)


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------


@env_app.command("check")
def env_check() -> None:
    """Verify git identity, native libraries and thread configuration."""
    _run_script("check_env.py")


@env_app.command("gpu")
def env_gpu() -> None:
    """Verify the CUDA stack launches kernels and returns correct results."""
    _run_script("check_gpu.py")


@env_app.command("info")
def env_info() -> None:
    """Print the resolved runtime configuration (secrets redacted)."""
    from mercury_rec import NUM_THREADS

    settings = get_settings()
    console.print(f"[bold]mercury-rec[/bold] {__version__}")
    console.print(f"  environment   {settings.environment}")
    console.print(f"  project root  {settings.project_root}")
    console.print(f"  data dir      {settings.data_dir}")
    console.print(f"  artifacts     {settings.artifacts_dir}")
    console.print(f"  seed          {settings.random_seed}")
    console.print(f"  threads       {NUM_THREADS}")
    console.print(f"  postgres      {'configured' if settings.postgres_dsn else 'not set'}")
    console.print(f"  redis         {'configured' if settings.redis_dsn else 'not set'}")
    console.print(f"  cache         {'enabled' if settings.cache_enabled else 'disabled'}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@data_app.command("download")
def data_download(
    force: Annotated[bool, typer.Option("--force", help="Re-download even if present.")] = False,
) -> None:
    """Fetch the Retailrocket dataset from Kaggle into ``data/raw``.

    The dataset is CC BY-NC-SA 4.0 and is not redistributed with this
    repository; this command retrieves it from the original source.
    """
    from mercury_rec.data.download import (
        DatasetDownloadError,
        KaggleAuthError,
        download_retailrocket,
    )

    settings = get_settings()
    try:
        files = download_retailrocket(
            settings.data_dir / "raw",
            dataset=settings.kaggle_dataset,
            force=force,
        )
    except (KaggleAuthError, DatasetDownloadError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Dataset ready[/green] - {len(files)} files in {settings.data_dir / 'raw'}"
    )
    for name, path in files.items():
        console.print(f"  {name:<28} {path.stat().st_size / 1024**2:>8.1f} MB")


@data_app.command("build")
def data_build(
    preset: Annotated[
        str, typer.Option("--preset", "-p", help="Dataset preset: 'demo' or 'full'.")
    ] = "full",
    force_download: Annotated[
        bool, typer.Option("--force-download", help="Re-fetch the raw dataset first.")
    ] = False,
) -> None:
    """Build the processed dataset: ingest, validate, augment, sessionise, split."""
    from mercury_rec.pipelines.build_data import build_dataset

    result = build_dataset(preset=preset, force_download=force_download)
    meta = result.metadata
    ingest_stats = meta["ingest"]
    split_stats = meta["split"]

    console.print(f"\n[green]Dataset built[/green] in {result.elapsed_seconds:.1f}s")
    console.print(f"  output        {result.output_dir}")
    console.print(f"  dataset hash  {meta['dataset_hash']}")
    console.print(
        f"  events        {ingest_stats['raw_events']:,} raw "
        f"-> {ingest_stats['final_events']:,} after k-core "
        f"({ingest_stats['retained_event_fraction']:.1%} retained)"
    )
    console.print(
        f"  users/items   {ingest_stats['final_users']:,} users, "
        f"{ingest_stats['final_items']:,} items"
    )
    console.print(
        f"  split         train {split_stats['train_events']:,} / "
        f"val {split_stats['validation_events']:,} / test {split_stats['test_events']:,}"
    )


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


@features_app.command("build")
def features_build(
    preset: Annotated[
        str, typer.Option("--preset", "-p", help="Dataset preset: 'demo' or 'full'.")
    ] = "full",
) -> None:
    """Compute leakage-free as-of features for every split."""
    from mercury_rec.pipelines.build_features import build_features

    result = build_features(preset=preset)
    console.print(f"[green]Features built[/green] in {result.elapsed_seconds:.1f}s")
    console.print(f"  output   {result.output_dir}")
    for split, rows in result.rows_per_split.items():
        console.print(f"  {split:<12} {rows:>9,} rows")
    console.print(f"  features {len(result.metadata['feature_columns'])} columns")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@train_app.command("baselines")
def train_baselines(
    preset: Annotated[str, typer.Option("--preset", "-p", help="Dataset preset.")] = "full",
    split: Annotated[str, typer.Option("--split", help="Evaluation split.")] = "test",
    max_users: Annotated[
        int | None, typer.Option("--max-users", help="Sample N evaluation users.")
    ] = None,
    quick: Annotated[
        bool, typer.Option("--quick", help="Shorten neural training (smoke test).")
    ] = False,
) -> None:
    """Fit and evaluate every retrieval model under one protocol."""
    from mercury_rec.pipelines.train_baselines import format_table, run_baselines

    result = run_baselines(preset=preset, split=split, max_users=max_users, quick=quick)
    payload = result.payload

    console.print(
        f"\n[bold]Model comparison[/bold]  preset={payload['preset']} split={payload['split']}  "
        f"users_scored={payload['users_scored']:,}  dataset={payload['dataset_hash']}"
    )
    console.print(format_table(payload, k=10))
    console.print(f"\n[green]Saved[/green] {result.output_path}")


@train_app.command("ranker")
def train_ranker_command(
    preset: Annotated[str, typer.Option("--preset", "-p", help="Dataset preset.")] = "full",
    quick: Annotated[bool, typer.Option("--quick", help="Short training (smoke test).")] = False,
    max_train_users: Annotated[
        int, typer.Option("--max-train-users", help="Cap users used to build the ranker set.")
    ] = 4000,
) -> None:
    """Train the ranker on real retrieval output and score the full pipeline."""
    from mercury_rec.pipelines.train_ranker import run_ranker_pipeline

    result = run_ranker_pipeline(preset=preset, quick=quick, max_train_users=max_train_users)
    payload = result.payload

    retrieval = payload["retrieval"]
    console.print(
        f"[bold]Retrieval ceiling[/bold]  recall@{retrieval['max_candidates']} candidates = "
        f"{retrieval['mean_recall_at_candidates_test']:.4f}"
    )
    console.print(
        f"[bold]Ranker[/bold]  {payload['ranker']['train_groups']:,} groups, "
        f"best_iteration={payload['ranker']['best_iteration']}"
    )
    console.print()
    header = f"{'stage':<28}{'R@10':>9}{'NDCG@10':>10}{'MAP@10':>9}{'HR@10':>9}{'cov':>8}"
    console.print(header)
    console.print("-" * len(header))
    for name, metrics in payload["stages"].items():
        console.print(
            f"{name:<28}{metrics['recall@10']:>9.4f}{metrics['ndcg@10']:>10.4f}"
            f"{metrics['map@10']:>9.4f}{metrics['hit_rate@10']:>9.4f}"
            f"{metrics['catalog_coverage']:>8.3f}"
        )
    console.print(f"[green]Saved[/green] {result.output_path}")


@train_app.command("experiment")
def run_experiment_command(
    preset: Annotated[str, typer.Option("--preset", "-p")] = "full",
    control: Annotated[str, typer.Option("--control", help="Baseline variant.")] = "popularity",
    treatment: Annotated[str, typer.Option("--treatment", help="Candidate variant.")] = "bpr_mf",
    no_track: Annotated[bool, typer.Option("--no-track", help="Skip MLflow logging.")] = False,
) -> None:
    """Run the offline A/B simulation and the promotion gate."""
    from mercury_rec.pipelines.run_experiment import run_experiment

    result = run_experiment(preset=preset, control=control, treatment=treatment, track=not no_track)
    sim = result.payload["simulation"]

    console.print(
        f"[bold yellow]OFFLINE SIMULATION[/bold yellow] - no live users. "
        f"{sim['control']} vs {sim['treatment']}, n={sim['n_users']:,}"
    )
    header = (
        f"{'metric':<16}{'control':>10}{'treatment':>11}{'lift':>10}{'95% CI':>22}{'signif':>8}"
    )
    console.print(header)
    console.print("-" * len(header))
    for m in sim["metrics"]:
        ci = f"[{m['ci_95'][0]:+.4f}, {m['ci_95'][1]:+.4f}]"
        console.print(
            f"{m['metric']:<16}{m['control']:>10.4f}{m['treatment']:>11.4f}"
            f"{m['relative_lift_pct']:>9.1f}%{ci:>22}{'yes' if m['is_significant'] else 'no':>8}"
        )

    gate = result.payload.get("gate") or {}
    if gate:
        console.print()
        verdict = "[green]PROMOTE[/green]" if gate["promoted"] else "[red]REJECT[/red]"
        console.print(f"Gate: {verdict}")
        for check in gate["checks"]:
            mark = {"pass": "ok  ", "fail": "FAIL", "inconclusive": "????"}[check["verdict"]]
            console.print(f"  {mark} {check['name']:<28} {check['detail']}")
    console.print(f"[green]Saved[/green] {result.output_path}")


if __name__ == "__main__":
    app()
