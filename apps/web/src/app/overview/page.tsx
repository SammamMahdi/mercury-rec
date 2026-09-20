"use client";

/**
 * Overview dashboard.
 *
 * Every figure here is read from the running API or from a committed
 * evaluation artifact. Where a number is not available the tile renders an em
 * dash and says why, rather than a zero that would read as a measurement.
 */

import {
  Boxes,
  Database,
  Gauge,
  Layers,
  Target,
  Users,
  Zap,
} from "lucide-react";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  ProvenanceBadge,
  SectionHeading,
} from "@/components/primitives";
import {
  useDataset,
  useEvaluation,
  useMetricsSummary,
  useModelStatus,
  useRanking,
} from "@/lib/api/hooks";
import { formatNumber, shortHash } from "@/lib/utils";
import type { ModelResultRow } from "@/lib/api/types";

/** Pull one metric off a result row, tolerating its dynamic metric keys. */
function metric(row: ModelResultRow | undefined, key: string): number | null {
  if (!row) return null;
  const value = row[key];
  return typeof value === "number" ? value : null;
}

export default function OverviewPage() {
  const dataset = useDataset();
  const evaluation = useEvaluation();
  const ranking = useRanking();
  const metrics = useMetricsSummary();
  const models = useModelStatus();

  const rows = evaluation.data?.results ?? [];
  const best = rows.reduce<ModelResultRow | undefined>((winner, row) => {
    const current = metric(row, "ndcg@10") ?? -1;
    const incumbent = metric(winner, "ndcg@10") ?? -1;
    return current > incumbent ? row : winner;
  }, undefined);

  const ingest = dataset.data?.ingest;
  const totalStage = metrics.data?.stages?.total;

  if (dataset.isError && evaluation.isError) {
    return (
      <div className="space-y-6">
        <SectionHeading title="Overview" />
        <EmptyState
          tone="warning"
          title="The MercuryRec API is not reachable"
          description="Start the service to see live system state. Nothing on this page is hard-coded, so there is nothing to show without it."
          command="uv run uvicorn mercury_rec.api.main:app --port 8000"
        />
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <SectionHeading
          title="Overview"
          description="Live system state and the most recent committed evaluation."
          action={
            <div className="flex flex-wrap items-center gap-2">
              <ProvenanceBadge provenance={dataset.data?.source?.provenance} />
              {dataset.data?.dataset_hash ? (
                <Badge tone="neutral" title="Dataset hash: every metric is traceable to this build">
                  <Database className="size-3" aria-hidden />
                  <span className="tabular">{shortHash(dataset.data.dataset_hash)}</span>
                </Badge>
              ) : null}
            </div>
          }
        />

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <MetricTile
            label="Users"
            value={dataset.data?.n_users ?? null}
            compact
            icon={Users}
            hint="after k-core filtering"
            emptyReason="API not reachable"
          />
          <MetricTile
            label="Items"
            value={dataset.data?.n_items ?? null}
            compact
            icon={Boxes}
            hint="catalogue size"
            emptyReason="API not reachable"
          />
          <MetricTile
            label="Interactions"
            value={ingest?.final_events ?? null}
            compact
            icon={Database}
            hint={
              ingest?.raw_events
                ? `${formatNumber(ingest.raw_events, { compact: true })} raw, ${formatNumber(
                    (ingest.retained_event_fraction ?? 0) * 100,
                    { decimals: 1 },
                  )}% retained`
                : undefined
            }
            emptyReason="API not reachable"
          />
          <MetricTile
            label="Models loaded"
            value={models.data ? models.data.retrieval.length + (models.data.ranking ? 1 : 0) : null}
            icon={Layers}
            hint={models.data?.ranking ? "retrieval + ranker" : "retrieval only"}
            emptyReason="API not reachable"
          />
        </div>
      </div>

      <div>
        <SectionHeading
          title="Model quality"
          description={
            evaluation.data
              ? `Best offline result on the ${evaluation.data.split} split, ${formatNumber(
                  evaluation.data.users_scored,
                )} scorable users.`
              : "Offline evaluation, committed to the repository."
          }
        />
        {evaluation.isLoading ? (
          <PanelSkeleton rows={2} />
        ) : evaluation.isError ? (
          <EmptyState
            title="No evaluation results yet"
            description="Model metrics appear once an evaluation has been run and committed."
            command="uv run mercury train baselines"
          />
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="NDCG@10"
              value={metric(best, "ndcg@10")}
              decimals={4}
              icon={Target}
              hint={best?.model ? `best: ${best.model}` : undefined}
              tone="positive"
            />
            <MetricTile
              label="Recall@10"
              value={metric(best, "recall@10")}
              decimals={4}
              hint={best?.model}
            />
            <MetricTile
              label="Catalogue coverage"
              value={best?.catalog_coverage ?? null}
              percent
              decimals={1}
              hint="share of items ever recommended"
            />
            <MetricTile
              label="Retrieval ceiling"
              value={ranking.data?.retrieval.mean_recall_at_candidates_test ?? null}
              decimals={4}
              hint="recall of the candidate pool"
              emptyReason="run the ranker pipeline"
            />
          </div>
        )}
      </div>

      <div>
        <SectionHeading
          title="Live serving"
          description="Measured from this API process. Percentiles appear once it has served traffic."
        />
        {metrics.data?.has_data ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Total p50"
              value={totalStage?.p50_ms ?? null}
              unit="ms"
              decimals={1}
              icon={Zap}
            />
            <MetricTile
              label="Total p95"
              value={totalStage?.p95_ms ?? null}
              unit="ms"
              decimals={1}
              tone="warning"
            />
            <MetricTile
              label="Total p99"
              value={totalStage?.p99_ms ?? null}
              unit="ms"
              decimals={1}
              tone="warning"
            />
            <MetricTile
              label="Cache hit rate"
              value={metrics.data.cache.hit_rate}
              percent
              decimals={1}
              icon={Gauge}
              hint={`${formatNumber(metrics.data.cache.hits + metrics.data.cache.stale_hits)} hits / ${formatNumber(
                metrics.data.cache.misses,
              )} misses`}
            />
          </div>
        ) : (
          <EmptyState
            title="No requests recorded in this process yet"
            description={
              metrics.data?.detail ??
              "Latency percentiles become available once the API has served traffic. Open the Explorer to generate some."
            }
          />
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <GlassPanel>
          <SectionHeading title="Dataset provenance" />
          {dataset.data ? (
            <div className="space-y-0">
              <DataRow label="Source" value={dataset.data.source?.dataset ?? "—"} mono={false} />
              <DataRow label="Licence" value={dataset.data.source?.license ?? "—"} mono={false} />
              <DataRow
                label="Window"
                value={
                  ingest?.first_event && ingest?.last_event
                    ? `${ingest.first_event.slice(0, 10)} → ${ingest.last_event.slice(0, 10)}`
                    : "—"
                }
              />
              <DataRow
                label="k-core iterations"
                value={formatNumber(ingest?.kcore_iterations ?? null)}
                title="Applied iteratively to a fixed point: removing a sparse user can push an item below threshold."
              />
              <DataRow label="Model version" value={dataset.data.model_version} />
              {dataset.data.columns?.synthesised?.length ? (
                <p className="pt-3 text-xs leading-relaxed text-[var(--color-ink-faint)]">
                  <span className="font-medium text-[var(--color-ink-muted)]">
                    Synthesised columns:
                  </span>{" "}
                  {dataset.data.columns.synthesised.join(", ")}. Behaviour, timestamps,
                  item ids and categories are real.
                </p>
              ) : null}
            </div>
          ) : (
            <PanelSkeleton rows={4} />
          )}
        </GlassPanel>

        <GlassPanel>
          <SectionHeading title="Loaded models" />
          {models.data ? (
            <div className="space-y-0">
              {models.data.retrieval.map((model) => (
                <DataRow
                  key={model.name}
                  label={model.name}
                  value={<Badge tone="positive">retrieval</Badge>}
                  mono={false}
                />
              ))}
              <DataRow
                label="lambdarank"
                value={
                  models.data.ranking ? (
                    <Badge tone="mercury">ranking</Badge>
                  ) : (
                    <Badge tone="warning">not loaded</Badge>
                  )
                }
                mono={false}
              />
              <DataRow
                label="Feature schema"
                value={`v${models.data.feature_schema_version}`}
                title="Checked on load: serving a model features from a different schema fails silently."
              />
            </div>
          ) : (
            <PanelSkeleton rows={4} />
          )}
        </GlassPanel>
      </div>

      {evaluation.data ? (
        <p className="text-xs leading-relaxed text-[var(--color-ink-faint)]">
          Offline metrics were produced by{" "}
          <code className="tabular text-[var(--color-mercury)]">mercury train baselines</code> on{" "}
          {new Date(evaluation.data.generated_at).toLocaleDateString()} against dataset{" "}
          <span className="tabular">{shortHash(evaluation.data.dataset_hash)}</span>. They
          describe the active core of the catalogue after k-core filtering, and are not
          comparable to published session-based figures on this dataset.
        </p>
      ) : null}
    </div>
  );
}
