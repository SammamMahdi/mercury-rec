"use client";

/**
 * Pipeline inspector.
 *
 * The request path, stage by stage, with the numbers each stage produced on a
 * real request rather than a diagram's idea of them. Two things are shown side
 * by side deliberately:
 *
 * - **Per-stage latency** from the response just received, which is what the
 *   architecture costs.
 * - **Retrieval recall**, read from the committed evaluation artifact, which
 *   is the ceiling everything downstream is measured against. An item no
 *   source proposes cannot be ranked, re-ranked or recommended, so this number
 *   decides whether effort belongs in retrieval or in ranking.
 */

import { ArrowDown, Clock, Filter, Layers, Target } from "lucide-react";
import { Suspense, useCallback, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import {
  usePipelineTrace,
  useRanking,
  useSampleUsers,
  useServingConfig,
} from "@/lib/api/hooks";
import { PIPELINE_STAGES, RETRIEVAL_SOURCES } from "@/lib/constants";
import { cn, formatNumber } from "@/lib/utils";
import type { PipelineTrace, StageLatency } from "@/lib/api/types";

/* -------------------------------------------------------------------------- */

/**
 * One stage of the funnel.
 *
 * The width of the bar is the stage's share of the candidates that entered the
 * pipeline, so the narrowing is proportional to what actually happened rather
 * than chosen to look like a funnel.
 */
function Stage({
  index,
  label,
  description,
  count,
  widthPct,
  ms,
  detail,
}: {
  index: number;
  label: string;
  description: string;
  count: number | null;
  widthPct: number;
  ms: number | null;
  detail?: React.ReactNode;
}) {
  return (
    <div className="relative">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:gap-4">
        <div className="flex w-full shrink-0 items-center gap-3 sm:w-52">
          <span className="tabular flex size-6 shrink-0 items-center justify-center rounded-full border border-[var(--color-hairline)] text-[10px] text-[var(--color-ink-faint)]">
            {index}
          </span>
          <span className="text-sm font-medium">{label}</span>
        </div>

        <div className="min-w-0 flex-1 space-y-2">
          <div className="flex items-center gap-3">
            <div className="h-7 flex-1 overflow-hidden rounded-[var(--radius-control)] bg-[var(--color-surface-1)]">
              <div
                className="flex h-full items-center rounded-[var(--radius-control)] px-2.5 transition-[width] duration-500"
                style={{
                  width: `${Math.max(widthPct, count === null ? 0 : 4)}%`,
                  background:
                    "linear-gradient(90deg, var(--color-mercury-dim), color-mix(in oklch, var(--color-mercury-dim) 45%, transparent))",
                }}
              >
                <span className="tabular whitespace-nowrap text-xs font-semibold text-[var(--color-void)]">
                  {count === null ? "—" : formatNumber(count)}
                </span>
              </div>
            </div>
            <span className="tabular w-16 shrink-0 text-right text-xs text-[var(--color-ink-muted)]">
              {ms === null ? "—" : `${ms.toFixed(2)} ms`}
            </span>
          </div>

          <p className="text-xs leading-relaxed text-[var(--color-ink-faint)]">{description}</p>
          {detail}
        </div>
      </div>

      <div className="flex justify-center py-1.5 sm:pl-52">
        <ArrowDown className="size-3 text-[var(--color-hairline-strong)]" aria-hidden />
      </div>
    </div>
  );
}

/** Per-source contribution, with overlap made visible. */
function SourceBreakdown({ trace }: { trace: PipelineTrace }) {
  const total = trace.candidate_ids.length;
  const proposed = Object.values(trace.candidate_sources).reduce(
    (sum, ids) => sum + ids.length,
    0,
  );

  return (
    <div className="space-y-1.5 pt-1">
      {Object.entries(trace.candidate_sources).map(([source, ids]) => (
        <div key={source} className="flex items-center gap-2 text-xs">
          <span
            className="w-36 shrink-0 truncate text-[var(--color-ink-muted)]"
            title={RETRIEVAL_SOURCES[source]?.description ?? source}
          >
            {RETRIEVAL_SOURCES[source]?.label ?? source}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
            <div
              className="h-full rounded-full bg-[var(--color-mercury)]"
              style={{ width: `${(ids.length / Math.max(total, 1)) * 100}%` }}
            />
          </div>
          <span className="tabular w-10 shrink-0 text-right text-[var(--color-ink-faint)]">
            {ids.length}
          </span>
        </div>
      ))}
      {proposed > total ? (
        <p className="pt-1 text-xs text-[var(--color-ink-faint)]">
          The sources proposed {formatNumber(proposed)} entries for{" "}
          {formatNumber(total)} distinct candidates, so{" "}
          {formatNumber(proposed - total)} were proposed by more than one source.
          Reciprocal rank fusion rewards exactly that agreement.
        </p>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function PipelineInner() {
  const router = useRouter();
  const params = useSearchParams();
  const userId = params.get("user") ?? "";

  const setUser = useCallback(
    (value: string | null) => {
      const next = new URLSearchParams(params.toString());
      if (value === null) next.delete("user");
      else next.set("user", value);
      router.replace(`/pipeline?${next.toString()}`, { scroll: false });
    },
    [params, router],
  );

  const samples = useSampleUsers(6);
  const traceQuery = usePipelineTrace(userId ? { userId, k: 10 } : null);
  const ranking = useRanking();
  const config = useServingConfig();

  const trace = traceQuery.data;
  const latency: StageLatency | null = trace?.latency ?? null;

  const filteredTotal = useMemo(
    () => Object.values(trace?.filtered ?? {}).reduce((sum, value) => sum + value, 0),
    [trace],
  );

  const entering = trace?.candidate_ids.length ?? 0;
  const pct = (count: number) => (entering > 0 ? (count / entering) * 100 : 0);

  const stages = trace
    ? [
        {
          ...PIPELINE_STAGES[0],
          count: null as number | null,
          widthPct: 100,
          ms: latency?.cache_lookup_ms ?? null,
          detail: (
            <p className="text-xs text-[var(--color-ink-faint)]">
              Bypassed: {trace.note}
            </p>
          ),
        },
        {
          ...PIPELINE_STAGES[1],
          count: null as number | null,
          widthPct: 100,
          ms: latency?.feature_lookup_ms ?? null,
          detail: undefined,
        },
        {
          ...PIPELINE_STAGES[2],
          count: trace.candidate_ids.length,
          widthPct: 100,
          ms: latency?.candidate_generation_ms ?? null,
          detail: <SourceBreakdown trace={trace} />,
        },
        {
          ...PIPELINE_STAGES[3],
          count: trace.ranked_ids.length,
          widthPct: pct(trace.ranked_ids.length),
          ms: latency?.ranking_ms ?? null,
          detail: (
            <p className="text-xs text-[var(--color-ink-faint)]">
              Ranking reorders the pool, it does not shrink it. The narrowing
              happens next.
            </p>
          ),
        },
        {
          ...PIPELINE_STAGES[4],
          count: trace.final_ids.length,
          widthPct: Math.max(pct(trace.final_ids.length), 3),
          ms: latency?.reranking_ms ?? null,
          detail:
            filteredTotal > 0 ? (
              <div className="flex flex-wrap gap-1.5 pt-1">
                {Object.entries(trace.filtered)
                  .filter(([, count]) => count > 0)
                  .map(([reason, count]) => (
                    <Badge key={reason} tone="warning">
                      <Filter className="size-3" aria-hidden />
                      {reason.replace(/_/g, " ")}: {count}
                    </Badge>
                  ))}
              </div>
            ) : undefined,
        },
      ]
    : [];

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Pipeline inspector"
        description="One request, stage by stage. Counts and timings come from the response; nothing here is illustrative."
      />

      <GlassPanel className="space-y-3">
        <span className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
          Trace a request
        </span>
        {samples.data?.users?.length ? (
          <div className="flex flex-wrap gap-2">
            {samples.data.users.map((user) => (
              <button
                key={user.user_id}
                type="button"
                onClick={() => setUser(user.user_id)}
                title={`${user.history_items} items in training history`}
                className={cn(
                  "tabular rounded-[var(--radius-control)] border px-3 py-1.5 text-xs transition-colors",
                  userId === user.user_id
                    ? "border-[var(--color-mercury)] bg-[color-mix(in_oklch,var(--color-mercury)_14%,transparent)] text-[var(--color-mercury)]"
                    : "border-[var(--color-hairline)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
                )}
              >
                {user.user_id}
              </button>
            ))}
          </div>
        ) : samples.isLoading ? (
          <PanelSkeleton rows={1} />
        ) : (
          <p className="text-xs text-[var(--color-ink-faint)]">
            No sample users available; the API is not reachable.
          </p>
        )}
      </GlassPanel>

      {!userId ? (
        <EmptyState
          title="Pick a user to trace"
          description="Every number on this page comes from a real, uncached request. There is nothing to show until one has been made."
        />
      ) : traceQuery.isLoading ? (
        <PanelSkeleton rows={8} />
      ) : traceQuery.isError ? (
        <EmptyState
          tone="warning"
          title="The trace could not be fetched"
          description="The API is not reachable."
          command="uv run uvicorn mercury_rec.api.main:app --port 8000"
        />
      ) : trace?.is_cold_start ? (
        <EmptyState
          title="This request took the cold-start path"
          description="With no usable history the engine answers from contextual popularity without running retrieval, so there are no stages to inspect. This is a real code path, not an error."
        />
      ) : trace ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Candidates"
              value={trace.candidate_ids.length}
              icon={Layers}
              hint="after fusion and history exclusion"
            />
            <MetricTile
              label="Returned"
              value={trace.final_ids.length}
              hint={`${formatNumber(
                (1 - trace.final_ids.length / Math.max(trace.candidate_ids.length, 1)) * 100,
                { decimals: 1 },
              )}% discarded`}
            />
            <MetricTile
              label="Total latency"
              value={latency?.total_ms ?? null}
              unit="ms"
              decimals={2}
              icon={Clock}
              tone="positive"
            />
            <MetricTile
              label="Filtered out"
              value={filteredTotal}
              icon={Filter}
              hint="hard rules, before diversity caps"
            />
          </div>

          <GlassPanel>
            <SectionHeading
              title="The request path"
              description="Bar width is each stage's share of the candidate pool."
            />
            <div className="space-y-1">
              {stages.map((stage, index) => (
                <Stage
                  key={stage.key}
                  index={index + 1}
                  label={stage.label}
                  description={stage.description}
                  count={stage.count}
                  widthPct={stage.widthPct}
                  ms={stage.ms}
                  detail={stage.detail}
                />
              ))}
            </div>
            <div className="flex items-center gap-3 pt-1 sm:pl-52">
              <Badge tone="mercury">
                {formatNumber(trace.final_ids.length)} recommendations
              </Badge>
              <span className="tabular text-xs text-[var(--color-ink-faint)]">
                {latency ? `${latency.total_ms.toFixed(2)} ms total` : null}
              </span>
            </div>
          </GlassPanel>
        </>
      ) : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <GlassPanel>
          <SectionHeading
            title="Retrieval is the ceiling"
            description="Measured offline over the evaluation split, not on this request."
          />
          {ranking.data ? (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <MetricTile
                  label="Recall at candidates"
                  value={ranking.data.retrieval.mean_recall_at_candidates_test}
                  decimals={4}
                  icon={Target}
                  hint="test split"
                  tone="positive"
                />
                <MetricTile
                  label="Recall at candidates"
                  value={ranking.data.retrieval.mean_recall_at_candidates_train}
                  decimals={4}
                  hint="train split"
                />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-[var(--color-ink-muted)]">
                {ranking.data.retrieval.note}
              </p>
              <div className="mt-3 space-y-0">
                <DataRow
                  label="Per-source k"
                  value={formatNumber(ranking.data.retrieval.per_source_k)}
                />
                <DataRow
                  label="Max candidates"
                  value={formatNumber(ranking.data.retrieval.max_candidates)}
                />
              </div>
            </>
          ) : ranking.isLoading ? (
            <PanelSkeleton rows={3} />
          ) : (
            <EmptyState
              title="No ranking results yet"
              description="The retrieval ceiling is reported once the ranking pipeline has run."
              command="uv run mercury train ranker"
            />
          )}
        </GlassPanel>

        <GlassPanel>
          <SectionHeading
            title="Effective configuration"
            description="What this process is actually running, not what a config file says."
          />
          {config.data ? (
            <div className="space-y-0">
              <DataRow label="Per-source k" value={formatNumber(config.data.per_source_k)} />
              <DataRow label="Max candidates" value={formatNumber(config.data.max_candidates)} />
              <DataRow
                label="Max per merchant"
                value={formatNumber(config.data.rerank.max_per_merchant)}
                title="Diversity cap. Prevents one merchant owning a slate."
              />
              <DataRow
                label="Max per category"
                value={formatNumber(config.data.rerank.max_per_category)}
              />
              <DataRow
                label="Availability enforced"
                value={config.data.rerank.enforce_availability ? "yes" : "no"}
                mono={false}
              />
              <DataRow
                label="Cache TTL"
                value={
                  config.data.cache.enabled
                    ? `${config.data.cache.ttl_seconds}s (soft ${config.data.cache.soft_ttl_seconds}s)`
                    : "disabled"
                }
              />
            </div>
          ) : (
            <PanelSkeleton rows={5} />
          )}
        </GlassPanel>
      </div>
    </div>
  );
}

export default function PipelinePage() {
  // useSearchParams requires a Suspense boundary in the App Router.
  return (
    <Suspense
      fallback={
        <div className="space-y-6">
          <SectionHeading title="Pipeline inspector" />
          <PanelSkeleton rows={6} />
        </div>
      }
    >
      <PipelineInner />
    </Suspense>
  );
}
