"use client";

/**
 * Live Recommendation Explorer.
 *
 * Every list on this page is a real API response. Changing a context control
 * re-issues the request and the ranking changes — which is what makes the
 * contextual-personalisation claim demonstrable rather than asserted.
 *
 * Context lives in the URL, so any configuration is shareable and
 * reproducible: someone can send a link to the exact state they are looking at.
 */

import { Clock, Layers, MapPin, RefreshCw, Sparkles, User } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useMemo, useState } from "react";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useRecommendations } from "@/lib/api/hooks";
import type { Recommendation, RecommendationResponse } from "@/lib/api/types";
import {
  HOUR_BUCKETS,
  RETRIEVAL_SOURCES,
  VERTICALS,
  featureLabel,
  verticalMeta,
} from "@/lib/constants";
import { cn, formatMs, formatNumber } from "@/lib/utils";

/* -------------------------------------------------------------------------- */

function ControlGroup({
  label,
  icon: Icon,
  children,
}: {
  label: string;
  icon: typeof Clock;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-2">
      <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
        <Icon className="size-3.5" aria-hidden />
        {label}
      </span>
      {children}
    </div>
  );
}

function ChipRow<T extends string | number>({
  options,
  value,
  onChange,
  ariaLabel,
}: {
  options: { value: T | null; label: string }[];
  value: T | null;
  onChange: (next: T | null) => void;
  ariaLabel: string;
}) {
  return (
    <div className="flex flex-wrap gap-1.5" role="group" aria-label={ariaLabel}>
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={String(option.value)}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={active}
            className={cn(
              "rounded-full border px-2.5 py-1 text-xs transition-colors",
              active
                ? "border-[color-mix(in_oklch,var(--color-mercury)_45%,transparent)] bg-[color-mix(in_oklch,var(--color-mercury)_14%,transparent)] text-[var(--color-mercury)]"
                : "border-[var(--color-hairline)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function StageFlow({ response }: { response: RecommendationResponse }) {
  const sources = Object.entries(response.candidate_sources);
  const totalFiltered = Object.values(response.filtered).reduce((a, b) => a + b, 0);

  return (
    <GlassPanel className="space-y-4">
      <SectionHeading
        title="What happened to this request"
        description="Counts and timings from the response itself."
      />

      <div className="space-y-2.5">
        {[
          {
            label: "Retrieval",
            detail: sources.length
              ? sources
                  .map(([name, count]) => `${RETRIEVAL_SOURCES[name]?.label ?? name} ${count}`)
                  .join(" · ")
              : "cold-start path",
            value: response.n_candidates,
            unit: "fused",
            ms: response.latency.candidate_generation_ms,
          },
          {
            label: "Ranking",
            detail: "LambdaRank over the fused candidates",
            value: response.n_candidates,
            unit: "scored",
            ms: response.latency.ranking_ms,
          },
          {
            label: "Business rerank",
            detail:
              totalFiltered > 0
                ? `${totalFiltered} removed: ${Object.entries(response.filtered)
                    .filter(([, n]) => n > 0)
                    .map(([reason, n]) => `${reason} ${n}`)
                    .join(", ")}`
                : "diversity caps applied, nothing filtered",
            value: response.recommendations.length,
            unit: "returned",
            ms: response.latency.reranking_ms,
          },
        ].map((stage) => (
          <div
            key={stage.label}
            className="flex flex-wrap items-baseline justify-between gap-2 rounded-[var(--radius-control)] bg-[var(--color-surface-1)] px-3 py-2"
          >
            <div className="min-w-0">
              <span className="text-sm font-medium">{stage.label}</span>
              <p className="truncate text-xs text-[var(--color-ink-faint)]">{stage.detail}</p>
            </div>
            <div className="flex items-baseline gap-3">
              <span className="tabular text-sm">
                {formatNumber(stage.value)}
                <span className="ml-1 text-xs text-[var(--color-ink-faint)]">{stage.unit}</span>
              </span>
              <span className="tabular w-16 text-right text-xs text-[var(--color-mercury)]">
                {formatMs(stage.ms)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </GlassPanel>
  );
}

/* -------------------------------------------------------------------------- */

function ExplanationPanel({
  response,
  selected,
}: {
  response: RecommendationResponse;
  selected: number | null;
}) {
  const explanation =
    response.explanations.find((e) => e.item_id === selected) ?? response.explanations[0];

  if (!explanation) {
    return (
      <GlassPanel>
        <SectionHeading title="Why recommended?" />
        <EmptyState
          title="No explanation for this request"
          description="Explanations come from the ranking model. A cold-start response has no ranker output to attribute."
        />
      </GlassPanel>
    );
  }

  const entries = Object.entries(explanation.contributions).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
  );
  const largest = Math.max(...entries.map(([, v]) => Math.abs(v)), 1e-9);

  return (
    <GlassPanel className="space-y-3">
      <SectionHeading
        title="Why recommended?"
        description={
          <>
            Exact TreeSHAP contributions from the ranking model for item{" "}
            <span className="tabular">{explanation.item_id}</span>. These are the model&apos;s
            real attributions, not a narrative written afterwards.
          </>
        }
      />
      <div className="space-y-2">
        {entries.map(([feature, contribution]) => {
          const positive = contribution >= 0;
          const width = `${(Math.abs(contribution) / largest) * 100}%`;
          return (
            <div key={feature} className="space-y-1">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-xs text-[var(--color-ink-muted)]">
                  {featureLabel(feature)}
                </span>
                <span
                  className={cn(
                    "tabular text-xs",
                    positive
                      ? "text-[var(--color-positive)]"
                      : "text-[var(--color-critical)]",
                  )}
                >
                  {positive ? "+" : ""}
                  {contribution.toFixed(4)}
                </span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
                <div
                  className="h-full rounded-full"
                  style={{
                    width,
                    backgroundColor: positive
                      ? "var(--color-positive)"
                      : "var(--color-critical)",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-[var(--color-ink-faint)]">
        Positive values pushed this item up the ranking; negative values pushed it down.
      </p>
    </GlassPanel>
  );
}

/* -------------------------------------------------------------------------- */

function RecommendationRow({
  item,
  selected,
  onSelect,
}: {
  item: Recommendation;
  selected: boolean;
  onSelect: () => void;
}) {
  const vertical = verticalMeta(item.vertical);
  const hasAdjustment = item.business_adjustment !== 0;

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        "flex w-full items-center gap-3 rounded-[var(--radius-control)] px-3 py-2 text-left transition-colors",
        selected
          ? "bg-[color-mix(in_oklch,var(--color-mercury)_10%,transparent)]"
          : "hover:bg-[var(--color-surface-2)]",
      )}
    >
      <span className="tabular w-6 shrink-0 text-xs text-[var(--color-ink-faint)]">
        {item.rank}
      </span>
      {vertical ? (
        <span
          className="size-2 shrink-0 rounded-full"
          style={{ backgroundColor: vertical.color }}
          title={vertical.label}
          aria-hidden
        />
      ) : (
        <span className="size-2 shrink-0" aria-hidden />
      )}
      <span className="tabular min-w-0 flex-1 truncate text-sm">
        item {item.item_id}
        {vertical ? (
          <span className="ml-2 text-xs text-[var(--color-ink-faint)]">{vertical.label}</span>
        ) : null}
      </span>

      {/* Model score and policy adjustment are shown separately, always. */}
      <span className="tabular shrink-0 text-xs text-[var(--color-ink-muted)]">
        {item.ml_relevance_score.toFixed(3)}
      </span>
      {hasAdjustment ? (
        <span
          className={cn(
            "tabular shrink-0 text-xs",
            item.business_adjustment > 0
              ? "text-[var(--color-positive)]"
              : "text-[var(--color-warning)]",
          )}
          title={Object.entries(item.adjustments)
            .map(([rule, value]) => `${rule}: ${value.toFixed(4)}`)
            .join("\n")}
        >
          {item.business_adjustment > 0 ? "+" : ""}
          {item.business_adjustment.toFixed(3)}
        </span>
      ) : (
        <span className="w-10 shrink-0" aria-hidden />
      )}
    </button>
  );
}

/* -------------------------------------------------------------------------- */

function ExplorerInner() {
  const router = useRouter();
  const params = useSearchParams();

  const [userId, setUserId] = useState(params.get("user") ?? "");
  const [pending, setPending] = useState(params.get("user") ?? "");
  const [selected, setSelected] = useState<number | null>(null);

  const hour = params.get("hour") ? Number(params.get("hour")) : null;
  const vertical = params.get("vertical") ? Number(params.get("vertical")) : null;
  const regionId = params.get("region") ? Number(params.get("region")) : null;

  /** Context lives in the URL so any view is shareable and reproducible. */
  const setParam = useCallback(
    (key: string, value: string | null) => {
      const next = new URLSearchParams(params.toString());
      if (value === null) next.delete(key);
      else next.set(key, value);
      router.replace(`/explorer?${next.toString()}`, { scroll: false });
    },
    [params, router],
  );

  const query = useMemo(
    () => (userId ? { userId, k: 10, hour, vertical, regionId } : null),
    [userId, hour, vertical, regionId],
  );
  const { data, isFetching, isError, error, refetch } = useRecommendations(query);

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Live recommendation explorer"
        description="Pick a user, change the context, and watch the ranking change. Every list here is a live API response."
      />

      <GlassPanel className="space-y-5">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            setUserId(pending.trim());
            setParam("user", pending.trim() || null);
            setSelected(null);
          }}
          className="flex flex-wrap items-end gap-3"
        >
          <ControlGroup label="User" icon={User}>
            <div className="flex gap-2">
              <input
                value={pending}
                onChange={(event) => setPending(event.target.value)}
                placeholder="e.g. 1150086"
                aria-label="User id"
                className="tabular w-48 rounded-[var(--radius-control)] border border-[var(--color-hairline)] bg-[var(--color-surface-1)] px-3 py-1.5 text-sm outline-none focus:border-[var(--color-mercury)]"
              />
              <button
                type="submit"
                className="rounded-[var(--radius-control)] bg-[var(--color-mercury)] px-3 py-1.5 text-sm font-medium text-[var(--color-void)]"
              >
                Recommend
              </button>
            </div>
          </ControlGroup>

          <ControlGroup label="Time of day" icon={Clock}>
            <ChipRow
              ariaLabel="Time of day"
              value={hour}
              onChange={(next) => setParam("hour", next === null ? null : String(next))}
              options={[
                { value: null, label: "Any" },
                ...HOUR_BUCKETS.map((bucket) => ({
                  value: bucket.representative,
                  label: bucket.label,
                })),
              ]}
            />
          </ControlGroup>

          <ControlGroup label="Vertical" icon={Layers}>
            <ChipRow
              ariaLabel="Vertical"
              value={vertical}
              onChange={(next) => setParam("vertical", next === null ? null : String(next))}
              options={[
                { value: null, label: "Any" },
                ...VERTICALS.map((v) => ({ value: v.id, label: v.label })),
              ]}
            />
          </ControlGroup>

          <ControlGroup label="Region" icon={MapPin}>
            <ChipRow
              ariaLabel="Region"
              value={regionId}
              onChange={(next) => setParam("region", next === null ? null : String(next))}
              options={[
                { value: null, label: "Any" },
                ...[0, 1, 2, 3, 4].map((id) => ({ value: id, label: `R${id}` })),
              ]}
            />
          </ControlGroup>

          <button
            type="button"
            onClick={() => refetch()}
            className="ml-auto inline-flex items-center gap-1.5 rounded-[var(--radius-control)] border border-[var(--color-hairline-strong)] px-3 py-1.5 text-xs"
            aria-label="Re-run the request"
          >
            <RefreshCw className={cn("size-3.5", isFetching && "animate-spin")} />
            Re-run
          </button>
        </form>
      </GlassPanel>

      {!userId ? (
        <EmptyState
          title="Choose a user to begin"
          description="Enter any user id from the dataset. Unknown ids are served by the cold-start path, which is a real code path rather than an error."
        />
      ) : isError ? (
        <EmptyState
          tone="warning"
          title="Could not fetch recommendations"
          description={error instanceof Error ? error.message : "Unknown error."}
          command="uv run uvicorn mercury_rec.api.main:app --port 8000"
        />
      ) : !data ? (
        <PanelSkeleton rows={6} />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Total latency"
              value={data.latency.total_ms}
              unit="ms"
              decimals={2}
              icon={Sparkles}
              tone={data.latency.total_ms < 50 ? "positive" : "warning"}
            />
            <MetricTile label="Candidates" value={data.n_candidates} hint="reached the ranker" />
            <MetricTile
              label="Cache"
              value={data.cache_hit ? 1 : 0}
              hint={data.cache_hit ? "served from cache" : "computed fresh"}
            />
            <MetricTile
              label="Path"
              value={data.is_cold_start ? 0 : 1}
              hint={data.is_cold_start ? "cold start" : "personalised"}
              tone={data.is_cold_start ? "warning" : "positive"}
            />
          </div>

          {data.is_cold_start ? (
            <GlassPanel className="border-[color-mix(in_oklch,var(--color-warning)_38%,transparent)]">
              <p className="text-sm text-[var(--color-ink-muted)]">
                <Badge tone="warning">Cold start</Badge>{" "}
                <span className="ml-2">
                  This user has no usable history, so contextual popularity served the
                  request. That is a deliberate path — an unknown visitor is the most
                  common request a live system receives, and an empty list is never the
                  right answer to it.
                </span>
              </p>
            </GlassPanel>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
            <GlassPanel className="space-y-1">
              <SectionHeading
                title="Final recommendations"
                description="Model score and business adjustment are shown as separate columns."
              />
              <div className="flex items-center gap-3 px-3 pb-1 text-[10px] uppercase tracking-wider text-[var(--color-ink-faint)]">
                <span className="w-6">#</span>
                <span className="size-2" aria-hidden />
                <span className="flex-1">item</span>
                <span>ml</span>
                <span className="w-10 text-right">policy</span>
              </div>
              {data.recommendations.map((item) => (
                <RecommendationRow
                  key={item.item_id}
                  item={item}
                  selected={selected === item.item_id}
                  onSelect={() => setSelected(item.item_id)}
                />
              ))}
            </GlassPanel>

            <div className="space-y-4">
              <StageFlow response={data} />
              <ExplanationPanel response={data} selected={selected} />
            </div>
          </div>

          <GlassPanel>
            <SectionHeading title="Request detail" />
            <DataRow label="Request id" value={data.request_id} />
            <DataRow label="Model version" value={data.model_version} />
            <DataRow label="Generated" value={new Date(data.generated_at).toLocaleString()} />
            <DataRow label="Provenance" value={data.data_provenance} mono={false} />
          </GlassPanel>
        </>
      )}
    </div>
  );
}

export default function ExplorerPage() {
  // useSearchParams requires a Suspense boundary in the App Router.
  return (
    <Suspense fallback={<PanelSkeleton rows={6} />}>
      <ExplorerInner />
    </Suspense>
  );
}
