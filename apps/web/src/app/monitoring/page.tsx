"use client";

/**
 * Monitoring.
 *
 * Two different kinds of number, kept visibly apart:
 *
 * - **Live telemetry** measured inside this API process since it started.
 *   Percentiles come from real requests; when none have been served the page
 *   says so rather than rendering "p95: 0.0ms", which looks like a
 *   measurement and is the absence of one.
 * - **Feature drift**, computed offline between the training window and the
 *   held-out window that follows it. There is no production traffic behind
 *   this system, so that chronological pair is the only genuine later data
 *   there is — and it is real data, not simulated drift.
 */

import { Activity, CircleCheck, CircleX, Database, Gauge, Zap } from "lucide-react";
import { useState } from "react";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useDrift, useHealth, useMetricsSummary, useReadiness } from "@/lib/api/hooks";
import { featureLabel } from "@/lib/constants";
import { cn, formatNumber } from "@/lib/utils";
import type { FeatureDrift } from "@/lib/api/types";

const SEVERITY_TONE = {
  stable: "text-[var(--color-positive)]",
  minor: "text-[var(--color-warning)]",
  major: "text-[var(--color-critical)]",
} as const;

const SEVERITY_BAR = {
  stable: "bg-[var(--color-positive)]",
  minor: "bg-[var(--color-warning)]",
  major: "bg-[var(--color-critical)]",
} as const;

/**
 * Time-indexed features shift by construction in a chronological split.
 *
 * Flagging them is the difference between a dashboard that teaches and one
 * that cries wolf: a reader who sees PSI 10.9 with no explanation concludes
 * the system is broken, when in fact the split is working.
 */
const STRUCTURAL = new Set([
  "user_tenure_days",
  "item_age_days",
  "user_days_since_last",
  "item_days_since_last",
  "ui_days_since_last",
]);

/* -------------------------------------------------------------------------- */

function DriftRow({ entry, maxPsi }: { entry: FeatureDrift; maxPsi: number }) {
  const structural = STRUCTURAL.has(entry.feature);

  return (
    <tr className="border-b border-[var(--color-hairline)] last:border-0">
      <td className="py-2.5 pr-4 align-top">
        <div className="flex items-center gap-2">
          <span className="text-sm">{featureLabel(entry.feature)}</span>
          {structural ? (
            <Badge
              tone="neutral"
              title="Shifts by construction: the split is chronological, so this feature has a mechanical reason to move."
            >
              expected
            </Badge>
          ) : null}
        </div>
        <div className="tabular mt-0.5 text-xs text-[var(--color-ink-faint)]">
          {entry.reference_mean.toFixed(3)} → {entry.current_mean.toFixed(3)}
        </div>
      </td>
      <td className="py-2.5 pr-4 align-top">
        <div className="flex items-center gap-2">
          <div className="h-1.5 w-24 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
            <div
              className={cn(
                "h-full rounded-full",
                structural ? "bg-[var(--color-surface-3)]" : SEVERITY_BAR[entry.severity],
              )}
              style={{ width: `${Math.min((entry.psi / maxPsi) * 100, 100)}%` }}
            />
          </div>
          <span
            className={cn(
              "tabular text-xs",
              structural ? "text-[var(--color-ink-faint)]" : SEVERITY_TONE[entry.severity],
            )}
          >
            {entry.psi.toFixed(3)}
          </span>
        </div>
      </td>
      <td className="tabular py-2.5 pr-4 text-right align-top text-xs text-[var(--color-ink-muted)]">
        {entry.ks_statistic.toFixed(3)}
      </td>
      <td className="tabular py-2.5 text-right align-top text-xs text-[var(--color-ink-muted)]">
        {entry.js_divergence.toFixed(3)}
      </td>
    </tr>
  );
}

/* -------------------------------------------------------------------------- */

export default function MonitoringPage() {
  const metrics = useMetricsSummary();
  const readiness = useReadiness();
  const health = useHealth();
  const drift = useDrift();
  const [hideStructural, setHideStructural] = useState(false);

  const stages = metrics.data?.stages ?? {};
  const total = stages.total;
  const cache = metrics.data?.cache;

  const driftFeatures = (drift.data?.features ?? []).filter(
    (entry) => !hideStructural || !STRUCTURAL.has(entry.feature),
  );
  const maxPsi = Math.max(...driftFeatures.map((entry) => entry.psi), 0.25);
  const signalMajor = (drift.data?.features ?? []).filter(
    (entry) => entry.severity === "major" && !STRUCTURAL.has(entry.feature),
  ).length;

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Monitoring"
        description="Live telemetry from this process, and feature drift measured offline between two time windows."
        action={
          <div className="flex items-center gap-2">
            <Badge tone={health.isError ? "critical" : "positive"}>
              <Activity className="size-3" aria-hidden />
              {health.isError ? "API offline" : `up ${formatNumber(health.data?.uptime_seconds ?? 0)}s`}
            </Badge>
          </div>
        }
      />

      {/* --- readiness --- */}
      <GlassPanel>
        <SectionHeading
          title="Readiness"
          description="Liveness and readiness are separate on purpose: restarting cannot fix a missing artifact, so only one of them should ever fail the process."
          action={
            readiness.data ? (
              <Badge tone={readiness.data.ready ? "positive" : "warning"}>
                {readiness.data.ready ? "ready" : "not ready"}
              </Badge>
            ) : null
          }
        />
        {readiness.data ? (
          <div className="grid gap-2 sm:grid-cols-2">
            {readiness.data.checks.map((check) => (
              <div
                key={check.name}
                className="flex items-start gap-2 rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-2.5"
              >
                {check.ready ? (
                  <CircleCheck
                    className="mt-0.5 size-4 shrink-0 text-[var(--color-positive)]"
                    aria-hidden
                  />
                ) : (
                  <CircleX
                    className="mt-0.5 size-4 shrink-0 text-[var(--color-warning)]"
                    aria-hidden
                  />
                )}
                <div className="min-w-0">
                  <div className="text-sm">{check.name.replace(/_/g, " ")}</div>
                  {check.detail ? (
                    <div className="text-xs text-[var(--color-ink-faint)]">{check.detail}</div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <PanelSkeleton rows={2} />
        )}
      </GlassPanel>

      {/* --- live latency --- */}
      <div>
        <SectionHeading
          title="Serving latency"
          description="Measured in this process since it started. Refreshes every five seconds."
        />
        {metrics.data?.has_data ? (
          <>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <MetricTile
                label="p50"
                value={total?.p50_ms ?? null}
                unit="ms"
                decimals={2}
                icon={Zap}
                tone="positive"
              />
              <MetricTile label="p95" value={total?.p95_ms ?? null} unit="ms" decimals={2} />
              <MetricTile
                label="p99"
                value={total?.p99_ms ?? null}
                unit="ms"
                decimals={2}
                tone="warning"
              />
              <MetricTile
                label="Requests observed"
                value={total?.count ?? null}
                compact
                icon={Gauge}
              />
            </div>

            <GlassPanel className="mt-4">
              <SectionHeading title="Where the time goes" />
              <div className="-mx-1 overflow-x-auto px-1">
                <table className="w-full min-w-[32rem] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-[var(--color-hairline)] text-left">
                      <th className="py-2 pr-4 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Stage
                      </th>
                      {["mean_ms", "p50_ms", "p95_ms", "p99_ms"].map((column) => (
                        <th
                          key={column}
                          className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
                        >
                          {column.replace("_ms", "")}
                        </th>
                      ))}
                      <th className="py-2 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        n
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(stages)
                      .filter(([stage]) => stage !== "total")
                      .map(([stage, values]) => (
                        <tr
                          key={stage}
                          className="border-b border-[var(--color-hairline)] last:border-0"
                        >
                          <td className="py-2 pr-4">{stage.replace(/_/g, " ")}</td>
                          {(["mean_ms", "p50_ms", "p95_ms", "p99_ms"] as const).map(
                            (column) => (
                              <td
                                key={column}
                                className="tabular py-2 pr-4 text-right text-[var(--color-ink-muted)]"
                              >
                                {values[column] === null || values[column] === undefined
                                  ? "—"
                                  : values[column].toFixed(2)}
                              </td>
                            ),
                          )}
                          <td className="tabular py-2 text-right text-[var(--color-ink-faint)]">
                            {formatNumber(values.count)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              {metrics.data.note ? (
                <p className="mt-3 text-xs leading-relaxed text-[var(--color-ink-faint)]">
                  {metrics.data.note}
                </p>
              ) : null}
            </GlassPanel>
          </>
        ) : (
          <EmptyState
            title="No requests recorded in this process yet"
            description={
              metrics.data?.detail ??
              "Percentiles appear once the API has served traffic. Open the Explorer or the Pipeline page to generate some."
            }
          />
        )}
      </div>

      {/* --- cache --- */}
      {cache ? (
        <div>
          <SectionHeading
            title="Cache"
            description="Keyed by model version, so a promotion invalidates atomically instead of scanning for keys to delete."
          />
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Hit rate"
              value={cache.hit_rate}
              percent
              decimals={1}
              icon={Database}
              hint={`${formatNumber(cache.hits + cache.stale_hits)} hits, ${formatNumber(cache.misses)} misses`}
            />
            <MetricTile
              label="Stale hits"
              value={cache.stale_hits}
              compact
              hint="served past soft TTL while a refresh runs"
            />
            <MetricTile
              label="Invalidations"
              value={cache.invalidations}
              compact
              hint="triggered by ingested events"
            />
            <MetricTile
              label="Errors"
              value={cache.errors}
              compact
              tone={cache.errors > 0 ? "warning" : "neutral"}
              hint="cache failures never fail a request"
            />
          </div>
        </div>
      ) : null}

      {/* --- drift --- */}
      <div>
        <SectionHeading
          title="Feature drift"
          description={
            drift.data
              ? `${drift.data.reference_split} window versus ${drift.data.current_split}, ${formatNumber(drift.data.reference_rows)} against ${formatNumber(drift.data.current_rows)} rows.`
              : "Distribution shift between the training window and the one that follows it."
          }
          action={
            drift.data ? (
              <button
                type="button"
                onClick={() => setHideStructural((current) => !current)}
                className="rounded-[var(--radius-control)] border border-[var(--color-hairline)] px-3 py-1.5 text-xs text-[var(--color-ink-muted)] transition-colors hover:bg-[var(--color-surface-2)]"
                aria-pressed={hideStructural}
              >
                {hideStructural ? "Show all features" : "Hide expected drift"}
              </button>
            ) : null
          }
        />

        {drift.isLoading ? (
          <PanelSkeleton rows={5} />
        ) : drift.isError ? (
          <EmptyState
            title="No drift report has been produced"
            description="Drift is computed offline between two time windows and committed as an artifact."
            command="uv run mercury monitor drift"
          />
        ) : drift.data ? (
          <>
            <div className="mb-4 grid gap-4 sm:grid-cols-3">
              <MetricTile
                label="Features compared"
                value={drift.data.features.length}
                hint="the full as-of feature vector"
              />
              <MetricTile
                label="Unexplained major shifts"
                value={signalMajor}
                tone={signalMajor > 0 ? "warning" : "positive"}
                hint="excluding time-indexed features"
              />
              <MetricTile
                label="Major threshold"
                value={drift.data.thresholds.major}
                decimals={2}
                hint="PSI, conventional rule of thumb"
              />
            </div>

            <GlassPanel>
              <div className="-mx-1 overflow-x-auto px-1">
                <table className="w-full min-w-[36rem] border-collapse">
                  <thead>
                    <tr className="border-b border-[var(--color-hairline)] text-left">
                      <th className="py-2 pr-4 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Feature
                      </th>
                      <th
                        title="Population Stability Index, binned on the reference window."
                        className="py-2 pr-4 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
                      >
                        PSI
                      </th>
                      <th
                        title="Largest gap between the two cumulative distributions."
                        className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
                      >
                        KS
                      </th>
                      <th
                        title="Jensen-Shannon divergence in bits; bounded to [0, 1]."
                        className="py-2 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
                      >
                        JS
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {driftFeatures.map((entry) => (
                      <DriftRow key={entry.feature} entry={entry} maxPsi={maxPsi} />
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 space-y-2 text-xs leading-relaxed text-[var(--color-ink-muted)]">
                {drift.data.expected_drift ? <p>{drift.data.expected_drift}</p> : null}
                <p className="text-[var(--color-ink-faint)]">{drift.data.interpretation}</p>
                <p className="text-[var(--color-ink-faint)]">{drift.data.note}</p>
              </div>
            </GlassPanel>
          </>
        ) : null}
      </div>

      <GlassPanel>
        <SectionHeading title="Scrape endpoints" />
        <div className="space-y-0">
          <DataRow label="Prometheus" value="/metrics" title="OpenMetrics exposition." />
          <DataRow label="Liveness" value="/health" />
          <DataRow label="Readiness" value="/ready" />
          <DataRow label="Version" value={health.data?.version ?? "—"} />
        </div>
        <p className="mt-3 text-xs leading-relaxed text-[var(--color-ink-faint)]">
          The percentiles above are computed in-process for this dashboard. Prometheus
          histograms are exported separately at <code className="tabular">/metrics</code>,
          where the buckets rather than the summary are what a real alerting rule should
          read.
        </p>
      </GlassPanel>
    </div>
  );
}
