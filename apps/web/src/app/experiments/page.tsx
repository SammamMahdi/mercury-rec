"use client";

/**
 * Offline A/B simulation and the promotion gate.
 *
 * The most important element on this page is the banner saying the experiment
 * is simulated. This project has no live traffic, so there is no online
 * result to report; what exists is a counterfactual comparison of two rankers
 * over the same held-out users, with confidence intervals from a bootstrap
 * resampled at the user level.
 *
 * That distinction is not a caveat to be softened. A simulated lift and a
 * measured one are different claims, and a portfolio that blurs them is
 * making the more impressive of the two without having earned it.
 */

import { CheckCircle2, CircleSlash, FlaskConical, ShieldCheck, XCircle } from "lucide-react";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useExperiment } from "@/lib/api/hooks";
import { cn, formatNumber, shortHash } from "@/lib/utils";
import type { MetricComparison } from "@/lib/api/types";

/* -------------------------------------------------------------------------- */

/**
 * One metric's control-versus-treatment comparison.
 *
 * The interval is drawn, not just printed, and the zero line is drawn with it.
 * Whether an interval crosses zero is the entire result, and it is far easier
 * to see than to read off two numbers in brackets.
 */
function ComparisonRow({ comparison }: { comparison: MetricComparison }) {
  const [low, high] = comparison.ci_95;

  // A symmetric scale around zero, wide enough to hold the whole interval.
  const extent = Math.max(Math.abs(low), Math.abs(high), 1e-9) * 1.15;
  const toPct = (value: number) => ((value + extent) / (2 * extent)) * 100;

  const significant = comparison.is_significant;

  return (
    <div className="border-b border-[var(--color-hairline)] py-3 last:border-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-medium">{comparison.metric}</span>
        <div className="flex items-baseline gap-3">
          <span className="tabular text-xs text-[var(--color-ink-faint)]">
            {comparison.control.toFixed(4)} → {comparison.treatment.toFixed(4)}
          </span>
          <span
            className={cn(
              "tabular text-sm font-semibold",
              !significant
                ? "text-[var(--color-ink-muted)]"
                : comparison.absolute_lift > 0
                  ? "text-[var(--color-positive)]"
                  : "text-[var(--color-critical)]",
            )}
          >
            {comparison.relative_lift_pct > 0 ? "+" : ""}
            {comparison.relative_lift_pct.toFixed(1)}%
          </span>
        </div>
      </div>

      <div className="relative mt-2.5 h-6">
        {/* Zero line: the only reference that matters. */}
        <div
          className="absolute top-0 h-full w-px bg-[var(--color-hairline-strong)]"
          style={{ left: `${toPct(0)}%` }}
          aria-hidden
        />
        <div
          className={cn(
            "absolute top-1/2 h-1.5 -translate-y-1/2 rounded-full",
            significant
              ? comparison.absolute_lift > 0
                ? "bg-[var(--color-positive)]"
                : "bg-[var(--color-critical)]"
              : "bg-[var(--color-surface-3)]",
          )}
          style={{ left: `${toPct(low)}%`, width: `${toPct(high) - toPct(low)}%` }}
        />
        <div
          className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--color-ink)]"
          style={{ left: `${toPct(comparison.absolute_lift)}%` }}
          aria-hidden
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--color-ink-faint)]">
        <span className="tabular">
          95% CI [{low.toFixed(5)}, {high.toFixed(5)}]
        </span>
        <span className="tabular">p = {comparison.p_value.toFixed(4)}</span>
        <span className="tabular">n = {formatNumber(comparison.n_users)}</span>
        {significant ? (
          <Badge tone="positive">significant</Badge>
        ) : (
          <Badge tone="neutral" title="The interval includes zero, so the sign of the effect is not established.">
            not significant
          </Badge>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

const VERDICT_ICON = {
  pass: CheckCircle2,
  fail: XCircle,
  inconclusive: CircleSlash,
} as const;

const VERDICT_TONE = {
  pass: "text-[var(--color-positive)]",
  fail: "text-[var(--color-critical)]",
  inconclusive: "text-[var(--color-warning)]",
} as const;

/* -------------------------------------------------------------------------- */

export default function ExperimentsPage() {
  const experiment = useExperiment();

  if (experiment.isError) {
    return (
      <div className="space-y-6">
        <SectionHeading title="Experiments" />
        <EmptyState
          tone="warning"
          title="No experiment results are available"
          description="This page reports a comparison that was actually run. There is nothing to show until one has been."
          command="uv run mercury train experiment"
        />
      </div>
    );
  }

  const data = experiment.data;
  const simulation = data?.simulation;
  const gate = data?.gate;

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Experiments"
        description="Counterfactual comparison of two rankers over the same held-out users, with bootstrap confidence intervals."
        action={
          data ? (
            <Badge tone="neutral">
              <span className="tabular">{shortHash(data.dataset_hash)}</span>
            </Badge>
          ) : null
        }
      />

      {/* The disclaimer is the first thing on the page, not a footnote. */}
      <div className="flex items-start gap-3 rounded-[var(--radius-panel)] border border-[color-mix(in_oklch,var(--color-warning)_40%,transparent)] bg-[color-mix(in_oklch,var(--color-warning)_7%,transparent)] p-4">
        <FlaskConical
          className="mt-0.5 size-4 shrink-0 text-[var(--color-warning)]"
          aria-hidden
        />
        <div className="space-y-1">
          <p className="text-sm font-medium text-[var(--color-ink)]">
            This experiment is simulated. It is not an online A/B test.
          </p>
          <p className="text-xs leading-relaxed text-[var(--color-ink-muted)]">
            {simulation?.disclaimer ??
              "There is no live traffic behind this system. Both arms are scored offline over the same held-out users, so the comparison measures a difference between models rather than a change in user behaviour."}
          </p>
        </div>
      </div>

      {experiment.isLoading ? (
        <PanelSkeleton rows={6} />
      ) : data && simulation && gate ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Users compared"
              value={simulation.n_users}
              compact
              hint="paired: both arms score the same users"
            />
            <MetricTile
              label="Bootstrap samples"
              value={simulation.bootstrap_samples}
              compact
              hint="resampled at the user level"
            />
            <MetricTile
              label="Metrics significant"
              value={simulation.metrics.filter((entry) => entry.is_significant).length}
              hint={`of ${simulation.metrics.length} compared`}
            />
            <MetricTile
              label="Cutoff"
              value={data.k}
              hint="top-k the comparison is evaluated at"
            />
          </div>

          <GlassPanel>
            <SectionHeading
              title={`${simulation.control} versus ${simulation.treatment}`}
              description="Point estimate with its 95% interval. An interval crossing the zero line means the direction of the effect is not established."
            />
            <div>
              {simulation.metrics.map((comparison) => (
                <ComparisonRow key={comparison.metric} comparison={comparison} />
              ))}
            </div>
            <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-faint)]">
              The bootstrap resamples users, not rows. Resampling rows would treat one
              user&apos;s twenty interactions as twenty independent observations and produce
              intervals several times too narrow — the most common way an offline comparison
              overstates its own certainty.
            </p>
          </GlassPanel>

          <GlassPanel>
            <SectionHeading
              title="Promotion gate"
              description="Every check a candidate must clear before it can be marked production."
              action={
                <Badge tone={gate.promoted ? "positive" : "warning"}>
                  <ShieldCheck className="size-3" aria-hidden />
                  {gate.promoted ? "promoted" : "held"}
                </Badge>
              }
            />

            <div className="mb-4 space-y-0">
              <DataRow label="Candidate" value={gate.candidate} />
              <DataRow label="Incumbent" value={gate.incumbent ?? "none"} />
              {gate.summary ? (
                <DataRow label="Verdict" value={gate.summary} mono={false} />
              ) : null}
            </div>

            <div className="space-y-2">
              {gate.checks.map((check) => {
                const Icon = VERDICT_ICON[check.verdict];
                return (
                  <div
                    key={check.name}
                    className="flex items-start gap-3 rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-3"
                  >
                    <Icon
                      className={cn("mt-0.5 size-4 shrink-0", VERDICT_TONE[check.verdict])}
                      aria-hidden
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-baseline justify-between gap-2">
                        <span className="text-sm font-medium">
                          {check.name.replace(/_/g, " ")}
                        </span>
                        {check.candidate !== null ? (
                          <span className="tabular text-xs text-[var(--color-ink-faint)]">
                            {check.candidate.toFixed(4)}
                            {check.incumbent !== null
                              ? ` vs ${check.incumbent.toFixed(4)}`
                              : null}
                            {check.threshold !== null
                              ? ` · threshold ${check.threshold.toFixed(4)}`
                              : null}
                          </span>
                        ) : null}
                      </div>
                      <p className="mt-0.5 text-xs leading-relaxed text-[var(--color-ink-muted)]">
                        {check.detail}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-faint)]">
              An inconclusive check never counts as a pass. Treating &ldquo;we could not
              tell&rdquo; as &ldquo;no harm found&rdquo; is how a regression reaches
              production with a green tick next to it.
            </p>
          </GlassPanel>

          <p className="text-xs text-[var(--color-ink-faint)]">
            Generated {new Date(data.generated_at).toLocaleString()} by{" "}
            <code className="tabular text-[var(--color-mercury)]">
              mercury train experiment
            </code>
            .
          </p>
        </>
      ) : null}
    </div>
  );
}
