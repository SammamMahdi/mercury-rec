"use client";

/**
 * Shared UI primitives.
 *
 * The important one is {@link MetricTile}. Its `value` prop is
 * `number | null` with **no default**, so there is no code path that renders a
 * placeholder figure. When data is absent it renders an em dash and a reason.
 * That is a structural guarantee rather than a convention: you cannot
 * accidentally show a zero that reads as a measurement, because you would have
 * to pass one deliberately.
 */

import { AlertTriangle, Info, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn, formatDelta, formatNumber } from "@/lib/utils";

/* -------------------------------------------------------------------------- */

export function GlassPanel({
  children,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "article" | "aside";
}) {
  return (
    <Tag
      className={cn(
        "glass rounded-[var(--radius-panel)] p-5",
        "transition-colors duration-200",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

/* -------------------------------------------------------------------------- */

export function SectionHeading({
  title,
  description,
  action,
}: {
  title: string;
  description?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div className="max-w-2xl">
        <h2 className="text-lg font-semibold tracking-tight text-[var(--color-ink)]">
          {title}
        </h2>
        {description ? (
          <p className="mt-1 text-sm leading-relaxed text-[var(--color-ink-muted)]">
            {description}
          </p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export type TileTone = "neutral" | "positive" | "warning" | "critical";

const TONE_TEXT: Record<TileTone, string> = {
  neutral: "text-[var(--color-ink)]",
  positive: "text-[var(--color-positive)]",
  warning: "text-[var(--color-warning)]",
  critical: "text-[var(--color-critical)]",
};

/**
 * A single headline figure.
 *
 * `value` is intentionally `number | null` with no default. Passing nothing is
 * a type error; passing null renders an em dash with `emptyReason`. There is
 * deliberately no way to make this component invent a number.
 */
export function MetricTile({
  label,
  value,
  unit,
  decimals = 0,
  compact = false,
  percent = false,
  delta,
  tone = "neutral",
  hint,
  emptyReason = "not measured yet",
  icon: Icon,
}: {
  label: string;
  value: number | null;
  unit?: string;
  decimals?: number;
  compact?: boolean;
  percent?: boolean;
  delta?: number | null;
  tone?: TileTone;
  hint?: string;
  emptyReason?: string;
  icon?: LucideIcon;
}) {
  const isEmpty = value === null || value === undefined || Number.isNaN(value);

  return (
    <GlassPanel className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
          {label}
        </span>
        {Icon ? (
          <Icon className="size-4 text-[var(--color-ink-faint)]" aria-hidden />
        ) : null}
      </div>

      {isEmpty ? (
        <>
          <span
            className="tabular text-2xl font-semibold text-[var(--color-ink-faint)]"
            aria-label={`${label}: ${emptyReason}`}
          >
            —
          </span>
          <span className="text-xs text-[var(--color-ink-faint)]">{emptyReason}</span>
        </>
      ) : (
        <>
          <span className={cn("tabular text-2xl font-semibold", TONE_TEXT[tone])}>
            {formatNumber(value, { decimals, compact, percent })}
            {unit ? (
              <span className="ml-1 text-sm font-normal text-[var(--color-ink-muted)]">
                {unit}
              </span>
            ) : null}
          </span>
          {delta !== undefined && delta !== null ? (
            <span
              className={cn(
                "tabular text-xs",
                delta >= 0
                  ? "text-[var(--color-positive)]"
                  : "text-[var(--color-critical)]",
              )}
            >
              {formatDelta(delta)}
            </span>
          ) : hint ? (
            <span className="text-xs text-[var(--color-ink-faint)]">{hint}</span>
          ) : null}
        </>
      )}
    </GlassPanel>
  );
}

/* -------------------------------------------------------------------------- */

export type BadgeTone = "neutral" | "mercury" | "positive" | "warning" | "critical";

const BADGE_TONE: Record<BadgeTone, string> = {
  neutral:
    "bg-[var(--color-surface-2)] text-[var(--color-ink-muted)] border-[var(--color-hairline)]",
  mercury:
    "bg-[color-mix(in_oklch,var(--color-mercury)_14%,transparent)] text-[var(--color-mercury)] border-[color-mix(in_oklch,var(--color-mercury)_35%,transparent)]",
  positive:
    "bg-[color-mix(in_oklch,var(--color-positive)_14%,transparent)] text-[var(--color-positive)] border-[color-mix(in_oklch,var(--color-positive)_35%,transparent)]",
  warning:
    "bg-[color-mix(in_oklch,var(--color-warning)_14%,transparent)] text-[var(--color-warning)] border-[color-mix(in_oklch,var(--color-warning)_38%,transparent)]",
  critical:
    "bg-[color-mix(in_oklch,var(--color-critical)_14%,transparent)] text-[var(--color-critical)] border-[color-mix(in_oklch,var(--color-critical)_35%,transparent)]",
};

export function Badge({
  children,
  tone = "neutral",
  className,
  title,
}: {
  children: ReactNode;
  tone?: BadgeTone;
  className?: string;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5",
        "text-[11px] font-medium leading-5",
        BADGE_TONE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* -------------------------------------------------------------------------- */

/**
 * States plainly where the underlying data came from.
 *
 * Shown wherever synthesised attributes are on screen. The dataset mixes real
 * Retailrocket behaviour with generated merchants, regions and prices, and a
 * reader must never have to guess which is which.
 */
export function ProvenanceBadge({ provenance }: { provenance?: string | null }) {
  const label =
    provenance === "retailrocket"
      ? "Real behavioural data"
      : provenance === "synthetic"
        ? "Fully synthetic"
        : "Real behaviour, synthesised attributes";

  return (
    <Badge
      tone="neutral"
      title="Behaviour, timestamps and item ids come from the Retailrocket dataset (CC BY-NC-SA 4.0). Merchants, regions, prices and verticals are generated. See docs/data-card.md."
    >
      <Info className="size-3" aria-hidden />
      {label}
    </Badge>
  );
}

/* -------------------------------------------------------------------------- */

/**
 * Rendered when data is genuinely unavailable.
 *
 * Always names the command that would produce it. "No data" leaves the reader
 * stuck; "run `mercury train baselines`" does not.
 */
export function EmptyState({
  title,
  description,
  command,
  tone = "neutral",
}: {
  title: string;
  description?: ReactNode;
  command?: string;
  tone?: "neutral" | "warning";
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-[var(--radius-panel)]",
        "border border-dashed p-8 text-center",
        tone === "warning"
          ? "border-[color-mix(in_oklch,var(--color-warning)_40%,transparent)]"
          : "border-[var(--color-hairline)]",
      )}
      role="status"
    >
      {tone === "warning" ? (
        <AlertTriangle className="size-5 text-[var(--color-warning)]" aria-hidden />
      ) : null}
      <p className="text-sm font-medium text-[var(--color-ink)]">{title}</p>
      {description ? (
        <p className="max-w-md text-sm text-[var(--color-ink-muted)]">{description}</p>
      ) : null}
      {command ? (
        <code className="tabular mt-1 rounded-[var(--radius-control)] bg-[var(--color-surface-2)] px-2.5 py-1 text-xs text-[var(--color-mercury)]">
          {command}
        </code>
      ) : null}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn("skeleton rounded-[var(--radius-control)]", className)}
      aria-hidden
    />
  );
}

export function PanelSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <GlassPanel className="space-y-3">
      <Skeleton className="h-4 w-1/3" />
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton key={index} className="h-3 w-full" />
      ))}
    </GlassPanel>
  );
}

/* -------------------------------------------------------------------------- */

/** A labelled key/value row, used throughout the detail panels. */
export function DataRow({
  label,
  value,
  mono = true,
  title,
}: {
  label: string;
  value: ReactNode;
  mono?: boolean;
  title?: string;
}) {
  return (
    <div
      className="flex items-baseline justify-between gap-4 border-b border-[var(--color-hairline)] py-1.5 last:border-0"
      title={title}
    >
      <span className="text-xs text-[var(--color-ink-muted)]">{label}</span>
      <span
        className={cn(
          "text-right text-xs text-[var(--color-ink)]",
          mono && "tabular",
        )}
      >
        {value}
      </span>
    </div>
  );
}
