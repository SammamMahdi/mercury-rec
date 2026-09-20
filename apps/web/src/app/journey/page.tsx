"use client";

/**
 * Customer journey.
 *
 * Personalisation is a claim until you can watch it change. This page builds a
 * session one item at a time and shows what the recommendations do in
 * response, then sweeps the same session across the day to show the contextual
 * half of the same story.
 *
 * The session is simulated: nobody is browsing. What is real is the code path
 * — every step is a live POST to the session endpoint, served by the same
 * engine, with the session items carried into the same feature emitter that
 * training used.
 */

import { ArrowDown, ArrowUp, Clock, Plus, RotateCcw, Sparkles } from "lucide-react";
import { Suspense, useCallback, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useHourSweep, useSampleUsers, useSessionRecommendations } from "@/lib/api/hooks";
import { HOUR_BUCKETS, verticalMeta } from "@/lib/constants";
import { cn, formatNumber } from "@/lib/utils";
import type { Recommendation } from "@/lib/api/types";

const SWEEP_HOURS = HOUR_BUCKETS.map((bucket) => bucket.representative);

/**
 * Sentinel for "no user id at all".
 *
 * Kept as a URL value rather than a separate flag so an anonymous journey is
 * as shareable as an identified one.
 */
const ANONYMOUS = "anon";

/* -------------------------------------------------------------------------- */

type Movement = "new" | "up" | "down" | "same";

function movementOf(item: Recommendation, previous: Recommendation[] | null): Movement {
  if (!previous) return "same";
  const before = previous.findIndex((entry) => entry.item_id === item.item_id);
  if (before === -1) return "new";
  if (before > item.rank - 1) return "up";
  if (before < item.rank - 1) return "down";
  return "same";
}

const MOVEMENT_BADGE: Record<Movement, { label: string; className: string } | null> = {
  new: {
    label: "new",
    className:
      "bg-[color-mix(in_oklch,var(--color-mercury)_16%,transparent)] text-[var(--color-mercury)]",
  },
  up: { label: "up", className: "text-[var(--color-positive)]" },
  down: { label: "down", className: "text-[var(--color-ink-faint)]" },
  same: null,
};

function RecommendationCard({
  item,
  movement,
  onAdd,
}: {
  item: Recommendation;
  movement: Movement;
  onAdd: () => void;
}) {
  const vertical = verticalMeta(item.vertical);
  const badge = MOVEMENT_BADGE[movement];

  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-[var(--radius-control)] border p-2.5 transition-colors",
        movement === "new"
          ? "border-[color-mix(in_oklch,var(--color-mercury)_35%,transparent)]"
          : "border-[var(--color-hairline)]",
      )}
    >
      <span className="tabular w-6 shrink-0 text-center text-xs text-[var(--color-ink-faint)]">
        {item.rank}
      </span>
      <span
        className="size-2 shrink-0 rounded-full"
        style={{ background: vertical?.color ?? "var(--color-surface-3)" }}
        title={vertical?.label ?? "Unknown vertical"}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <div className="tabular truncate text-sm">Item {item.item_id}</div>
        <div className="truncate text-xs text-[var(--color-ink-faint)]">
          {vertical?.label ?? "—"}
          {item.price !== null ? ` · ${item.price.toFixed(2)}` : ""}
          {item.sources.length > 0 ? ` · ${item.sources.join(", ")}` : ""}
        </div>
      </div>

      {badge ? (
        <span
          className={cn(
            "flex shrink-0 items-center gap-0.5 rounded-full px-1.5 py-0.5 text-[10px] font-medium",
            badge.className,
          )}
        >
          {movement === "up" ? (
            <ArrowUp className="size-2.5" aria-hidden />
          ) : movement === "down" ? (
            <ArrowDown className="size-2.5" aria-hidden />
          ) : null}
          {badge.label}
        </span>
      ) : null}

      <span className="tabular w-12 shrink-0 text-right text-xs text-[var(--color-ink-muted)]">
        {item.ml_relevance_score.toFixed(3)}
      </span>

      <button
        type="button"
        onClick={onAdd}
        title="Add this item to the session and re-run"
        className="shrink-0 rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-1.5 text-[var(--color-ink-muted)] transition-colors hover:bg-[var(--color-surface-2)] hover:text-[var(--color-ink)]"
      >
        <Plus className="size-3.5" aria-hidden />
        <span className="sr-only">Add item {item.item_id} to the session</span>
      </button>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

/** The same session, scored at six times of day. */
function HourSweep({
  userId,
  sessionItems,
}: {
  userId: string | null;
  sessionItems: number[];
}) {
  const results = useHourSweep(userId, SWEEP_HOURS, sessionItems);

  const anyLoading = results.some((result) => result.isLoading);
  const rows = results.map((result, index) => ({
    bucket: HOUR_BUCKETS[index],
    top: result.data?.recommendations.slice(0, 3) ?? [],
    error: result.isError,
  }));

  const distinct = new Set(
    rows.flatMap((row) => row.top.map((item) => item.item_id)),
  ).size;
  const maxDistinct = rows.reduce((sum, row) => sum + row.top.length, 0);

  if (anyLoading) return <PanelSkeleton rows={4} />;

  return (
    <div className="space-y-3">
      <div className="space-y-1.5">
        {rows.map((row) => (
          <div key={row.bucket.label} className="flex items-center gap-3">
            <span className="w-20 shrink-0 text-xs text-[var(--color-ink-muted)]">
              {row.bucket.label}
            </span>
            <span className="tabular w-10 shrink-0 text-xs text-[var(--color-ink-faint)]">
              {String(row.bucket.representative).padStart(2, "0")}:00
            </span>
            <div className="flex min-w-0 flex-1 gap-1.5">
              {row.error ? (
                <span className="text-xs text-[var(--color-ink-faint)]">unavailable</span>
              ) : (
                row.top.map((item) => (
                  <span
                    key={item.item_id}
                    className="tabular truncate rounded-[var(--radius-control)] border border-[var(--color-hairline)] px-2 py-1 text-xs text-[var(--color-ink-muted)]"
                    style={{
                      borderLeftColor:
                        verticalMeta(item.vertical)?.color ?? "var(--color-hairline)",
                      borderLeftWidth: 2,
                    }}
                    title={`${verticalMeta(item.vertical)?.label ?? "Unknown"} · score ${item.ml_relevance_score.toFixed(3)}`}
                  >
                    {item.item_id}
                  </span>
                ))
              )}
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs leading-relaxed text-[var(--color-ink-faint)]">
        {distinct} distinct items across {maxDistinct} slots.{" "}
        {distinct <= 1 + maxDistinct / 6
          ? "Time of day barely moves this user's slate, which is the honest reading: their own history dominates the contextual prior."
          : "Time of day visibly reorders the slate for this user."}
      </p>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function JourneyInner() {
  const router = useRouter();
  const params = useSearchParams();
  const userId = params.get("user") ?? "";

  const [session, setSession] = useState<number[]>([]);
  const [previous, setPrevious] = useState<Recommendation[] | null>(null);

  const setUser = useCallback(
    (value: string) => {
      const next = new URLSearchParams(params.toString());
      next.set("user", value);
      router.replace(`/journey?${next.toString()}`, { scroll: false });
      setSession([]);
      setPrevious(null);
    },
    [params, router],
  );

  const samples = useSampleUsers(6);
  const isAnonymous = userId === ANONYMOUS;
  const sessionParams = useMemo(
    () =>
      userId
        ? { userId: isAnonymous ? null : userId, sessionItems: session, k: 8 }
        : null,
    [userId, isAnonymous, session],
  );
  const { data, isFetching, isError } = useSessionRecommendations(sessionParams);

  const addToSession = (itemId: number) => {
    setPrevious(data?.recommendations ?? null);
    setSession((current) => (current.includes(itemId) ? current : [...current, itemId]));
  };

  const reset = () => {
    setSession([]);
    setPrevious(null);
  };

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Customer journey"
        description="Build a session one item at a time and watch the slate respond. Every step is a live request to the session endpoint."
      />

      <GlassPanel className="space-y-3">
        <span className="text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
          Start from a user
        </span>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => setUser(ANONYMOUS)}
            title="No user id at all: the request carries nothing but the session"
            className={cn(
              "rounded-[var(--radius-control)] border px-3 py-1.5 text-xs transition-colors",
              isAnonymous
                ? "border-[var(--color-warning)] bg-[color-mix(in_oklch,var(--color-warning)_14%,transparent)] text-[var(--color-warning)]"
                : "border-[var(--color-hairline)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
            )}
          >
            Anonymous visitor
          </button>
        </div>

        {samples.data?.users?.length ? (
          <div className="flex flex-wrap gap-2">
            {samples.data.users.map((user) => (
              <button
                key={user.user_id}
                type="button"
                onClick={() => setUser(user.user_id)}
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
          title="Pick a user to begin a session"
          description="The journey is simulated, but the requests behind it are real. Nothing is shown until one has been made."
        />
      ) : isError ? (
        <EmptyState
          tone="warning"
          title="The session request failed"
          description="The API is not reachable."
          command="uv run uvicorn mercury_rec.api.main:app --port 8000"
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <GlassPanel>
            <SectionHeading
              title={session.length === 0 ? "Opening slate" : `After ${session.length} interaction${session.length === 1 ? "" : "s"}`}
              description={
                session.length === 0
                  ? "What this user sees before doing anything in the session."
                  : "Re-scored with the session items carried into the feature vector."
              }
              action={
                session.length > 0 ? (
                  <button
                    type="button"
                    onClick={reset}
                    className="inline-flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
                  >
                    <RotateCcw className="size-3.5" aria-hidden />
                    Restart
                  </button>
                ) : null
              }
            />

            {data ? (
              <div className={cn("space-y-1.5 transition-opacity", isFetching && "opacity-60")}>
                {data.recommendations.map((item) => (
                  <RecommendationCard
                    key={item.item_id}
                    item={item}
                    movement={movementOf(item, previous)}
                    onAdd={() => addToSession(item.item_id)}
                  />
                ))}
              </div>
            ) : (
              <PanelSkeleton rows={5} />
            )}

            {data?.is_cold_start ? (
              <p className="mt-3 text-xs leading-relaxed text-[var(--color-ink-faint)]">
                {session.length > 0
                  ? "Served by the cold-start path, but not by popularity alone: with no user id, the session IS the signal, so the slate is item-similarity to what has been viewed, blended with the contextual prior."
                  : "Served by the cold-start path. With nothing at all to go on, contextual popularity is the honest best guess. Add an item and the slate stops being generic."}
              </p>
            ) : session.length > 0 ? (
              <p className="mt-3 text-xs leading-relaxed text-[var(--color-ink-faint)]">
                This user has real history, and it dominates a three-item session — the
                slate barely moves. That is the honest result, and the reason the anonymous
                path above is where session signal earns its place.
              </p>
            ) : null}
          </GlassPanel>

          <div className="space-y-4">
            <GlassPanel>
              <SectionHeading title="Session so far" />
              {session.length === 0 ? (
                <p className="text-sm text-[var(--color-ink-muted)]">
                  Empty. Add an item with the{" "}
                  <Plus className="inline size-3" aria-hidden /> button to advance the
                  journey.
                </p>
              ) : (
                <ol className="space-y-1.5">
                  {session.map((itemId, index) => (
                    <li key={itemId} className="flex items-center gap-2 text-xs">
                      <span className="tabular w-5 shrink-0 text-[var(--color-ink-faint)]">
                        {index + 1}
                      </span>
                      <span className="tabular flex-1 truncate">Item {itemId}</span>
                    </li>
                  ))}
                </ol>
              )}
            </GlassPanel>

            {data ? (
              <GlassPanel>
                <SectionHeading title="This request" />
                <div className="space-y-0">
                  <DataRow
                    label="Candidates"
                    value={formatNumber(data.n_candidates)}
                    title="The session context is unique per step, so none of these are cache hits."
                  />
                  <DataRow label="Latency" value={`${data.latency.total_ms.toFixed(1)} ms`} />
                  <DataRow
                    label="Cache"
                    value={
                      <Badge tone="neutral">{data.cache_hit ? "hit" : "bypassed"}</Badge>
                    }
                    mono={false}
                  />
                  <DataRow
                    label="Cold start"
                    value={
                      <Badge tone={data.is_cold_start ? "warning" : "positive"}>
                        {data.is_cold_start ? "yes" : "no"}
                      </Badge>
                    }
                    mono={false}
                  />
                </div>
              </GlassPanel>
            ) : null}
          </div>
        </div>
      )}

      {userId ? (
        <GlassPanel>
          <SectionHeading
            title="The same session, across the day"
            description="Six requests, identical except for the hour. The contextual popularity prior is the only thing that changes."
            action={
              <Badge tone="neutral">
                <Clock className="size-3" aria-hidden />
                {SWEEP_HOURS.length} requests
              </Badge>
            }
          />
          <HourSweep userId={isAnonymous ? null : userId} sessionItems={session} />
        </GlassPanel>
      ) : null}

      <div className="flex items-start gap-3 rounded-[var(--radius-panel)] border border-[var(--color-hairline)] p-4">
        <Sparkles className="mt-0.5 size-4 shrink-0 text-[var(--color-ink-faint)]" aria-hidden />
        <p className="text-xs leading-relaxed text-[var(--color-ink-muted)]">
          The session is simulated — no one is browsing this catalogue. The requests are
          not: each step is a real POST served by the same engine, through the same feature
          emitter the training pipeline used. What this page demonstrates is that the code
          path exists and responds, not that a customer behaved this way. Compare the two
          modes: for a user with history the session barely moves the slate, while for an
          anonymous visitor it decides it entirely.
        </p>
      </div>
    </div>
  );
}

export default function JourneyPage() {
  // useSearchParams requires a Suspense boundary in the App Router.
  return (
    <Suspense
      fallback={
        <div className="space-y-6">
          <SectionHeading title="Customer journey" />
          <PanelSkeleton rows={6} />
        </div>
      }
    >
      <JourneyInner />
    </Suspense>
  );
}
