"use client";

/**
 * The Recommendation Galaxy.
 *
 * Six thousand real item embeddings from the two-tower model, projected to 3D
 * offline and rendered as one point cloud. Picking a user runs a real request
 * and lights up the items each stage of the pipeline actually held, so the
 * funnel that the other pages describe in counts is visible here as shape.
 *
 * What this view is and is not:
 *
 * - The coordinates are the **model's** geometry. Nearby points are items the
 *   two-tower learned to place near each other; the clusters were not
 *   arranged to look like clusters.
 * - The projection is **lossy**. 64 dimensions do not fit in three, and both
 *   UMAP and PCA discard real structure to get there. Two points that look
 *   adjacent may not be neighbours in the space the model actually scores in.
 *
 * Both statements are on the page, not only in this comment, because a
 * convincing 3D visualisation invites more confidence than a projection can
 * carry.
 */

import { Loader2, Pause, Play, RotateCcw, Users } from "lucide-react";
import { Suspense, useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

import { GalaxyFallback } from "@/components/galaxy/GalaxyFallback";
import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { decodeFloat32, decodeInt32 } from "@/lib/api/client";
import { usePipelineTrace, useProjection, useSampleUsers, useUserPosition } from "@/lib/api/hooks";
import { RETRIEVAL_SOURCES, VERTICALS, verticalMeta } from "@/lib/constants";
import { cn, formatNumber } from "@/lib/utils";
import type { GalaxyPoints, GalaxyView } from "@/components/galaxy/GalaxyCanvas";
import dynamic from "next/dynamic";

/**
 * The WebGL scene is loaded only in the browser and only when it is going to
 * be shown. three.js and its React bindings are by far the heaviest thing in
 * this application, and eight of the nine pages never touch them - bundling
 * that cost into the shared chunk would slow every one of them down.
 */
const GalaxyCanvas = dynamic(
  () => import("@/components/galaxy/GalaxyCanvas").then((module) => module.GalaxyCanvas),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-[var(--color-ink-faint)]">
        <Loader2 className="size-4 animate-spin" aria-hidden />
        Loading the scene
      </div>
    ),
  },
);

const VIEWS: { key: GalaxyView; label: string; hint: string }[] = [
  { key: "all", label: "All", hint: "The whole projected catalogue." },
  {
    key: "candidates",
    label: "Candidates",
    hint: "Everything the four retrieval sources proposed, after fusion.",
  },
  {
    key: "ranked",
    label: "Top ranked",
    hint: "The ranker's best candidates, before business policy.",
  },
  { key: "final", label: "Final", hint: "What the request actually returned." },
];

/** How many ranker-ordered candidates count as "top ranked" in the scene. */
const TOP_RANKED = 50;

/* -------------------------------------------------------------------------- */

function useDecodedProjection(method: "umap" | "pca") {
  const query = useProjection(method, 6000);

  /* Decoded once per payload rather than per render: this allocates six typed
     arrays over 6,000 elements, and doing it on every keystroke in the user
     field would be pure waste. */
  const points = useMemo<GalaxyPoints | null>(() => {
    const data = query.data;
    if (!data) return null;
    return {
      x: decodeFloat32(data.x),
      y: decodeFloat32(data.y),
      z: decodeFloat32(data.z),
      itemId: decodeInt32(data.item_id),
      vertical: decodeInt32(data.vertical),
      price: decodeFloat32(data.price),
    };
  }, [query.data]);

  return { ...query, points };
}

/**
 * Decide whether to render WebGL at all.
 *
 * Three independent reasons not to: the visitor asked for reduced motion, the
 * viewport is too small for orbit controls to be usable with a thumb, or the
 * browser cannot give us a WebGL context. The 2D projection covers all three
 * and shows the same data.
 */
function useCanRender3D(): { ready: boolean; use3D: boolean; reason: string | null } {
  const [state, setState] = useState<{ ready: boolean; use3D: boolean; reason: string | null }>({
    ready: false,
    use3D: false,
    reason: null,
  });

  useEffect(() => {
    const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    const widthQuery = window.matchMedia("(min-width: 768px)");

    const evaluate = () => {
      if (motionQuery.matches) {
        setState({
          ready: true,
          use3D: false,
          reason: "Reduced motion is enabled, so the 2D projection is shown instead.",
        });
        return;
      }
      if (!widthQuery.matches) {
        setState({
          ready: true,
          use3D: false,
          reason: "The viewport is too narrow for orbit controls; showing the 2D projection.",
        });
        return;
      }

      let supported = false;
      try {
        const canvas = document.createElement("canvas");
        supported = Boolean(
          canvas.getContext("webgl2") ?? canvas.getContext("webgl"),
        );
      } catch {
        supported = false;
      }

      setState({
        ready: true,
        use3D: supported,
        reason: supported ? null : "This browser has no WebGL context; showing the 2D projection.",
      });
    };

    evaluate();
    motionQuery.addEventListener("change", evaluate);
    widthQuery.addEventListener("change", evaluate);
    return () => {
      motionQuery.removeEventListener("change", evaluate);
      widthQuery.removeEventListener("change", evaluate);
    };
  }, []);

  return state;
}

/* -------------------------------------------------------------------------- */

function Legend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5">
      {VERTICALS.map((vertical) => (
        <span
          key={vertical.id}
          className="flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)]"
        >
          <span
            className="size-2 rounded-full"
            style={{ background: vertical.color }}
            aria-hidden
          />
          {vertical.label}
        </span>
      ))}
    </div>
  );
}

function ToggleGroup<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { key: T; label: string; hint?: string }[];
  onChange: (next: T) => void;
  label: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-0.5"
    >
      {options.map((option) => (
        <button
          key={option.key}
          type="button"
          role="radio"
          aria-checked={value === option.key}
          title={option.hint}
          onClick={() => onChange(option.key)}
          className={cn(
            "rounded-[calc(var(--radius-control)-2px)] px-3 py-1.5 text-xs font-medium transition-colors",
            value === option.key
              ? "bg-[var(--color-mercury)] text-[var(--color-void)]"
              : "text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function GalaxyInner() {
  const router = useRouter();
  const params = useSearchParams();

  const userId = params.get("user") ?? "";
  const method = (params.get("method") === "pca" ? "pca" : "umap") as "umap" | "pca";
  const view = (VIEWS.find((entry) => entry.key === params.get("view"))?.key ??
    "all") as GalaxyView;

  const [autoRotate, setAutoRotate] = useState(true);
  const [hovered, setHovered] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);

  const setParam = useCallback(
    (key: string, value: string | null) => {
      const next = new URLSearchParams(params.toString());
      if (value === null) next.delete(key);
      else next.set(key, value);
      router.replace(`/galaxy?${next.toString()}`, { scroll: false });
    },
    [params, router],
  );

  const projection = useDecodedProjection(method);
  const samples = useSampleUsers(6);
  const traceQuery = usePipelineTrace(userId ? { userId, k: 10 } : null);
  const userPositionQuery = useUserPosition(userId || null, method);
  const render3D = useCanRender3D();

  const trace = traceQuery.data;

  /* Sets, not arrays: the tier pass tests membership once per point, and a
     linear scan of 600 candidates for each of 6,000 points would be 3.6
     million comparisons on every toggle. */
  const { candidates, ranked, finals } = useMemo(() => {
    if (!trace) {
      return { candidates: new Set<number>(), ranked: new Set<number>(), finals: new Set<number>() };
    }
    return {
      candidates: new Set(trace.candidate_ids),
      ranked: new Set(trace.ranked_ids.slice(0, TOP_RANKED)),
      finals: new Set(trace.final_ids),
    };
  }, [trace]);

  const userPosition = useMemo<[number, number, number] | null>(() => {
    const position = userPositionQuery.data?.position;
    return position ? [position[0], position[1], position[2]] : null;
  }, [userPositionQuery.data]);

  /* Look-up by item id for the readout. Built once per projection, because
     scanning 6,000 entries on every pointer move is the kind of thing that
     turns a smooth scene into a stuttering one. */
  const indexById = useMemo(() => {
    const map = new Map<number, number>();
    if (!projection.points) return map;
    for (let i = 0; i < projection.points.itemId.length; i += 1) {
      map.set(projection.points.itemId[i], i);
    }
    return map;
  }, [projection.points]);

  const inspected = hovered ?? selected;

  if (projection.isError) {
    return (
      <div className="space-y-6">
        <SectionHeading title="Recommendation galaxy" />
        <EmptyState
          tone="warning"
          title="No embedding projection is available"
          description="The galaxy plots real two-tower embeddings, so there is nothing honest to show until the model has been trained and projected. Nothing here is generated."
          command="uv run mercury viz project"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Recommendation galaxy"
        description="Six thousand item embeddings from the two-tower model, projected to three dimensions. Distance is similarity the model learned, not a layout."
        action={
          <div className="flex flex-wrap items-center gap-2">
            <ToggleGroup
              label="Projection method"
              value={method}
              onChange={(next) => setParam("method", next)}
              options={[
                {
                  key: "umap",
                  label: "UMAP",
                  hint: "Non-linear; preserves neighbourhoods. A user has no exact projection.",
                },
                {
                  key: "pca",
                  label: "PCA",
                  hint: "Linear; preserves global variance. A user projects in exactly.",
                },
              ]}
            />
            <button
              type="button"
              onClick={() => setAutoRotate((current) => !current)}
              className="inline-flex items-center gap-1.5 rounded-[var(--radius-control)] border border-[var(--color-hairline)] px-3 py-1.5 text-xs text-[var(--color-ink-muted)] transition-colors hover:bg-[var(--color-surface-2)]"
              aria-pressed={autoRotate}
            >
              {autoRotate ? (
                <Pause className="size-3.5" aria-hidden />
              ) : (
                <Play className="size-3.5" aria-hidden />
              )}
              {autoRotate ? "Pause" : "Rotate"}
            </button>
          </div>
        }
      />

      {/* --- user picker --- */}
      <GlassPanel className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
            <Users className="size-3.5" aria-hidden />
            Light up a request
          </div>
          {userId ? (
            <button
              type="button"
              onClick={() => setParam("user", null)}
              className="inline-flex items-center gap-1.5 text-xs text-[var(--color-ink-muted)] hover:text-[var(--color-ink)]"
            >
              <RotateCcw className="size-3.5" aria-hidden />
              Clear
            </button>
          ) : null}
        </div>

        {samples.data?.users?.length ? (
          <div className="flex flex-wrap gap-2">
            {samples.data.users.map((user) => (
              <button
                key={user.user_id}
                type="button"
                onClick={() => setParam("user", user.user_id)}
                title={`${user.history_items} items in training history · ${user.percentile}th percentile of activity`}
                className={cn(
                  "tabular rounded-[var(--radius-control)] border px-3 py-1.5 text-xs transition-colors",
                  userId === user.user_id
                    ? "border-[var(--color-mercury)] bg-[color-mix(in_oklch,var(--color-mercury)_14%,transparent)] text-[var(--color-mercury)]"
                    : "border-[var(--color-hairline)] text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
                )}
              >
                {user.user_id}
                <span className="ml-1.5 text-[var(--color-ink-faint)]">
                  {formatNumber(user.history_items, { compact: true })}
                </span>
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

        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup
            label="Pipeline stage"
            value={view}
            onChange={(next) => setParam("view", next)}
            options={VIEWS}
          />
          <span className="text-xs text-[var(--color-ink-faint)]">
            {userId
              ? VIEWS.find((entry) => entry.key === view)?.hint
              : "Pick a user to light up the stages."}
          </span>
        </div>
      </GlassPanel>

      {/* --- the scene --- */}
      <GlassPanel className="relative overflow-hidden p-0">
        <div className="relative h-[clamp(20rem,62vh,40rem)] w-full">
          {!projection.points || !render3D.ready ? (
            <div className="flex h-full items-center justify-center gap-2 text-sm text-[var(--color-ink-faint)]">
              <Loader2 className="size-4 animate-spin" aria-hidden />
              Loading {formatNumber(6000, { compact: true })} embeddings
            </div>
          ) : render3D.use3D ? (
            <GalaxyCanvas
              points={projection.points}
              candidates={candidates}
              ranked={ranked}
              finals={finals}
              view={view}
              userPosition={userPosition}
              autoRotate={autoRotate}
              onHoverItem={setHovered}
              onSelectItem={setSelected}
            />
          ) : (
            <GalaxyFallback
              points={projection.points}
              candidates={candidates}
              ranked={ranked}
              finals={finals}
              view={view}
              userPosition={userPosition}
            />
          )}

          {/* Hover readout, pinned rather than following the cursor: a tooltip
              chasing the pointer across a rotating scene is unreadable. */}
          {inspected !== null && projection.points && indexById.has(inspected) ? (
            <div className="glass pointer-events-none absolute bottom-3 left-3 rounded-[var(--radius-control)] px-3 py-2 text-xs">
              <div className="tabular font-medium">Item {inspected}</div>
              <div className="text-[var(--color-ink-muted)]">
                {verticalMeta(projection.points.vertical[indexById.get(inspected)!])?.label ??
                  "Unknown vertical"}
                {" · "}
                <span className="tabular">
                  {projection.points.price[indexById.get(inspected)!].toFixed(2)}
                </span>
              </div>
              <div className="mt-0.5 text-[var(--color-ink-faint)]">
                {finals.has(inspected)
                  ? `Returned, rank ${(trace?.final_ids.indexOf(inspected) ?? 0) + 1}`
                  : ranked.has(inspected)
                    ? "Top-ranked candidate"
                    : candidates.has(inspected)
                      ? "Retrieved candidate"
                      : "Not in this request"}
              </div>
            </div>
          ) : null}

          {render3D.reason ? (
            <div className="glass absolute right-3 top-3 max-w-[18rem] rounded-[var(--radius-control)] px-3 py-2 text-xs text-[var(--color-ink-muted)]">
              {render3D.reason}
            </div>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--color-hairline)] px-4 py-3">
          <Legend />
          <span className="text-xs text-[var(--color-ink-faint)]">
            {render3D.use3D ? "Drag to orbit, scroll to zoom" : "2D projection"}
          </span>
        </div>
      </GlassPanel>

      {/* --- what the request did --- */}
      <div className="grid gap-4 lg:grid-cols-2">
        <GlassPanel>
          <SectionHeading title="This request" />
          {!userId ? (
            <p className="text-sm text-[var(--color-ink-muted)]">
              Pick a user above. The galaxy will light up the items each stage held, drawn
              from a real uncached request.
            </p>
          ) : traceQuery.isLoading ? (
            <PanelSkeleton rows={4} />
          ) : traceQuery.isError ? (
            <p className="text-sm text-[var(--color-ink-muted)]">
              The trace could not be fetched. The API may not be running.
            </p>
          ) : trace ? (
            <div className="space-y-0">
              <DataRow
                label="Candidates retrieved"
                value={formatNumber(trace.candidate_ids.length)}
              />
              <DataRow
                label="Shown in the galaxy"
                value={formatNumber(
                  trace.candidate_ids.filter((id) => indexById.has(id)).length,
                )}
                title="The projection is a 6,000-item sample of the 36,044-item catalogue, so most candidates have no point to light up."
              />
              <DataRow label="Top ranked highlighted" value={formatNumber(TOP_RANKED)} />
              <DataRow label="Returned" value={formatNumber(trace.final_ids.length)} />
              <DataRow
                label="Total latency"
                value={`${trace.latency.total_ms.toFixed(1)} ms`}
                title={trace.note}
              />
              {trace.is_cold_start ? (
                <p className="pt-3 text-xs text-[var(--color-ink-faint)]">
                  This user was served by the cold-start path, which answers from contextual
                  popularity without running retrieval. There are no stages to light up.
                </p>
              ) : null}
            </div>
          ) : null}

          {trace && !trace.is_cold_start ? (
            <div className="mt-4 space-y-1.5">
              {Object.entries(trace.candidate_sources).map(([source, ids]) => (
                <div key={source} className="flex items-center gap-2 text-xs">
                  <span
                    className="w-40 shrink-0 text-[var(--color-ink-muted)]"
                    title={RETRIEVAL_SOURCES[source]?.description}
                  >
                    {RETRIEVAL_SOURCES[source]?.label ?? source}
                  </span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
                    <div
                      className="h-full rounded-full bg-[var(--color-mercury-dim)]"
                      style={{
                        width: `${(ids.length / Math.max(trace.candidate_ids.length, 1)) * 100}%`,
                      }}
                    />
                  </div>
                  <span className="tabular w-10 shrink-0 text-right text-[var(--color-ink-faint)]">
                    {ids.length}
                  </span>
                </div>
              ))}
            </div>
          ) : null}
        </GlassPanel>

        <GlassPanel>
          <SectionHeading title="What you are looking at" />
          <div className="space-y-3 text-sm leading-relaxed text-[var(--color-ink-muted)]">
            <p>
              Each point is one catalogue item, positioned by its 64-dimensional two-tower
              embedding reduced to three dimensions. The reduction runs offline:{" "}
              <span className="text-[var(--color-ink)]">UMAP takes tens of seconds</span> for
              6,000 points and has no cheap transform for a new one, so it cannot live in a
              request.
            </p>
            <p>
              <span className="text-[var(--color-ink)]">The projection is lossy.</span> Sixty-four
              dimensions do not fit in three, and both methods discard real structure to get
              there. Points that look adjacent here are not necessarily neighbours in the space
              the model scores in.
            </p>
            {userId && userPositionQuery.data ? (
              <p>
                <span className="text-[var(--color-ink)]">The white marker is the user.</span>{" "}
                {userPositionQuery.data.note}
              </p>
            ) : null}
          </div>

          <div className="mt-4 space-y-0">
            <DataRow
              label="Points plotted"
              value={formatNumber(projection.points?.itemId.length ?? null)}
            />
            <DataRow label="Projection" value={method.toUpperCase()} />
            <DataRow
              label="User placement"
              value={
                !userId ? (
                  "—"
                ) : userPositionQuery.data?.position ? (
                  <Badge tone={userPositionQuery.data.is_exact ? "positive" : "warning"}>
                    {userPositionQuery.data.is_exact ? "exact" : "approximate"}
                  </Badge>
                ) : (
                  <Badge tone="neutral">no embedding</Badge>
                )
              }
              mono={false}
            />
            <DataRow
              label="Transfer encoding"
              value="base64 Float32Array"
              title="Roughly 75KB of binary rather than 450KB of JSON text, decoded straight into a BufferAttribute."
            />
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}

export default function GalaxyPage() {
  // useSearchParams requires a Suspense boundary in the App Router.
  return (
    <Suspense
      fallback={
        <div className="space-y-6">
          <SectionHeading title="Recommendation galaxy" />
          <PanelSkeleton rows={6} />
        </div>
      }
    >
      <GalaxyInner />
    </Suspense>
  );
}
