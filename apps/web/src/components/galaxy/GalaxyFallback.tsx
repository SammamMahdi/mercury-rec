"use client";

/**
 * The galaxy without WebGL.
 *
 * Shown when the visitor has asked for reduced motion, when the viewport is
 * too small for orbit controls to be usable, or when WebGL is unavailable.
 *
 * This is not a placeholder. It plots the same coordinates from the same
 * projection and keeps the same stage highlighting, dropping only the third
 * dimension and the rotation. Someone who cannot use the 3D view still learns
 * what the page is for: that the model's embedding space has structure, and
 * that a request lands in part of it.
 */

import { useMemo } from "react";

import { VERTICALS } from "@/lib/constants";

import type { GalaxyPoints, GalaxyView } from "./GalaxyCanvas";

const SIZE = 600;
const PADDING = 12;

export function GalaxyFallback({
  points,
  candidates,
  ranked,
  finals,
  view,
  userPosition,
}: {
  points: GalaxyPoints;
  candidates: Set<number>;
  ranked: Set<number>;
  finals: Set<number>;
  view: GalaxyView;
  userPosition: [number, number, number] | null;
}) {
  /* Drop the z axis and map x/y into the viewBox. The projection is already
     normalised to a unit sphere, so the extents are known without a scan. */
  const project = useMemo(() => {
    const span = SIZE - PADDING * 2;
    return (x: number, y: number) => ({
      cx: PADDING + ((x + 1) / 2) * span,
      // SVG y grows downward; flipping keeps the shape the same as in 3D.
      cy: PADDING + ((1 - y) / 2) * span,
    });
  }, []);

  const rendered = useMemo(() => {
    const hasSelection = candidates.size > 0 || ranked.size > 0 || finals.size > 0;
    const background: { cx: number; cy: number; fill: string }[] = [];
    const highlighted: { cx: number; cy: number; fill: string; r: number }[] = [];

    for (let i = 0; i < points.itemId.length; i += 1) {
      const id = points.itemId[i];
      const fill = VERTICALS[points.vertical[i]]?.color ?? VERTICALS[0].color;
      const { cx, cy } = project(points.x[i], points.y[i]);

      const isFinal = finals.has(id);
      const isRanked = ranked.has(id);
      const isCandidate = candidates.has(id);

      if (isFinal) highlighted.push({ cx, cy, fill, r: 5 });
      else if (isRanked && view !== "final") highlighted.push({ cx, cy, fill, r: 3.5 });
      else if (isCandidate && (view === "all" || view === "candidates"))
        highlighted.push({ cx, cy, fill, r: 2.5 });
      else if (!hasSelection || view === "all") background.push({ cx, cy, fill });
    }
    return { background, highlighted };
  }, [points, candidates, ranked, finals, view, project]);

  const user = userPosition ? project(userPosition[0], userPosition[1]) : null;

  return (
    <svg
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="h-full w-full"
      role="img"
      aria-label={
        `Two-dimensional projection of ${points.itemId.length.toLocaleString()} item ` +
        `embeddings, coloured by vertical` +
        (rendered.highlighted.length > 0
          ? `, with ${rendered.highlighted.length.toLocaleString()} items highlighted as the current request's candidates.`
          : ".")
      }
    >
      {/* Background items first, so highlights are never painted under them. */}
      <g opacity={rendered.highlighted.length > 0 ? 0.22 : 0.55}>
        {rendered.background.map((point, index) => (
          <circle key={index} cx={point.cx} cy={point.cy} r={1.4} fill={point.fill} />
        ))}
      </g>
      <g>
        {rendered.highlighted.map((point, index) => (
          <circle
            key={index}
            cx={point.cx}
            cy={point.cy}
            r={point.r}
            fill={point.fill}
            opacity={0.95}
          />
        ))}
      </g>
      {user ? (
        <g>
          <circle
            cx={user.cx}
            cy={user.cy}
            r={9}
            fill="none"
            stroke="var(--color-ink)"
            strokeWidth={1}
            opacity={0.5}
          />
          <circle cx={user.cx} cy={user.cy} r={3.5} fill="var(--color-ink)" />
        </g>
      ) : null}
    </svg>
  );
}
