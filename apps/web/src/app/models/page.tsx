"use client";

/**
 * Model comparison.
 *
 * The offline table, reported as measured. The two-tower model does not beat
 * matrix factorisation on this dataset, and that is stated on the page rather
 * than buried: a portfolio that only shows the runs which went well is not
 * showing evaluation, it is showing selection.
 *
 * Accuracy is deliberately not the only column. A recommender that scores well
 * on NDCG while serving the same fifty items to everyone has a real problem
 * that NDCG cannot see, so coverage, Gini and mean popularity rank sit in the
 * same table at the same weight.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Boxes, Gauge, Timer, Trophy } from "lucide-react";
import { useMemo, useState } from "react";

import {
  Badge,
  DataRow,
  EmptyState,
  GlassPanel,
  MetricTile,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useEvaluation, useModelStatus, useRanking } from "@/lib/api/hooks";
import { featureLabel } from "@/lib/constants";
import { cn, formatNumber, shortHash } from "@/lib/utils";
import type { ModelResultRow } from "@/lib/api/types";

/** Pull a dynamically-keyed metric off a result row. */
function metric(row: ModelResultRow, key: string): number | null {
  const value = row[key];
  return typeof value === "number" ? value : null;
}

const MODEL_LABELS: Record<string, string> = {
  popularity: "Popularity",
  popularity_contextual: "Contextual popularity",
  popularity_trending: "Trending",
  item_cf: "Item-CF",
  bpr_mf: "Matrix factorisation",
  two_tower: "Two-tower",
};

const MODEL_NOTES: Record<string, string> = {
  popularity: "The floor. Everything else has to justify its cost against this.",
  popularity_contextual:
    "Popularity conditioned on region, vertical and time of day, falling back when a cell is too thin to estimate.",
  popularity_trending: "Recency-weighted counts, so a rising item is not buried by all-time totals.",
  item_cf: "Cosine similarity between item columns, damped for popularity.",
  bpr_mf: "Pairwise ranking loss over latent factors, trained with negative sampling.",
  two_tower:
    "Neural retrieval with separate towers. Serving cost is independent of catalogue size, which is the architectural point even where the accuracy is not the best here.",
};

/* -------------------------------------------------------------------------- */

function ComparisonChart({ rows, metricKey }: { rows: ModelResultRow[]; metricKey: string }) {
  const data = useMemo(
    () =>
      rows
        .map((row) => ({
          name: MODEL_LABELS[row.model] ?? row.model,
          value: metric(row, metricKey) ?? 0,
          coverage: row.catalog_coverage,
        }))
        .sort((a, b) => b.value - a.value),
    [rows, metricKey],
  );

  const best = data[0]?.value ?? 0;

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24, top: 4, bottom: 4 }}>
          <CartesianGrid horizontal={false} stroke="var(--color-hairline)" strokeDasharray="2 4" />
          <XAxis
            type="number"
            tick={{ fill: "var(--color-ink-faint)", fontSize: 11 }}
            stroke="var(--color-hairline)"
            tickFormatter={(value: number) => value.toFixed(3)}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={140}
            tick={{ fill: "var(--color-ink-muted)", fontSize: 11 }}
            stroke="var(--color-hairline)"
          />
          <Tooltip
            cursor={{ fill: "var(--color-surface-1)" }}
            contentStyle={{
              background: "var(--color-surface-1)",
              border: "1px solid var(--color-hairline)",
              borderRadius: 10,
              fontSize: 12,
            }}
            formatter={(value) => [Number(value).toFixed(4), metricKey]}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry) => (
              <Cell
                key={entry.name}
                fill={
                  entry.value === best ? "var(--color-mercury)" : "var(--color-mercury-dim)"
                }
                opacity={entry.value === best ? 1 : 0.55}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

function ResultsTable({ rows, kValues }: { rows: ModelResultRow[]; kValues: number[] }) {
  const k = kValues.includes(10) ? 10 : (kValues[kValues.length - 1] ?? 10);

  const columns: { key: string; label: string; decimals: number; title: string }[] = [
    {
      key: `ndcg@${k}`,
      label: `NDCG@${k}`,
      decimals: 4,
      title: "Position-weighted relevance. The headline ranking metric.",
    },
    {
      key: `recall@${k}`,
      label: `Recall@${k}`,
      decimals: 4,
      title: "Share of a user's held-out novel items that appeared in the top k.",
    },
    {
      key: `map@${k}`,
      label: `MAP@${k}`,
      decimals: 4,
      title: "Mean average precision; rewards putting several hits high rather than one.",
    },
  ];

  const bestOf = (key: string) =>
    Math.max(...rows.map((row) => metric(row, key) ?? Number.NEGATIVE_INFINITY));

  return (
    <div className="-mx-1 overflow-x-auto px-1">
      <table className="w-full min-w-[46rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-[var(--color-hairline)] text-left">
            <th className="py-2 pr-4 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
              Model
            </th>
            {columns.map((column) => (
              <th
                key={column.key}
                title={column.title}
                className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
              >
                {column.label}
              </th>
            ))}
            <th
              title="Share of the catalogue that was ever recommended to anyone."
              className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
            >
              Coverage
            </th>
            <th
              title="Concentration of exposure. 0 is perfectly even, 1 is one item taking everything."
              className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
            >
              Gini
            </th>
            <th
              title="Wall-clock training time on the hardware documented in the benchmarks."
              className="py-2 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]"
            >
              Train
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.model}
              className="border-b border-[var(--color-hairline)] last:border-0"
            >
              <td className="py-2.5 pr-4">
                <div className="font-medium">{MODEL_LABELS[row.model] ?? row.model}</div>
                <div className="mt-0.5 max-w-sm text-xs leading-relaxed text-[var(--color-ink-faint)]">
                  {MODEL_NOTES[row.model]}
                </div>
              </td>
              {columns.map((column) => {
                const value = metric(row, column.key);
                const isBest = value !== null && value === bestOf(column.key);
                return (
                  <td
                    key={column.key}
                    className={cn(
                      "tabular py-2.5 pr-4 text-right align-top",
                      isBest
                        ? "font-semibold text-[var(--color-mercury)]"
                        : "text-[var(--color-ink-muted)]",
                    )}
                  >
                    {value === null ? "—" : value.toFixed(column.decimals)}
                  </td>
                );
              })}
              <td className="tabular py-2.5 pr-4 text-right align-top text-[var(--color-ink-muted)]">
                {(row.catalog_coverage * 100).toFixed(1)}%
              </td>
              <td className="tabular py-2.5 pr-4 text-right align-top text-[var(--color-ink-muted)]">
                {row.gini.toFixed(3)}
              </td>
              <td className="tabular py-2.5 text-right align-top text-[var(--color-ink-muted)]">
                {row.train_seconds < 1
                  ? `${(row.train_seconds * 1000).toFixed(0)} ms`
                  : `${row.train_seconds.toFixed(1)} s`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* -------------------------------------------------------------------------- */

export default function ModelsPage() {
  const evaluation = useEvaluation();
  const ranking = useRanking();
  const status = useModelStatus();
  const [metricKey, setMetricKey] = useState("ndcg@10");

  // Memoised so the empty-array fallback keeps a stable identity; a fresh
  // [] on every render would invalidate every downstream memo.
  const rows = useMemo(() => evaluation.data?.results ?? [], [evaluation.data]);
  const kValues = evaluation.data?.k_values ?? [10];

  const best = useMemo(
    () =>
      rows.reduce<ModelResultRow | null>((winner, row) => {
        const current = metric(row, "ndcg@10") ?? -1;
        const incumbent = winner ? (metric(winner, "ndcg@10") ?? -1) : -1;
        return current > incumbent ? row : winner;
      }, null),
    [rows],
  );

  const broadest = useMemo(
    () =>
      rows.reduce<ModelResultRow | null>(
        (winner, row) =>
          !winner || row.catalog_coverage > winner.catalog_coverage ? row : winner,
        null,
      ),
    [rows],
  );

  const topFeatures = useMemo(() => {
    const features = ranking.data?.ranker.top_features ?? {};
    const total = Object.values(features).reduce((sum, value) => sum + value, 0);
    return Object.entries(features)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([name, gain]) => ({ name, gain, share: total > 0 ? gain / total : 0 }));
  }, [ranking.data]);

  if (evaluation.isError) {
    return (
      <div className="space-y-6">
        <SectionHeading title="Models" />
        <EmptyState
          tone="warning"
          title="No evaluation results are available"
          description="This page reports measured results only. Until an evaluation has been run and committed there is nothing to show."
          command="uv run mercury train baselines"
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Models"
        description="Every model evaluated under one protocol, on one split, with the same ground-truth definition."
        action={
          evaluation.data ? (
            <Badge tone="neutral" title="Dataset the results were produced on">
              <span className="tabular">{shortHash(evaluation.data.dataset_hash)}</span>
            </Badge>
          ) : null
        }
      />

      {evaluation.isLoading ? (
        <PanelSkeleton rows={6} />
      ) : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricTile
              label="Best NDCG@10"
              value={best ? metric(best, "ndcg@10") : null}
              decimals={4}
              icon={Trophy}
              hint={best ? (MODEL_LABELS[best.model] ?? best.model) : undefined}
              tone="positive"
            />
            <MetricTile
              label="Best recall@10"
              value={best ? metric(best, "recall@10") : null}
              decimals={4}
              hint={best ? (MODEL_LABELS[best.model] ?? best.model) : undefined}
            />
            <MetricTile
              label="Widest coverage"
              value={broadest?.catalog_coverage ?? null}
              percent
              decimals={1}
              icon={Boxes}
              hint={broadest ? (MODEL_LABELS[broadest.model] ?? broadest.model) : undefined}
            />
            <MetricTile
              label="Users scored"
              value={evaluation.data?.users_scored ?? null}
              compact
              icon={Gauge}
              hint={
                evaluation.data
                  ? `of ${formatNumber(evaluation.data.n_users, { compact: true })} in the dataset`
                  : undefined
              }
            />
          </div>

          <GlassPanel>
            <SectionHeading
              title="Offline comparison"
              description={
                evaluation.data
                  ? `${evaluation.data.split} split · ground truth is held-out items the user had not already interacted with.`
                  : undefined
              }
              action={
                <div className="inline-flex rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-0.5">
                  {["ndcg@10", "recall@10", "map@10"].map((key) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => setMetricKey(key)}
                      className={cn(
                        "rounded-[calc(var(--radius-control)-2px)] px-2.5 py-1 text-xs font-medium transition-colors",
                        metricKey === key
                          ? "bg-[var(--color-mercury)] text-[var(--color-void)]"
                          : "text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)]",
                      )}
                    >
                      {key}
                    </button>
                  ))}
                </div>
              }
            />
            <ComparisonChart rows={rows} metricKey={metricKey} />
          </GlassPanel>

          <GlassPanel>
            <SectionHeading title="Full results" />
            <ResultsTable rows={rows} kValues={kValues} />
            <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-faint)]">
              These figures describe the active core of the catalogue after k-core
              filtering, and are not comparable to published session-based results on this
              dataset. Accuracy and coverage are shown together because a model can win the
              first by collapsing the second.
            </p>
          </GlassPanel>

          {evaluation.data?.index_benchmarks?.length ? (
            <GlassPanel>
              <SectionHeading
                title="Approximate versus exact search"
                description="Measured on this catalogue, on the hardware documented in the benchmarks."
              />
              <div className="-mx-1 overflow-x-auto px-1">
                <table className="w-full min-w-[34rem] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-[var(--color-hairline)] text-left">
                      <th className="py-2 pr-4 text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Index
                      </th>
                      <th className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Build
                      </th>
                      <th className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Mean query
                      </th>
                      <th className="py-2 pr-4 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        p95
                      </th>
                      <th className="py-2 text-right text-xs font-medium uppercase tracking-wider text-[var(--color-ink-faint)]">
                        Recall
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {evaluation.data.index_benchmarks.map((benchmark) => (
                      <tr
                        key={benchmark.index}
                        className="border-b border-[var(--color-hairline)] last:border-0"
                      >
                        <td className="py-2.5 pr-4 font-medium">{benchmark.index}</td>
                        <td className="tabular py-2.5 pr-4 text-right text-[var(--color-ink-muted)]">
                          {benchmark.build_seconds.toFixed(2)} s
                        </td>
                        <td className="tabular py-2.5 pr-4 text-right text-[var(--color-ink-muted)]">
                          {benchmark.mean_query_ms.toFixed(3)} ms
                        </td>
                        <td className="tabular py-2.5 pr-4 text-right text-[var(--color-ink-muted)]">
                          {benchmark.p95_query_ms.toFixed(3)} ms
                        </td>
                        <td className="tabular py-2.5 text-right text-[var(--color-ink-muted)]">
                          {benchmark.recall_at_k.toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-4 text-xs leading-relaxed text-[var(--color-ink-faint)]">
                At this catalogue size exact search is competitive, which is the honest
                answer rather than the expected one. The approximate index is kept because
                the gap opens with scale, and a system that has never run one has no idea
                what it would cost to add.
              </p>
            </GlassPanel>
          ) : null}

          <div className="grid gap-4 lg:grid-cols-2">
            <GlassPanel>
              <SectionHeading
                title="What the ranker leans on"
                description="Split gain from the trained LambdaRank model, not an estimate."
              />
              {topFeatures.length > 0 ? (
                <div className="space-y-1.5">
                  {topFeatures.map((feature) => (
                    <div key={feature.name} className="flex items-center gap-2 text-xs">
                      <span
                        className="w-44 shrink-0 truncate text-[var(--color-ink-muted)]"
                        title={feature.name}
                      >
                        {featureLabel(feature.name)}
                      </span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-surface-2)]">
                        <div
                          className="h-full rounded-full bg-[var(--color-mercury)]"
                          style={{ width: `${feature.share * 100}%` }}
                        />
                      </div>
                      <span className="tabular w-12 shrink-0 text-right text-[var(--color-ink-faint)]">
                        {(feature.share * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <PanelSkeleton rows={4} />
              )}
            </GlassPanel>

            <GlassPanel>
              <SectionHeading title="Training and serving" />
              {ranking.data ? (
                <div className="space-y-0">
                  <DataRow
                    label="Best iteration"
                    value={formatNumber(ranking.data.ranker.best_iteration)}
                    title="Chosen by early stopping on the validation split, not by a fixed budget."
                  />
                  <DataRow
                    label="Training groups"
                    value={formatNumber(ranking.data.ranker.train_groups)}
                    title="One group per (user, as-of) candidate list. LambdaRank optimises within a group."
                  />
                  <DataRow
                    label="Training rows"
                    value={formatNumber(ranking.data.ranker.train_rows)}
                  />
                  <DataRow
                    label="Positive rate"
                    value={`${(ranking.data.ranker.positive_rate * 100).toFixed(2)}%`}
                  />
                  <DataRow
                    label="Ranker train time"
                    value={`${ranking.data.ranker.train_seconds.toFixed(1)} s`}
                  />
                </div>
              ) : (
                <PanelSkeleton rows={4} />
              )}

              {status.data ? (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {status.data.retrieval.map((model) => (
                    <Badge key={model.name} tone="positive">
                      {MODEL_LABELS[model.name] ?? model.name}
                    </Badge>
                  ))}
                  {status.data.ranking ? <Badge tone="mercury">LambdaRank</Badge> : null}
                </div>
              ) : null}
              <p className="mt-3 text-xs text-[var(--color-ink-faint)]">
                Badges show what this API process currently has loaded.
              </p>
            </GlassPanel>
          </div>

          {evaluation.data ? (
            <GlassPanel>
              <SectionHeading title="Protocol" />
              <div className="grid gap-x-8 sm:grid-cols-2">
                {Object.entries(evaluation.data.protocol).map(([key, value]) => (
                  <DataRow
                    key={key}
                    label={key.replace(/_/g, " ")}
                    value={String(value)}
                    mono={false}
                  />
                ))}
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-2 text-xs text-[var(--color-ink-faint)]">
                <Timer className="size-3.5" aria-hidden />
                Generated {new Date(evaluation.data.generated_at).toLocaleString()}
                {Object.entries(evaluation.data.environment).map(([key, value]) => (
                  <span key={key} className="tabular">
                    · {key} {value}
                  </span>
                ))}
              </div>
            </GlassPanel>
          ) : null}
        </>
      )}
    </div>
  );
}
