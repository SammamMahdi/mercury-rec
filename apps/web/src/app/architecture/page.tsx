"use client";

/**
 * Architecture.
 *
 * The parts of the system and why they are arranged this way. Where a claim on
 * this page can be checked against the running process — which models are
 * loaded, what the serving configuration actually is — it is read from the API
 * rather than written down, because a diagram that drifts from the code is
 * worse than no diagram.
 */

import {
  Boxes,
  Database,
  GitBranch,
  Layers,
  Lock,
  Server,
  Share2,
  Workflow,
} from "lucide-react";

import {
  Badge,
  DataRow,
  GlassPanel,
  PanelSkeleton,
  SectionHeading,
} from "@/components/primitives";
import { useDataset, useModelStatus, useServingConfig } from "@/lib/api/hooks";
import { PIPELINE_STAGES } from "@/lib/constants";
import { formatNumber, shortHash } from "@/lib/utils";

/**
 * The package layering contract.
 *
 * Not decoration: `tests/unit/test_layering.py` parses every module's imports
 * and fails the build on any upward edge, naming both ends. A documented
 * architecture that nothing checks decays within weeks — someone imports the
 * API's settings object into a model for convenience and the model layer
 * silently acquires a dependency on the web framework.
 */
const LAYERS = [
  { rank: 9, names: ["cli"], note: "The canonical entry point. Typer, not Make." },
  { rank: 8, names: ["pipelines"], note: "Batch jobs: build, train, evaluate, project, monitor." },
  { rank: 7, names: ["api"], note: "FastAPI. The only layer that knows HTTP exists." },
  { rank: 6, names: ["services"], note: "Artifact loading and composition." },
  { rank: 5, names: ["recommender"], note: "The staged engine. Holds no framework types." },
  { rank: 4, names: ["experimentation"], note: "Tracking, gating, offline A/B." },
  {
    rank: 3,
    names: ["models", "retrieval", "reranking", "evaluation", "monitoring"],
    note: "Everything that scores, ranks or measures.",
  },
  { rank: 2, names: ["features"], note: "The as-of engine. One definition, two call sites." },
  { rank: 1, names: ["data", "db", "cache"], note: "Ingest, persistence, Redis." },
  { rank: 0, names: ["core", "config"], note: "Logging, enums, settings. Depends on nothing." },
];

const DECISIONS = [
  {
    icon: Share2,
    title: "One feature definition, two call sites",
    body: "Training and serving both call the same emit function against the same state object. There are not two implementations meant to agree — there is one, invoked from two places, with a parity test asserting it. Training/serving skew is the most expensive bug in applied ML precisely because nothing surfaces it.",
  },
  {
    icon: GitBranch,
    title: "Features computed as-of, by forward scan",
    body: "Aggregates are emitted before the event that would update them is applied, in one chronological pass. A groupby over the whole frame would let an event contribute to its own features, which inflates offline metrics and cannot be reproduced online.",
  },
  {
    icon: Layers,
    title: "Retrieval before ranking",
    body: "Four sources propose a few hundred candidates, fused by reciprocal rank; the ranker scores only those. Ranking the whole catalogue per request is impossible at any real size, and retrieval recall becomes the ceiling on everything downstream.",
  },
  {
    icon: Lock,
    title: "Relevance and policy stay separate",
    body: "Every recommendation carries its model score and its business adjustment as distinct fields, with each adjustment attributed to the rule that produced it. Blend them into one number and 'why did this appear here?' stops being answerable.",
  },
  {
    icon: Boxes,
    title: "Artifacts swap as one bundle",
    body: "Models are loaded together and replaced atomically. Swapping them individually risks serving a ranker against embeddings from a different training run, which does not raise — it produces plausible scores that are quietly wrong.",
  },
  {
    icon: Database,
    title: "The cache key carries the model version",
    body: "A promotion invalidates every cached slate atomically, with no key scan and no stale-after-deploy window. Redis KEYS and SCAN are both absent from this codebase on purpose.",
  },
];

/* -------------------------------------------------------------------------- */

function LayerStack() {
  return (
    <div className="space-y-1">
      {LAYERS.map((layer) => (
        <div
          key={layer.rank}
          className="flex flex-col gap-1 rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-2.5 sm:flex-row sm:items-center sm:gap-3"
        >
          <span className="tabular flex size-6 shrink-0 items-center justify-center rounded-full border border-[var(--color-hairline)] text-[10px] text-[var(--color-ink-faint)]">
            {layer.rank}
          </span>
          <div className="flex w-full flex-wrap gap-1.5 sm:w-72 sm:shrink-0">
            {layer.names.map((name) => (
              <code
                key={name}
                className="tabular rounded-[var(--radius-control)] bg-[var(--color-surface-2)] px-1.5 py-0.5 text-xs text-[var(--color-mercury)]"
              >
                {name}
              </code>
            ))}
          </div>
          <span className="text-xs text-[var(--color-ink-faint)]">{layer.note}</span>
        </div>
      ))}
    </div>
  );
}

function RequestPath() {
  return (
    <ol className="space-y-1">
      {PIPELINE_STAGES.map((stage, index) => (
        <li
          key={stage.key}
          className="flex gap-3 rounded-[var(--radius-control)] border border-[var(--color-hairline)] p-2.5"
        >
          <span className="tabular flex size-6 shrink-0 items-center justify-center rounded-full border border-[var(--color-hairline)] text-[10px] text-[var(--color-ink-faint)]">
            {index + 1}
          </span>
          <div className="min-w-0">
            <div className="text-sm font-medium">{stage.label}</div>
            <p className="mt-0.5 text-xs leading-relaxed text-[var(--color-ink-faint)]">
              {stage.description}
            </p>
          </div>
        </li>
      ))}
    </ol>
  );
}

/* -------------------------------------------------------------------------- */

export default function ArchitecturePage() {
  const models = useModelStatus();
  const config = useServingConfig();
  const dataset = useDataset();

  return (
    <div className="space-y-6">
      <SectionHeading
        title="Architecture"
        description="How the system is put together, and which parts of that are verified rather than asserted."
        action={
          dataset.data ? (
            <Badge tone="neutral" title="Model version this process is serving">
              <span className="tabular">{shortHash(dataset.data.dataset_hash)}</span>
            </Badge>
          ) : null
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <GlassPanel>
          <SectionHeading
            title="The request path"
            description="Each stage is timed separately, into both the response body and a Prometheus histogram."
          />
          <RequestPath />
        </GlassPanel>

        <GlassPanel>
          <SectionHeading
            title="Package layering"
            description="Dependencies point strictly downward. A static import check enforces it in CI."
            action={
              <Badge tone="positive" title="tests/unit/test_layering.py">
                enforced by test
              </Badge>
            }
          />
          <LayerStack />
        </GlassPanel>
      </div>

      <div>
        <SectionHeading
          title="Decisions that shaped this"
          description="The choices that would be expensive to reverse, and what each one buys."
        />
        <div className="grid gap-4 sm:grid-cols-2">
          {DECISIONS.map((decision) => (
            <GlassPanel key={decision.title} className="space-y-2">
              <div className="flex items-center gap-2">
                <decision.icon className="size-4 text-[var(--color-mercury)]" aria-hidden />
                <h3 className="text-sm font-semibold">{decision.title}</h3>
              </div>
              <p className="text-sm leading-relaxed text-[var(--color-ink-muted)]">
                {decision.body}
              </p>
            </GlassPanel>
          ))}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <GlassPanel>
          <SectionHeading title="Runtime" />
          <div className="space-y-0">
            <DataRow label="API" value="FastAPI + uvicorn" mono={false} />
            <DataRow label="Relational" value="PostgreSQL 17 + SQLAlchemy 2" mono={false} />
            <DataRow label="Cache" value="Redis 8" mono={false} />
            <DataRow label="Tracking" value="MLflow" mono={false} />
            <DataRow label="Metrics" value="Prometheus + Grafana" mono={false} />
          </div>
          <p className="mt-3 flex items-center gap-1.5 text-xs text-[var(--color-ink-faint)]">
            <Server className="size-3.5" aria-hidden />
            Everything runs from one compose file.
          </p>
        </GlassPanel>

        <GlassPanel>
          <SectionHeading title="Loaded now" />
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
                    <Badge tone="warning">absent</Badge>
                  )
                }
                mono={false}
              />
              <DataRow
                label="Feature schema"
                value={`v${models.data.feature_schema_version}`}
                title="Checked on load. A mismatch is fatal, because it is silent otherwise: shapes align, scores are produced, and they are meaningless."
              />
            </div>
          ) : (
            <PanelSkeleton rows={4} />
          )}
        </GlassPanel>

        <GlassPanel>
          <SectionHeading title="Scale served" />
          {dataset.data ? (
            <div className="space-y-0">
              <DataRow label="Users" value={formatNumber(dataset.data.n_users)} />
              <DataRow label="Items" value={formatNumber(dataset.data.n_items)} />
              <DataRow
                label="Candidates"
                value={
                  config.data ? formatNumber(config.data.max_candidates) : "—"
                }
                title="Per request, after fusion."
              />
              <DataRow
                label="Cache TTL"
                value={
                  config.data?.cache.enabled
                    ? `${config.data.cache.ttl_seconds}s`
                    : "disabled"
                }
              />
            </div>
          ) : (
            <PanelSkeleton rows={4} />
          )}
          <p className="mt-3 flex items-start gap-1.5 text-xs leading-relaxed text-[var(--color-ink-faint)]">
            <Workflow className="mt-0.5 size-3.5 shrink-0" aria-hidden />
            These are the counts after k-core filtering, which is what the models are
            actually trained and served on.
          </p>
        </GlassPanel>
      </div>

      <GlassPanel>
        <SectionHeading title="What this system is not" />
        <ul className="space-y-2 text-sm leading-relaxed text-[var(--color-ink-muted)]">
          <li>
            <span className="text-[var(--color-ink)]">It has no production traffic.</span> Every
            latency figure is measured against local load, and the A/B comparison is an
            offline counterfactual, not an online test.
          </li>
          <li>
            <span className="text-[var(--color-ink)]">Its data is partly synthesised.</span>{" "}
            Behaviour, timestamps, item ids and categories are real Retailrocket. Merchants,
            regions, prices and verticals are generated, and the dataset card says exactly
            how.
          </li>
          <li>
            <span className="text-[var(--color-ink)]">
              It does not claim the neural model won.
            </span>{" "}
            Matrix factorisation outperforms the two-tower on this dataset, and the
            comparison page reports that rather than omitting the run.
          </li>
        </ul>
      </GlassPanel>
    </div>
  );
}
