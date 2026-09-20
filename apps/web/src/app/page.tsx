"use client";

/**
 * Landing page.
 *
 * The funnel is the one piece of decoration in the product that earns its
 * place: it shows the actual shape of the system — a catalogue narrowing
 * through retrieval to a ranked handful — and its stage widths are driven by
 * the real candidate counts the API reports, not by chosen proportions.
 */

import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight, Boxes, Orbit, Route } from "lucide-react";
import Link from "next/link";

import { Badge, GlassPanel, ProvenanceBadge } from "@/components/primitives";
import { useDataset, useRanking } from "@/lib/api/hooks";
import { formatNumber } from "@/lib/utils";

/** Stage widths come from real counts when the API is up. */
function Funnel() {
  const reduceMotion = useReducedMotion();
  const { data: dataset } = useDataset();
  const { data: ranking } = useRanking();

  const catalogue = dataset?.n_items ?? null;
  const candidates = ranking?.retrieval.max_candidates ?? null;
  const shown = 10;

  const stages = [
    {
      label: "Catalogue",
      value: catalogue,
      width: "100%",
      colour: "var(--color-surface-3)",
      note: "every item that could be shown",
    },
    {
      label: "Retrieval",
      value: candidates,
      width: "46%",
      colour: "var(--color-mercury-dim)",
      note: "four sources, fused by reciprocal rank",
    },
    {
      label: "Ranked",
      value: shown,
      width: "16%",
      colour: "var(--color-mercury)",
      note: "LambdaRank, then business policy",
    },
  ];

  return (
    <div className="space-y-2.5" aria-label="Recommendation funnel">
      {stages.map((stage, index) => (
        <div key={stage.label} className="flex items-center gap-4">
          <span className="w-20 shrink-0 text-right text-xs text-[var(--color-ink-muted)]">
            {stage.label}
          </span>
          <motion.div
            initial={reduceMotion ? false : { width: 0, opacity: 0 }}
            animate={{ width: stage.width, opacity: 1 }}
            transition={{
              duration: reduceMotion ? 0 : 0.7,
              delay: reduceMotion ? 0 : index * 0.14,
              ease: [0.22, 1, 0.36, 1],
            }}
            className="flex h-9 items-center rounded-[var(--radius-control)] px-3"
            style={{
              background: `linear-gradient(90deg, ${stage.colour}, color-mix(in oklch, ${stage.colour} 55%, transparent))`,
            }}
          >
            <span className="tabular text-xs font-semibold text-[var(--color-void)] mix-blend-normal">
              {formatNumber(stage.value, { compact: true })}
            </span>
          </motion.div>
          <span className="hidden text-xs text-[var(--color-ink-faint)] md:inline">
            {stage.note}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function HomePage() {
  const { data: dataset } = useDataset();

  return (
    <div className="space-y-8">
      <section className="pt-6">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="mercury">Multi-stage retrieval → ranking → policy</Badge>
          <ProvenanceBadge provenance={dataset?.source?.provenance} />
        </div>

        <h1 className="mt-5 max-w-3xl text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
          Real-time recommendation intelligence
        </h1>
        <p className="mt-4 max-w-2xl text-pretty text-base leading-relaxed text-[var(--color-ink-muted)]">
          A production-shaped recommendation platform: candidate retrieval from four
          sources, neural two-tower embeddings, learning-to-rank, business policy,
          caching and observability — with every number on this site traced back to an
          experiment committed in the repository.
        </p>

        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/explorer"
            className="group inline-flex items-center gap-2 rounded-[var(--radius-control)] bg-[var(--color-mercury)] px-4 py-2.5 text-sm font-medium text-[var(--color-void)] transition-transform hover:scale-[1.02]"
          >
            <ScatterIcon />
            Explore recommendations
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
          </Link>
          <Link
            href="/pipeline"
            className="inline-flex items-center gap-2 rounded-[var(--radius-control)] border border-[var(--color-hairline-strong)] px-4 py-2.5 text-sm font-medium text-[var(--color-ink)] transition-colors hover:bg-[var(--color-surface-2)]"
          >
            <Route className="size-4" />
            Open the pipeline
          </Link>
          <Link
            href="/galaxy"
            className="inline-flex items-center gap-2 rounded-[var(--radius-control)] border border-[var(--color-hairline-strong)] px-4 py-2.5 text-sm font-medium text-[var(--color-ink)] transition-colors hover:bg-[var(--color-surface-2)]"
          >
            <Orbit className="size-4" />
            View the galaxy
          </Link>
          <Link
            href="/architecture"
            className="inline-flex items-center gap-2 rounded-[var(--radius-control)] border border-[var(--color-hairline-strong)] px-4 py-2.5 text-sm font-medium text-[var(--color-ink)] transition-colors hover:bg-[var(--color-surface-2)]"
          >
            <Boxes className="size-4" />
            Architecture
          </Link>
        </div>
      </section>

      <GlassPanel className="glow">
        <h2 className="mb-1 text-sm font-semibold">How a request narrows</h2>
        <p className="mb-5 text-xs text-[var(--color-ink-muted)]">
          Widths are illustrative; the counts are read from the running system.
        </p>
        <Funnel />
      </GlassPanel>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {[
          {
            title: "Retrieval is the ceiling",
            body: "An item no source proposes can never be recommended, so retrieval recall is measured and reported separately from end-to-end quality. It is the number that says whether to invest in retrieval or in ranking.",
          },
          {
            title: "One feature definition",
            body: "Training and serving call the same emit function against the same state object. There are not two implementations meant to agree — there is one, called from two places, with a parity test asserting it.",
          },
          {
            title: "Policy stays separate",
            body: "Every recommendation carries its model score and its business adjustment as distinct fields, with each adjustment attributed to the rule that produced it. Merge them and 'why did this appear here?' stops being answerable.",
          },
        ].map((card) => (
          <GlassPanel key={card.title} className="space-y-2">
            <h3 className="text-sm font-semibold">{card.title}</h3>
            <p className="text-sm leading-relaxed text-[var(--color-ink-muted)]">
              {card.body}
            </p>
          </GlassPanel>
        ))}
      </section>
    </div>
  );
}

function ScatterIcon() {
  return (
    <svg viewBox="0 0 16 16" className="size-4" fill="currentColor" aria-hidden>
      <circle cx="3.5" cy="12" r="1.6" />
      <circle cx="7.5" cy="7" r="1.6" />
      <circle cx="12" cy="4" r="1.6" />
      <circle cx="12" cy="11" r="1.2" opacity="0.6" />
    </svg>
  );
}
