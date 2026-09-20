/**
 * Shared vocabulary: the enum values the API returns, and how they are named
 * and coloured in the UI.
 *
 * Defined once so the galaxy, the charts and the badges agree. A vertical that
 * is orange in the 3D view and blue in a table forces the reader to hold a
 * second mapping in their head.
 */

export interface VerticalMeta {
  id: number;
  label: string;
  /** CSS custom property, resolved at render so the theme can swap it. */
  color: string;
}

export const VERTICALS: readonly VerticalMeta[] = [
  { id: 0, label: "Food delivery", color: "var(--color-v-food)" },
  { id: 1, label: "Restaurant", color: "var(--color-v-restaurant)" },
  { id: 2, label: "Grocery", color: "var(--color-v-grocery)" },
  { id: 3, label: "Pharmacy", color: "var(--color-v-pharmacy)" },
  { id: 4, label: "Commerce", color: "var(--color-v-commerce)" },
  { id: 5, label: "Lifestyle", color: "var(--color-v-lifestyle)" },
] as const;

export function verticalMeta(id: number | null | undefined): VerticalMeta | null {
  if (id === null || id === undefined) return null;
  return VERTICALS.find((v) => v.id === id) ?? null;
}

/**
 * Time-of-day buckets, matching `HOUR_BUCKETS` in the popularity model.
 *
 * Six bands rather than 24 hours: hourly slices fragment a sparse dataset into
 * cells too thin to estimate, and consumption genuinely clusters into
 * meal-shaped bands.
 */
export const HOUR_BUCKETS = [
  { label: "Night", start: 0, end: 6, representative: 3 },
  { label: "Breakfast", start: 6, end: 10, representative: 8 },
  { label: "Midday", start: 10, end: 14, representative: 12 },
  { label: "Afternoon", start: 14, end: 17, representative: 15 },
  { label: "Dinner", start: 17, end: 21, representative: 19 },
  { label: "Late", start: 21, end: 24, representative: 22 },
] as const;

export const WEEKDAYS = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
] as const;

/** Retrieval sources, with the one-line description the UI shows on hover. */
export const RETRIEVAL_SOURCES: Record<string, { label: string; description: string }> = {
  popularity: {
    label: "Popularity",
    description:
      "Intent-weighted, recency-decayed counts, optionally conditioned on region, vertical and time of day. The floor every other model must clear.",
  },
  item_cf: {
    label: "Item-CF",
    description:
      "Cosine similarity between item columns of the weighted interaction matrix, damped for popularity. Strongest where a user has rich history.",
  },
  matrix_factorization: {
    label: "Matrix factorisation",
    description:
      "BPR-trained latent factors. Optimises a pairwise ranking objective rather than reconstructing unobserved entries as dislikes.",
  },
  two_tower: {
    label: "Two-tower",
    description:
      "Neural retrieval with separate user and item towers. Serving cost is independent of catalogue size, which is the architectural point.",
  },
  cold_start: {
    label: "Cold start",
    description:
      "Contextual popularity, used when a user has no usable history. A real path, not an error case.",
  },
};

/**
 * The stages of the request path, in order, with what each one does.
 *
 * Drives the Pipeline Inspector. The keys match the `StageLatency` fields the
 * API returns, so the diagram's numbers come from the response rather than
 * from a hard-coded illustration.
 */
export const PIPELINE_STAGES = [
  {
    key: "cache_lookup_ms",
    label: "Cache",
    description:
      "Redis lookup keyed by model version, user and a bucketed context fingerprint. Because the model version is in the key, a promotion invalidates atomically with no scan.",
  },
  {
    key: "feature_lookup_ms",
    label: "Features",
    description:
      "As-of feature vector for the user and each candidate, emitted by the same function the training pipeline used. One definition, two call sites.",
  },
  {
    key: "candidate_generation_ms",
    label: "Retrieval",
    description:
      "Four sources queried in parallel and fused by reciprocal rank. Retrieval recall is the ceiling on everything downstream: an item never proposed cannot be recommended.",
  },
  {
    key: "ranking_ms",
    label: "Ranking",
    description:
      "LightGBM LambdaRank over the fused candidates, optimising NDCG directly rather than per-item probability.",
  },
  {
    key: "reranking_ms",
    label: "Business rerank",
    description:
      "Availability filters, then merchant and category diversity caps. Relevance and policy stay in separate fields so any placement is attributable.",
  },
] as const;

export type PipelineStageKey = (typeof PIPELINE_STAGES)[number]["key"];

/** Human-readable names for the as-of feature columns, used in SHAP panels. */
export const FEATURE_LABELS: Record<string, string> = {
  user_event_count: "User activity",
  user_purchase_count: "User purchases",
  user_conversion_rate: "User conversion rate",
  user_distinct_items: "User breadth",
  user_days_since_last: "Days since user's last visit",
  user_tenure_days: "User tenure",
  user_avg_interaction_price: "User price level",
  user_avg_order_value: "User order value",
  item_event_count: "Item popularity",
  item_purchase_count: "Item purchases",
  item_conversion_rate: "Item conversion rate",
  item_distinct_users: "Item reach",
  item_days_since_last: "Item recency",
  item_age_days: "Item age",
  ui_prior_events: "Prior interactions with this item",
  ui_prior_purchases: "Prior purchases of this item",
  ui_days_since_last: "Days since this pair last interacted",
  user_category_affinity: "Category affinity",
  user_merchant_affinity: "Merchant affinity",
  price_ratio_to_user_avg: "Price vs user's usual",
  retrieval_score_two_tower: "Two-tower score",
  retrieval_score_item_cf: "Item-CF score",
  retrieval_score_popularity: "Popularity score",
  retrieval_score_matrix_factorization: "Matrix-factorisation score",
  retrieval_rank_fused: "Fused retrieval rank",
  retrieval_n_sources: "Sources proposing this item",
};

export function featureLabel(name: string): string {
  return FEATURE_LABELS[name] ?? name;
}
