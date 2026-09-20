/**
 * Response shapes returned by the MercuryRec API.
 *
 * These mirror the Pydantic models in `src/mercury_rec/api/schemas/`. They can
 * be regenerated from the live OpenAPI schema with `npm run gen:api`, which CI
 * runs followed by `git diff --exit-code` so backend/frontend drift becomes a
 * failing build rather than a runtime surprise.
 */

export interface StageLatency {
  cache_lookup_ms: number;
  feature_lookup_ms: number;
  candidate_generation_ms: number;
  ranking_ms: number;
  reranking_ms: number;
  total_ms: number;
}

export interface Recommendation {
  item_id: number;
  rank: number;
  /** The ranking model's output. An ordinal utility, NOT a probability. */
  ml_relevance_score: number;
  /** Deterministic policy adjustment, kept separate so it stays attributable. */
  business_adjustment: number;
  final_score: number;
  adjustments: Record<string, number>;
  sources: string[];
  category_id: number | null;
  merchant_id: number | null;
  vertical: number | null;
  price: number | null;
}

export interface RecommendationExplanation {
  item_id: number;
  /** Exact TreeSHAP contributions. Signed: negative pushed the item down. */
  contributions: Record<string, number>;
}

export interface RecommendationResponse {
  user_id: string;
  request_id: string;
  model_version: string;
  generated_at: string;
  recommendations: Recommendation[];
  explanations: RecommendationExplanation[];
  latency: StageLatency;
  cache_hit: boolean;
  n_candidates: number;
  candidate_sources: Record<string, number>;
  filtered: Record<string, number>;
  is_cold_start: boolean;
  data_provenance: "retailrocket" | "augmented" | "synthetic";
}

export interface ReadinessCheck {
  name: string;
  ready: boolean;
  detail: string | null;
}

export interface ReadinessResponse {
  ready: boolean;
  checks: ReadinessCheck[];
}

export interface HealthResponse {
  status: string;
  version: string;
  uptime_seconds: number;
}

export interface ModelStatus {
  name: string;
  version: string;
  trained_at: string | null;
  dataset_hash: string | null;
  is_loaded: boolean;
  params: Record<string, string | number | boolean | null>;
}

export interface ModelsStatusResponse {
  retrieval: ModelStatus[];
  ranking: ModelStatus | null;
  feature_schema_version: number;
  dataset_hash: string | null;
}

export interface DatasetInsights {
  n_users: number;
  n_items: number;
  model_version: string;
  dataset_hash: string | null;
  ingest: {
    raw_events?: number;
    raw_users?: number;
    raw_items?: number;
    final_events?: number;
    final_users?: number;
    final_items?: number;
    retained_event_fraction?: number;
    kcore_iterations?: number;
    first_event?: string;
    last_event?: string;
    events_by_type?: Record<string, number>;
  };
  split: Record<string, number | string>;
  source: {
    dataset?: string;
    license?: string;
    url?: string;
    provenance?: string;
    note?: string;
  };
  columns: { real?: string[]; synthesised?: string[] };
}

export interface ModelResultRow {
  model: string;
  n_users_evaluated: number;
  n_users_skipped: number;
  catalog_coverage: number;
  gini: number;
  intra_list_diversity: number;
  mean_popularity_rank: number;
  n_distinct_items: number;
  train_seconds: number;
  scoring_ms_per_user: number;
  params: Record<string, unknown>;
  fit_extra: Record<string, unknown>;
  [metric: string]: unknown;
}

export interface EvaluationResults {
  generated_at: string;
  preset: string;
  split: string;
  dataset_hash: string | null;
  n_users: number;
  n_items: number;
  users_scored: number;
  k_values: number[];
  protocol: Record<string, unknown>;
  environment: Record<string, string>;
  results: ModelResultRow[];
  index_benchmarks?: IndexBenchmark[];
}

export interface IndexBenchmark {
  index: string;
  n_items: number;
  dim: number;
  build_seconds: number;
  mean_query_ms: number;
  p95_query_ms: number;
  recall_at_k: number;
}

export interface StageMetrics {
  [stage: string]: Record<string, number>;
}

export interface RankingResults {
  generated_at: string;
  dataset_hash: string | null;
  retrieval: {
    per_source_k: number;
    max_candidates: number;
    mean_recall_at_candidates_test: number;
    mean_recall_at_candidates_train: number;
    note: string;
  };
  ranker: {
    best_iteration: number;
    train_seconds: number;
    train_groups: number;
    train_rows: number;
    positive_rate: number;
    top_features: Record<string, number>;
    best_score: Record<string, number>;
  };
  stages: Record<string, Record<string, number | string>>;
}

export interface MetricComparison {
  metric: string;
  control: number;
  treatment: number;
  absolute_lift: number;
  relative_lift_pct: number;
  ci_95: [number, number];
  p_value: number;
  is_significant: boolean;
  n_users: number;
}

export interface ExperimentResults {
  generated_at: string;
  dataset_hash: string | null;
  k: number;
  simulation: {
    control: string;
    treatment: string;
    n_users: number;
    bootstrap_samples: number;
    /** Always true. This project has no live traffic. */
    is_simulated: boolean;
    disclaimer: string;
    metrics: MetricComparison[];
  };
  gate: {
    promoted: boolean;
    candidate: string;
    incumbent: string | null;
    summary?: string;
    checks: {
      name: string;
      verdict: "pass" | "fail" | "inconclusive";
      detail: string;
      candidate: number | null;
      incumbent: number | null;
      threshold: number | null;
    }[];
  };
}

export interface MetricsSummary {
  has_data: boolean;
  detail?: string;
  note?: string;
  stages?: Record<
    string,
    { count: number; mean_ms: number | null; p50_ms: number | null; p95_ms: number | null; p99_ms: number | null }
  >;
  cache: {
    hits: number;
    misses: number;
    stale_hits: number;
    errors: number;
    writes: number;
    invalidations: number;
    hit_rate: number;
  };
}

export interface ProjectionResponse {
  projection: string;
  count: number;
  dataset_hash: string | null;
  encoding: string;
  note: string;
  /** base64 Float32Array */
  x: string;
  y: string;
  z: string;
  /** base64 Int32Array */
  item_id: string;
  category_id: string;
  vertical: string;
  price: string;
}

export interface PipelineTrace {
  user_id: string;
  request_id: string;
  model_version: string;
  generated_at: string;
  candidate_ids: number[];
  /** Item ids per source; an item appears under every source that proposed it. */
  candidate_sources: Record<string, number[]>;
  ranked_ids: number[];
  ranked_scores: number[];
  final_ids: number[];
  latency: StageLatency;
  n_candidates: number;
  filtered: Record<string, number>;
  is_cold_start: boolean;
  note: string;
}

export interface UserPositionResponse {
  user_id: string;
  /** null when the user has no two-tower embedding at all. */
  position: [number, number, number] | null;
  method: "umap" | "pca";
  /** True only for PCA, which is a linear map and projects a user exactly. */
  is_exact: boolean;
  dataset_hash?: string | null;
  note: string;
}

export interface SampleUser {
  user_id: string;
  history_items: number;
  /** Position in the history distribution; 100 is the most active user. */
  percentile: number;
  has_two_tower_embedding: boolean;
}

export interface SampleUsersResponse {
  users: SampleUser[];
  n_users_with_history?: number;
  detail?: string;
}

export interface FeatureDrift {
  feature: string;
  /** Population Stability Index, binned on the reference window. */
  psi: number;
  ks_statistic: number;
  js_divergence: number;
  reference_mean: number;
  current_mean: number;
  reference_n: number;
  current_n: number;
  severity: "stable" | "minor" | "major";
}

export interface DriftReport {
  generated_at: string;
  preset: string;
  reference_split: string;
  current_split: string;
  reference_rows: number;
  current_rows: number;
  bins: number;
  thresholds: { minor: number; major: number };
  note: string;
  interpretation: string;
  expected_drift?: string;
  features: FeatureDrift[];
}

export interface ServingConfig {
  per_source_k: number;
  max_candidates: number;
  rerank: {
    max_per_merchant: number;
    max_per_category: number;
    enforce_availability: boolean;
    sponsored_boost: number;
    popularity_penalty: number;
  };
  cache: { enabled: boolean; ttl_seconds: number; soft_ttl_seconds: number };
}
