"use client";

/**
 * TanStack Query hooks. Every hook in the application lives here.
 *
 * Stale times are chosen per resource rather than left at a global default,
 * because these resources change at wildly different rates: a committed
 * evaluation artifact changes when someone retrains a model, while live
 * latency percentiles change with every request.
 *
 * Retries are disabled for "the API is not running", which is the common local
 * case. Retrying three times with backoff just delays the honest message by
 * several seconds while the user watches a spinner.
 */

import { useQueries, useQuery, type UseQueryResult } from "@tanstack/react-query";

import { ApiUnreachableError, apiGet, apiPost } from "./client";
import type {
  DatasetInsights,
  DriftReport,
  EvaluationResults,
  ExperimentResults,
  HealthResponse,
  MetricsSummary,
  ModelsStatusResponse,
  PipelineTrace,
  ProjectionResponse,
  RankingResults,
  ReadinessResponse,
  RecommendationResponse,
  SampleUsersResponse,
  ServingConfig,
  UserPositionResponse,
} from "./types";

/** Never retry an unreachable API; the message is more useful than the wait. */
const retryUnlessUnreachable = (failureCount: number, error: Error) =>
  !(error instanceof ApiUnreachableError) && failureCount < 2;

const MINUTE = 60_000;

export function useHealth(): UseQueryResult<HealthResponse> {
  return useQuery({
    queryKey: ["health"],
    queryFn: () => apiGet<HealthResponse>("/health"),
    // Polled, because this drives the status pill and a stale "connected"
    // indicator is worse than none.
    refetchInterval: 15_000,
    retry: false,
  });
}

export function useReadiness(): UseQueryResult<ReadinessResponse> {
  return useQuery({
    queryKey: ["ready"],
    queryFn: () => apiGet<ReadinessResponse>("/ready"),
    refetchInterval: 30_000,
    retry: false,
  });
}

export function useDataset(): UseQueryResult<DatasetInsights> {
  return useQuery({
    queryKey: ["insights", "dataset"],
    queryFn: () => apiGet<DatasetInsights>("/api/v1/insights/dataset"),
    // The dataset only changes when someone rebuilds it.
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useEvaluation(): UseQueryResult<EvaluationResults> {
  return useQuery({
    queryKey: ["insights", "evaluation"],
    queryFn: () => apiGet<EvaluationResults>("/api/v1/insights/evaluation"),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useRanking(): UseQueryResult<RankingResults> {
  return useQuery({
    queryKey: ["insights", "ranking"],
    queryFn: () => apiGet<RankingResults>("/api/v1/insights/ranking"),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useExperiment(): UseQueryResult<ExperimentResults> {
  return useQuery({
    queryKey: ["insights", "experiment"],
    queryFn: () => apiGet<ExperimentResults>("/api/v1/insights/experiment"),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useMetricsSummary(enabled = true): UseQueryResult<MetricsSummary> {
  return useQuery({
    queryKey: ["insights", "metrics"],
    queryFn: () => apiGet<MetricsSummary>("/api/v1/insights/metrics/summary"),
    // Live telemetry: refresh often, never serve from a long cache.
    refetchInterval: 5_000,
    staleTime: 0,
    enabled,
    retry: false,
  });
}

export function useServingConfig(): UseQueryResult<ServingConfig> {
  return useQuery({
    queryKey: ["insights", "config"],
    queryFn: () => apiGet<ServingConfig>("/api/v1/insights/config"),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useModelStatus(): UseQueryResult<ModelsStatusResponse> {
  return useQuery({
    queryKey: ["models", "status"],
    queryFn: () => apiGet<ModelsStatusResponse>("/api/v1/models/status"),
    staleTime: 5 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export interface RecommendationParams {
  userId: string;
  k?: number;
  hour?: number | null;
  weekday?: number | null;
  regionId?: number | null;
  vertical?: number | null;
}

export function useRecommendations(
  params: RecommendationParams | null,
): UseQueryResult<RecommendationResponse> {
  return useQuery({
    queryKey: ["recommendations", params],
    queryFn: () =>
      apiGet<RecommendationResponse>(
        `/api/v1/recommendations/${encodeURIComponent(params!.userId)}`,
        {
          k: params!.k ?? 10,
          hour: params!.hour ?? undefined,
          weekday: params!.weekday ?? undefined,
          region_id: params!.regionId ?? undefined,
          vertical: params!.vertical ?? undefined,
        },
      ),
    enabled: params !== null && params.userId.length > 0,
    // Keeps the previous result on screen while a new context loads, so
    // adjusting a slider does not flash the whole panel to a skeleton.
    placeholderData: (previous) => previous,
    staleTime: 0,
    retry: retryUnlessUnreachable,
  });
}

export function useProjection(
  method: "umap" | "pca",
  maxItems: number,
): UseQueryResult<ProjectionResponse> {
  return useQuery({
    queryKey: ["projection", method, maxItems],
    queryFn: () =>
      apiGet<ProjectionResponse>("/api/v1/embeddings/projection", {
        method,
        max_items: maxItems,
      }),
    // A static artifact that only changes on retrain, and the payload is large
    // enough that refetching it would be visibly wasteful.
    staleTime: Infinity,
    gcTime: 60 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

export function useSampleUsers(n = 8): UseQueryResult<SampleUsersResponse> {
  return useQuery({
    queryKey: ["insights", "sample-users", n],
    queryFn: () => apiGet<SampleUsersResponse>("/api/v1/insights/sample-users", { n }),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}

/**
 * Where a user sits among the projected items.
 *
 * Separate from the projection query because it changes per user while the
 * 75KB item projection does not, and refetching the galaxy every time someone
 * picks a different user would be absurd.
 */
export function useUserPosition(
  userId: string | null,
  method: "umap" | "pca",
): UseQueryResult<UserPositionResponse> {
  return useQuery({
    queryKey: ["projection", "user", userId, method],
    queryFn: () =>
      apiGet<UserPositionResponse>(
        `/api/v1/embeddings/user/${encodeURIComponent(userId!)}`,
        { method },
      ),
    enabled: Boolean(userId),
    staleTime: 5 * MINUTE,
    // A 404 here means "this user has no embedding", which is an answer, not
    // a transient failure worth retrying.
    retry: false,
  });
}

/**
 * Stage-by-stage membership for one request.
 *
 * Deliberately a separate request from {@link useRecommendations}: a trace
 * bypasses the cache and is two orders of magnitude larger than the result it
 * explains, so pages that only need the recommendations must not pay for it.
 */
export function usePipelineTrace(
  params: RecommendationParams | null,
): UseQueryResult<PipelineTrace> {
  return useQuery({
    queryKey: ["trace", params],
    queryFn: () =>
      apiGet<PipelineTrace>(
        `/api/v1/recommendations/${encodeURIComponent(params!.userId)}/trace`,
        {
          k: params!.k ?? 10,
          hour: params!.hour ?? undefined,
          weekday: params!.weekday ?? undefined,
          region_id: params!.regionId ?? undefined,
          vertical: params!.vertical ?? undefined,
        },
      ),
    enabled: params !== null && params.userId.length > 0,
    placeholderData: (previous) => previous,
    staleTime: 0,
    retry: retryUnlessUnreachable,
  });
}

export interface SessionParams {
  userId: string | null;
  sessionItems: number[];
  k?: number;
  hour?: number | null;
  vertical?: number | null;
}

/**
 * Recommendations from in-session behaviour.
 *
 * A POST, because the session item list is unbounded and belongs in a body
 * rather than a query string. Never cached server-side either: a session
 * context is effectively unique per request, so caching it would fill Redis
 * with entries nothing ever reads again.
 */
export function useSessionRecommendations(
  params: SessionParams | null,
): UseQueryResult<RecommendationResponse> {
  return useQuery({
    queryKey: ["session", params],
    queryFn: () =>
      apiPost<RecommendationResponse>("/api/v1/session/recommend", {
        user_id: params!.userId,
        session_items: params!.sessionItems,
        k: params!.k ?? 8,
        context: {
          hour: params!.hour ?? null,
          vertical: params!.vertical ?? null,
        },
      }),
    enabled: params !== null,
    placeholderData: (previous) => previous,
    staleTime: 0,
    retry: retryUnlessUnreachable,
  });
}

/**
 * The same user, the same session, at several times of day.
 *
 * Issued as parallel queries rather than one request because the API models a
 * request as having exactly one context, and faking a batch endpoint purely
 * for a visualisation would put a shape in the API that serving never uses.
 */
export function useHourSweep(
  userId: string | null,
  hours: readonly number[],
  sessionItems: number[],
  enabled = true,
) {
  return useQueries({
    queries: hours.map((hour) => ({
      queryKey: ["session", "hour-sweep", userId, hour, sessionItems],
      queryFn: () =>
        apiPost<RecommendationResponse>("/api/v1/session/recommend", {
          user_id: userId,
          session_items: sessionItems,
          k: 5,
          context: { hour, vertical: null },
        }),
      // Explicit rather than derived from `userId`, because a null id is a
      // legitimate anonymous request rather than a missing parameter.
      enabled,
      staleTime: 0,
      retry: false,
    })),
  });
}

/**
 * The committed drift report.
 *
 * A static artifact regenerated by `mercury monitor drift`, so it is cached
 * hard. Recomputing it is seconds of work over two quarter-million-row frames,
 * which is why it is an artifact and not a live endpoint.
 */
export function useDrift(): UseQueryResult<DriftReport> {
  return useQuery({
    queryKey: ["insights", "drift"],
    queryFn: () => apiGet<DriftReport>("/api/v1/insights/drift"),
    staleTime: 30 * MINUTE,
    retry: retryUnlessUnreachable,
  });
}
