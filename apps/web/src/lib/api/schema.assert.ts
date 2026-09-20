/**
 * Compile-time check that the hand-written response types still match the
 * backend's OpenAPI schema.
 *
 * `types.ts` is hand-written on purpose: the generated types are accurate but
 * unpleasant to read and to import, and the comments explaining what a field
 * means belong next to the field. The cost of that choice is drift — the
 * backend renames something, the frontend keeps compiling, and the mismatch
 * surfaces as `undefined` at runtime in front of whoever opened the page.
 *
 * This file pays that cost down. Each assertion below fails to compile if the
 * generated schema and the hand-written type stop agreeing, so `tsc --noEmit`
 * catches the drift. Regenerate with:
 *
 *     uv run python scripts/export_openapi.py && npm run gen:api
 *
 * Nothing imports this module; it exists for the type checker. That is why
 * every binding is `void`-consumed at the bottom rather than exported.
 */

import type { components } from "./schema";
import type {
  HealthResponse,
  ModelsStatusResponse,
  PipelineTrace,
  ReadinessResponse,
  Recommendation,
  RecommendationResponse,
  StageLatency,
} from "./types";

type Schemas = components["schemas"];

/**
 * Assignable in both directions.
 *
 * One direction alone is not enough. If the hand-written type were only
 * assignable *to* the generated one, it could be missing fields the API
 * sends; if only *from*, it could invent fields the API never returns. Both
 * are bugs, and both are silent.
 *
 * Expressed as a conditional rather than as `A extends B, B extends A`
 * constraints, which TypeScript rejects as circular.
 */
type Extends<A, B> = [A] extends [B] ? true : false;

type MutuallyAssignable<A, B> = Extends<A, B> extends true
  ? Extends<B, A> extends true
    ? true
    : false
  : false;

/** Fails to compile when handed `false`, which is the whole mechanism. */
type Assert<T extends true> = T;

type AssertStageLatency = Assert<MutuallyAssignable<StageLatency, Schemas["StageLatency"]>>;
type AssertRecommendation = Assert<MutuallyAssignable<Recommendation, Schemas["Recommendation"]>>;
type AssertRecommendationResponse = Assert<
  MutuallyAssignable<RecommendationResponse, Schemas["RecommendationResponse"]>
>;
type AssertPipelineTrace = Assert<
  MutuallyAssignable<PipelineTrace, Schemas["PipelineTraceResponse"]>
>;
type AssertHealth = Assert<MutuallyAssignable<HealthResponse, Schemas["HealthResponse"]>>;
type AssertReadiness = Assert<MutuallyAssignable<ReadinessResponse, Schemas["ReadinessResponse"]>>;
type AssertModelsStatus = Assert<
  MutuallyAssignable<ModelsStatusResponse, Schemas["ModelsStatusResponse"]>
>;

/* The insights endpoints return untyped dictionaries on the backend, so they
   have no schema to check against. That is a deliberate asymmetry: those
   responses assemble committed artifacts whose shape is owned by the pipeline
   that writes them, not by a Pydantic model. Their frontend types are checked
   by the pages that consume them and by nothing else, which is worth stating
   rather than leaving someone to discover. */

export type {
  AssertHealth,
  AssertModelsStatus,
  AssertPipelineTrace,
  AssertReadiness,
  AssertRecommendation,
  AssertRecommendationResponse,
  AssertStageLatency,
};
