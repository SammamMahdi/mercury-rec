/**
 * The single source of data for the entire application.
 *
 * Nothing else in the frontend fetches, and nothing else invents a number.
 * That is enforced rather than merely intended: `scripts/check-no-fake-data.mjs`
 * fails CI on numeric literals appearing where data should be.
 *
 * The client distinguishes three states, because they mean different things to
 * someone looking at the screen:
 *
 * - `live`       - the API answered.
 * - `snapshot`   - the API is down, and a recorded evaluation artifact is being
 *                  shown instead. The numbers are real; they are just not live.
 * - `unavailable`- neither. The UI says so and names the command that fixes it,
 *                  rather than rendering zeros that read as measurements.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

/** How long a request may take before we treat the API as unreachable. */
const DEFAULT_TIMEOUT_MS = 8000;

export type DataSourceState = "live" | "snapshot" | "unavailable";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
    readonly requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ApiUnreachableError extends Error {
  constructor(readonly cause: unknown) {
    super("The MercuryRec API is not reachable.");
    this.name = "ApiUnreachableError";
  }
}

interface RequestOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * Perform a typed GET against the API.
 *
 * Distinguishes "the server said no" (ApiError, with the server's own detail
 * and request id) from "the server is not there" (ApiUnreachableError). The UI
 * renders those very differently: one is a problem with the request, the other
 * is a problem with the deployment.
 */
export async function apiGet<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined | null>,
  options: RequestOptions = {},
): Promise<T> {
  const url = new URL(path, API_BASE);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }
  }

  // A timeout is essential: without one a hung API leaves the page in a
  // loading state forever, which looks like a frontend bug.
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  );
  if (options.signal) {
    options.signal.addEventListener("abort", () => controller.abort());
  }

  let response: Response;
  try {
    response = await fetch(url, {
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
  } catch (cause) {
    throw new ApiUnreachableError(cause);
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    let detail: string | undefined;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = typeof body.detail === "string" ? body.detail : undefined;
    } catch {
      // A non-JSON error body is not itself an error worth surfacing.
    }
    throw new ApiError(
      `${response.status} ${response.statusText}`,
      response.status,
      detail,
      response.headers.get("X-Request-ID") ?? undefined,
    );
  }

  return (await response.json()) as T;
}

export async function apiPost<T>(
  path: string,
  body: unknown,
  options: RequestOptions = {},
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
  );

  let response: Response;
  try {
    response = await fetch(new URL(path, API_BASE), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (cause) {
    throw new ApiUnreachableError(cause);
  } finally {
    clearTimeout(timeout);
  }

  if (!response.ok) {
    throw new ApiError(
      `${response.status} ${response.statusText}`,
      response.status,
      undefined,
      response.headers.get("X-Request-ID") ?? undefined,
    );
  }
  return (await response.json()) as T;
}

/**
 * Decode a base64 Float32Array sent by the projection endpoint.
 *
 * The galaxy receives coordinates as binary rather than JSON numbers: 72KB
 * instead of ~450KB for 6000 points, and the decoded buffer goes straight into
 * a BufferAttribute with no per-point allocation on the main thread.
 */
export function decodeFloat32(base64: string): Float32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Float32Array(bytes.buffer);
}

/** Decode a base64 Int32Array. Same rationale as {@link decodeFloat32}. */
export function decodeInt32(base64: string): Int32Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Int32Array(bytes.buffer);
}
