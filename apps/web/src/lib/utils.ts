import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge class names, letting later Tailwind utilities win over earlier ones. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Format a number for display, or return a dash when it is absent.
 *
 * Returning "—" rather than "0" is deliberate and load-bearing: a zero reads
 * as a measurement, and rendering one for missing data is exactly the kind of
 * invented figure this project forbids.
 */
export function formatNumber(
  value: number | null | undefined,
  options: { decimals?: number; compact?: boolean; percent?: boolean } = {},
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";

  const { decimals = 0, compact = false, percent = false } = options;
  if (percent) {
    return `${(value * 100).toFixed(decimals)}%`;
  }
  if (compact && Math.abs(value) >= 1000) {
    return new Intl.NumberFormat("en", {
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value);
  }
  return new Intl.NumberFormat("en", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

/** Format milliseconds, choosing a unit that keeps the number readable. */
export function formatMs(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (value < 1) return `${(value * 1000).toFixed(0)}µs`;
  if (value < 1000) return `${value.toFixed(value < 10 ? 2 : 1)}ms`;
  return `${(value / 1000).toFixed(2)}s`;
}

/** Format a signed relative change, always showing the sign. */
export function formatDelta(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/** Shorten a hash for display while keeping it recognisable. */
export function shortHash(hash: string | null | undefined, length = 8): string {
  if (!hash) return "—";
  return hash.length <= length ? hash : hash.slice(0, length);
}
