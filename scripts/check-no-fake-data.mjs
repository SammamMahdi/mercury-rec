#!/usr/bin/env node
/**
 * Fail the build if the frontend starts inventing numbers.
 *
 * The whole product rests on one claim: every figure on screen came from the
 * API or from a committed artifact. That claim is easy to make and easy to
 * erode — someone hard-codes a plausible latency while the backend is down,
 * it looks right, and it ships. This turns that from a review question into a
 * failing build.
 *
 * What it looks for, and why each one:
 *
 *  - `Math.random()` in application source. There is no legitimate reason for
 *    a dashboard to generate a number, and it is the fastest way to produce a
 *    chart that moves convincingly and means nothing.
 *  - Identifiers named after fabrication (mockData, FAKE_METRICS, dummyRows,
 *    sampleLatency). Naming is the honest signal here: nobody calls real data
 *    `fakeStats`.
 *  - A numeric literal handed to a metric component. `<MetricTile value={0.87}/>`
 *    renders exactly like a measurement.
 *  - Metric-shaped literals in visible text: "+42%", "p95: 12ms", "3.2x
 *    faster". These are the claims a reader will quote back.
 *
 * Escape hatch: a line carrying `no-fake-data-ok: <reason>` is exempt. The
 * reason is mandatory, because an unexplained suppression is how a gate stops
 * meaning anything. Structural constants — a point budget, a bin count, a
 * chart's pixel height — are the legitimate use.
 *
 *   node scripts/check-no-fake-data.mjs [rootDir]
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

const REPO_ROOT = fileURLToPath(new URL("..", import.meta.url));
const ROOT = process.argv[2] ?? join(REPO_ROOT, "apps", "web", "src");

const SOURCE_EXTENSIONS = [".ts", ".tsx", ".js", ".jsx"];
const SKIP_DIRECTORIES = new Set(["node_modules", ".next", "dist", "build", "coverage"]);

/** Generated files are not hand-written claims about the world. */
const SKIP_FILES = [/schema\.d\.ts$/];

const ALLOW_MARKER = /no-fake-data-ok:\s*\S+/;

/**
 * Lines that are entirely a comment.
 *
 * Skipped deliberately. This gate is about what reaches the screen, and a
 * comment explaining why a fabricated figure would be wrong is part of the
 * defence rather than a breach of it. A number in a comment renders nowhere.
 */
const COMMENT_ONLY = /^\s*(\/\/|\/\*|\*)/;

const RULES = [
  {
    id: "random",
    pattern: /Math\s*\.\s*random\s*\(/,
    message: "Math.random() generates data. A dashboard has nothing to generate.",
  },
  {
    id: "fabricated-identifier",
    // Word-boundary anchored, so `sampleUsers` (a real API response) does not
    // trip it while `sampleData` and `mockMetrics` do.
    //
    // `placeholder` is absent from the camelCase alternation on purpose:
    // TanStack Query's `placeholderData` holds the PREVIOUS REAL RESPONSE
    // while the next one loads, which is the opposite of fabrication. The
    // screaming-case form still catches a `PLACEHOLDER_METRICS` constant.
    pattern:
      /\b(mock|fake|dummy|stub|fixture)(Data|Metrics?|Stats|Rows|Values?|Results?|Response|Items?|Latency|Numbers?)\b|\b(MOCK|FAKE|DUMMY|STUB|PLACEHOLDER)_[A-Z_]+\b/,
    message:
      "An identifier named after fabricated data. If it is real, name it for what it is.",
  },
  {
    id: "literal-metric-prop",
    // value={0.87} or value={1234} on a component, but NOT value={expression}.
    pattern: /\bvalue=\{\s*-?\d+(\.\d+)?\s*\}/,
    message:
      "A numeric literal passed as a metric value renders identically to a measurement.",
  },
  {
    id: "quoted-percentage-claim",
    // A signed percentage literal, in the three shapes a claim actually
    // takes. Each alternative is narrowed to stay clear of CSS, which uses
    // percentages constantly and legitimately.
    //
    //  1. JSX text: `>+42%`. Nothing renders CSS after a `>`, so either sign.
    //  2. A quoted string, but only with an explicit `+`. `"-50%"` is a
    //     transform offset; `"+42%"` is somebody stating a lift, because CSS
    //     has no use for a leading plus.
    //  3. Mid-sentence: `improved by +38%`. The preceding letter is what
    //     excludes `translate(-50%, -50%)` and `left: "-50%"`, which follow a
    //     bracket, comma or colon.
    //
    // A computed percentage trips none of them: it has no literal digits
    // beside the sign at all, reading `{value.toFixed(1)}%`.
    pattern:
      />\s*[+-]\d+(\.\d+)?\s*%|["']\s*\+\d+(\.\d+)?\s*%|[A-Za-z]\s+[+-]\d+(\.\d+)?\s*%/,
    message: "A hard-coded percentage change. Lifts must come from a computed comparison.",
  },
  {
    id: "quoted-latency-claim",
    pattern: /\b(p50|p95|p99|latency|throughput|rps|qps)\b\s*[:=]\s*["']?\s*\d+(\.\d+)?\s*(ms|s|rps|qps)\b/i,
    message: "A hard-coded latency or throughput figure. These must be measured.",
  },
  {
    id: "quoted-speedup-claim",
    pattern: /\b\d+(\.\d+)?\s*(x|×)\s*(faster|slower|speedup|improvement)/i,
    message: "A hard-coded speed-up claim. Benchmarks belong in docs/benchmarks.md.",
  },
];

function* walk(directory) {
  for (const entry of readdirSync(directory)) {
    const path = join(directory, entry);
    if (statSync(path).isDirectory()) {
      if (!SKIP_DIRECTORIES.has(entry)) yield* walk(path);
      continue;
    }
    if (!SOURCE_EXTENSIONS.some((extension) => entry.endsWith(extension))) continue;
    if (SKIP_FILES.some((pattern) => pattern.test(path))) continue;
    yield path;
  }
}

function scan(path) {
  const findings = [];
  const lines = readFileSync(path, "utf8").split(/\r?\n/);

  lines.forEach((line, index) => {
    if (ALLOW_MARKER.test(line) || COMMENT_ONLY.test(line)) return;

    for (const rule of RULES) {
      if (!rule.pattern.test(line)) continue;
      findings.push({
        file: relative(REPO_ROOT, path).split(sep).join("/"),
        line: index + 1,
        rule: rule.id,
        message: rule.message,
        source: line.trim().slice(0, 120),
      });
    }
  });

  return findings;
}

let scanned = 0;
const findings = [];
for (const path of walk(ROOT)) {
  scanned += 1;
  findings.push(...scan(path));
}

if (findings.length === 0) {
  console.log(`no-fake-data: ${scanned} files clean`);
  process.exit(0);
}

console.error(`no-fake-data: ${findings.length} finding(s) across ${scanned} files\n`);
for (const finding of findings) {
  console.error(`  ${finding.file}:${finding.line}  [${finding.rule}]`);
  console.error(`    ${finding.message}`);
  console.error(`    ${finding.source}\n`);
}
console.error(
  "If one of these is a structural constant rather than a claim about the\n" +
    "world, add `no-fake-data-ok: <reason>` to that line.",
);
process.exit(1);
