import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /**
   * Trace the modules the server actually needs and emit a self-contained
   * server.js. Without this the container image has to carry the whole
   * node_modules tree - typescript, eslint, every @types package - to run a
   * build that needs almost none of it.
   */
  output: "standalone",

  /**
   * The repository root holds the Python project, so Next.js finds two
   * lockfiles and has to guess which one marks the workspace root. Saying so
   * explicitly keeps the standalone trace rooted at this app rather than at a
   * directory it inferred.
   */
  outputFileTracingRoot: __dirname,
};

export default nextConfig;
