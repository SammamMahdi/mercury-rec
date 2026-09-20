# MercuryRec frontend.
#
# Three stages, so the shipped image contains the built application and
# nothing that built it. A single-stage Next.js image carries the full
# node_modules tree including every dev dependency - typescript, eslint, the
# @types packages - which is roughly an order of magnitude more surface area
# than the thing actually being run.
#
# Build from the repository root or from apps/web; the compose file sets
# ./apps/web as the context.

# --- dependencies ------------------------------------------------------------
FROM node:24-alpine AS deps
WORKDIR /app

# Only the manifests, so this layer is reused across every source change. A
# COPY of the whole tree here would invalidate the install on every edit.
COPY package.json package-lock.json ./
RUN npm ci

# --- build -------------------------------------------------------------------
FROM node:24-alpine AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Baked in at build time, because Next.js inlines NEXT_PUBLIC_* into the
# client bundle. Changing it later requires a rebuild, which is a property of
# the framework rather than a choice made here; compose passes the value that
# a browser on the host can reach.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE

ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --- runtime -----------------------------------------------------------------
FROM node:24-alpine AS runtime
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0

# A dedicated unprivileged user with a fixed uid. Fixed, because a
# host-mounted volume's ownership has to be predictable, and root inside a
# container is root on the host kernel for anything that escapes.
RUN addgroup --system --gid 10002 mercury \
 && adduser --system --uid 10002 --ingroup mercury mercury

# The standalone output traces exactly the modules the server needs and ships
# its own minimal server.js. Static assets are not traced into it, so they are
# copied separately - a missing .next/static is the classic "the page renders
# with no styles" bug in a containerised Next.js app.
COPY --from=builder --chown=mercury:mercury /app/.next/standalone ./
COPY --from=builder --chown=mercury:mercury /app/.next/static ./.next/static
COPY --from=builder --chown=mercury:mercury /app/public ./public

USER mercury
EXPOSE 3000

# Hits the app itself rather than a static file, so the check fails when the
# server is up but not serving.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "server.js"]
