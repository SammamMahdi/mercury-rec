"use client";

/**
 * Application shell: navigation, backend status, and the honesty banner.
 *
 * The banner is not decoration. This project mixes real behavioural data with
 * synthesised attributes and runs entirely offline, and a visitor must be able
 * to tell that from the screen without reading the repository. It is always
 * present, never dismissible on pages showing metrics.
 */

import {
  Activity,
  Boxes,
  FlaskConical,
  GitBranch,
  Layers,
  LayoutDashboard,
  Orbit,
  Route,
  ScatterChart,
  Menu,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";

import { useHealth } from "@/lib/api/hooks";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/overview", label: "Overview", icon: LayoutDashboard },
  { href: "/explorer", label: "Explorer", icon: ScatterChart },
  { href: "/galaxy", label: "Galaxy", icon: Orbit },
  { href: "/pipeline", label: "Pipeline", icon: Route },
  { href: "/models", label: "Models", icon: Layers },
  { href: "/journey", label: "Journey", icon: GitBranch },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/monitoring", label: "Monitoring", icon: Activity },
  { href: "/architecture", label: "Architecture", icon: Boxes },
] as const;

/**
 * Live backend status.
 *
 * Polled rather than checked once at mount: a pill that said "connected" three
 * minutes ago and has not looked since is worse than no pill, because it
 * actively misleads while the page sits open.
 */
function BackendStatusPill() {
  const { data, isError, isLoading } = useHealth();

  const state = isLoading
    ? { label: "Connecting", colour: "var(--color-ink-faint)" }
    : isError
      ? { label: "API offline", colour: "var(--color-critical)" }
      : { label: `API v${data?.version ?? "?"}`, colour: "var(--color-positive)" };

  return (
    <div
      className="flex items-center gap-2 rounded-full border border-[var(--color-hairline)] bg-[var(--color-surface-1)] px-3 py-1"
      role="status"
      aria-live="polite"
    >
      <span
        className="size-1.5 rounded-full"
        style={{ backgroundColor: state.colour }}
        aria-hidden
      />
      <span className="text-[11px] font-medium text-[var(--color-ink-muted)]">
        {state.label}
      </span>
    </div>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();

  return (
    <nav aria-label="Primary" className="flex flex-col gap-0.5">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group flex items-center gap-2.5 rounded-[var(--radius-control)] px-3 py-2",
              "text-sm transition-colors duration-150",
              active
                ? "bg-[color-mix(in_oklch,var(--color-mercury)_12%,transparent)] text-[var(--color-mercury)]"
                : "text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)] hover:text-[var(--color-ink)]",
            )}
          >
            <Icon className="size-4 shrink-0" aria-hidden />
            <span>{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-dvh">
      {/* Skip link: the first tab stop, so keyboard users are not forced
          through nine navigation items on every page. */}
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-[var(--radius-control)] focus:bg-[var(--color-surface-2)] focus:px-4 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>

      <header className="sticky top-0 z-40 border-b border-[var(--color-hairline)] bg-[color-mix(in_oklch,var(--color-void)_88%,transparent)] backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-4 px-4">
          <button
            type="button"
            onClick={() => setMobileOpen((open) => !open)}
            className="rounded-[var(--radius-control)] p-1.5 text-[var(--color-ink-muted)] hover:bg-[var(--color-surface-2)] lg:hidden"
            aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>

          <Link href="/" className="flex items-center gap-2.5">
            <span
              className="size-2.5 rounded-full"
              style={{
                background:
                  "radial-gradient(circle at 30% 30%, var(--color-mercury-glow), var(--color-mercury-dim))",
                boxShadow: "0 0 12px -2px var(--color-mercury)",
              }}
              aria-hidden
            />
            <span className="text-sm font-semibold tracking-tight">MercuryRec</span>
          </Link>

          <div className="ml-auto flex items-center gap-3">
            <span className="hidden text-[11px] text-[var(--color-ink-faint)] sm:inline">
              Real-time recommendation intelligence
            </span>
            <BackendStatusPill />
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-[1600px] gap-6 px-4 py-6">
        <aside
          className={cn(
            "w-56 shrink-0",
            mobileOpen
              ? "fixed inset-x-0 top-14 z-30 h-[calc(100dvh-3.5rem)] overflow-y-auto border-b border-[var(--color-hairline)] bg-[var(--color-void)] p-4 lg:static lg:z-auto lg:h-auto lg:border-0 lg:bg-transparent lg:p-0"
              : "hidden lg:block",
          )}
        >
          <div className="lg:sticky lg:top-20">
            <NavLinks onNavigate={() => setMobileOpen(false)} />
          </div>
        </aside>

        <main id="main" className="min-w-0 flex-1 pb-16">
          {children}
        </main>
      </div>
    </div>
  );
}
