"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

/**
 * Application providers.
 *
 * The QueryClient is created inside `useState` rather than at module scope.
 * A module-level client is shared across every request on the server, which
 * leaks one user's cached data into another's render — the classic Next.js
 * App Router mistake. Creating it per component instance keeps each render
 * tree isolated.
 */
export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Refetching on every tab focus makes a dashboard flicker and
            // hammers the API for data that mostly does not change. Hooks that
            // genuinely need freshness set their own refetchInterval.
            refetchOnWindowFocus: false,
            staleTime: 60_000,
            retry: 1,
          },
        },
      }),
  );

  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
