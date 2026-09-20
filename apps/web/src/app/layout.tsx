import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { AppShell } from "@/components/layout/AppShell";

import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

/**
 * Every numeral in the product is monospaced so digits align in tables and a
 * changing latency figure does not make the layout twitch.
 */
const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "MercuryRec",
    template: "%s · MercuryRec",
  },
  description:
    "Multi-stage recommendation and personalization platform: candidate retrieval, " +
    "two-tower neural embeddings, learning-to-rank, experimentation, serving and monitoring.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrains.variable} antialiased`}
        style={{
          // Bind the loaded fonts to the design tokens, so components refer to
          // --font-sans/--font-mono and never to a framework-specific variable.
          ["--font-sans" as string]: "var(--font-inter)",
          ["--font-mono" as string]: "var(--font-jetbrains)",
        }}
      >
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
