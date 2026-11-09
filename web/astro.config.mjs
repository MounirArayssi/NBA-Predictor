// @ts-check
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";

// Static output only. Set SITE_URL when a production domain exists so canonical
// URLs and the sitemap use it; without it, canonical tags are omitted.
export default defineConfig({
  site: process.env.SITE_URL || undefined,
  integrations: [mdx(), ...(process.env.SITE_URL ? [sitemap()] : [])],
  output: "static",
  trailingSlash: "always",
  build: { format: "directory" },
  devToolbar: { enabled: false },
  markdown: { shikiConfig: { theme: "github-light" } },
});
