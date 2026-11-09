// @ts-check
import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import mdx from "@astrojs/mdx";

// Static output only. The production domain is thehoopledger.com; SITE_URL can
// override it (e.g. for a preview deployment). Canonical URLs and the sitemap use it.
export default defineConfig({
  site: process.env.SITE_URL || "https://thehoopledger.com",
  integrations: [mdx(), sitemap()],
  output: "static",
  trailingSlash: "always",
  build: { format: "directory" },
  devToolbar: { enabled: false },
  markdown: { shikiConfig: { theme: "github-light" } },
});
