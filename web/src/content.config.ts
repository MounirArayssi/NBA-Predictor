import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Notebook entries are plain Markdown files in src/content/notebook/.
const notebook = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/notebook" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    topic: z.enum(["Model Development", "Forecast Evaluation", "Team Trends", "Data Quality", "Experiments", "Research Notes"]),
    summary: z.string(),
    outcome: z.enum(["Kept", "Rejected", "Open", "Note"]),
  }),
});

export const collections = { notebook };
