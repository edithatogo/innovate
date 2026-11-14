import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { docsVersionsLoader } from 'starlight-versions/loader';

const docs = defineCollection({
  loader: docsLoader(),
  schema: docsSchema({ extend: z.object({ description: z.string().trim().min(1) }) }),
});

const versions = defineCollection({
  loader: docsVersionsLoader(),
});

export const collections = { docs, versions };
