import { defineCollection } from 'astro:content';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';
import { docsVersionsLoader } from 'starlight-versions/loader';

const docs = defineCollection({
  loader: docsLoader(),
  schema: docsSchema(),
});

const versions = defineCollection({
  loader: docsVersionsLoader(),
});

export const collections = { docs, versions };
