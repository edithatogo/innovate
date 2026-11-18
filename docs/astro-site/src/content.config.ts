import { defineCollection } from 'astro:content';
import { file } from 'astro/loaders';
import { z } from 'astro/zod';
import { docsLoader } from '@astrojs/starlight/loaders';
import { docsSchema } from '@astrojs/starlight/schema';

const docs = defineCollection({
  loader: docsLoader(),
  schema: docsSchema({ extend: z.object({ description: z.string().trim().min(1) }) }),
});

// Generated from native discovery and the content-quality example bindings.
// A catalogue entry records advertised source metadata, not qualification.
const catalogue = '../contracts/learning-catalogue.json';
const identifierList = z.array(z.string().min(1));

const learningModels = defineCollection({
  loader: file(catalogue, { parser: (text) => JSON.parse(text).models }),
  schema: z.object({
    title: z.string().min(1),
    summary: z.string().min(1),
    advertised: z.looseObject({
      key: z.string().regex(/^[a-z][a-z0-9_]*$/),
      family: z.string().min(1),
      stability: z.string().min(1),
      supported_operations: identifierList.min(1),
      supported_backends: identifierList.min(1),
      simulation_engines: identifierList.min(1),
      simulation_semantics: z.string().min(1),
      variants: identifierList,
    }).superRefine((advertised, context) => {
      for (const [key, value] of Object.entries(advertised)) {
        if (key.startsWith('supports_') && typeof value !== 'boolean') {
          context.addIssue({ code: 'custom', path: [key], message: 'Capability flags must be boolean.' });
        }
      }
    }),
  }),
});

const learningTutorials = defineCollection({
  loader: file(catalogue, { parser: (text) => JSON.parse(text).tutorials }),
  schema: z.object({
    route: z.string().startsWith('/innovate/').endsWith('/'),
    title: z.string().min(1),
    description: z.string().min(1),
    source: z.string().min(1),
    example_id: z.string().min(1),
    language: z.string().min(1),
    network: z.boolean(),
    execution_status: z.literal('not_run_by_generator'),
  }),
});

export const collections = { docs, learningModels, learningTutorials };
