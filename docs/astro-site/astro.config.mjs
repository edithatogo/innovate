import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import polyglot from 'starlight-polyglot';
import starlightConfig from './starlight.config.mjs';

export default defineConfig({
  site: 'https://docs.innovate.example',
  trailingSlash: 'always',
  integrations: [
    starlight({
      ...starlightConfig,
      plugins: [
        ...(starlightConfig.plugins ?? []),
        polyglot({
          python: {
            entryPoints: ['src/innovate'],
            output: 'api/python',
          },
        }),
      ],
    }),
    sitemap(),
  ],
});
