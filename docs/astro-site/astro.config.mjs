import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';
import starlightConfig from './starlight.config.mjs';

export default defineConfig({
  site: 'https://docs.innovate.example',
  trailingSlash: 'always',
  integrations: [
    starlight(starlightConfig),
    sitemap(),
  ],
});
