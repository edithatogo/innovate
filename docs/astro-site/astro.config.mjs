import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightDocSearch from '@astrojs/starlight-docsearch';
import remarkSingleTitle from './scripts/remark-single-title.mjs';
import rehypeFocusableTables from './scripts/rehype-focusable-tables.mjs';
import starlightLinksValidator from 'starlight-links-validator';
import starlightVersions from 'starlight-versions';

const docSearchPlugins =
  process.env.ALGOLIA_APP_ID && process.env.ALGOLIA_API_KEY && process.env.ALGOLIA_INDEX_NAME
    ? [
        starlightDocSearch({
          appId: process.env.ALGOLIA_APP_ID,
          apiKey: process.env.ALGOLIA_API_KEY,
          indexName: process.env.ALGOLIA_INDEX_NAME,
          searchParameters: {},
        }),
      ]
    : [];

export default defineConfig({
  site: 'https://edithatogo.github.io',
  base: '/innovate',
  trailingSlash: 'always',
  markdown: {
    processor: unified({
      remarkPlugins: [remarkSingleTitle],
      rehypePlugins: [rehypeFocusableTables],
    }),
  },
  integrations: [
    starlight({
      title: 'Innovate',
      description: 'Rust-native innovation diffusion modelling with Polars.',
      customCss: ['./src/styles/custom.css'],
      ...(docSearchPlugins.length === 0
        ? { components: { Search: './src/components/AccessibleSearch.astro' } }
        : {}),
      editLink: {
        baseUrl: 'https://github.com/edithatogo/innovate/edit/main/docs/astro-site/src/content/docs/',
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/edithatogo/innovate',
        },
      ],
      plugins: [
        starlightLinksValidator(),
        ...docSearchPlugins,
        starlightVersions({
          current: { label: '0.7 Development' },
          versions: [{ slug: 'latest', label: '0.6 Reference' }],
        }),
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Quickstart', link: '/user-guide/getting-started/' },
            { label: 'Installation', link: '/user-guide/installation/' },
            { label: 'Native capabilities and limits', link: '/native-migration/' },
          ],
        },
        {
          label: 'User Guide',
          items: [
            { label: 'Core concepts', link: '/core/' },
            { label: 'Fitting models', link: '/user-guide/fitting/' },
            { label: 'Forecasting', link: '/user-guide/forecasting/' },
            { label: 'Native execution and backends', link: '/user-guide/backends/' },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            { label: 'Choose a tutorial', link: '/tutorials/' },
            { label: 'Scenario experiments', link: '/tutorials/scenario-experiment-workflows/' },
            { label: 'Norton–Bass model', link: '/tutorials/norton-bass/' },
            { label: 'Bayesian inference', link: '/tutorials/bayesian-fitter/' },
            { label: 'Counterfactual analysis', link: '/tutorials/counterfactual-analysis/' },
            { label: 'Arrow scenario results', link: '/tutorials/arrow-interchange/' },
            { label: 'Data provenance', link: '/tutorials/data-ingestion-provenance/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Python API', link: '/api/python/' },
            { label: 'Julia API', link: '/api/generated/julia/' },
            { label: 'R API', link: '/api/generated/r/' },
            { label: 'Native bindings', link: '/bindings/' },
            { label: 'Kernel requests', link: '/core/kernel/' },
            { label: 'Arrow interchange contract', link: '/core/arrow-interchange/' },
          ],
        },
        {
          label: 'Maintainers',
          items: [
            { label: 'Documentation quality contracts', link: '/maintainers/documentation-quality/' },
            { label: 'Test acceleration', link: '/maintainers/test-acceleration/' },
            { label: 'Registry readiness', link: '/operations/registry-readiness/' },
          ],
        },
        {
          label: 'Historical Reference',
          collapsed: true,
          items: [
            { label: 'Earlier documentation and current replacements', link: '/historical-documentation/' },
          ],
        },
      ],
    }),
  ],
});
