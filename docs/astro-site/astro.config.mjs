import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightDocSearch from '@astrojs/starlight-docsearch';
import remarkSingleTitle from './scripts/remark-single-title.mjs';
import rehypeFocusableTables from './scripts/rehype-focusable-tables.mjs';
import starlightLinksValidator from 'starlight-links-validator';

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
            { label: 'Model atlas', link: '/models/' },
            { label: 'Fitting models', link: '/user-guide/fitting/' },
            { label: 'Forecasting', link: '/user-guide/forecasting/' },
            { label: 'Native execution and backends', link: '/user-guide/backends/' },
          ],
        },
        {
          label: 'Tutorials',
          items: [
            { label: 'Choose a tutorial', link: '/tutorials/' },
            { label: 'Fit public observations', link: '/tutorials/public-data-fit/' },
            { label: 'Explore in marimo', link: '/tutorials/marimo/' },
            { label: 'Batch execution', link: '/tutorials/native-batch/' },
            { label: 'Forcing and time', link: '/tutorials/forcing-and-time/' },
            { label: 'Network scenarios', link: '/tutorials/network-scenarios/' },
            { label: 'Kairos simulation', link: '/tutorials/kairos-simulation/' },
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
            { label: 'Language documentation conventions', link: '/maintainers/language-conventions/' },
            { label: 'Open science documentation standards', link: '/operations/documentation-standards/' },
            { label: 'Test acceleration', link: '/maintainers/test-acceleration/' },
            { label: 'Registry readiness', link: '/operations/registry-readiness/' },
          ],
        },
        {
          label: 'Migration',
          collapsed: true,
          items: [
            { label: 'Retired documentation and current routes', link: '/historical-documentation/' },
          ],
        },
      ],
    }),
  ],
});
