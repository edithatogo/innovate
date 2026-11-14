import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import starlight from '@astrojs/starlight';
import starlightDocSearch from '@astrojs/starlight-docsearch';
import remarkSingleTitle from './scripts/remark-single-title.mjs';
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
    processor: unified({ remarkPlugins: [remarkSingleTitle] }),
  },
  integrations: [
    starlight({
      title: 'Innovate',
      description: 'Rust-native innovation diffusion modelling with Polars.',
      customCss: ['./src/styles/custom.css'],
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
            { label: 'Scenario experiments', link: '/tutorials/scenario-experiment-workflows/' },
            { label: 'Installation', link: '/user-guide/installation/' },
            { label: 'Native capabilities and limits', link: '/native-migration/' },
          ],
        },
        {
          label: 'Earlier user guide (pre-0.7)',
          collapsed: true,
          items: [
            { label: 'Core Concepts', link: '/core/' },
            { label: 'Fitting Models', link: '/user-guide/fitting/' },
            { label: 'Predicting & Forecasting', link: '/user-guide/forecasting/' },
            { label: 'Diagnostics', link: '/core/diagnostics-contract/' },
            { label: 'Backends', link: '/user-guide/backends/' },
          ],
        },
        {
          label: 'API Reference',
          items: [
            { label: 'Python API', link: '/api/python/' },
            { label: 'Julia API', link: '/api/generated/julia/' },
            { label: 'R API', link: '/api/generated/r/' },
            { label: 'Native bindings', link: '/bindings/' },
          ],
        },
        {
          label: 'Earlier API design (pre-0.7)',
          collapsed: true,
          items: [
            { label: 'Kernel', link: '/core/kernel/' },
            { label: 'Arrow Interchange', link: '/core/arrow-interchange/' },
          ],
        },
        {
          label: 'Maintainers',
          items: [
            { label: 'Overview', link: '/maintainers/' },
            { label: 'Documentation quality contracts', link: '/maintainers/documentation-quality/' },
            { label: 'Publication', link: '/maintainers/publication/' },
            { label: 'DocSearch Gate', link: '/maintainers/docsearch/' },
            { label: 'Package Health', link: '/maintainers/package-health/' },
            { label: 'Compatibility', link: '/maintainers/compatibility/' },
            { label: 'Deprecation', link: '/maintainers/deprecation/' },
            { label: 'Support', link: '/maintainers/support/' },
            { label: 'Maintenance', link: '/maintainers/maintenance/' },
            { label: 'Deployment Readiness', link: '/maintainers/deployment-readiness/' },
            { label: 'Release Notes', link: '/maintainers/release-notes/' },
            { label: 'Plugins', link: '/maintainers/plugins/' },
            { label: 'Runtime Logging', link: '/maintainers/runtime-logging/' },
            { label: 'Stability', link: '/maintainers/stability/' },
          ],
        },
        {
          label: 'Operations',
          collapsed: true,
          items: [
            { label: 'Roadmap', link: '/operations/roadmap/' },
            { label: 'Release Maturity', link: '/operations/release-maturity/' },
            { label: 'HPC Readiness', link: '/operations/hpc-readiness/' },
            { label: 'Rust Core', link: '/operations/rust-core/' },
            { label: 'Governance', link: '/operations/governance/' },
            { label: 'Polyglot Registry', link: '/operations/polyglot-registry/' },
          ],
        },
        {
          label: 'Architecture',
          collapsed: true,
          items: [
            { label: 'Overview', link: '/architecture/' },
            { label: 'ADR Log', link: '/architecture/adr/' },
            { label: 'Polyglot Repo', link: '/architecture/polyglot-repo/' },
          ],
        },
        {
          label: 'Migration',
          collapsed: true,
          items: [
            { label: 'Overview', link: '/migration/' },
            { label: 'Redirects', link: '/migration/redirects/' },
            { label: 'Validation', link: '/migration/validation/' },
            { label: 'Archive', link: '/migration/archive/' },
            { label: 'References', link: '/migration/references/' },
          ],
        },
      ],
    }),
  ],
});
