import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightDocSearch from '@astrojs/starlight-docsearch';
import polyglot from 'starlight-polyglot';
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
  site: 'https://edithatogo.github.io/innovate',
  base: '/innovate',
  trailingSlash: 'always',
  integrations: [
    starlight({
      title: 'Innovate',
      description: 'A Python library for simplifying innovation/policy diffusion modelling.',
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
          current: { label: 'Current' },
          versions: [{ slug: 'latest', label: 'Latest' }],
        }),
        polyglot({
          python: {
            entryPoints: ['../../src/innovate'],
            pythonExecutable: process.env.STARLIGHT_POLYGLOT_PYTHON ?? 'python',
            output: 'api/python',
          },
        }),
      ],
      sidebar: [
        {
          label: 'Getting Started',
          items: [
            { label: 'Overview', link: '/' },
            { label: 'Quickstart', link: '/user-guide/getting-started/' },
            { label: 'Installation', link: '/user-guide/installation/' },
          ],
        },
        {
          label: 'User Guide',
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
            { label: 'Kernel', link: '/core/kernel/' },
            { label: 'Arrow Interchange', link: '/core/arrow-interchange/' },
            { label: 'Bindings', link: '/bindings/' },
          ],
        },
        {
          label: 'Maintainers',
          items: [
            { label: 'Overview', link: '/maintainers/' },
            { label: 'Publication', link: '/maintainers/publication/' },
            { label: 'DocSearch Gate', link: '/maintainers/docsearch/' },
            { label: 'Release Notes', link: '/maintainers/release-notes/' },
            { label: 'Plugins', link: '/maintainers/plugins/' },
            { label: 'Runtime Logging', link: '/maintainers/runtime-logging/' },
            { label: 'Stability', link: '/maintainers/stability/' },
          ],
        },
        {
          label: 'Operations',
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
          items: [
            { label: 'Overview', link: '/architecture/' },
            { label: 'ADR Log', link: '/architecture/adr/' },
            { label: 'Polyglot Repo', link: '/architecture/polyglot-repo/' },
          ],
        },
        {
          label: 'Migration',
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
