import starlightDocSearch from '@astrojs/starlight-docsearch';
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

export default {
  title: 'Innovate',
  customCss: ['./src/styles/custom.css'],
  components: {},
  plugins: [
    starlightLinksValidator(),
    // Credentials are supplied by the deployment environment.
    ...docSearchPlugins,
  ],
  sidebar: [
    {
      label: 'Core',
      items: [
        { label: 'Kernel', link: '/core/kernel/' },
        { label: 'Arrow Interchange', link: '/core/arrow-interchange/' },
        { label: 'Diagnostics', link: '/core/diagnostics-contract/' },
        { label: 'Bindings', link: '/bindings/' },
        { label: 'C# Bindings', link: '/bindings/csharp/' },
        { label: 'Go Bindings', link: '/bindings/go/' },
        { label: 'Julia Bindings', link: '/bindings/julia/' },
        { label: 'Rust Bindings', link: '/bindings/rust/' },
      ],
    },
    {
      label: 'Operations',
      items: [
        { label: 'ABI Compatibility', link: '/operations/abi-compatibility/' },
        { label: 'Community Readiness', link: '/operations/community-readiness/' },
        { label: 'Accelerator Parallel Execution', link: '/operations/accelerator-parallel-execution/' },
        { label: 'Governance', link: '/operations/governance/' },
        { label: 'HPC Registry Contract', link: '/operations/hpc-registry/' },
        { label: 'HPC Readiness', link: '/operations/hpc-readiness/' },
        { label: 'HPC Submission Packet', link: '/operations/hpc-submission-packet/' },
        { label: 'HPC Submission Workflow', link: '/operations/hpc-submission-workflow/' },
        { label: 'Polyglot Registry Plan', link: '/operations/polyglot-registry/' },
        { label: 'Remote Execution', link: '/operations/remote-execution/' },
        { label: 'Registry Submissions', link: '/operations/registry-submissions/' },
        { label: 'Release Maturity', link: '/operations/release-maturity/' },
        { label: 'Rust Core', link: '/operations/rust-core/' },
        { label: 'Roadmap', link: '/operations/roadmap/' },
        { label: 'Scientific HPC', link: '/operations/scientific-hpc/' },
        { label: 'Submission Dossiers', link: '/operations/submission-dossiers/' },
        { label: 'XLA Backend', link: '/operations/xla-backend/' },
      ],
    },
    {
      label: 'Maintainers',
      items: [
        { label: 'Publication', link: '/maintainers/publication/' },
        { label: 'DocSearch Gate', link: '/maintainers/docsearch/' },
        { label: 'Package Health', link: '/maintainers/package-health/' },
        { label: 'Compatibility', link: '/maintainers/compatibility/' },
        { label: 'Deprecation', link: '/maintainers/deprecation/' },
        { label: 'Support', link: '/maintainers/support/' },
        { label: 'Maintenance', link: '/maintainers/maintenance/' },
        { label: 'Deployment Readiness', link: '/maintainers/deployment-readiness/' },
        { label: 'Release Readiness', link: '/maintainers/release-readiness/' },
        { label: 'Release Notes', link: '/maintainers/release-notes/' },
        { label: 'Binding Conformance', link: '/maintainers/binding-conformance/' },
        { label: 'Plugins', link: '/maintainers/plugins/' },
        { label: 'Runtime Logging', link: '/maintainers/runtime-logging/' },
        { label: 'Stability', link: '/maintainers/stability/' },
        { label: 'Migration', link: '/migration/' },
        { label: 'Redirects', link: '/migration/redirects/' },
        { label: 'Validation', link: '/migration/validation/' },
        { label: 'Archive', link: '/migration/archive/' },
        { label: 'References', link: '/migration/references/' },
      ],
    },
    {
      label: 'Roadmap',
      items: [
        { label: 'Diagnostics Uncertainty', link: '/roadmap/diagnostics-uncertainty/' },
        { label: 'Probabilistic Inference', link: '/roadmap/probabilistic-inference/' },
        { label: 'DataFrame Engine Experiments', link: '/roadmap/dataframe-engine/' },
      ],
    },
    {
      label: 'Architecture',
      items: [
        { label: 'Architecture', link: '/architecture/' },
        { label: 'ADR', link: '/architecture/adr/' },
        { label: 'Polyglot Repo', link: '/architecture/polyglot-repo/' },
      ],
    },
    {
      label: 'Tutorials',
      items: [
        { label: 'Tutorials', link: '/tutorials/' },
        { label: 'Advanced Runtime', link: '/tutorials/advanced-runtime/' },
        { label: 'Benchmark Workflows', link: '/tutorials/benchmark-workflows/' },
        { label: 'Kairos Simulation Adapter', link: '/tutorials/kairos-simulation-adapter/' },
        { label: 'Plugin API Stability', link: '/tutorials/plugin-api-stability/' },
      ],
    },
  ],
  editLink: {
    baseUrl: 'https://github.com/doughnut/innovate/edit/main/docs/astro-site/src/content/docs/',
  },
  social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/doughnut/innovate' }],
  markdown: {
    // The pinned plugin baseline is documented in package.json and the manifest.
    // starlight-versions 0.9.0
    // starlight-links-validator 0.24.1
    // @astrojs/starlight-docsearch 0.7.0 (Algolia DocSearch)
  },
};
