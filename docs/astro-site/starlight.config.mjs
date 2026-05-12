import starlightDocSearch from '@astrojs/starlight-docsearch';
import starlightLinksValidator from 'starlight-links-validator';

export default {
  title: 'Innovate',
  customCss: ['./src/styles/custom.css'],
  components: {},
  plugins: [
    starlightLinksValidator(),
    // Credentials are supplied by the deployment environment.
    starlightDocSearch({
      appId: process.env.ALGOLIA_APP_ID ?? 'YOUR_APP_ID',
      apiKey: process.env.ALGOLIA_API_KEY ?? 'YOUR_SEARCH_API_KEY',
      indexName: process.env.ALGOLIA_INDEX_NAME ?? 'YOUR_INDEX_NAME',
      searchParameters: {},
    }),
  ],
  sidebar: [
    {
      label: 'Core',
      items: [
        { label: 'Kernel', link: '/core/kernel/' },
        { label: 'Arrow Interchange', link: '/core/arrow-interchange/' },
        { label: 'Diagnostics', link: '/core/diagnostics-contract/' },
        { label: 'Bindings', link: '/bindings/' },
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
        { label: 'Release Notes', link: '/maintainers/release-notes/' },
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
      items: [{ label: 'Tutorials', link: '/tutorials/' }],
    },
  ],
  editLink: {
    baseUrl: 'https://github.com/doughnut/innovate/edit/main/docs/astro-site/src/content/docs/',
  },
  social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/doughnut/innovate' }],
  markdown: {
    // The pinned plugin baseline is documented in package.json and the manifest.
    // starlight-versions 0.5.4
    // starlight-links-validator 0.24.0
    // @astrojs/starlight-docsearch 0.6.1 (Algolia DocSearch)
  },
};
