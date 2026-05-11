export default {
  title: 'Innovate',
  logo: {
    src: '/favicon.svg',
  },
  customCss: ['./src/styles/custom.css'],
  components: {},
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
      items: [{ label: 'Rust Core', link: '/operations/rust-core/' }],
    },
    {
      label: 'Maintainers',
      items: [
        { label: 'Publication', link: '/maintainers/publication/' },
        { label: 'Migration', link: '/migration/' },
      ],
    },
  ],
  editLink: {
    baseUrl: 'https://github.com/doughnut/innovate/edit/main/docs/astro-site/src/content/docs/',
  },
  social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/doughnut/innovate' }],
  markdown: {
    // The pinned plugin baseline is documented in package.json and the manifest.
    // starlight-versions 0.5.4
    // starlight-links-validator 0.18.0
    // @astrojs/starlight-docsearch 0.6.1 (Algolia DocSearch)
  },
};
