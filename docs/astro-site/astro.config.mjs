import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';
import starlight from '@astrojs/starlight';

export default defineConfig({
  site: 'https://docs.innovate.example',
  integrations: [
    starlight({
      title: 'Innovate',
      description:
        'Astro/Starlight migration scaffold for the Innovate documentation site.',
      social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/doughnut/innovate' }],
      sidebar: [
        {
          label: 'Core',
          items: [{ label: 'Kernel', link: '/core/kernel/' }],
        },
      ],
    }),
    sitemap(),
  ],
});
