import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: "/ruslingam/",
  title: "ruslingam",
  description: "Accelerating LiNGAM with Rust.",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Getting Started', link: '/getting-started' },
      { text: 'API', link: '/direct-lingam' }
    ],

    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting Started', link: '/getting-started' },
          { text: 'Threading', link: '/threading' },
          { text: 'Differences from lingam', link: '/differences' }
        ]
      },
      {
        text: 'API Reference',
        items: [
          { text: 'DirectLiNGAM', link: '/direct-lingam' },
          { text: 'BootstrapResult', link: '/bootstrap' },
          { text: 'CAMUV', link: '/camuv' },
          { text: 'Module functions', link: '/module-functions' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/ikeuchi-screen/ruslingam' }
    ]
  }
})
