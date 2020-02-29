// .vuepress/config.js

const sidebar = [
  {
    title: 'Project',
    collapsable: true,
    children: [
      '/project-overview/',
      '/community/',
    ],
  },
  {
    title: 'Contributor Guide',
    collapsable: true,
    children: [
      '/design-overview/',
      '/oss-dev/',
      '/dev-fx/',
      '/dev-ml/',
      '/dev-platform/'
    ],
  },
  {
    title: 'Object docs',
    collapsable: true,
    children: [
      '/obj-docs/',
      '/obj-docs/clarify/',
      '/obj-docs/ftw-ui/',
      '/obj-docs/hc/'
    ]
  },
  {
    title: 'API docs',
    collapsable: true,
    children: [
      '/api-docs/',
      '/ftw-api/',
      '/models-api/'
    ]
  },
];

module.exports = {
  title: '',
  description: 'An open community devoted to using ML to benefit people.',
  themeConfig: {
    logo: '/logo.png',
    displayAllHeaders: false,
    sidebarDepth: 2,
    sidebar: {
      '/project-overview/': sidebar,
      '/community/': sidebar,
      '/design-overview/': sidebar,
      '/oss-dev/': sidebar,
      '/dev-fx/': sidebar,
      '/dev-ml/': sidebar,
      '/dev-platform/': sidebar,
      '/obj-docs/': sidebar,
      '/obj-docs/clarify/': sidebar,
      '/obj-docs/ftw-ui/': sidebar,
      '/obj-docs/hc': sidebar,
      '/api-docs/': sidebar,
      '/ftw-api/': sidebar,
      '/models-api/': sidebar
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Docs', link: '/project-overview/' },
    ],
    docsDir: 'docs',
    editLinks: true,
    editLinkText: 'Edit Page',
    lastUpdated: 'Last Updated',
  },
  dest: '_site',
};
