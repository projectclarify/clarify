// .vuepress/config.js

const sidebar = [
  {
    title: 'Project',
    collapsable: true,
    children: [
      '/project-overview/',
      {
        title: 'Community',
	collapsable: true,
	children: [
	  ['/community/', 'Overview']
	]
      },
      {
        title: 'Design and proposals',
        collapsable: true,
        children: [
          ['/community/proposals/', 'Overview'],
          '/community/proposals/ftw',
        ],
      },
  ]},
  {
    title: 'Contributor Guide',
    collapsable: true,
    children: [
      '/contributing/',
      '/contributing/oss-dev/',
      '/contributing/dev-fx/',
      '/contributing/dev-ml/',
      '/contributing/dev-platform/'
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
      '/contributing/': sidebar,
      '/contributing/oss-dev/': sidebar,
      '/contributing/dev-fx/': sidebar,
      '/contributing/dev-ml/': sidebar,
      '/contributing/dev-platform/': sidebar,
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
