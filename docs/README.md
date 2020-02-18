# docs

View our docs at [docs.cl4rify.org](https://docs.cl4rify.org)

The docs are deployed automatically on merges into master.

## Local development

Install dependencies:

```bash
yarn install
```

Build the docs

```bash
yarn docs:build
```

Run the local docs server

```bash
yarn docs:dev
```

If no other process is using 8080 you will be able to view the docs locally at localhost:8080. Some kinds of changes require restarting the dev server while others will hot reload.

