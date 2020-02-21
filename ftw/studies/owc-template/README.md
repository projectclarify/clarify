
# FTW 

<a href="https://open-wc.org/guide/" target="_blank"><img src="https://img.shields.io/badge/thanks-open--wc-blueviolet"></img></a>


## Docs

Our project was built using the open-wc initializer and aims to continue to conform to those standards. Thus the primary open-wc documentation should cover most needs in that regard: https://open-wc.org/guide/.


#### Install Node Version Manager (nvm)

Please refer to the [nvm repo README](https://github.com/nvm-sh/nvm) for instructions on setting up nvm on your system. This involves running the following as well as adding some content to the relevant profile file.

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.35.2/install.sh | bash
```

#### Upgrade node and npm

```bash
# Install latest
nvm install --latest-npm

# Show npm version
npm --version
6.13.7

# Show node version
node --version
13.8.0
```

#### Install Chrome (Linux; for headless browser testing)

```bash

# Get chrome .deb
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

# Install
sudo dpkg -i google-chrome-stable_current_amd64.deb

```

#### Yarn install

From the root of this directory, simply run:

```bash
yarn install
```

Please note that we are not using npm in this project and installing package with npm instead of yarn will create a second, potentially conflicting package lock file. A PR prepared in this way will have to roll back to using Yarn and confirm the same functionality.

#### Testing and coverage

Tests can be run with the following:

```bash
npm run test
```

This also generates a test coverage report that is helpful for us to ensure all code is properly tested. The above command is run at PR submission time.

During development, you may also find it handy to run tests continuously with each file change. The test runner can be started in watch mode via the following:

```bash
npm run test:watch
```

#### Linting and formatting

You can apply our formatting standard by running the formatting command

```bash
npm run format
```

As well as to run the same linter checks that will be run on the CI upon submitting a pull request:

```bash
npm run lint
```

#### Building

Building is only required when preparing to deploy to production so most developers won't need to be concerned with it. This process involves producing a version of the app that is optimized for different browsers and tries to minimize the size of files. A build can be run via the following:

```bash
npm run build
```

#### Deployment

The @firebase-tools package is only needed for deployment so we don't include it in our main set of dependencies. It can be installed globally with `npm` via the following:

```bash
npm install -g firebase-tools
```

Upon running this command you should not see any changes in the ftw-ui sub-repo i.e. no new package lock files, no additional lines in the yarn.lock or subdirectories of node_modules including the `firebase` string.

#### Development

##### Forwarding

For developers within Project Clarify, we advise development within a JupyterLab notebook instance running on our Kubeflow deployment as well as the forwarding of relevant ports to your personal Google Cloud Shell instance to permit (1) actually viewing the served page as well as (2) this page having OAuth origin and redirect domains that are within a controlled set.

In order to view served pages, you will need to forward served ports from your jlab development container over to your Cloud Shell container which firstly will require that within cloud shell you have pulled the necessary Kubernetes credentials, via the following:

```bash
gcloud container clusters get-credentials --zone us-central1-a [current cluster name]
```

where current cluster name is replaced with the cluster name obtained from a system admin, peer, or by looking at the prefix of the URL for JupyterLab instances, i.e. having the structure:

```
https://[1. current cluster name].endpoints.[our project name].cloud.goog/notebook/[2. your chosen namespace name]/[3. your chosen notebook and volume name]/lab?
```

⚠️ Please note: you will also need the name of your (2) namespace and (3) notebook/volume name in the following steps so if you don't remember the one you chose upon initializing your Kubeflow account please make note of it (as it is also shown in this URL). ⚠️

After obtaining credentials, you may determine the ID of your notebook pod by listing the pods in your namespace:

```bash
kubectl get pods -n [2. your chosen namespace name]
```

Your notebook ID will be listed as the pod name prefixed by (3) (your notebook/volume name) above.

Given that, you can initiate a port forward from your Cloud Shell by running the following:

```bash
kubectl port-forward -n [2. your namespace] [3. your notebook pod name] [cs port]:[jlab port]
# e.g. for me something like:
# kubectl port-forward -n kubeflow-chris-w-beitel cwbeitel-0 8080:8000
# with the ui serving on jlab at 8000 and web preview launched from cs on port 8080.
```

Where [jlab port] is the port being served from JupyterLab and [cs port] is the port you will display via web preview.  What port is forwarded will depend on what port is being served within your JupyterLab instance (see the following sections). Also, Cloud Shell only supports a limited set of ports for Web Preview so please ensure you are forwarding to one of those (e.g. both 8000 and 8080 are available).

##### Application



The application can be served locally via the following command, indicating the application is being served on the lowest-numbered port available on localhost at or above 8000:

```bash
npm start
```

You can forward 

##### Storybook

Storybook is a valuable tool for team development of UI components, read more [here](https://open-wc.org/demoing/). You can start a local Storybook server running via the following which you will see available on the lowest-numbered port available on localhost at or above 8000:

```bash
npm run storybook
```
