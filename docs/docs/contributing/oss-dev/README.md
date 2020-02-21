# Workflow and setup

In order to contribute effectively it is important to understand our development workflow such as how we organize tasks, track progress, and integrate code. There are also general environment setup steps to describe here common to each of the development areas.

## Git and GitHub

### Org & repo permissions

Project-internal contributors are encouraged to push to branches off of the main repository instead of from separate forks. For this to work you will have to have been added to the repository and organization with the correct permissions. Please contact a project admin to help you verify this configuration.

### GitHub ssh config (linux)

Create a new SSH key and chmod

```bash
ssh-keygen

# Follow the provided steps to specify a full path that avoids conflicts with existing
# and probably uses a passphrase you can store e.g. in a password manager.

chmod 400 </full/path/to/your/id_rsa_key_name>

```

Copy the contents of your public key, excluding the trailing host name, e.g.

```bash

# The full public key
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCxchx9YiFNl9om4J8m57JG+iF7jVNk2m6n9aA5CwvjB/4nS6nqy4BWvgdKF/0uxntyZY0bDCcz9bKJU3oNXlriddpB24ZVMOFtepNX9j1YwCT02vLA/jrmCXZeBUioSCSmQ9r5ExEBuIyV5CLx+3+QkyJwIjPWGoJt/F/lS5SJwrEA4JBMueVK8WejjeA/fJTwCRSRouA+Ak4BvT9bflcIayPKEdn0FlzfNJ0+0NzPcczoZ98/NQXVheZNU9q5enH1m0Jh9xclgfQkN61mH8ZSfknCzLPqHBwDbB7K4B/GyMtLRzCo1zyaljg1qA+y/5tS7xN6jfXIggMtuUZ95Ror jovyan@cwbeitel-0

# The portion to copy (notice lack of trailing whitespace)
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCxchx9YiFNl9om4J8m57JG+iF7jVNk2m6n9aA5CwvjB/4nS6nqy4BWvgdKF/0uxntyZY0bDCcz9bKJU3oNXlriddpB24ZVMOFtepNX9j1YwCT02vLA/jrmCXZeBUioSCSmQ9r5ExEBuIyV5CLx+3+QkyJwIjPWGoJt/F/lS5SJwrEA4JBMueVK8WejjeA/fJTwCRSRouA+Ak4BvT9bflcIayPKEdn0FlzfNJ0+0NzPcczoZ98/NQXVheZNU9q5enH1m0Jh9xclgfQkN61mH8ZSfknCzLPqHBwDbB7K4B/GyMtLRzCo1zyaljg1qA+y/5tS7xN6jfXIggMtuUZ95Ror
```

Navigate to your [GitHub SSH and GPG key settings](https://github.com/settings/keys), click "New SSH key", provide any name you choose, and paste in the public key from above, followed by confirming with "Add SSH Key".

Return to your development machine and clone the repository if you haven't already:

```bash
git clone git@github.com:projectclarify/clarify.git
cd clarify
```

From the root of the repository, your git remotes should look like the following:

```bash
#git remote -v
origin git@github.com:projectclarify/clarify.git (fetch)
origin git@github.com:projectclarify/clarify.git (push)
```

If they do not, for example including https instead of git@, remove and replace these remotes:
```bash
git remote remove origin
git remote add origin git@github.com:projectclarify/clarify.git
```

Lastly, confirm the configuration is correct by modifying a file on your branch (which can be viewed via `git branch`), for example under the exp/ directory, staging it (e.g. by running `git add *`), add a commit message (e.g. git commit -m 'your commit message'), and push your branch to the repository (`git push origin [your branch name]`).

#### Debug

If problems arise with the previous please follow these steps and retry from the beginning of the section:

1. Re-start your machine. This will ensure any running ssh-agent processes are terminated. You may also attempt to terminate them manually.
2. Move your current ~/.ssh directory to ~/.ssh-backup. You can add your existing keys back in upon verifying things are functioning with GitHub as desired.
3. Remove any public keys currently added to your GitHub account (from previous steps).
4. When you retry the above, try using full paths at each step (as well as choosing a path that will not conflict with old keys once they are added back in).

TODO: We are looking for a refinement of these debug steps. If you have a deeper understanding of this and can contribute a refinement of these steps please [file an issue](https://github.com/projectclarify/clarify/issues) or send us a PR.
