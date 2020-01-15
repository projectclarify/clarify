
# Fyre 🔥

Fyre is a tool for Fyring off Kubernetes Job's that will run code versioned via git. Currently it's only designed to work with the Project Clarify codebase and runtime environment.

Usage (command line):

```bash

# Print hello world, in batch
fyre --batch --autocommit -- echo "hello world"
# The --autocommit flag indicates fyre should mint a commit message

```

Usage (Python object):

```python

f = Fyre(autocommit=True, command="echo 'hello world'")
f.run()

```

Both cases trigger the following to run in batch:

```bash

fyre --commit=< the commit ID generated when launching job > \
  -- echo "hello world"

```

In more detail, let's understand the steps that occur, first for developing a command and next for running that in batch:

```python

# A FyreLocal object is created. We do not specify a commit so this will
# refer to whatever code version is currently on the machine at
# workspace_root.
flocal = FyreLocal(commit=None,
                   command="echo 'hello world'",
                   workspace_root="/home/jovyan/pcml")

# Run the command. Prints hello world.
flocal.run()

```

And now in batch

```python

# A Fyre object is created.
f = Fyre(autocommit=True,
         command="echo 'hello world'",
         workspace_root="/home/jovyan/pcml")

# Trigger a batch run
f.batch_run()

# Behind the scenes, Fyre._commit() is run and if --autocommit is 
# specified a commit message is minted; otherwise fails if 
# there are uncommitted changes.
f._commit()

# Instantiates a Job object whose command is the f.command and
# container is the latest version of the Project Clarify runtime.
f._fetch_latest_runtime_id()
job = f._make_job_object()
job.command = """

fyre --local \
     --workspace_root=/home/jovyan/pcml \
     --commit=< my new commit > -- \
     echo 'hello world'"

"""

# The .batch_run() method is then called on the Job object, running
# the job in batch.
job.batch_run()

# In batch, the above command is run which triggers the instantiation
# of a FyreLocal object with the same configuration as above along with
# a commit ID. 
flocal = FyreLocal(...)

# This is then run, as above.
flocal.run()

# But in this case, because the --commit attribute is specified we 
# check out the specified commit for the git repository at
# --workspace_root
flocal._checkout()

# Now we finally run the command to print hello world, with the
# current working directory set to --workspace_root.
echo "hello world"

```

## Wait... why?

Does that sound a bit convoluted? It is. This relates to the context of our data science / research users working within a JupyterLab notebook environment that does not have Docker available. Another piece of context is that what needs to run in batch will probably have changed recently, e.g. a model design, and is not already built into a container. A third piece of context is that the codebase that will be run spans many languages (e.g. not just Python, see the first case). Given those, our options for packaging code for a batch job include the following:

(1) If our project is a pure python project we can simply produce a wheel, copy that to Google Cloud Storage, stage the wheel in at job startup, and pip install it. But our project unforunately isn't that simple and this approach only ensures jobs can be reproduced as long as the whl is never deleted. This also requires setup at job time which is slow.

(2) We could just tgz the whole codebase and stage that through GCS. Unfortunately that is problematic for the latter two reasons of the former.

(3) We could trigger a container build using something like Google Container Builder that bundles the codebase into the container image. This is a fine option except for having to wait a fairly long time to pull, decompress, build, and push the resulting image that only has to again be pulled and decompressed at training time. So that option is slow but it works pretty well. Alternatively if we built with a container builder that maintained an image cache this wouldn't be as much of an issue.

(4) We could use a Bazel container_image rule to package the codebase into an image and run the job using that. This is a fine option but requires doing a lot of configuration with Bazel we haven't done yet.

(5) Use the fyre approach to get code into the container via a git repository. First code is pushed, then the job is launched and pulls the relevant version, then runs the command. This isn't necessarily leagues better than (4) but we've found it easier to set up and has the added benefit of GitHub being a source of truth about what jobs were run when. When as in (4) code is baked into a container we can reproduce the job but we don't have the information as readily available as we would if it were stored in a remote git repo. Seems like that could be beneficial with additional tooling but at present the primary motivation is that (5) was easier for us to implement than (4).
