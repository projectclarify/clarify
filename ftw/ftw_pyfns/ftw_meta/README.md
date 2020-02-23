# ftw loader functions

## Usage

All of the functions can be deployed at once using the following.

```bash
# Deploy
bazel run //ftw/loader_fns:deploy -- \
  --project_id=[your project] \
  --region=[your region]

```

Deployed functions can be test queried via curl in the usual way providing the relevant REGION, PROJECT_ID, FUNCTION_NAME, and ID_TOKEN fields.

```bash
# Test query

curl --request POST https://[REGION]-[PROJECT_ID].cloudfunctions.net/[FUNCTION_NAME] -H "Authorization: bearer [ID_TOKEN]"

```

