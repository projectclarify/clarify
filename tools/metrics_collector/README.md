# metrics_collector

Minor mods to https://github.com/kubeflow/katib/tree/master/cmd/tfevent-metricscollector to tailor to our case.

Build in the regular way, i.e.

```bash
docker build -t gcr.io/project/name:tag .

gcloud docker -- push gcr.io/project/name:tag

```

Used by the metrics collector CronJob that runs in parallel to Katib StudyJob trails which calls metrics_collector.py in the following way:

```yaml
apiVersion: batch/v1beta1
kind: CronJob
metadata:
  name: "{{.WorkerID}}"
  namespace: kubeflow
spec:
  schedule: "*/1 * * * *"
  successfulJobsHistoryLimit: 0
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: "{{.WorkerID}}"
            image: gcr.io/project/name:tag
            args:
            - "python"
            - "metrics_collector.py"
            - "--study_id"
            - "{{.StudyID}}"
            - "--worker_id"
            - "{{.WorkerID}}"
            - "--log_dir"
            - "<log_dir>/{{.WorkerID}}"
          restartPolicy: Never
          serviceAccountName: metrics-collector
```

where '<log_dir>' is templated at StudyJob creation time on the user side (see pcml.launcher.study.get_metrics_collector_template) as opposed to the other goTemplate variables that are templated at trial creation time.