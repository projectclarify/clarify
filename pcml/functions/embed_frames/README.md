## Video frames embedding predictor

The code integrates sensory modality data that are delivered to Google Cloud Firestore by way of the web UI with models served via TFServing on Kubeflow/Kubernetes/GKE. The function is triggered by the arrival of new video frames. When the function is called it constructs the new frames into a query, sends that query to the served model, parses the result, and writes the prediction back to the relevant part of the originating Firestore database.

### Deployment

The stock deployment expects there to be a Cloud FireStore database with document structure `users/{username}/modalities/video`.

And can be performed via the following:

```bash
python -m pcml.functions.embed_frames.deploy \
  --service_account={your service account} \
  --region={ the gcp region of your served model }
```
