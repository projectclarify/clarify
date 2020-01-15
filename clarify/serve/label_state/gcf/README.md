## State labeler

The code integrates sensory modality data that are delivered to Google Cloud Firestore by way of the web UI with models served via TFServing on Kubeflow/Kubernetes/GKE. The function is triggered by the arrival of a new audio/video data packet. When the function is called it constructs the new frames into a query, sends that query to the served model, parses the result, and writes the prediction back to the relevant part of the originating Firestore database, that being `users/{uid}` with data structure `Array<StateItem>`:

```typescript
interface StateItem {
    label: string;
    target: number;
    current: number;
}
```

e.g.

```json
    let initialState = [
        {"label": "happiness", "target": 0.5, "current": 0.5},
        {"label": "calm", "target": 0.5, "current": 0.5},
        {"label": "confidence", "target": 0.5, "current": 0.5},
        {"label": "kindness", "target": 0.5, "current": 0.5},
        {"label": "focus", "target": 0.5, "current": 0.5},
        {"label": "posture", "target": 0.5, "current": 0.5},
        {"label": "flow", "target": 0.5, "current": 0.5}]
```

There labels here are hypothetical placeholders meant to provide context for the ML program not existing capabilities.

### Deployment

The function can be deployed with the following:

```bash
python -m pcml.functions.label_state.deploy \
  --service_account={e.g. <your-project>@appspot.gserviceaccount.com} \
  --region={ the gcp region of your served model e.g. us-central1 }
```

Where the specified service account has been granted (1) Service Account Token Creator and (2) IAP-secured Web App User, and (3) Firebase Admin permissions and where your Kubeflow deployment has [IAP](https://cloud.google.com/iap/docs/) enabled (see [here](https://www.kubeflow.org/docs/gke/deploy/oauth-setup/) for instructions).

(TODO: Does it need these first two or just iap.httpsResourceAccessor per [this](https://www.kubeflow.org/docs/components/serving/tfserving_new/#sending-prediction-request-through-ingress-and-iap)).

The stock deployment expects the following data structure at path `users/{uid}/modalities/av`:

```typescript


interface VideoMeta {

  // Frames per second (not yet applicable)
  fps: Number;

  // The length of the x and y dimensions of a frame in pixels
  xDim: Number;
  yDim: Number;

  // The number of color channels of the frame (probably 3)
  numChannels: Number

}

interface AVMeta {

  // Milliseconds since the Unix epoch
  timestamp: Number;

  // Duration of the clip in seconds
  duration: Number;

}

interface AVDataPacket {

    // The raw video data, for now for a single frame,
    // flattened
    videoData: string;

    // Single channel of raw audio data corresponding to 
    // video data
    audioData: string;

    // Metadata specific to audio or video
    videoMeta: VideoMeta;
    audioMeta: Map<string, any>;

    // Packet level metadata (timestamp and duration)
    meta: AVMeta;

}

```