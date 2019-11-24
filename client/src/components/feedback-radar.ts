
import { LitElement, css, html, property, customElement } from 'lit-element';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

import { connect } from 'pwa-helpers/connect-mixin.js';
import { store, RootState } from '../store.js';

interface StateItem {
    label: string;
    target: number;
    current: number;
}

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

@customElement('feedback-radar')
export class FeedbackRadar extends connect(store)(LitElement) {

  @property({ type: Object })
  userFirestoreDocument = null;

  @property({ type: Number })
  radius = 5;

  @property({ type: Number })
  height = 400;

  @property({ type: Number })
  width = 400;

  @property({ type: Number })
  factor = 1;

  @property({ type: Number })
  factorLegend = 0.85;

  @property({ type: Number })
  levels = 3;

  @property({ type: Number })
  maxValue = 0;

  @property({ type: Number })
  radians = 2 * Math.PI;

  @property({ type: Number })
  opacityArea = 0.5;

  @property({ type: Number })
  toRight = 5;

  @property({ type: Number })
  translateX = 80;

  @property({ type: Number })
  translateY = 30;

  @property({ type: Number })
  extraWidthX = 100;

  @property({ type: Number })
  extraWidthY = 100;

  @property({ type: Number })
  color = 1; /* todo: these should be given by label type model */

  @property({ type: Object })
  targetAreaChartRef = null;

  @property({ type: Object })
  currentAreaChartRef = null; 

  @property({ type: String })
  lastStateDataTimestamp = "foo";

  @property({ type: Object })
  _avModalityDocRef = null;

  @property({ type: Number })
  videoSecondsPerPacket = 2;

  @property({ type: Number })
  videoFps = 1;

  @property({ type: Number })
  videoXDim = 256;

  @property({ type: Number })
  videoYDim = 256;

  @property({ type: Number })
  audioSecondsPerPacket = 2;

  @property({ type: Number })
  videoNumChannels = 3;

  @property({ type: Boolean })
  shouldContinueCapturingFrames = true;

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

        :host {
          width: 100%;
          display: flex;
          flex-direction: row;
          justify-content: center;
        }

        .feedback-card {
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
            max-width: 400px;
            max-height: 400px;
            height: 80vw;
            width: 80vw;
        }

        .feedback-legend {
          position: relative;
          width: 120px;
          top: 10px;
          left: 10px;
        }

        .feedback-legend-icon-current, .feedback-legend-icon-target {
          height: 14.8px;
          display: flex;
          flex-direction: column;
          justify-content: center;
        }

        .feedback-legend-icon-current > svg > circle {
          fill: var(--app-primary-color);
        }

        .feedback-legend-icon-target > svg > circle {
          fill: none;
          stroke: var(--app-primary-color);
        }

        .feedback-legend {
        }

        .feedback-legend-item {
          display: flex;
          flex-direction: row;
        }

        .feedback-legend-item-text {
          margin-left: 10px;
          color: black;
          font-size: 10px;
          font-weight: 100;
        }

        .feedback-viz {
            fill: #242424;
            text-align: center;
            cursor: default;
        }

        .feedback-viz-container {
            display:block;
        }

        .feedback-viz-tooltip {
            fill: #333333;
        }

        .target-point {
          fill: none;
          stroke: var(--app-primary-color);
        }

        .current-area {
          fill: var(--app-primary-color);
          fill-opacity: 0.8;
          stroke-width: 2;
          stroke: var(--app-primary-color);
        }
    
        .target-area {
          fill: none;
          fill-opacity: 0.8;
          stroke-width: 2;
          stroke: var(--app-primary-color);
          stroke-dasharray: 4 16;
        }
        
        .text-label {
          font-size: 8px;
          text-anchor: middle
        }

        video {
            position: fixed;
            top: -500px;
            height: 256px;
            width: 256px;
        }
        canvas {
            position: fixed;
            top: -500px;
        }

      `];
  }

  renderSmallCircle() {
    return html`
      <svg width="8" height="8" viewbox="0 0 8 8">
        <circle cx="4" cy="4" r="3" fill="white" />
      </svg>
    `
  }

  getFeedbackVizRef() {
    let feedbackViz = this.shadowRoot.querySelector("#feedback-viz");
    var svg = d3.select(feedbackViz);
    return svg
  }

  firstRenderDataviz() {

    // Remove any existing
    this.shadowRoot.querySelector("#feedback-viz").innerHTML = "";

    let initialState = [
        {"label": "happiness", "target": 0.5, "current": 0.5},
        {"label": "calm", "target": 0.5, "current": 0.5},
        {"label": "confidence", "target": 0.5, "current": 0.5},
        {"label": "kindness", "target": 0.5, "current": 0.5},
        {"label": "focus", "target": 0.5, "current": 0.5},
        {"label": "posture", "target": 0.5, "current": 0.5},
        {"label": "flow", "target": 0.5, "current": 0.5}]

    this._renderDataviz(initialState);

  }

  _radiansPerIdx(data: Array<StateItem>) {
    return (2*Math.PI/(data.length))
  }

  _curveGeneratorGenerator(data: Array<StateItem>, key: string, height: number, width: number) {

    let radiusMax = 0.75*Math.min(height/2, width/2)

    let centerFraction = 0.25;
    let innerRadius = centerFraction * radiusMax;
    let outerThis = this

    return d3.areaRadial()
	  .angle(function(d: StateItem, i: number) {
		return i * outerThis._radiansPerIdx(data) })
	  .innerRadius(function() {return 0})
	  .outerRadius(function(d: StateItem) {
		return innerRadius + (1 - centerFraction) * d[key] * radiusMax;
	  })
      .curve(d3.curveCardinalClosed.tension(0.7));

  }

  _renderAxes(data: Array<StateItem>, height: number, width: number) {

    let svg = this.getFeedbackVizRef();

    let text = svg.selectAll("text")
                  .data(data)
                  .enter()
                  .append("text");

    let _this = this

    let labelTerminusRadius = 0.75

    let xText = function(d: StateItem, i: number): number {
    
      let radians: number = i * _this._radiansPerIdx(data);
      let xUnscaled: number = Math.cos(radians) * labelTerminusRadius;
      return xUnscaled * width/2 + width/2

    }

    let yText = function(d: StateItem, i: number): number {
    
      let radians: number = i * _this._radiansPerIdx(data);
      console.log(`yrad: ${radians}`);
      let xUnscaled: number = Math.sin(radians) * labelTerminusRadius;
      console.log(`yunscaled: ${xUnscaled}`);
      return xUnscaled * height/2 + height/2

    }

    let textLabels = text
        .attr("x", xText)
        .attr("y", yText)
        .text( function (d: StateItem) { return d["label"]; })
        .attr("class", "text-label")

  }

  _renderDataviz(data: Array<StateItem>) {

    let container = this.shadowRoot.querySelector(".feedback-viz-container");
    let bounds = container.getBoundingClientRect();
    let height = bounds.width
    let width = bounds.width
    this.height = height;
    this.width = width;

    let svg = this.getFeedbackVizRef();

    this._renderAxes(data, height, width)

    let targetAreaGenerator = this._curveGeneratorGenerator(
        data, "target", height, width
    )

	let targetAreaChart = svg.append('path')
        .attr('d', targetAreaGenerator(data))
        .attr("class", "target-area")
        .attr("transform", `translate(${width/2}, ${height/2})`);

    let currentAreaGenerator = this._curveGeneratorGenerator(
        data, "current", height, width
    )

	let currentAreaChart = svg.append('path')
        .attr('d', currentAreaGenerator(data))
        .attr("class", "current-area")
        .attr("transform", `translate(${width/2}, ${height/2})`);

    this.targetAreaChartRef = targetAreaChart;
    this.currentAreaChartRef = currentAreaChart;

  }

  _renderUpdate(data: Array<StateItem>) {

      console.log("update")

      let t = d3.transition()
        .duration(750)
        .ease(d3.easeLinear);

      /*let updateData = [
        {"label": "happiness", "target": 0.9, "current": Math.random()},
        {"label": "calm", "target": 0.6, "current": Math.random()},
        {"label": "confidence", "target": 0.75, "current": Math.random()},
        {"label": "kindness", "target": 0.8, "current": Math.random()},
        {"label": "focus", "target": 0.8, "current": Math.random()},
        {"label": "posture", "target": 0.8, "current": Math.random()},
        {"label": "flow", "target": 0.9, "current": Math.random()}
        ]*/

      let updateCurrentGenerator = this._curveGeneratorGenerator(
          data, "current", this.height, this.width
      );

      this.currentAreaChartRef!.transition(t)
        .attr('d', updateCurrentGenerator(data));

      let updateTargetGenerator = this._curveGeneratorGenerator(
          data, "target", this.height, this.width
      );

      this.targetAreaChartRef!.transition(t)
        .attr('d', updateTargetGenerator(data));

  }

  firstUpdated() {
    this.firstRenderDataviz();
    let thisRef = this;
    window.addEventListener("resize", function(e: Event){
        thisRef.firstRenderDataviz()
    });
    this.configureVideoStream();
  }

  protected render() {
    return html`

      <div class="feedback-card">

        <div class="feedback-legend">
          <div class="feedback-legend-item"><div class="feedback-legend-icon-current">${this.renderSmallCircle()}</div><div class="feedback-legend-item-text">Current</div></div>
          <div class="feedback-legend-item"><div class="feedback-legend-icon-target">${this.renderSmallCircle()}</div><div class="feedback-legend-item-text">Goal</div></div>
        </div>

        <div class="feedback-viz-container">
          <svg id="feedback-viz" height=${this.height} width=${this.width}></svg>
          <video></video>
          <canvas id="video-canvas"></canvas>
        </div>

      </div>

    `;
  }

  async handleStateChanged(userData: Object) {

      if ( userData["state"] != null ) {

        let ts = userData["state"]["meta"]["timestamp"];

        if ( ts != null) {

            if (ts != this.lastStateDataTimestamp) {

                // The data is different so we should use it to trigger
                // a render.
                console.log("state data change triggering render");
                this._renderUpdate(userData["state"]["data"]);

                this.lastStateDataTimestamp = ts;

            }

        }

      }

  }

  async handleDBRefChanged(db: any) {

    console.log("handling possible db ref change");

    if ( db != null ) {

        if ( this._avModalityDocRef == null ) {

            let avDocRef = db.collection("modalities").doc("av");

            this._avModalityDocRef = avDocRef;

            let initialPacket = {
                    "videoData": "placeholder",
                    "audioData": "placeholder",
                    "videoMeta": {
                        "fps": this.videoFps,
                        "xDim": this.videoXDim,
                        "yDim": this.videoYDim,
                        "numChannels": this.videoNumChannels
                    },
                    "audioMeta" : {
                        "placeholder": 0
                    },
                    "meta": {
                        "timestamp": Date.now(),
                        "duration": this.audioSecondsPerPacket
                    }
                }

            console.log("Writing initial AV packet")
            this.writeAvPacket(initialPacket);
            
        }

    }

  }

  writeAvPacket(dataPacket: AVDataPacket) {

    console.log("Writing AV packet.")
    console.log(dataPacket)

    let avDocRef = this._avModalityDocRef;
    if (avDocRef != null) {
      avDocRef.set(dataPacket).then(function() {
            console.log("Document successfully written!");
        });
    } else {
        console.log("Can't write to null av modality doc ref.")
    }

  }

  stateChanged(state: RootState) {
    this.handleStateChanged(state.user!.userData);
    this.handleDBRefChanged(state.user!.userFirestoreDB);
  }

  async configureVideoStream() {

    let stream = null;

    const supports = navigator.mediaDevices.getSupportedConstraints();
    if (!supports.aspectRatio || !supports.facingMode || !supports.width || !supports.height) {
        // Treat like an error.
        console.log("Unsupported video configuration.")
    }
    const constraints = {
        width: this.videoXDim,
        height: this.videoYDim,
        advanced: [
          {frameRate: this.videoFps},
          {facingMode: 'user'},
          {channelCount: 3},
        ]
    };

    try {
        stream = await navigator.mediaDevices.getUserMedia({
        video: constraints, audio: false
        });
    } catch(err) {
        console.log("An error occurred: " + err);
    }

    let video = this.shadowRoot.querySelector('video');
    if (!video) {
        throw new Error("Can't find the video element.");
    }

    video.srcObject = stream;

    await video.play();

    this.continuouslyCaptureFrames();
    video.onloadedmetadata = function(e) {
    }

  }

  getPacketTemplate(): AVDataPacket {
    return {
            "videoData": "placeholder",
            "audioData": "placeholder",
            "videoMeta": {
                "fps": this.videoFps,
                "xDim": this.videoXDim,
                "yDim": this.videoYDim,
                "numChannels": this.videoNumChannels
            },
            "audioMeta" : {
                "placeholder": 0
            },
            "meta": {
                "timestamp": Date.now(),
                "duration": this.audioSecondsPerPacket
            }
        }    
  }

  async continuouslyCaptureFrames() {

    try {

        let packetTemplate = this.getPacketTemplate();

        let canvas: HTMLCanvasElement = this.shadowRoot.querySelector('canvas');
        let video: HTMLVideoElement = this.shadowRoot.querySelector('video');

        console.log("Capturing frame");

        var context = canvas.getContext('2d');

        let width = packetTemplate["videoMeta"]["xDim"];
        let height = packetTemplate["videoMeta"]["yDim"];

        canvas.width = width;
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);
        let videoData = canvas.toDataURL('image/jpeg'); 

        packetTemplate["videoData"] = videoData;
        packetTemplate["meta"]["timestamp"] = Date.now()

        this.writeAvPacket(packetTemplate);

    } catch (error) {

        console.log("Caught error " + error)

    }

    if (this.videoFps == 0) {
        throw new Error("Video fps must not be zero");
    }

    let timeout = 1000/this.videoFps;

    setTimeout(this.continuouslyCaptureFrames.bind(this), timeout);

  }

}
