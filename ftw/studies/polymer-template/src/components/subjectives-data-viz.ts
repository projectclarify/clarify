import { LitElement, css, html, property, customElement } from 'lit-element';
import { connect } from 'pwa-helpers/connect-mixin.js';

import { store, RootState } from '../store.js';

import { SharedStyles } from './shared-styles.js';
import { ButtonSharedStyles } from './button-shared-styles.js';

// This element is *not* connected to the Redux store.
@customElement('subjectives-data-viz')
export class SubjectivesDataViz extends connect(store)(LitElement) {

  @property({ type: Number })
  extraWidthY = 100;

  @property({ type: Map })
  userData = new Map();

  @property({ type: Map })
  stateLabelData = new Map();

  // Dimensions of chart, updates on window resize
  @property({ type: Array })
  chartDimensions = [600, 400];

  static get styles() {
    return [
      SharedStyles,
      ButtonSharedStyles,
      css`

      :host {
        width: 100%;
        height: 100%;
      }

      #dataviz-container {
        display: flex;
        flex-direction: row;
        justify-content: center;
      }

      .data-point {
        fill: var(--app-primary-color);
      }

      .oops {
        display: none;
      }

      .oops-inner {
        width: 100%;
        text-align: center
      }

      @media (max-width: 1048px) {

          .oops {
             display: flex;
             flex-direction: column;
             justify-content: center;
             height: 50vh
          }

          #dataviz-container {
              display: none
          }

      }


      `];
  }

  renderScatterplot() {

	let dataset = [];

    let measures = this.userData["measures"]["energy-subjective"];

    for (let timestamp in measures){
      dataset.push([parseInt(timestamp),
                    parseFloat(measures[timestamp])])
    }

    console.log(dataset)

    let w = this.chartDimensions[0];
    let h = this.chartDimensions[1];
    let padding = 40;

    let xMin = d3.min(dataset, function(d) { return d[0]; });
    let xMax = d3.max(dataset, function(d) { return d[0]; });

    let yMin = d3.min(dataset, function(d) { return d[1]; });
    let yMax = d3.max(dataset, function(d) { return d[1]; });

	let xScale = d3.scaleLinear()
			.domain([xMin, xMax])
			.range([padding, w - padding * 2]);
        
    let yScale = d3.scaleLinear()
        .domain([yMin, yMax])
        .range([h - padding, padding]);
    
    let xAxis = d3.axisBottom().scale(xScale).ticks(5);
    
    let yAxis = d3.axisLeft().scale(yScale).ticks(5);
    
    //create svg element
    var svg = d3.select(this.shadowRoot!.querySelector("#dataviz"));
    svg.attr("width", w).attr("height", h);

	svg.selectAll("circle")
			.data(dataset)
			.enter()
			.append("circle")
			.attr("cx", function(d) {
				return xScale(d[0]);
			})
			.attr("cy", function(d) {
				return yScale(d[1]);
			})
			.attr("r", 5)
            .attr("class", "data-point")	

    svg.append("g")
        .attr("class", "x axis")	
        .attr("transform", "translate(0," + (h - padding) + ")")
        .call(xAxis);
    
    svg.append("g")
        .attr("class", "y axis")	
        .attr("transform", "translate(" + padding + ", 0)")
        .call(yAxis);

  }

  protected render() {
    return html`

      <div id="dataviz-container">
        <svg id="dataviz"></svg>
      </div>

      <div class="oops">
      
        <p class="oops-inner">
          Currently this view is designed for displays >= 1048px wide.
        </p>

      </div>

      <div id="right-bar-menu"></div>

    `;
  }

  firstUpdated() {
    this.renderScatterplot();
  }

  stateChanged(state: RootState) {
    this.userData = state.user!.userData;
  }

}
