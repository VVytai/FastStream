import {VirtualAudioNode} from '../../ui/audio/VirtualAudioNode.mjs';

export class LinkwitzRileyCrossoverNetwork {
  constructor(ctx) {
    this.ctx = ctx;
    this.sampleRate = ctx.sampleRate;
    this._outputIndexToNode = new Map();
    this.input = new VirtualAudioNode('Crossover input');
  }

  getInputNode() {
    return this.input;
  }

  getOutputNode(id) {
    const node = this._outputIndexToNode.get(id);
    if (!node) {
      const newNode = new VirtualAudioNode(`Crossover output ${id}`);
      this._outputIndexToNode.set(id, newNode);
      return newNode;
    }
    return node;
  }

  _build() {
    // Cache per section nodes used in the processing cascade
    this._sectionNodes = []; // array of { lp: IIRFilterNode, hp: IIRFilterNode, allpass: IIRFilterNode[] }

    // Build the crossover network
    let stageInput = this.input;
    for (let i = 0; i < this.sections.length; i++) {
      const {low, high, allpass} = this.sections[i];

      const lp = new IIRFilterNode(this.ctx, {feedforward: low.ff, feedback: low.fb});
      const hp = new IIRFilterNode(this.ctx, {feedforward: high.ff, feedback: high.fb});

      stageInput.connect(lp);
      stageInput.connect(hp);

      let lpOut = lp;
      const apNodes = [];
      for (let k = 0; k < allpass.length; k++) {
        const ap = new IIRFilterNode(this.ctx, {feedforward: allpass[k].ff, feedback: allpass[k].fb});
        lpOut.connect(ap);
        lpOut = ap;
        apNodes.push(ap);
      }

      // Route band i to the GainNode associated with mappings[i]
      const outIndex = this.mappings[i];
      const outNode = this.getOutputNode(outIndex);
      outNode.connectFrom(lpOut);

      // Store for analysis
      this._sectionNodes.push({lp, hp, allpass: apNodes});

      // Next stage uses the highpassed branch
      stageInput = hp;
    }

    // Route final high band to its mapped output index
    const lastMapIndex = this.mappings[this.sections.length];
    const lastNode = this.getOutputNode(lastMapIndex);
    lastNode.connectFrom(stageInput);
  }

  configure({cutoffs, mappings}) {
    if (mappings.length !== cutoffs.length + 1) {
      throw new Error('mappings must be one longer than cutoffs');
    }

    this._destroy();

    this.cutoffs = cutoffs.slice();
    this.mappings = mappings.slice();

    // Prepare IIR coefficients per crossover section
    this.sections = this.cutoffs.map((fc, i) => ({
      low: this._lr4Coeffs(fc, 'lowpass'),
      high: this._lr4Coeffs(fc, 'highpass'),
      allpass: this.cutoffs.slice(i + 1).map((f2) => this._allpass2Coeffs(f2, 1.0)),
    }));

    // Rebuild the graph
    this._build();
  }

  destroy() {
    this._destroy();
  }

  _destroy() {
    if (!this._sectionNodes) {
      return;
    }

    let stageInput = this.input;
    for (let i = 0; i < this._sectionNodes.length; i++) {
      const {lp, hp, allpass} = this._sectionNodes[i];
      stageInput.disconnect(lp);
      stageInput.disconnect(hp);
      let lpOut = lp;
      for (let k = 0; k < allpass.length; k++) {
        lpOut.disconnect(allpass[k]);
        lpOut = allpass[k];
      }

      const outIndex = this.mappings[i];
      const outNode = this.getOutputNode(outIndex);
      outNode.disconnectFrom(lpOut);

      stageInput = hp;
    }

    const lastMapIndex = this.mappings[this.sections.length];
    const lastNode = this.getOutputNode(lastMapIndex);
    lastNode.disconnectFrom(stageInput);

    this._sectionNodes = [];
  }

  // 4th order Linkwitz Riley section coefficients
  _lr4Coeffs(fc, type) {
    const fs = this.sampleRate;
    const wc = 2.0 * Math.PI * fc;
    const wc2 = wc * wc;
    const wc3 = wc2 * wc;
    const wc4 = wc2 * wc2;

    const k = wc / Math.tan(Math.PI * fc / fs);
    const k2 = k * k;
    const k3 = k2 * k;
    const k4 = k2 * k2;

    const sqrt2 = Math.sqrt(2.0);
    const sq1 = sqrt2 * wc3 * k;
    const sq2 = sqrt2 * wc * k3;

    const a_tmp = 4.0 * wc2 * k2 + 2.0 * sq1 + k4 + 2.0 * sq2 + wc4;

    const b1 = (4.0 * (wc4 + sq1 - k4 - sq2)) / a_tmp;
    const b2 = (6.0 * wc4 - 8.0 * wc2 * k2 + 6.0 * k4) / a_tmp;
    const b3 = (4.0 * (wc4 - sq1 + sq2 - k4)) / a_tmp;
    const b4 = (k4 - 2.0 * sq1 + wc4 - 2.0 * sq2 + 4.0 * wc2 * k2) / a_tmp;

    let a0; let a1; let a2; let a3; let a4;

    if (type === 'lowpass') {
      a0 = wc4 / a_tmp;
      a1 = 4.0 * wc4 / a_tmp;
      a2 = 6.0 * wc4 / a_tmp;
      a3 = a1;
      a4 = a0;
    } else if (type === 'highpass') {
      a0 = k4 / a_tmp;
      a1 = -4.0 * k4 / a_tmp;
      a2 = 6.0 * k4 / a_tmp;
      a3 = a1;
      a4 = a0;
    } else {
      throw new Error('type must be lowpass or highpass');
    }

    return {
      ff: [a0, a1, a2, a3, a4],
      fb: [1, b1, b2, b3, b4],
    };
  }

  // Biquad allpass coefficients (RBJ)
  _allpass2Coeffs(fc, Q = 1.0) {
    const w0 = 2 * Math.PI * (fc / this.sampleRate);
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const alpha = sinw / (2 * Q);

    const b0 = 1 - alpha;
    const b1 = -2 * cosw;
    const b2 = 1 + alpha;

    const a0 = 1 + alpha;
    const a1 = -2 * cosw;
    const a2 = 1 - alpha;

    return {ff: [b0, b1, b2], fb: [a0, a1, a2]};
  }
}
