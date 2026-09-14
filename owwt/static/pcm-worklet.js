// Captures microphone audio, low-pass filters it and resamples to 16 kHz mono Int16,
// posting 1280-sample (80 ms) chunks: the frame size openWakeWord works with.
class Pcm16k extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.inputRate = (options.processorOptions && options.processorOptions.inputRate) || sampleRate;
    this.step = this.inputRate / 16000;

    // Windowed-sinc low-pass at 7.2 kHz so nothing aliases when we drop to 16 kHz
    const taps = 63;
    const fc = Math.min(7200 / this.inputRate, 0.45);
    this.kernel = new Float32Array(taps);
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const n = i - (taps - 1) / 2;
      const sinc = n === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * n) / (Math.PI * n);
      const hamming = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (taps - 1));
      this.kernel[i] = sinc * hamming;
      sum += this.kernel[i];
    }
    for (let i = 0; i < taps; i++) this.kernel[i] /= sum;

    this.history = new Float32Array(taps);
    this.historyPos = 0;
    this.readPos = 0; // position of the next output sample, in input samples
    this.inputIndex = 0;
    this.previous = 0;
    this.out = new Int16Array(1280);
    this.outLength = 0;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    const size = this.history.length;
    for (let i = 0; i < input.length; i++) {
      this.history[this.historyPos] = input[i];
      this.historyPos = (this.historyPos + 1) % size;
      let y = 0;
      for (let k = 0, j = this.historyPos; k < size; k++, j = (j + 1) % size) y += this.kernel[k] * this.history[j];

      while (this.readPos <= this.inputIndex) {
        const frac = this.readPos - (this.inputIndex - 1);
        this.push(this.previous + (y - this.previous) * frac);
        this.readPos += this.step;
      }
      this.previous = y;
      this.inputIndex++;
    }
    return true;
  }

  push(sample) {
    const v = Math.max(-1, Math.min(1, sample));
    this.out[this.outLength++] = v < 0 ? v * 32768 : v * 32767;
    if (this.outLength === this.out.length) {
      this.port.postMessage(this.out, [this.out.buffer]);
      this.out = new Int16Array(1280);
      this.outLength = 0;
    }
  }
}

registerProcessor("pcm16k", Pcm16k);
