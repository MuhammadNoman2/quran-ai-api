/**
 * Captures microphone audio and posts it as 16-bit PCM.
 *
 * Runs on the audio thread, so it must not allocate carelessly or block. The
 * AudioContext is created at 16 kHz, so no resampling is needed here - the
 * browser does it upstream, correctly and for free.
 */
class PcmWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    // ~100 ms at 16 kHz. Frame size is not otherwise significant: the server
    // decides when to run recognition, not the client.
    this.frameSize = 1600;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;

    for (let i = 0; i < channel.length; i++) this.buffer.push(channel[i]);

    while (this.buffer.length >= this.frameSize) {
      const slice = this.buffer.splice(0, this.frameSize);
      const pcm = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        const clamped = Math.max(-1, Math.min(1, slice[i]));
        pcm[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}

registerProcessor("pcm-worklet", PcmWorklet);
