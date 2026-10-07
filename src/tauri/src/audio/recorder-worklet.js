// Mono mic tap for the voice pipeline. Posts copied Float32 chunks to the
// main thread and emits near-silence (no output channel is used).
class VoicertRecorder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs && inputs[0] && inputs[0][0];
    if (ch && ch.length > 0) {
      this.port.postMessage(ch.slice(0));
    }
    return true;
  }
}
registerProcessor("voicert-recorder", VoicertRecorder);
