# Capacity benchmark

_GPU: NVIDIA GeForce RTX 3050 Laptop GPU | rate: $0.05/hr | 2026-09-12T22:49:30_

## LLM (TTFT / TPO / throughput / efficiency / cost)

| genlen | conc | TTFT p50 | TPO | TPS | req/s | tok/s | util% | W | tok/s/W | $/1M |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1 | 214ms | 20.3ms | 26.4 | 2.40 | 26.4 | 2.0 | 19.8 | 1.33 | $0.5265 |
| 16 | 2 | 30ms | 29.4ms | 34.6 | 4.85 | 65.5 | 1.0 | 22.1 | 2.96 | $0.2122 |
| 48 | 1 | 14ms | 14.4ms | 69.6 | 6.31 | 69.4 | 1.0 | 22.1 | 3.14 | $0.2000 |
| 48 | 2 | 28ms | 25.8ms | 40.3 | 3.44 | 67.1 | 16.0 | 27.5 | 2.44 | $0.2070 |

## E2E round-trip (TTS synth -> stream_turn)

TTFA 308ms, E2E 827ms, VRAM 1957MB

## STT/TTS spot

STT 59ms (RTF 0.020), TTS 141ms (RTF 0.045)
