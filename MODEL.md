````

 [PROB]                                           Architecture: Silero VAD [ONNX CRNN]
    │                                              .
    + ──────────┐                                   \_ [2.3 MB ONNX]
    |   ┌──────────────────────────┐               .
    │   |      LSTM CLASSIFIER     │                \_ [WIN 512 @16kHz = 32ms]
    │   └──────────────────────────┘               .
    |──[SIGMOID]───┘                                \_ [CTX 64 + State 2x128]
    + ──────────┐                                  .
    │   ┌──────────────────────────┐                \_ [SR 16000 Hz]
    │   |      FRAME ENCODER       │               .
    │   └──────────────────────────┘                \_ [RTF 0.01 CPU]
    └──[CONTEXT]───┘                              .
    │                                               \_ [Threshold 0.5]
 [INPUT]  (16kHz mono, 32ms frames)

````

````

 [SPEECH / NON-SPEECH PROB]                         Architecture: Silero VAD v4 [ONNX]
        │                                      .
        + ──────────┐                          \_ [~2.3 MB ONNX]
        │   ┌──────────────────────────┐       .
        │   |     OUTPUT PROBABILITY   |        \_ [CPU Execution Provider]
        │   └──────────────────────────┘       .
        │              ▲                       .
        │      [RECURRENT / CONTEXT]            \_ [Window: 512 samples]
        │              ▲                       .
        │   ┌──────────────────────────┐       .
        │   |     VAD NEURAL NETWORK   |        \_ [Context: 64 samples]
        │   └──────────────────────────┘       .
        │              ▲                       .
        │      [AUDIO FEATURES]                 \_ [Sample Rate: 16 kHz]
        │              ▲                       .
   ┌──────────────────────────┐
   |      AUDIO WINDOW        |
   └──────────────────────────┘
        │
 [PCM AUDIO @ 16 kHz]

````
````

 [TEXT]                                           Architecture: Whisper [Encoder-Decoder Transformer]
    │                                              .
    + ──────────┐                                   \_ [74M Parameters]
    |   ┌──────────────────────────┐               .
    │   |      DECODER 6× [512]    │                \_ [D=512 H=8 Dh=64]
    │   └──────────────────────────┘               .
    |──[LAYER-NORM]──┘                              \_ [FF 2048 / XN 1500]
    + ──────────┐                                  .
    │   ┌──────────────────────────┐                \_ [80 Mel Bins / 3000 Frames]
    │   |      ENCODER 6× [512]    │               .
    │   └──────────────────────────┘                \_ [N_MELS 80 / Hop 160]
    └──[LAYER-NORM]───┘                            .
    │                                               \_ [51865 Vocab Size]
 [INPUT]  (log-mel 80x3000)



````

 [OUTPUT]                                         Architecture: Qwen3 [Decoder-only Transformer]
    │                                              .
    + ──────────┐                                   \_ [~598M Parameters]
    |   ┌──────────────────────────┐               .
    │   |        FEEDFORWARD       │                \_ [1024 D_model | 28 Layers | 16 Query Heads | 8 KV Heads — GQA | 128 Head Dim ]
    │   └──────────────────────────┘               .
    |──[RMS-NORM]───┘                               \_ [RoPE, θ = 1,000,000]
    + ──────────┐                                  .
    │   ┌───────────────────────────┐               \_ [2048 Context Length]
    │   |   GROUPED-QUERY ATTENTION |              .
    │   └───────────────────────────┘               \_ [QK-Norm]
    └──[RMS-NORM]───┘                              
    │                                              
 [INPUT]

````

````

 [WAV 24kHz]                                      Architecture: Kokoro [StyleTTS2 + ISTFTNet]
    │                                              .
    + ──────────┐                                   \_ [82M Parameters]
    |   ┌──────────────────────────┐               .
    │   |     ISTFT DECODER [512]  │                \_ [SR 24000 Hz / Hop 5]
    │   └──────────────────────────┘               .
    |──[AdaIN+Snake]──┘                              \_ [Style 128 / Hidden 512]
    + ──────────┐                                  .
    │   ┌──────────────────────────┐                \_ [N_layer 3 / Max Dur 50]
    │   |    PROSODY PREDICTOR     │               .
    │   └──────────────────────────┘                \_ [PLBERT 12×768]
    + ──────────┐                                  .
    │   ┌──────────────────────────┐                \_ [TextEnc K5 / Depth 3]
    │   |      TEXT ENCODER        │               .
    │   └──────────────────────────┘                \_ [178 Vocab Size]
    └──[LAYER-NORM]───┘                            .
    │                                               \_ [RTF 0.045 Eager]
 [INPUT]  (phonemes → 510 chunk)

````
