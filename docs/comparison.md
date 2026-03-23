# Comparison with Original Implementations

Packages with original implementations:

* `transformers` for GigaAM models ([Hugging Face](https://huggingface.co/ai-sage/GigaAM-v3))
* `nemo-toolkit` for NeMo models ([GitHub](https://github.com/nvidia/nemo))
* `openai-whisper` for Whisper models ([GitHub](https://github.com/openai/whisper))
* `sherpa-onnx` for Vosk models ([GitHub](https://github.com/k2-fsa/sherpa-onnx), [Docs](https://k2-fsa.github.io/sherpa/onnx/index.html))
* `T-one` for T-Tech T-one model ([GitHub](https://github.com/voicekit-team/T-one))

## Test Hardware

1. CPU tests were run on AMD Ryzen™ 7 9800X3D processor.
2. GPU tests were run on NVIDIA GeForce RTX 5070 Ti GPU.

## Russian ASR Models

Tests of Russian ASR models were performed on a *test* subset of the [Russian LibriSpeech](https://huggingface.co/datasets/istupakov/russian_librispeech) dataset.

| Model                     | Package / decoding   | CER    | WER    | RTFx (CPU) | RTFx (GPU)   |
|---------------------------|----------------------|--------|--------|------------|--------------|
|       GigaAM v3 CTC       |        default       | 0.98%  | 4.72%  |       39.1 | 163.2        |
|       GigaAM v3 CTC       |       onnx-asr       | 0.98%  | 4.72%  |       58.5 | 859.5        |
|      GigaAM v3 RNN-T      |        default       | 0.93%  | 4.39%  |       30.7 | 68.4         |
|      GigaAM v3 RNN-T      |       onnx-asr       | 0.93%  | 4.39%  |       42.4 | 122.8        |
|     GigaAM v3 E2E CTC     |        default       | 1.50%  | 7.10%  |       38.4 | 449.8        |
|     GigaAM v3 E2E CTC     |       onnx-asr       | 1.56%  | 7.80%  |       58.7 | 853.8        |
|    GigaAM v3 E2E RNN-T    |        default       | 1.61%  | 6.94%  |       30.7 | 80.0         |
|    GigaAM v3 E2E RNN-T    |       onnx-asr       | 1.67%  | 7.60%  |       42.4 | 156.6        |
|  Nemo FastConformer CTC   |        default       | 3.11%  | 13.06% |       92.6 | 329.8        |
|  Nemo FastConformer CTC   |       onnx-asr       | 3.13%  | 13.10% |       96.1 | 1270.0       |
| Nemo FastConformer RNN-T  |        default       | 2.62%  | 11.60% |       66.4 | 181.2        |
| Nemo FastConformer RNN-T  |       onnx-asr       | 2.62%  | 11.57% |       59.0 | 158.8        |
| Nemo Parakeet TDT 0.6B V3 |        default       | 2.34%  | 10.95% |       20.5 | 178.1        |
| Nemo Parakeet TDT 0.6B V3 |       onnx-asr       | 2.38%  | 10.95% |       34.1 | 237.7        |
|     Nemo Canary 1B V2     |        default       | 4.89%  | 20.00% |        4.5 | 32.3         |
|     Nemo Canary 1B V2     |       onnx-asr       | 5.01%  | 19.97% |        7.4 | 27.3         |
|       T-Tech T-one        |        default       | 1.28%  | 6.56%  |       27.6 | N/A          |
|       T-Tech T-one        |       onnx-asr       | 1.28%  | 6.57%  |       25.4 | 77.0         |
|      Vosk 0.52 small      |     greedy_search    | 3.64%  | 14.53% |       94.4 | 135.8        |
|      Vosk 0.52 small      | modified_beam_search | 3.57%  | 14.41% |       83.0 | 81.4         |
|      Vosk 0.52 small      |       onnx-asr       | 3.64%  | 14.53% |       72.6 | 148.0        |
|         Vosk 0.54         |     greedy_search    | 2.21%  | 9.89%  |       66.7 | 121.2        |
|         Vosk 0.54         | modified_beam_search | 2.21%  | 9.84%  |       61.5 | 76.6         |
|         Vosk 0.54         |       onnx-asr       | 2.21%  | 9.89%  |       62.3 | 130.9        |
|       Whisper base        |        default       | 10.12% | 38.34% |       11.6 | 43.3          |
|       Whisper base        |       onnx-asr[^1]   | 10.64% | 38.33% |       23.5 | 63.1          |
|  Whisper large-v3-turbo   |        default       | 2.96%  | 10.27% |        N/A | 37.5          |
|  Whisper large-v3-turbo   |       onnx-asr       | 2.64%  | 10.10% |        N/A | 16.8/26.4[^2] |

[^1]: `whisper-ort` model.
[^2]: With `fp32`/`fp16` precision.

## English ASR Models

Tests of English ASR models were performed on a *test* subset of the [Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset.

| Model                     | Package / decoding   | CER    | WER    | RTFx (CPU) | RTFx (GPU)    |
|---------------------------|----------------------|--------|--------|------------|---------------|
|  Nemo Parakeet CTC 0.6B   |        default       | 4.10%  | 7.21%  | 26.3       | 290.9         |
|  Nemo Parakeet CTC 0.6B   |       onnx-asr       | 4.10%  | 7.22%  | 48.2       | 829.6         |
| Nemo Parakeet RNN-T 0.6B  |        default       | 3.64%  | 6.34%  | 23.1       | 181.6         |
| Nemo Parakeet RNN-T 0.6B  |       onnx-asr       | 3.64%  | 6.34%  | 36.6       | 146.6         |
| Nemo Parakeet TDT 0.6B V2 |        default       | 3.87%  | 6.52%  | 23.0       | 228.2         |
| Nemo Parakeet TDT 0.6B V2 |       onnx-asr       | 3.87%  | 6.51%  | 40.6       | 298.8         |
| Nemo Parakeet TDT 0.6B V3 |        default       | 3.97%  | 6.76%  | 21.4       | 234.3         |
| Nemo Parakeet TDT 0.6B V3 |       onnx-asr       | 3.97%  | 6.75%  | 41.2       | 294.8         |
|     Nemo Canary 1B V2     |        default       | 4.62%  | 7.42%  | 5.2        | 42.6          |
|     Nemo Canary 1B V2     |       onnx-asr       | 4.62%  | 7.42%  | 9.8        | 36.7          |
|       Whisper base        |        default       | 6.88%  | 12.26% | 17.7       | 69.7          |
|       Whisper base        |       onnx-asr[^1]   | 7.52%  | 12.76% | 35.0       | 77.9          |
|  Whisper large-v3-turbo   |        default       | 7.23%  | 11.54% | N/A        | 55.3          |
|  Whisper large-v3-turbo   |       onnx-asr       | 10.49% | 14.81% | N/A        | 51.4/69.2[^2] |
