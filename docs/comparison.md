# Comparison with Original Implementations

Packages with original implementations:

* `gigaam` for GigaAM models ([github](https://github.com/salute-developers/GigaAM))
* `nemo-toolkit` for NeMo models ([github](https://github.com/nvidia/nemo))
* `openai-whisper` for Whisper models ([github](https://github.com/openai/whisper))
* `sherpa-onnx` for Vosk models ([github](https://github.com/k2-fsa/sherpa-onnx), [docs](https://k2-fsa.github.io/sherpa/onnx/index.html))
* `T-one` for T-Tech T-one model ([github](https://github.com/voicekit-team/T-one))

## Test Hardware

1. CPU tests were run on a laptop with an Intel i7-7700HQ processor.
2. GPU tests were run in Google Colab on Nvidia T4.

## Russian ASR Models

Tests of Russian ASR models were performed on a *test* subset of the [Russian LibriSpeech](https://huggingface.co/datasets/istupakov/russian_librispeech) dataset.

| Model                     | Package / decoding   | CER    | WER    | RTFx (CPU) | RTFx (GPU)   |
|---------------------------|----------------------|--------|--------|------------|--------------|
|       GigaAM v2 CTC       |        default       | 1.06%  | 5.23%  |        7.2 | 44.2         |
|       GigaAM v2 CTC       |       onnx-asr       | 1.06%  | 5.23%  |       11.6 | 197.0        |
|      GigaAM v2 RNN-T      |        default       | 1.10%  | 5.22%  |        5.5 | 23.3         |
|      GigaAM v2 RNN-T      |       onnx-asr       | 1.10%  | 5.22%  |       10.7 | 84.1         |
|       GigaAM v3 CTC       |        default       | 0.98%  | 4.72%  |       12.2 | 73.3         |
|       GigaAM v3 CTC       |       onnx-asr       | 0.98%  | 4.72%  |       14.5 | 223.1        |
|      GigaAM v3 RNN-T      |        default       | 0.93%  | 4.39%  |        8.2 | 41.6         |
|      GigaAM v3 RNN-T      |       onnx-asr       | 0.93%  | 4.39%  |       13.3 | 92.1         |
|     GigaAM v3 E2E CTC     |        default       | 1.50%  | 7.10%  |        N/A | 178.0        |
|     GigaAM v3 E2E CTC     |       onnx-asr       | 1.56%  | 7.80%  |        N/A | 222.8        |
|    GigaAM v3 E2E RNN-T    |        default       | 1.61%  | 6.94%  |        N/A | 47.6         |
|    GigaAM v3 E2E RNN-T    |       onnx-asr       | 1.67%  | 7.60%  |        N/A | 98.5         |
|  Nemo FastConformer CTC   |        default       | 3.11%  | 13.12% |       29.1 | 143.0        |
|  Nemo FastConformer CTC   |       onnx-asr       | 3.13%  | 13.10% |       45.8 | 484.7        |
| Nemo FastConformer RNN-T  |        default       | 2.63%  | 11.62% |       17.4 | 111.6        |
| Nemo FastConformer RNN-T  |       onnx-asr       | 2.62%  | 11.57% |       27.2 | 119.4        |
| Nemo Parakeet TDT 0.6B V3 |        default       | 2.34%  | 10.95% |        5.6 | 75.4         |
| Nemo Parakeet TDT 0.6B V3 |       onnx-asr       | 2.38%  | 10.95% |        9.7 | 97.3         |
|     Nemo Canary 1B V2     |        default       | 4.89%  | 20.00% |        N/A | 14.0         |
|     Nemo Canary 1B V2     |       onnx-asr       | 5.00%  | 20.03% |        N/A | 18.6         |
|       T-Tech T-one        |        default       | 1.28%  | 6.56%  |       11.9 | N/A          |
|       T-Tech T-one        |       onnx-asr       | 1.28%  | 6.57%  |       11.7 | 40.6         |
|      Vosk 0.52 small      |     greedy_search    | 3.64%  | 14.53% |       48.2 | 71.4         |
|      Vosk 0.52 small      | modified_beam_search | 3.50%  | 14.25% |       29.0 | 24.7         |
|      Vosk 0.52 small      |       onnx-asr       | 3.64%  | 14.53% |       45.5 | 115.0        |
|         Vosk 0.54         |     greedy_search    | 2.21%  | 9.89%  |       34.8 | 64.2         |
|         Vosk 0.54         | modified_beam_search | 2.21%  | 9.85%  |       23.9 | 24           |
|         Vosk 0.54         |       onnx-asr       | 2.21%  | 9.89%  |       33.6 | 97.6         |
|       Whisper base        |        default       | 10.61% | 38.89% |        5.4 | 17.3         |
|       Whisper base        |       onnx-asr*      | 10.64% | 38.33% |        6.6 | 58.0         |
|  Whisper large-v3-turbo   |        default       | 2.96%  | 10.27% |        N/A | 13.6         |
|  Whisper large-v3-turbo   |       onnx-asr**     | 2.63%  | 10.13% |        N/A | 19.5         |

## English ASR Models

Tests of English ASR models were performed on a *test* subset of the [Voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) dataset.

| Model                     | Package / decoding   | CER    | WER    | RTFx (CPU) | RTFx (GPU)   |
|---------------------------|----------------------|--------|--------|------------|--------------|
|  Nemo Parakeet CTC 0.6B   |        default       | 4.09%  | 7.20%  | 8.3        | 107.7        |
|  Nemo Parakeet CTC 0.6B   |       onnx-asr       | 4.10%  | 7.22%  | 11.5       | 154.7        |
| Nemo Parakeet RNN-T 0.6B  |        default       | 3.64%  | 6.32%  | 6.7        | 85.0         |
| Nemo Parakeet RNN-T 0.6B  |       onnx-asr       | 3.64%  | 6.33%  | 8.7        | 69.7         |
| Nemo Parakeet TDT 0.6B V2 |        default       | 3.88%  | 6.52%  | 6.5        | 87.6         |
| Nemo Parakeet TDT 0.6B V2 |       onnx-asr       | 3.87%  | 6.52%  | 10.5       | 116.7        |
| Nemo Parakeet TDT 0.6B V3 |        default       | 3.97%  | 6.76%  | 6.1        | 90.0         |
| Nemo Parakeet TDT 0.6B V3 |       onnx-asr       | 3.97%  | 6.75%  | 9.5        | 106.2        |
|     Nemo Canary 1B V2     |        default       | 4.62%  | 7.42%  | N/A        | 17.5         |
|     Nemo Canary 1B V2     |       onnx-asr       | 4.67%  | 7.47%  | N/A        | 22.1         |
|       Whisper base        |        default       | 7.81%  | 13.24% | 8.4        | 27.7         |
|       Whisper base        |       onnx-asr*      | 7.52%  | 12.76% | 9.2        | 92.2         |
|  Whisper large-v3-turbo   |        default       | 6.85%  | 11.16% | N/A        | 20.4         |
|  Whisper large-v3-turbo   |       onnx-asr**     | 10.31% | 14.65% | N/A        | 29.2         |

## Notes

> [!NOTE]
> 1. \* `whisper-ort` model.
> 2. ** `whisper` model with `fp16` precision.
> 3. All other models were run with the default precision - `fp32` on CPU and `fp32` or `fp16` (some of the original models) on GPU.