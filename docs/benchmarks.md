# Benchmarks

## Hardware

1. Arm tests were run on an Orange Pi Zero 3 with a Cortex-A53 processor.
2. x64 tests were run on a laptop with an Intel i7-7700HQ processor.
3. T4 tests were run in Google Colab on Nvidia T4 with CUDA and TensorRT.

> [!NOTE]
> In T4 tests, preprocessors are always run using the TensorRT provider.

## Russian ASR Models

Notebook with benchmark code - [benchmark-ru](https://github.com/istupakov/onnx-asr/blob/main/examples/benchmark-ru.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/istupakov/onnx-asr/blob/main/examples/benchmark-ru.ipynb)

| Model                     | Arm RTFx   | x64 RTFx   | T4 RTFx (CUDA) | T4 RTFx (TensorRT) | T4 RTFx (TensorRT, fp16) |
|---------------------------|------------|------------|----------------|--------------------|--------------------------|
| GigaAM v2 CTC             | 0.8        | 11.6       | 127.6          | 197.0              | 619.8                    |
| GigaAM v2 RNN-T           | 0.8        | 10.7       | 52.6           | 84.1               | 101.6                    |
| GigaAM v3 CTC             | N/A        | 14.5       | 134.8          | 223.1              | 706.3                    |
| GigaAM v3 RNN-T           | N/A        | 13.3       | 52.4           | 92.1               | 99.6                     |
| GigaAM v3 E2E CTC         | N/A        | N/A        | 135.6          | 222.8              | 716.5                    |
| GigaAM v3 E2E RNN-T       | N/A        | N/A        | 63.8           | 98.5               | 119.3                    |
| Nemo FastConformer CTC    | 4.0        | 45.8       | 127.7          | 484.7              | 777.7                    |
| Nemo FastConformer RNN-T  | 3.2        | 27.2       | 57.1           | 119.4              | 124.9                    |
| Nemo Parakeet TDT 0.6B V3 | N/A        | 9.7        | 63.5           | 97.3               | 181.3                    |
| Nemo Canary 1B V2         | N/A        | N/A        | 18.6           | N/A                | N/A                      |
| T-Tech T-one              | N/A        | 11.7       | 15.2           | 40.6               | N/A                      |
| Vosk 0.52 small           | 5.1        | 45.5       | 115.0          | N/A                | N/A                      |
| Vosk 0.54                 | 3.8        | 33.6       | 97.6           | N/A                | N/A                      |
| Whisper base              | 0.8        | 6.6        | 58.0           | N/A                | N/A                      |
| Whisper large-v3-turbo    | N/A        | N/A        | 19.5           | N/A                | N/A                      |

## English ASR Models

Notebook with benchmark code - [benchmark-en](https://github.com/istupakov/onnx-asr/blob/main/examples/benchmark-en.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/istupakov/onnx-asr/blob/main/examples/benchmark-en.ipynb)

| Model                     | Arm RTFx   | x64 RTFx   | T4 RTFx (CUDA)  | T4 RTFx (TensorRT) | T4 RTFx (TensorRT, fp16) |
|---------------------------|------------|------------|-----------------|--------------------|--------------------------|
| Nemo Parakeet CTC 0.6B    | 1.1        | 11.5       | 106.1           | 154.7              | N/A                      |
| Nemo Parakeet RNN-T 0.6B  | 1.0        | 8.7        | 49.7            | 69.7               | N/A                      |
| Nemo Parakeet TDT 0.6B V2 | 1.1        | 10.5       | 77.9            | 116.7              | 233.8                    |
| Nemo Parakeet TDT 0.6B V3 | N/A        | 9.5        | 77.4            | 106.2              | 227.4                    |
| Nemo Canary 1B V2         | N/A        | N/A        | 22.1            | N/A                | N/A                      |
| Whisper base              | 1.2        | 9.2        | 92.2            | N/A                | N/A                      |
| Whisper large-v3-turbo    | N/A        | N/A        | 29.2            | N/A                | N/A                      |