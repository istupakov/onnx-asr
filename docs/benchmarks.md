# Benchmarks

## Hardware

1. x64 tests were run on a PC with an AMD Ryzen™ 7 9800X3D processor.
2. Arm tests were run on an Orange Pi Zero 3 with a Cortex-A53 processor.
3. RTX tests were run on a PC with NVIDIA GeForce RTX 5070 Ti GPU.
4. T4 tests were run in Google Colab on NVIDIA T4 GPU.

## Metrics

**Inverse Real-Time Factor (RTFx)** measures how fast an ASR system processes audio relative to its duration.

$$\text{RTFx} = \frac{\text{Audio Duration}}{\text{Processing Time}}$$

- **RTFx > 1**: Processing faster than real-time (audio duration is longer than processing time)
- **RTFx = 1**: Real-time processing (audio duration equals processing time)
- **RTFx < 1**: Processing slower than real-time

For example, an RTFx of 10 means the model can process 10 seconds of audio in 1 second.

## Russian ASR models
Notebook with benchmark code - [benchmark-ru](https://github.com/istupakov/onnx-asr/blob/main/examples/benchmark-ru.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/istupakov/onnx-asr/blob/main/examples/benchmark-ru.ipynb)

### CPU Benchmarks (`CPUExecutionProvider`)

| Model                     | x64 RTFx | x64 RTFx (int8) | Arm RTFx (int8) |
|---------------------------|----------|-----------------|-----------------|
| GigaAM v2 CTC             | 53.0     | 16.7            | 0.8             |
| GigaAM v2 RNN-T           | 39.7     | 15.9            | 0.8             |
| GigaAM v3 CTC             | 58.8     | 52.2            | 1.6             |
| GigaAM v3 RNN-T           | 42.6     | 42.8            | 1.5             |
| GigaAM v3 E2E CTC         | 59.0     | 52.2            | 1.6             |
| GigaAM v3 E2E RNN-T       | 41.9     | 42.5            | 1.5             |
| Nemo FastConformer CTC    | 90.8     | 70.6            | 4.1             |
| Nemo FastConformer RNN-T  | 57.2     | 54.2            | 3.3             |
| Nemo Parakeet TDT 0.6B V3 | 33.4     | 30.5            | 1.1             |
| Nemo Canary 1B V2         | 7.7      | 14.4            | N/A             |
| T-Tech T-one              | 26.3     | N/A             | 1.3[^1]         |
| Vosk 0.52 small           | 72.6     | 83.5            | 6.2             |
| Vosk 0.54                 | 62.1     | 70.5            | 4.5             |
| Whisper base              | 23.7     | 51.6            | 0.9             |
| Whisper large-v3-turbo    | 2.9      | 3.9             | N/A             |

[^1]: With default precision.

### CUDA Benchmarks (`CUDAExecutionProvider`)

| Model                      | RTX RTFx  | T4 RTFx  |
|----------------------------|-----------|----------|
| GigaAM v2 CTC              | 245.2     | 82.5     |
| GigaAM v2 RNN-T            | 51.4      | 38.7     |
| GigaAM v3 CTC              | 240.9     | 83.7     |
| GigaAM v3 RNN-T            | 50.6      | 39.8     |
| GigaAM v3 E2E CTC          | 239.6     | 83.8     |
| GigaAM v3 E2E RNN-T        | 62.5      | 45.9     |
| Nemo FastConformer CTC     | 77.8      | 77.8     |
| Nemo FastConformer RNN-T   | 41.8      | 42.3     |
| Nemo Parakeet TDT 0.6B V3  | 74.8      | 44.0     |
| Nemo Canary 1B V2          | 28.7      | 18.7     |
| T-Tech T-one               | 13.3      | 14.6     |
| Vosk 0.52 small            | 149.9     | 119.4    |
| Vosk 0.54                  | 134.3     | 100.3    |
| Whisper base               | 66.4      | 57.9     |
| Whisper large-v3-turbo[^2] | 34.9/49.8 | 7.5/19.8 |

[^2]: With `fp32`/`fp16` precision.

### TensorRT Benchmarks (`TensorrtExecutionProvider`)

| Model                     | RTX RTFx | RTX RTFx (fp16) | T4 RTFx | T4 RTFx (fp16) |
|---------------------------|----------|-----------------|---------|----------------|
| GigaAM v2 CTC             | 851.0    | 1373.9          | 182.0   | 646.8          |
| GigaAM v2 RNN-T           | 124.2    | 129.4           | 82.6    | 93.8           |
| GigaAM v3 CTC             | 934.3    | 1372.5          | 216.8   | 717.2          |
| GigaAM v3 RNN-T           | 124.1    | 130.3           | 90.2    | 90.2           |
| GigaAM v3 E2E CTC         | 931.3    | 1392.8          | 216.9   | 702.6          |
| GigaAM v3 E2E RNN-T       | 164.7    | 170.1           | 103.3   | 112.4          |
| Nemo FastConformer CTC    | 1395.1   | 1510.4          | 501.6   | 817.9          |
| Nemo FastConformer RNN-T  | 162.1    | 160.8           | 119.3   | 119.6          |
| Nemo Parakeet TDT 0.6B V3 | 252.2    | 278.6           | 82.7    | 189.5          |
| T-Tech T-one              | 82.3     | N/A             | 42.0    | N/A            |

## English ASR models

Notebook with benchmark code - [benchmark-en](https://github.com/istupakov/onnx-asr/blob/main/examples/benchmark-en.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/istupakov/onnx-asr/blob/main/examples/benchmark-en.ipynb)


### CPU Benchmarks (`CPUExecutionProvider`)

| Model                     | x64 RTFx | x64 RTFx (int8) | Arm RTFx (int8) |
|---------------------------|----------|-----------------|-----------------|
| Nemo Parakeet CTC 0.6B    | 43.2     | 34.8            | 1.1             |
| Nemo Parakeet RNN-T 0.6B  | 33.5     | 30.2            | 1.0             |
| Nemo Parakeet TDT 0.6B V2 | 36.8     | 30.5            | 1.1             |
| Nemo Parakeet TDT 0.6B V3 | 35.4     | 31.4            | 1.0             |
| Nemo Canary 1B V2         | 8.3      | 14.9            | N/A             |
| Whisper base              | 31.4     | 61.8            | 1.4             |
| Whisper large-v3-turbo    | 3.9      | 5.4             | N/A             |

### CUDA Benchmarks (`CUDAExecutionProvider`)

| Model                      | RTX RTFx  | T4 RTFx  |
|----------------------------|-----------|----------|
| Nemo Parakeet CTC 0.6B     | 100.5     | 71.7     |
| Nemo Parakeet RNN-T 0.6B   | 43.1      | 40.2     |
| Nemo Parakeet TDT 0.6B V2  | 88.7      | 57.6     |
| Nemo Parakeet TDT 0.6B V3  | 91.4      | 57.5     |
| Nemo Canary 1B V2          | 35.7      | 21.4     |
| Whisper base               | 108.2     | 88.9     |
| Whisper large-v3-turbo[^2] | 54.8/76.0 | 9.2/25.9 |

### TensorRT Benchmarks (`TensorrtExecutionProvider`)

| Model                     | RTX RTFx | RTX RTFx (fp16) | T4 RTFx | T4 RTFx (fp16) |
|---------------------------|----------|-----------------|---------|----------------|
| Nemo Parakeet CTC 0.6B    | 794.2    | N/A             | 146.5   | N/A            |
| Nemo Parakeet RNN-T 0.6B  | 133.0    | N/A             | 70.4    | N/A            |
| Nemo Parakeet TDT 0.6B V2 | 298.3    | 329.2           | 96.0    | 237.2          |
| Nemo Parakeet TDT 0.6B V3 | 290.8    | 318.5           | 99.1    | 232.3          |
