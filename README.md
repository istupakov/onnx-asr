# ONNX ASR

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Types](https://img.shields.io/pypi/types/onnx-asr)](https://pypi.org/project/onnx-asr)
[![GitHub - License](https://img.shields.io/github/license/istupakov/onnx-asr)](https://github.com/istupakov/onnx-asr/blob/main/LICENSE)
[![GitHub - CI](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml/badge.svg)](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml)
[![GitHub - Release Date](https://img.shields.io/github/release-date/istupakov/onnx-asr)](https://github.com/istupakov/onnx-asr/releases/latest)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg)](https://istupakov-onnx-asr.hf.space/)

**onnx-asr** is a Python package for Automatic Speech Recognition using ONNX models. It's a simple and fast pure Python package with minimal dependencies (PyTorch, Transformers, and FFmpeg are not required):

[![numpy](https://img.shields.io/badge/numpy-required-blue?logo=numpy)](https://pypi.org/project/numpy/)
[![onnxruntime](https://img.shields.io/badge/onnxruntime-required-blue?logo=onnx)](https://pypi.org/project/onnxruntime/)
[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-optional-blue?logo=huggingface)](https://pypi.org/project/huggingface-hub/)

> [!TIP]
> Supports **Parakeet v2 (En) / v3 (Multilingual)**, **Canary v2 (Multilingual)** and **GigaAM v2/v3 (Ru)** models!

The **onnx-asr** package supports many modern ASR [models](#supported-models-architectures) and the following features:
 * Works on a wide range of devices, from small IoT devices to servers with powerful GPUs ([benchmarks](#benchmarks))
 * Runs on Windows, Linux, and macOS on x86 or Arm CPUs and can use CUDA, TensorRT, CoreML, ROCm, and DirectML
 * NumPy versions from 1.21.6 to 2.4+ and Python versions from 3.10 to 3.14 are supported
 * Loading models from Hugging Face or local folders (including quantized versions)
 * Accepts WAV files or NumPy arrays (built-in support for reading and resampling files)
 * Custom models (if their architecture is supported)
 * Batch processing
 * (experimental) Longform recognition using VAD (Voice Activity Detection)
 * (experimental) Returns token timestamps and logprobs
 * Simple CLI
 * Online demo in [HF Spaces](https://istupakov-onnx-asr.hf.space/)

## Supported models architectures

The package supports the following modern ASR model architectures ([comparison](#comparison-with-original-implementations) with original implementations):
* Nvidia NeMo Conformer/FastConformer/Parakeet/Canary (with CTC, RNN-T, TDT and Transformer decoders)
* Kaldi Icefall Zipformer (with stateless RNN-T decoder) including Alpha Cephei Vosk 0.52+
* Sber GigaAM v2/v3 (with CTC and RNN-T decoders, including E2E versions)
* T-Tech T-one (with CTC decoder, no streaming support yet)
* OpenAI Whisper

When saving these models in ONNX format, usually only the encoder and decoder are saved. To run them, the corresponding preprocessor and decoding must be implemented. Therefore, the package contains these implementations for all supported models:
* Log-mel spectrogram preprocessors
* Greedy search decoding

## Installation

The package can be installed from [PyPI](https://pypi.org/project/onnx-asr/):

1. With CPU `onnxruntime` and `huggingface-hub`:
```shell
pip install onnx-asr[cpu,hub]
```

2. With `onnxruntime` for NVIDIA GPUs and `huggingface-hub`:
```shell
pip install onnx-asr[gpu,hub]
```

> [!IMPORTANT]
> First, you need to install the [required](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) version of CUDA / TensorRT.

You can alse install `onnxruntime` dependencies and TensorRT via Pip:
```shell
pip install onnxruntime-gpu[cuda,cudnn] tensorrt-cu12-libs
```

3. Without `onnxruntime` and `huggingface-hub` (if you already have some version of `onnxruntime` installed and prefer to download the models yourself):
```shell
pip install onnx-asr
```

To install latest version `onnx-asr` from **GitHub**, use `pip` (or `uv pip`):
```shell
pip install git+https://github.com/istupakov/onnx-asr
```

## Usage examples

### Load ONNX model from Hugging Face

Load ONNX model from Hugging Face and recognize wav file:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize("test.wav"))
```

> [!IMPORTANT]
> Supported wav file formats: PCM_U8, PCM_16, PCM_24 and PCM_32 formats. For other formats, you either need to convert them first, or use a library that can read them into a numpy array.

#### Supported model names:
* `gigaam-v2-ctc` for Sber GigaAM v2 CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `gigaam-v2-rnnt` for Sber GigaAM v2 RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `gigaam-v3-ctc` for Sber GigaAM v3 CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-rnnt` for Sber GigaAM v3 RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-e2e-ctc` for Sber GigaAM v3 E2E CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-e2e-rnnt` for Sber GigaAM v3 E2E RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `nemo-fastconformer-ru-ctc` for Nvidia FastConformer-Hybrid Large (ru) with CTC decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-fastconformer-ru-rnnt` for Nvidia FastConformer-Hybrid Large (ru) with RNN-T decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-parakeet-ctc-0.6b` for Nvidia Parakeet CTC 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-ctc-0.6b-onnx))
* `nemo-parakeet-rnnt-0.6b` for Nvidia Parakeet RNNT 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-rnnt-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-rnnt-0.6b-onnx))
* `nemo-parakeet-tdt-0.6b-v2` for Nvidia Parakeet TDT 0.6B V2 (en) ([origin](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx))
* `nemo-parakeet-tdt-0.6b-v3` for Nvidia Parakeet TDT 0.6B V3 (multilingual) ([origin](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx))
* `nemo-canary-1b-v2` for Nvidia Canary 1B V2 (multilingual) ([origin](https://huggingface.co/nvidia/canary-1b-v2), [onnx](https://huggingface.co/istupakov/canary-1b-v2-onnx))
* `whisper-base` for OpenAI Whisper Base exported with onnxruntime ([origin](https://huggingface.co/openai/whisper-base), [onnx](https://huggingface.co/istupakov/whisper-base-onnx))
* `alphacep/vosk-model-ru` for Alpha Cephei Vosk 0.54-ru ([origin](https://huggingface.co/alphacep/vosk-model-ru))
* `alphacep/vosk-model-small-ru` for Alpha Cephei Vosk 0.52-small-ru ([origin](https://huggingface.co/alphacep/vosk-model-small-ru))
* `t-tech/t-one` for T-Tech T-one ([origin](https://huggingface.co/t-tech/T-one))
* `onnx-community/whisper-tiny`, `onnx-community/whisper-base`, `onnx-community/whisper-small`, `onnx-community/whisper-large-v3-turbo`, etc. for OpenAI Whisper exported with Hugging Face optimum ([onnx-community](https://huggingface.co/onnx-community?search_models=whisper))

> [!IMPORTANT]
> Some long-ago converted `onnx-community` models have a broken `fp16` precision version.

Example with `soundfile`:
```py
import onnx_asr
import soundfile as sf

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

waveform, sample_rate = sf.read("test.wav", dtype="float32")
model.recognize(waveform, sample_rate=sample_rate)
```

Batch processing is also supported:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize(["test1.wav", "test2.wav", "test3.wav", "test4.wav"]))
```

Most models have a quantized versions:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
print(model.recognize("test.wav"))
```

Return tokens, timestamps and logprobs:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_timestamps()
print(model.recognize("test1.wav"))
```

### TensorRT

Running ONNX model on the TensorRT provider and fp16 precision:
```py
import onnx_asr
import tensorrt_libs

providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_max_workspace_size": 6 * 1024**3,
            "trt_fp16_enable": True,
        },
    )
]
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=providers)
print(model.recognize("test.wav"))
```

### VAD

Load VAD ONNX model from Hugging Face and recognize wav file:
```py
import onnx_asr
vad = onnx_asr.load_vad("silero")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_vad(vad)
for res in model.recognize("test.wav"):
    print(res)
```

> [!NOTE]
> You will most likely need to adjust VAD parameters to get the correct results.

#### Supported VAD names:
* `silero` for Silero VAD ([origin](https://github.com/snakers4/silero-vad), [onnx](https://huggingface.co/onnx-community/silero-vad))

### CLI

Package has simple CLI interface
```shell
onnx-asr nemo-parakeet-tdt-0.6b-v3 test.wav
```

For full usage parameters, see help:
```shell
onnx-asr -h
```

### Gradio

Create simple web interface with Gradio:
```py
import onnx_asr
import gradio as gr

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

def recognize(audio):
    if not audio:
        return None

    sample_rate, waveform = audio
    waveform = waveform / 2**15
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    return model.recognize(waveform, sample_rate=sample_rate)

demo = gr.Interface(fn=recognize, inputs="audio", outputs="text")
demo.launch()
```

### Load ONNX model from local directory

Load ONNX model from local directory and recognize wav file:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "models/parakeet-v3")
print(model.recognize("test.wav"))
```

> [!NOTE]
> If the directory does not exist, it will be created and the model will be loaded into it.

### Load a custom ONNX model from Hugging Face

Load the Canary 180M Flash model from Hugging Face [repo](https://huggingface.co/istupakov/canary-180m-flash-onnx) and recognize the wav file:
```py
import onnx_asr
model = onnx_asr.load_model("istupakov/canary-180m-flash-onnx")
print(model.recognize("test.wav"))
```

#### Supported model types:
* All models from [supported model names](#supported-model-names)
* `nemo-conformer-ctc` for NeMo Conformer/FastConformer/Parakeet with CTC decoder
* `nemo-conformer-rnnt` for NeMo Conformer/FastConformer/Parakeet with RNN-T decoder
* `nemo-conformer-tdt` for NeMo Conformer/FastConformer/Parakeet with TDT decoder
* `nemo-conformer-aed` for NeMo Canary with Transformer decoder
* `kaldi-rnnt` or `vosk` for Kaldi Icefall Zipformer with stateless RNN-T decoder
* `whisper-ort` for Whisper (exported with [onnxruntime](#openai-whisper-with-onnxruntime-export))
* `whisper` for Whisper (exported with [optimum](#openai-whisper-with-optimum-export))

## Comparison with original implementations

Packages with original implementations:
* `gigaam` for GigaAM models ([github](https://github.com/salute-developers/GigaAM))
* `nemo-toolkit` for NeMo models ([github](https://github.com/nvidia/nemo))
* `openai-whisper` for Whisper models ([github](https://github.com/openai/whisper))
* `sherpa-onnx` for Vosk models ([github](https://github.com/k2-fsa/sherpa-onnx), [docs](https://k2-fsa.github.io/sherpa/onnx/index.html))
* `T-one` for T-Tech T-one model ([github](https://github.com/voicekit-team/T-one))

Hardware:
1. CPU tests were run on a laptop with an Intel i7-7700HQ processor.
2. GPU tests were run in Google Colab on Nvidia T4.

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

> [!NOTE]
> 1. \* `whisper-ort` model ([model types](#supported-model-types)).
> 2. ** `whisper` model ([model types](#supported-model-types)) with `fp16` precision.
> 3. All other models were run with the default precision - `fp32` on CPU and `fp32` or `fp16` (some of the original models) on GPU.

## Benchmarks

Hardware:
1. Arm tests were run on an Orange Pi Zero 3 with a Cortex-A53 processor.
2. x64 tests were run on a laptop with an Intel i7-7700HQ processor.
3. T4 tests were run in Google Colab on Nvidia T4 with CUDA and TensorRT.

> [!NOTE]
> In T4 tests, preprocessors are always run using the TensorRT provider.

### Russian ASR models
Notebook with benchmark code - [benchmark-ru](examples/benchmark-ru.ipynb)

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

### English ASR models

Notebook with benchmark code - [benchmark-en](examples/benchmark-en.ipynb)

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

## Convert model to ONNX

Save the model according to the instructions below and add config.json:

```json
{
    "model_type": "nemo-conformer-rnnt", // See "Supported model types"
    "features_size": 80, // Size of preprocessor features for Whisper or Nemo models, supported 80 and 128
    "subsampling_factor": 8, // Subsampling factor - 4 for conformer models and 8 for fastconformer and parakeet models
    "max_tokens_per_step": 10 // Max tokens per step for RNN-T decoder
}
```
Then you can upload the model into Hugging Face and use `load_model` to download it.

### Nvidia NeMo Conformer/FastConformer/Parakeet
Install **NeMo Toolkit**
```shell
pip install nemo_toolkit['asr']
```

Download model and export to ONNX format
```py
import nemo.collections.asr as nemo_asr
from pathlib import Path

model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ru_fastconformer_hybrid_large_pc")

# For export Hybrid models with CTC decoder
# model.set_export_config({"decoder_type": "ctc"})

onnx_dir = Path("nemo-onnx")
onnx_dir.mkdir(exist_ok=True)
model.export(str(Path(onnx_dir, "model.onnx")))

with Path(onnx_dir, "vocab.txt").open("wt") as f:
    for i, token in enumerate([*model.tokenizer.vocab, "<blk>"]):
        f.write(f"{token} {i}\n")
```

### Sber GigaAM v2/v3
Install **GigaAM**
```shell
git clone https://github.com/salute-developers/GigaAM.git
pip install ./GigaAM --extra-index-url https://download.pytorch.org/whl/cpu
```

Download model and export to ONNX format
```py
import gigaam
from pathlib import Path

onnx_dir = "gigaam-onnx"
model_type = "rnnt"  # or "ctc"

model = gigaam.load_model(
    model_type,
    fp16_encoder=False,  # only fp32 tensors
    use_flash=False,  # disable flash attention
)
model.to_onnx(dir_path=onnx_dir)

with Path(onnx_dir, "v2_vocab.txt").open("wt") as f:
    for i, token in enumerate(["\u2581", *(chr(ord("а") + i) for i in range(32)), "<blk>"]):
        f.write(f"{token} {i}\n")
```

### OpenAI Whisper (with `onnxruntime` export)

Read onnxruntime [instruction](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md) for convert Whisper to ONNX.

Download model and export with *Beam Search* and *Forced Decoder Input Ids*:
```shell
python3 -m onnxruntime.transformers.models.whisper.convert_to_onnx -m openai/whisper-base --output ./whisper-onnx --use_forced_decoder_ids --optimize_onnx --precision fp32
```

Save tokenizer config
```py
from transformers import WhisperTokenizer

processor = WhisperTokenizer.from_pretrained("openai/whisper-base")
processor.save_pretrained("whisper-onnx")
```

### OpenAI Whisper (with `optimum` export)

Export model to ONNX with Hugging Face `optimum-cli`
```shell
optimum-cli export onnx --model openai/whisper-base ./whisper-onnx
```
