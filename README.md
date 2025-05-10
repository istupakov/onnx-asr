# ONNX ASR

[![PyPI - Version](https://img.shields.io/pypi/v/onnx-asr.svg)](https://pypi.org/project/onnx-asr)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/onnx-asr)](https://pypi.org/project/onnx-asr)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/onnx-asr.svg)](https://pypi.org/project/onnx-asr)
[![PyPI - Types](https://img.shields.io/pypi/types/onnx-asr)](https://pypi.org/project/onnx-asr)
[![GitHub License](https://img.shields.io/github/license/istupakov/onnx-asr)](https://github.com/istupakov/onnx-asr/blob/main/LICENSE)
[![CI](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml/badge.svg)](https://github.com/istupakov/onnx-asr/actions/workflows/python-package.yml)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-xl-dark.svg)](https://istupakov-onnx-asr.hf.space/)

**onnx-asr** is a Python package for Automatic Speech Recognition using ONNX models. The package is written in pure Python with minimal dependencies (no `pytorch` or `transformers`):

[![numpy](https://img.shields.io/badge/numpy-required-blue?logo=numpy)](https://pypi.org/project/numpy/)
[![onnxruntime](https://img.shields.io/badge/onnxruntime-required-blue?logo=onnx)](https://pypi.org/project/onnxruntime/)
[![huggingface-hub](https://img.shields.io/badge/huggingface--hub-optional-blue?logo=huggingface)](https://pypi.org/project/huggingface-hub/)

The **onnx-asr** package supports many modern ASR [models](#supported-models-architectures) and the following features:
 * Loading models from hugging face or local folders (including quantized versions)
 * Accepts wav files or NumPy arrays (built-in support for file reading and resampling)
 * Batch processing
 * (experimental) Longform recognition with VAD (Voice Activity Detection)
 * (experimental) Returns token timestamps
 * Simple CLI
 * Online demo in [HF Spaces](https://istupakov-onnx-asr.hf.space/)

## Supported models architectures

The package supports the following modern ASR model architectures ([comparison](#comparison-with-original-implementations) with original implementations):
* Nvidia NeMo Conformer/FastConformer/Parakeet (with CTC, RNN-T and TDT decoders)
* Kaldi Icefall Zipformer (with stateless RNN-T decoder) including Alpha Cephei Vosk 0.52+
* Sber GigaAM v2 (with CTC and RNN-T decoders)
* OpenAI Whisper

When saving these models in onnx format, usually only the encoder and decoder are saved. To run them, the corresponding preprocessor and decoding must be implemented. Therefore, the package contains these implementations for all supported models:
* Log-mel spectrogram preprocessors
* Greedy search decoding

## Installation

The package can be installed from [PyPI](https://pypi.org/project/onnx-asr/):

1. With CPU `onnxruntime` and `huggingface-hub`
```shell
pip install onnx-asr[cpu,hub]
```
2. With GPU `onnxruntime` and `huggingface-hub`
```shell
pip install onnx-asr[gpu,hub]
```
3. Without `onnxruntime` and `huggingface-hub` (if you already have some version of `onnxruntime` installed and prefer to download the models yourself)
```shell
pip install onnx-asr
```

## Usage examples

### Load ONNX model from Hugging Face

Load ONNX model from Hugging Face and recognize wav file:
```py
import onnx_asr
model = onnx_asr.load_model("gigaam-v2-rnnt")
print(model.recognize("test.wav"))
```

#### Supported model names:
* `gigaam-v2-ctc` for Sber GigaAM v2 CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `gigaam-v2-rnnt` for Sber GigaAM v2 RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `nemo-fastconformer-ru-ctc` for Nvidia FastConformer-Hybrid Large (ru) with CTC decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-fastconformer-ru-rnnt` for Nvidia FastConformer-Hybrid Large (ru) with RNN-T decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-parakeet-ctc-0.6b` for Nvidia Parakeet CTC 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-ctc-0.6b-onnx))
* `nemo-parakeet-rnnt-0.6b` for Nvidia Parakeet RNNT 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-rnnt-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-rnnt-0.6b-onnx))
* `nemo-parakeet-tdt-0.6b-v2` for Nvidia Parakeet TDT 0.6B V2 (en) ([origin](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx))
* `whisper-base` for OpenAI Whisper Base exported with onnxruntime ([origin](https://huggingface.co/openai/whisper-base), [onnx](https://huggingface.co/istupakov/whisper-base-onnx))
* `alphacep/vosk-model-ru` for Alpha Cephei Vosk 0.54-ru ([origin](https://huggingface.co/alphacep/vosk-model-ru))
* `alphacep/vosk-model-small-ru` for Alpha Cephei Vosk 0.52-small-ru ([origin](https://huggingface.co/alphacep/vosk-model-small-ru))
* `onnx-community/whisper-tiny`, `onnx-community/whisper-base`, `onnx-community/whisper-small`, `onnx-community/whisper-large-v3-turbo`, etc. for OpenAI Whisper exported with Hugging Face optimum ([onnx-community](https://huggingface.co/onnx-community?search_models=whisper))

Supported wav file formats: PCM_U8, PCM_16, PCM_24 and PCM_32 formats. For other formats, you either need to convert them first, or use a library that can read them into a numpy array.

Example with `soundfile`:
```py
import onnx_asr
import soundfile as sf

model = onnx_asr.load_model("whisper-base")

waveform, sample_rate = sf.read("test.wav", dtype="float32")
model.recognize(waveform)
```

Batch processing is also supported:
```py
import onnx_asr
model = onnx_asr.load_model("nemo-fastconformer-ru-ctc")
print(model.recognize(["test1.wav", "test2.wav", "test3.wav", "test4.wav"]))
```

Some models have a quantized versions:
```py
import onnx_asr
model = onnx_asr.load_model("alphacep/vosk-model-ru", quantization="int8")
print(model.recognize("test.wav"))
```

Return tokens and timestamps:
```py
import onnx_asr
model = onnx_asr.load_model("alphacep/vosk-model-ru").with_timestamps()
print(model.recognize("test1.wav"))
```

### VAD

Load VAD ONNX model from Hugging Face and recognize wav file:
```py
import onnx_asr
vad = onnx_asr.load_vad("silero")
model = onnx_asr.load_model("gigaam-v2-rnnt").with_vad(vad)
for res in model.recognize("test.wav"):
    print(res)
```

#### Supported VAD names:
* `silero` for Silero VAD ([origin](https://github.com/snakers4/silero-vad), [onnx](https://huggingface.co/onnx-community/silero-vad))

### CLI

Package has simple CLI interface
```shell
onnx-asr nemo-fastconformer-ru-ctc test.wav
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

model = onnx_asr.load_model("gigaam-v2-rnnt")

def recognize(audio):
    if audio:
        sample_rate, waveform = audio
        waveform = waveform / 2**15
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        return model.recognize(waveform, sample_rate=sample_rate)

demo = gr.Interface(fn=recognize, inputs=gr.Audio(min_length=1, max_length=30), outputs="text")
demo.launch()
```

### Load ONNX model from local directory

Load ONNX model from local directory and recognize wav file:
```py
import onnx_asr
model = onnx_asr.load_model("gigaam-v2-ctc", "models/gigaam-onnx")
print(model.recognize("test.wav"))
```
#### Supported model types:
* All models from [supported model names](#supported-model-names)
* `nemo-conformer-ctc` for NeMo Conformer/FastConformer/Parakeet with CTC decoder
* `nemo-conformer-rnnt` for NeMo Conformer/FastConformer/Parakeet with RNN-T decoder
* `nemo-conformer-tdt` for NeMo Conformer/FastConformer/Parakeet with TDT decoder
* `kaldi-rnnt` or `vosk` for Kaldi Icefall Zipformer with stateless RNN-T decoder
* `whisper-ort` for Whisper (exported with [onnxruntime](#openai-whisper-with-onnxruntime-export))
* `whisper` for Whisper (exported with [optimum](#openai-whisper-with-optimum-export))

## Comparison with original implementations

Packages with original implementations:
* `gigaam` for GigaAM models ([github](https://github.com/salute-developers/GigaAM))
* `nemo-toolkit` for NeMo models ([github](https://github.com/nvidia/nemo))
* `openai-whisper` for Whisper models ([github](https://github.com/openai/whisper))
* `sherpa-onnx` for Vosk models ([github](https://github.com/k2-fsa/sherpa-onnx), [docs](https://k2-fsa.github.io/sherpa/onnx/index.html))

Tests were performed on a *test* subset of the [Russian LibriSpeech](https://openslr.org/96/) dataset.

Hardware:
1. CPU tests were run on a laptop with an Intel i7-7700HQ processor.
2. GPU tests were run in Google Colab on Nvidia T4

| Model                    | Package / decoding   | CER    | WER    | RTFx (CPU) | RTFx (GPU)   |
|--------------------------|----------------------|--------|--------|------------|--------------|
|       GigaAM v2 CTC      |        default       | 1.06%  | 5.23%  |        7.2 | 44.2         |
|       GigaAM v2 CTC      |       onnx-asr       | 1.06%  | 5.23%  |       11.4 | 58.6         |
|      GigaAM v2 RNN-T     |        default       | 1.10%  | 5.22%  |        5.5 | 23.3         |
|      GigaAM v2 RNN-T     |       onnx-asr       | 1.10%  | 5.22%  |       10.4 | 26           |
|  Nemo FastConformer CTC  |        default       | 3.11%  | 13.12% |       22.7 | 71.7         |
|  Nemo FastConformer CTC  |       onnx-asr       | 3.11%  | 13.12% |       43.1 | 88.8         |
| Nemo FastConformer RNN-T |        default       | 2.63%  | 11.62% |       15.9 | 13.9         |
| Nemo FastConformer RNN-T |       onnx-asr       | 2.63%  | 11.62% |       26.0 | 49           |
|      Vosk 0.52 small     |     greedy_search    | 3.64%  | 14.53% |       48.2 | 71.4         |
|      Vosk 0.52 small     | modified_beam_search | 3.50%  | 14.25% |       29.0 | 24.7         |
|      Vosk 0.52 small     |       onnx-asr       | 3.64%  | 14.53% |       42.5 | 60.2         |
|         Vosk 0.54        |     greedy_search    | 2.21%  | 9.89%  |       34.8 | 64.2         |
|         Vosk 0.54        | modified_beam_search | 2.21%  | 9.85%  |       23.9 | 24           |
|         Vosk 0.54        |       onnx-asr       | 2.21%  | 9.89%  |       32.2 | 55.9         |
|       Whisper base       |        default       | 10.53% | 38.82% |        5.4 | 13.6         |
|       Whisper base       |       onnx-asr       | 10.64% | 38.33% |      6.3** | 16.1*/19.4** |
|  Whisper large-v3-turbo  |        default       | 2.96%  | 10.27% |        N/A | 11           |
|  Whisper large-v3-turbo  |       onnx-asr       | 2.63%  | 10.08% |        N/A | 9.8*         |

1. \* `whisper` model ([model types](#supported-model-types)) with `fp16` quantization.
2. ** `whisper-ort` model ([model types](#supported-model-types)).
3. All other models were run with the default precision - `fp32` on CPU and `fp32` or `fp16` (some of the original models) on GPU.

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

### Sber GigaAM v2
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
