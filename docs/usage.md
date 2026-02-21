# Usage Examples

## Load ONNX model from Hugging Face

Load ONNX model from Hugging Face and recognize WAV file:

```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize("test.wav"))
```

> [!WARNING]
> Supported WAV file formats: PCM_U8, PCM_16, PCM_24, and PCM_32 formats. For other formats, you either need to convert them first, or use a library that can read them into a NumPy array.

### Supported model names

* `gigaam-v2-ctc` for GigaChat GigaAM v2 CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `gigaam-v2-rnnt` for GigaChat GigaAM v2 RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v2-onnx))
* `gigaam-v3-ctc` for GigaChat GigaAM v3 CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-rnnt` for GigaChat GigaAM v3 RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-e2e-ctc` for GigaChat GigaAM v3 E2E CTC ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `gigaam-v3-e2e-rnnt` for GigaChat GigaAM v3 E2E RNN-T ([origin](https://github.com/salute-developers/GigaAM), [onnx](https://huggingface.co/istupakov/gigaam-v3-onnx))
* `nemo-fastconformer-ru-ctc` for Nvidia FastConformer-Hybrid Large (ru) with CTC decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-fastconformer-ru-rnnt` for Nvidia FastConformer-Hybrid Large (ru) with RNN-T decoder ([origin](https://huggingface.co/nvidia/stt_ru_fastconformer_hybrid_large_pc), [onnx](https://huggingface.co/istupakov/stt_ru_fastconformer_hybrid_large_pc_onnx))
* `nemo-parakeet-ctc-0.6b` for Nvidia Parakeet CTC 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-ctc-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-ctc-0.6b-onnx))
* `nemo-parakeet-rnnt-0.6b` for Nvidia Parakeet RNNT 0.6B (en) ([origin](https://huggingface.co/nvidia/parakeet-rnnt-0.6b), [onnx](https://huggingface.co/istupakov/parakeet-rnnt-0.6b-onnx))
* `nemo-parakeet-tdt-0.6b-v2` for Nvidia Parakeet TDT 0.6B V2 (en) ([origin](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v2-onnx))
* `nemo-parakeet-tdt-0.6b-v3` for Nvidia Parakeet TDT 0.6B V3 (multilingual) ([origin](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx))
* `nemo-canary-1b-v2` for Nvidia Canary 1B V2 (multilingual) ([origin](https://huggingface.co/nvidia/canary-1b-v2), [onnx](https://huggingface.co/istupakov/canary-1b-v2-onnx))
* `istupakov/canary-180m-flash-onnx` for Nvidia Canary 180M Flash (multilingual) ([origin](https://huggingface.co/nvidia/canary-180m-flash), [onnx](https://huggingface.co/istupakov/canary-180m-flash-onnx))
* `istupakov/canary-1b-flash-onnx` for Nvidia Canary 1B Flash (multilingual) ([origin](https://huggingface.co/nvidia/canary-1b-flash), [onnx](https://huggingface.co/istupakov/canary-1b-flash-onnx))
* `whisper-base` for OpenAI Whisper Base exported with onnxruntime ([origin](https://huggingface.co/openai/whisper-base), [onnx](https://huggingface.co/istupakov/whisper-base-onnx))
* `alphacep/vosk-model-ru` for Alpha Cephei Vosk 0.54-ru ([origin](https://huggingface.co/alphacep/vosk-model-ru))
* `alphacep/vosk-model-small-ru` for Alpha Cephei Vosk 0.52-small-ru ([origin](https://huggingface.co/alphacep/vosk-model-small-ru))
* `t-tech/t-one` for T-Tech T-one ([origin](https://huggingface.co/t-tech/T-one))
* `onnx-community/whisper-tiny`, `onnx-community/whisper-base`, `onnx-community/whisper-small`, `onnx-community/whisper-large-v3-turbo`, etc. for OpenAI Whisper exported with Hugging Face optimum ([onnx-community](https://huggingface.co/onnx-community?search_models=whisper))

> [!WARNING]
> Some long-ago converted `onnx-community` models have a broken `fp16` precision version.

### Using soundfile

```py
import onnx_asr
import soundfile as sf

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")

waveform, sample_rate = sf.read("test.wav", dtype="float32")
model.recognize(waveform, sample_rate=sample_rate)
```

### Batch processing

```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize(["test1.wav", "test2.wav", "test3.wav", "test4.wav"]))
```

### Quantized models

Most models have quantized versions:

```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8")
print(model.recognize("test.wav"))
```

### Timestamps and log probabilities

Return tokens, timestamps and log probabilities:

```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_timestamps()
print(model.recognize("test1.wav"))
```

## TensorRT

Running an ONNX model on the TensorRT provider with fp16 precision:

```py
import onnx_asr
import tensorrt_libs # If installed via pip tensorrt-cu12-libs

providers = [
    (
        "TensorrtExecutionProvider",
        {
            "trt_max_workspace_size": 6 * 1024**3, # for big models
            "trt_fp16_enable": True,               # for auto conversion to fp16 
        },
    )
]
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=providers)
print(model.recognize("test.wav"))
```

## VAD (Voice Activity Detection)

Load a VAD ONNX model from Hugging Face and recognize a WAV file:

```py
import onnx_asr
vad = onnx_asr.load_vad("silero")
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3").with_vad(vad)
for res in model.recognize("test.wav"):
    print(res)
```

> [!TIP]
> You will most likely need to adjust VAD parameters to get the correct results.

### Supported VAD names

* `silero` for Silero VAD ([origin](https://github.com/snakers4/silero-vad), [onnx](https://huggingface.co/onnx-community/silero-vad))

## CLI

The package has a simple CLI interface:

```sh
onnx-asr nemo-parakeet-tdt-0.6b-v3 test.wav
```

For full usage parameters, see help:

```sh
onnx-asr -h
```

## Gradio

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
    return model.recognize(waveform, sample_rate=sample_rate, channel="mean")

demo = gr.Interface(fn=recognize, inputs="audio", outputs="text")
demo.launch()
```

## Load ONNX model from local directory

Load ONNX model from local directory and recognize WAV file:

```py
import onnx_asr
model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", "models/parakeet-v3")
print(model.recognize("test.wav"))
```

> [!NOTE]
> If the directory does not exist, it will be created and the model will be loaded into it.

## Load a custom ONNX model from Hugging Face

Load the Canary 180M Flash model from Hugging Face [repo](https://huggingface.co/istupakov/canary-180m-flash-onnx) and recognize the WAV file:

```py
import onnx_asr
model = onnx_asr.load_model("istupakov/canary-180m-flash-onnx")
print(model.recognize("test.wav"))
```

### Supported model types

* All models from [supported model names](#supported-model-names)
* `kaldi-rnnt` or `vosk` for Kaldi Icefall Zipformer with stateless RNN-T decoder
* `nemo-conformer-ctc` for NeMo Conformer/FastConformer/Parakeet with CTC decoder
* `nemo-conformer-rnnt` for NeMo Conformer/FastConformer/Parakeet with RNN-T decoder
* `nemo-conformer-tdt` for NeMo Conformer/FastConformer/Parakeet with TDT decoder
* `nemo-conformer-aed` for NeMo Canary with Transformer decoder
* `t-one-ctc` for T-Tech T-one with CTC decoder
* `whisper-ort` for Whisper (exported with [onnxruntime](#openai-whisper-with-onnxruntime-export))
* `whisper` for Whisper (exported with [optimum](#openai-whisper-with-optimum-export))