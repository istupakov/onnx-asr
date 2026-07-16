# Community Models

The models in [supported model names](usage.md#supported-model-names) are maintained or
selected by the onnx-asr project. The wider community also publishes compatible models
on Hugging Face, including fine-tunes for additional languages and alternative
quantizations.

[Browse all models tagged `onnx-asr`](https://huggingface.co/models?other=onnx-asr&sort=trending)

Use the full Hugging Face repository ID to load a community model:

```py
import onnx_asr

model = onnx_asr.load_model("xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr")
print(model.recognize("test.wav"))
```

> [!IMPORTANT]
> This is a curated snapshot inspected on **July 16, 2026**, not an exhaustive registry
> or compatibility guarantee. The entries below passed metadata and file-layout
> inspection only: they were not downloaded, executed, benchmarked, or checked for
> transcription quality. Community models are maintained and supported by their
> publishers.

## Language fine-tunes

These repositories convert distinct upstream fine-tunes rather than copying an existing
onnx-asr export.

| Repository | Owner | Language | Architecture | Upstream model | Precision | License | Verification |
|---|---|---|---|---|---|---|---|
| [`xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr`](https://huggingface.co/xezpeleta/parakeet-tdt-0.6b-v3-basque-onnx-asr) | xezpeleta | Basque | NeMo Conformer TDT | [`itzune/parakeet-tdt-0.6b-v3-basque`](https://huggingface.co/itzune/parakeet-tdt-0.6b-v3-basque) | FP32, INT8 encoder | CC-BY-4.0 | Metadata inspected |
| [`alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx`](https://huggingface.co/alefiury/parakeet-tdt-0.6b-v3-ptBR-TAGARELA-onnx) | alefiury | Brazilian Portuguese | NeMo Conformer TDT | [`alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus`](https://huggingface.co/alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus) | FP32 | CC-BY-4.0 | Metadata inspected |
| [`Buttermilk03/parakeet-primeline-onnx`](https://huggingface.co/Buttermilk03/parakeet-primeline-onnx) | Buttermilk03 | German | NeMo Conformer TDT | [`primeline/parakeet-primeline`](https://huggingface.co/primeline/parakeet-primeline) | FP32, INT8 | CC-BY-4.0 | Metadata inspected |
| [`AlinClaudiu/SpeD-ParakeetRo-110M-onnx`](https://huggingface.co/AlinClaudiu/SpeD-ParakeetRo-110M-onnx) | AlinClaudiu | Romanian | NeMo Conformer CTC | [`gabrielpirlo/SpeD_ParakeetRo_110M_TDT-CTC`](https://huggingface.co/gabrielpirlo/SpeD_ParakeetRo_110M_TDT-CTC) | FP32 | Apache-2.0 | Metadata inspected |
| [`AigizK/GigaAM-Bashkir-CV25-ONNX`](https://huggingface.co/AigizK/GigaAM-Bashkir-CV25-ONNX) | AigizK | Bashkir | GigaAM Multilingual CTC | [`AigizK/GigaAM-Bashkir-CV25`](https://huggingface.co/AigizK/GigaAM-Bashkir-CV25) | FP32, INT8 | MIT | Metadata inspected |

## OpenVoiceOS model families

[OpenVoiceOS](https://huggingface.co/OpenVoiceOS) publishes a large coordinated set of
conversions. The table uses one representative repository for each family; follow its
owner link to find other languages and model sizes.

| Family and representative | Languages | Architecture | Upstream publisher | Precision in representative | License | Verification |
|---|---|---|---|---|---|---|
| [AI4Bharat IndicConformer](https://huggingface.co/OpenVoiceOS/ai4bharat-indicconformer-hi-onnx) | Indic languages | NeMo Conformer CTC | [AI4Bharat](https://huggingface.co/ai4bharat) | FP32, INT8 | MIT | Metadata inspected |
| [IISc Vaani FastConformer](https://huggingface.co/OpenVoiceOS/artpark-iisc-vaani-fastconformer-multi-onnx) | Multilingual and individual Indic languages | NeMo Conformer TDT | [ARTPARK-IISc](https://huggingface.co/ARTPARK-IISc) | FP32, INT8 | MIT | Metadata inspected |
| [NVIDIA monolingual Conformer](https://huggingface.co/OpenVoiceOS/nvidia-en-conformer-ctc-large-onnx) | Multiple languages | NeMo Conformer CTC/RNN-T | [NVIDIA](https://huggingface.co/nvidia) | FP32, INT8 | CC-BY-4.0 | Metadata inspected |
| [Localized Parakeet](https://huggingface.co/OpenVoiceOS/yuriyvnv-parakeet-tdt-0.6b-pl-onnx) | Polish, Estonian, Dutch, Slovenian, Portuguese, and others | NeMo Conformer TDT | Community fine-tunes | FP32 | CC-BY-4.0 | Metadata inspected |
| [Wav2Vec2](https://huggingface.co/OpenVoiceOS/wav2vec2-xlsr-300m-finnish-onnx) | Multiple languages | Wav2Vec2 CTC | Multiple publishers | FP32 | Apache-2.0 | Metadata inspected |

Licenses can differ between repositories in a family. Always check the selected model
card and its upstream model before redistribution or commercial use.

## Optimized variants

Some community repositories retain an existing model but provide a materially different
ONNX optimization. These are listed separately from language fine-tunes.

| Repository | Owner | Language | Architecture | Upstream model | Precision or optimization | License | Verification |
|---|---|---|---|---|---|---|---|
| [`Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx`](https://huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx) | Olicorne | Multilingual | NeMo Conformer TDT | [`istupakov/parakeet-tdt-0.6b-v3-onnx`](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | FP32, FP16, SmoothQuant INT8 | CC-BY-4.0 | Metadata inspected |
| [`gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc`](https://huggingface.co/gvij/parakeet-tdt-0.6b-v3-onnx-static-qdq-pc) | gvij | Multilingual | NeMo Conformer TDT | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Static QDQ per-channel INT8 | CC-BY-4.0 | Metadata inspected |

## Curation criteria

A repository is included when it is public and ungated, declares a model type supported
by onnx-asr, contains a valid `config.json`, and provides the expected ONNX graphs and
tokenizer or vocabulary files. It must also identify its purpose, upstream lineage, and
license.

A different fine-tuned upstream model, language, architecture, or documented
quantization method is considered meaningful. Unchanged forks, mirrors, re-uploads,
incomplete repositories, and repositories that differ only by owner are omitted. Large
coordinated conversion sets are represented as families to keep this page maintainable.

Metadata inspection does not validate ONNX graph integrity or recognition quality. If a
model fails to load, report the problem to its publisher and see the
[troubleshooting guide](troubleshooting.md).
