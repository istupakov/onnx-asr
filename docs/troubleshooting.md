# Troubleshooting / FAQ

## Common Issues

- **Model download fails**: Ensure Hugging Face is accessible. To improve download speed set the `HF_TOKEN` environment variable.
- **Model loading fails**: Ensure you have the latest `onnxruntime` version compatible with your setup. For GPU, verify CUDA / TensorRT installation. Try a different provider (not all models compatible with all providers).
- **Model loading fails on onnxruntime 1.24.1**: ONNX Runtime 1.24.1 does not support symlinks to data files used in the HuggingFace cache for large models. Please upgrade to 1.24.2+ or downgrade to 1.23.
- **Audio loading issues**: Check that your WAV file is in a supported format (PCM_U8, PCM_16, PCM_24, PCM_32). Use `soundfile` for other formats.
- **Audio recognition fails**: Most models support up to 20-30 seconds of audio. For longer files, use [VAD](../usage/#vad-voice-activity-detection) for segmentation.
- **Slow performance**: Try quantized models (e.g., `quantization="int8"`) on CPU or TensorRT for GPU acceleration.
- **Incorrect segmentation with VAD**: Adjust VAD parameters like `threshold` or `min_speech_duration_ms` for your audio.

## Getting Help

For more help, check the [GitHub Issues](https://github.com/istupakov/onnx-asr/issues) or open a new one.