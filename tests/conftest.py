import huggingface_hub as hf
import numpy as np
import onnxruntime as rt


def pytest_report_header() -> str:
    return (
        f"onnx-asr deps: numpy-{np.__version__}, onnxruntime-{rt.__version__}, huggingface-hub-{hf.__version__}"
        f"\nonnxruntime providers: {rt.get_available_providers()}"
    )
