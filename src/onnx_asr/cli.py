"""CLI for ASR models."""

import argparse
import pathlib
from typing import get_args

from onnx_asr import load_model
from onnx_asr.loader import ModelNames, ModelTypes


def run() -> None:
    """Run CLI for ASR models."""
    parser = argparse.ArgumentParser(prog="onnx_asr", description="Automatic Speech Recognition in Python using ONNX models.")
    parser.add_argument(
        "model",
        choices=get_args(ModelNames) + get_args(ModelTypes),
        help="Model name or type",
    )
    parser.add_argument(
        "filename",
        help="Path to wav file (only PCM_U8, PCM_16, PCM_24 and PCM_32 formats with 16 kHz sample rate are supported).",
        nargs="+",
    )
    parser.add_argument("-p", "--model_path", type=pathlib.Path, help="Path to directory with model files")
    parser.add_argument("-q", "--quantization", help="Model quantization ('int8' for example)")
    args = parser.parse_args()

    model = load_model(args.model, args.model_path, args.quantization)
    for text in model.recognize(args.filename):
        print(text)
