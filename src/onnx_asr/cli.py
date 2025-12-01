"""CLI for ASR models."""

import argparse
import pathlib
from importlib.metadata import version
from typing import get_args

import onnx_asr
from onnx_asr.loader import ModelNames, ModelTypes, VadNames


def run() -> None:
    """Run CLI for ASR models."""
    parser = argparse.ArgumentParser(prog="onnx_asr", description="Automatic Speech Recognition in Python using ONNX models.")
    parser.add_argument(
        "model",
        help=f"Model name or type {(*get_args(ModelNames), *get_args(ModelTypes), 'onnx-community/whisper-...')}",
    )
    parser.add_argument(
        "filename",
        help="Path to wav file (only PCM_U8, PCM_16, PCM_24 and PCM_32 formats are supported).",
        nargs="+",
    )
    parser.add_argument("-p", "--model_path", type=pathlib.Path, help="Path to directory with model files")
    parser.add_argument("-q", "--quantization", help="Model quantization ('int8' for example)")
    parser.add_argument("--vad", help="Use VAD model", choices=get_args(VadNames))
    parser.add_argument("--version", action="version", version=f"%(prog)s {version('onnx_asr')}")
    args = parser.parse_args()

    model = onnx_asr.load_model(args.model, args.model_path, quantization=args.quantization)
    if args.vad:
        vad = onnx_asr.load_vad(args.vad)
        for segment in model.with_vad(vad, batch_size=1).recognize(args.filename):
            for res in segment:
                print(f"[{res.start:5.1f}, {res.end:5.1f}]: {res.text}")  # noqa: T201
            print()  # noqa: T201
    else:
        for text in model.recognize(args.filename):
            print(text)  # noqa: T201
