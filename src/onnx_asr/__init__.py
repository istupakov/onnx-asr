"""A lightweight Python package for Automatic Speech Recognition using ONNX models."""

from .loader import load_model, load_vad

__all__ = ["load_model", "load_vad"]
