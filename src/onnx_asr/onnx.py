"""Helpers for ONNX."""

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, TypedDict

import onnxruntime as rt


class OnnxSessionOptions(TypedDict, total=False):
    """Options for onnxruntime InferenceSession."""

    sess_options: rt.SessionOptions | None
    providers: Sequence[str | tuple[str, dict[Any, Any]]] | None
    provider_options: Sequence[dict[Any, Any]] | None


class TensorRtOptions:
    """Options for onnxruntime TensorRT providers."""

    profile_min_shapes: ClassVar[dict[str, int]] = {"batch": 1, "waveform_len_ms": 50}
    profile_max_shapes: ClassVar[dict[str, int]] = {"batch": 16, "waveform_len_ms": 30_000}
    profile_opt_shapes: ClassVar[dict[str, int]] = {"batch": 1, "waveform_len_ms": 20_000}

    @classmethod
    def _generate_trt_profile(cls, transform_shapes: Callable[..., str]) -> dict[str, str]:
        return {
            "trt_profile_min_shapes": transform_shapes(**cls.profile_min_shapes),
            "trt_profile_max_shapes": transform_shapes(**cls.profile_max_shapes),
            "trt_profile_opt_shapes": transform_shapes(**cls.profile_opt_shapes),
        }

    @classmethod
    def _generate_nv_profile(cls, transform_shapes: Callable[..., str]) -> dict[str, str]:
        return {
            "nv_profile_min_shapes": transform_shapes(**cls.profile_min_shapes),
            "nv_profile_max_shapes": transform_shapes(**cls.profile_max_shapes),
            "nv_profile_opt_shapes": transform_shapes(**cls.profile_opt_shapes),
        }

    @classmethod
    def add_profile(cls, onnx_options: OnnxSessionOptions, transform_shapes: Callable[..., str]) -> OnnxSessionOptions:
        """Add TensorRT trt_profile options."""
        return update_onnx_providers(
            onnx_options,
            default_options={
                "TensorrtExecutionProvider": cls._generate_trt_profile(transform_shapes),
                "NvTensorRtRtxExecutionProvider": cls._generate_nv_profile(transform_shapes),
            },
        )

    @staticmethod
    def get_provider_names() -> list[str]:
        """Get TensorRT provider names."""
        return ["TensorrtExecutionProvider", "NvTensorRtRtxExecutionProvider"]


def _merge_onnx_provider_options(onnx_options: OnnxSessionOptions) -> dict[str, dict[Any, Any]]:
    providers = onnx_options.get("providers")
    if providers is None:
        return {}

    provider_options = onnx_options.get("provider_options")
    merged_providers: dict[str, dict[Any, Any]] = {}
    if provider_options is None:
        for provider in providers:
            if isinstance(provider, tuple):
                merged_providers[provider[0]] = provider[1]
            else:
                merged_providers[provider] = {}
    else:
        for name, options in zip(providers, provider_options, strict=True):
            assert isinstance(name, str)
            merged_providers[name] = options

    return merged_providers


def update_onnx_providers(
    onnx_options: OnnxSessionOptions,
    *,
    default_options: dict[str, dict[Any, Any]] | None = None,
    new_options: dict[str, dict[Any, Any]] | None = None,
    excluded_providers: list[str] | None = None,
) -> OnnxSessionOptions:
    """Update onnxruntime providers."""
    providers_dict = _merge_onnx_provider_options(onnx_options)
    if not providers_dict:
        return onnx_options
    default_options = default_options or {}
    new_options = new_options or {}
    excluded_providers = excluded_providers or []

    providers_dict = {
        name: default_options.get(name, {}) | options | new_options.get(name, {})
        for name, options in providers_dict.items()
        if name not in excluded_providers
    }
    return {**onnx_options, "providers": list(providers_dict.keys()), "provider_options": list(providers_dict.values())}


def get_onnx_providers(onnx_options: OnnxSessionOptions) -> list[str]:
    """Get providers list from onnxruntime options."""
    return list(_merge_onnx_provider_options(onnx_options).keys())


def get_onnx_device(session: rt.InferenceSession) -> tuple[str, int]:
    """Get onnxruntime device type and id from session."""
    provider = session.get_providers()[0]
    match provider:
        case "CUDAExecutionProvider" | "ROCMExecutionProvider":
            device_type = "cuda"
        case "DmlExecutionProvider":
            device_type = "dml"
        case _:
            device_type = "cpu"

    return device_type, int(session.get_provider_options()[provider].get("device_id", 0))
