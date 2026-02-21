"""Build ONNX preprocessors."""

import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class _BuildPreprocessorsHook(BuildHookInterface):  # type: ignore[type-arg]
    artifacts_path = Path("src/onnx_asr/preprocessors/data")

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self.app.display_info(f"Build ONNX and NumPy preprocessors ({self.artifacts_path})")
        sys.path.append(self.root)

        from preprocessors.build import build  # noqa: PLC0415

        self.artifacts_path.mkdir(exist_ok=True)
        build(self.artifacts_path, self.metadata.version)
        build_data["artifacts"] = [
            str(self.artifacts_path.joinpath("*.onnx")),
            str(self.artifacts_path.joinpath("*.npz")),
        ]

    def dependencies(self) -> list[str]:
        return self.metadata.config["dependency-groups"]["build"]  # type: ignore[no-any-return]
