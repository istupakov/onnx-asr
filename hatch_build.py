"""Build ONNX preprocessors."""

import sys
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class _BuildPreprocessorsHook(BuildHookInterface):  # type: ignore[type-arg]
    artifacts_path = "src/onnx_asr/preprocessors/*.onnx"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        self.app.display_info(f"Build ONNX preprocessor models ({self.artifacts_path})")
        build_data["artifacts"] = [self.artifacts_path]
        sys.path.append(self.root)

        from preprocessors.build import build_models  # noqa: PLC0415

        build_models(Path(self.artifacts_path).parent, self.metadata.version)

    def dependencies(self) -> list[str]:
        return self.metadata.config["dependency-groups"]["build"]  # type: ignore[no-any-return]
