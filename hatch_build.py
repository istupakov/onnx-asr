"""Build ONNX preprocessors."""

import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class _BuildPreprocessorsHook(BuildHookInterface):  # type: ignore[type-arg]
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        frosen = Path("./uv.lock").exists()
        print(f"Build preprocessors hook (frosen dependencies: {frosen})...", flush=True)  # noqa: T201
        subprocess.run(  # noqa: S603
            ["uv", "run", *(["--frozen"] if frosen else []), "--isolated", "--only-group", "build", "-m", "preprocessors.build"],  # noqa: S607
            check=True,
        )
