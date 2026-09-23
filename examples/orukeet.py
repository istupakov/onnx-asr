"""Download a verified Orukeet release and transcribe a local WAV file.

Install ``onnx-asr[cpu,hub]`` and ``soundfile`` before running this example.
"""

# ruff: noqa: INP001

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

import onnx_asr

REPO_ID = "oruk/orukeet"
REVISION = "eac739d754bb171287930e6e63386f5b88f8179e"
SUBFOLDER = "onnx/combined-v0.1.0-int8"
MANIFEST_SHA256 = "77f9f8e9fadeddbbf98b1dfd9e51f90a8e109c795b1679d9d85e3a5e9ef205f4"
FILES = (
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
    "config.json",
    "LICENSE-WEIGHTS",
    "LICENSE-CONVERTER.txt",
    "NOTICE.md",
)


def sha256(path: Path) -> str:
    """Hash a model without reading all its weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(cache_dir: str | None = None, *, offline: bool = False) -> Path:
    """Return a verified Hub snapshot directory, reusing cached files first."""

    def fetch(filename: str) -> Path:
        try:
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    revision=REVISION,
                    filename=f"{SUBFOLDER}/{filename}",
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError:
            if offline:
                raise
            return Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    revision=REVISION,
                    filename=f"{SUBFOLDER}/{filename}",
                    cache_dir=cache_dir,
                )
            )

    manifest_path = fetch("manifest.json")
    if sha256(manifest_path) != MANIFEST_SHA256:
        msg = "Orukeet release manifest checksum mismatch"
        raise ValueError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in FILES:
        expected = manifest["files"][filename]
        path = fetch(filename)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            msg = f"Orukeet checksum mismatch: {filename}"
            raise ValueError(msg)
    return manifest_path.parent


def main() -> None:
    """Transcribe 16 kHz mono WAV audio with the existing NeMo TDT loader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--cache-dir")
    parser.add_argument("--offline", action="store_true", help="Require an already downloaded model")
    args = parser.parse_args()
    model_dir = download_model(args.cache_dir, offline=args.offline)
    model = onnx_asr.load_model(
        "nemo-conformer-tdt", path=model_dir, quantization="int8", providers=["CPUExecutionProvider"]
    )
    print(model.recognize(str(args.audio)))  # noqa: T201


if __name__ == "__main__":
    main()
