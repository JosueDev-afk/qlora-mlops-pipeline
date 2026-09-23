"""Load a versioned prompt and pin exactly which bytes ran.

Prompts are experiments (H3): a run is only reproducible if it records the
prompt it used, and an in-place edit must be detectable. The hash is therefore
taken over the raw file bytes, not the parsed YAML, so that even a comment
change produces a different fingerprint.

Loading validates the file eagerly. Nodes load their prompt at import time, so
a malformed file fails when the service starts, never in the middle of a call.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
REQUIRED_KEYS = ("id", "version", "schema", "language")


class PromptError(ValueError):
    """A prompt file exists but cannot be trusted as the artifact it claims to be."""


@dataclass(frozen=True)
class Prompt:
    id: str
    version: int
    schema: Path
    language: str
    sha256: str
    body: dict[str, Any]


def load_prompt(prompt_id: str, version: int, *, root: Path = PROMPTS_DIR) -> Prompt:
    """Load `<root>/<prompt_id>/v<version>.yaml`.

    `schema` in the file is relative to the repository root, which is taken to
    be `root.parent` so the same layout works in tests.

    The id and version inside the file must match its path: a `v2.yaml` copied
    from `v1.yaml` without bumping `version` would otherwise log the wrong
    experiment.
    """
    path = root / prompt_id / f"v{version}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"prompt {prompt_id} v{version} not found at {path}")

    raw = path.read_bytes()
    body = yaml.safe_load(raw)
    if not isinstance(body, dict):
        raise PromptError(f"{path}: expected a mapping at the top level")

    missing = [key for key in REQUIRED_KEYS if key not in body]
    if missing:
        raise PromptError(f"{path}: missing required keys {missing}")
    if body["id"] != prompt_id:
        raise PromptError(f"{path}: id is {body['id']!r}, directory says {prompt_id!r}")
    if body["version"] != version:
        raise PromptError(f"{path}: version is {body['version']!r}, filename says {version}")

    schema = (root.parent / body["schema"]).resolve()
    if not schema.is_file():
        raise PromptError(f"{path}: schema {body['schema']!r} does not exist")

    return Prompt(
        id=prompt_id,
        version=version,
        schema=schema,
        language=body["language"],
        sha256=hashlib.sha256(raw).hexdigest(),
        body=body,
    )
