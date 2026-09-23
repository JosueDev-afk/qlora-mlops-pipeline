"""Prompts are versioned artifacts: loading one must pin exactly which bytes ran.

A prompt change is a new experiment (H3), so the hash has to move on any edit,
and a malformed file has to fail at start-up, never in the middle of a call.
"""

import hashlib
from pathlib import Path

import pytest

from src.common.prompts import PROMPTS_DIR, PromptError, load_prompt

COMMITTED = sorted(PROMPTS_DIR.glob("*/v*.yaml"))


def _write_prompt(root: Path, body: str, *, prompt_id: str = "demo", version: int = 1) -> Path:
    """Lay out a minimal repo (prompts/ + schemas/) so paths resolve like the real one."""
    (root / "schemas").mkdir(exist_ok=True)
    (root / "schemas" / "demo.schema.json").write_text("{}")
    prompt_dir = root / "prompts" / prompt_id
    prompt_dir.mkdir(parents=True, exist_ok=True)
    path = prompt_dir / f"v{version}.yaml"
    path.write_text(body)
    return path


VALID = """\
id: demo
version: 1
schema: schemas/demo.schema.json
language: es-MX
system: |
  Responde únicamente con JSON.
"""


@pytest.mark.parametrize("path", COMMITTED, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_committed_prompt_loads(path: Path) -> None:
    prompt = load_prompt(path.parent.name, version=int(path.stem[1:]))
    assert prompt.schema.is_file()


def test_loads_extract_entity_as_the_node_expects() -> None:
    prompt = load_prompt("extract_entity", version=1)
    assert prompt.id == "extract_entity"
    assert prompt.version == 1
    assert prompt.schema.name == "extract_entity.schema.json"
    assert prompt.language == "es-MX"
    assert "system" in prompt.body


def test_hash_is_sha256_of_the_file_bytes(tmp_path: Path) -> None:
    path = _write_prompt(tmp_path, VALID)
    prompt = load_prompt("demo", version=1, root=tmp_path / "prompts")
    assert prompt.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_hash_moves_on_any_edit_even_a_comment(tmp_path: Path) -> None:
    path = _write_prompt(tmp_path, VALID)
    before = load_prompt("demo", version=1, root=tmp_path / "prompts").sha256
    path.write_text(VALID + "# editado en sitio\n")
    after = load_prompt("demo", version=1, root=tmp_path / "prompts").sha256
    assert before != after


def test_missing_version_raises(tmp_path: Path) -> None:
    _write_prompt(tmp_path, VALID)
    with pytest.raises(FileNotFoundError, match="v2"):
        load_prompt("demo", version=2, root=tmp_path / "prompts")


def test_id_must_match_directory(tmp_path: Path) -> None:
    """Catches a v2 copied from another prompt without updating its id."""
    _write_prompt(tmp_path, VALID.replace("id: demo", "id: other"))
    with pytest.raises(PromptError, match="id"):
        load_prompt("demo", version=1, root=tmp_path / "prompts")


def test_version_must_match_filename(tmp_path: Path) -> None:
    """Catches a v2.yaml whose body still says version 1."""
    _write_prompt(tmp_path, VALID, version=2)
    with pytest.raises(PromptError, match="version"):
        load_prompt("demo", version=2, root=tmp_path / "prompts")


def test_missing_schema_file_raises(tmp_path: Path) -> None:
    _write_prompt(tmp_path, VALID.replace("demo.schema.json", "missing.schema.json"))
    with pytest.raises(PromptError, match="schema"):
        load_prompt("demo", version=1, root=tmp_path / "prompts")


@pytest.mark.parametrize("key", ["id", "version", "schema", "language"])
def test_missing_required_key_raises(tmp_path: Path, key: str) -> None:
    body = "\n".join(line for line in VALID.splitlines() if not line.startswith(f"{key}:"))
    _write_prompt(tmp_path, body)
    with pytest.raises(PromptError, match=key):
        load_prompt("demo", version=1, root=tmp_path / "prompts")


def test_non_mapping_file_raises(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "- just\n- a list\n")
    with pytest.raises(PromptError, match="mapping"):
        load_prompt("demo", version=1, root=tmp_path / "prompts")
