"""The corpus sources in params.yaml, validated before anything is downloaded.

A source must be pinned: a Hub commit for datasets on the Hub, a sha256 for
plain URLs. Branch names and "latest" are refused, because bronze must be
rebuildable byte for byte from params.yaml alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, cast

Kind = Literal["url", "hf_dataset"]
NAME_RE = re.compile(r"^[a-z0-9_]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class SourceError(ValueError):
    """params.yaml describes a source that cannot be ingested reproducibly."""


@dataclass(frozen=True)
class Source:
    name: str
    kind: Kind
    license: str
    purpose: str
    verified: bool
    url: str | None = None
    sha256: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    allow_patterns: tuple[str, ...] | None = None

    @property
    def pin(self) -> str:
        """What makes this download immutable: the file hash or the Hub commit."""
        return (self.sha256 if self.kind == "url" else self.revision) or ""

    @property
    def origin(self) -> str:
        return self.url or f"hf://datasets/{self.repo_id}"


def _source(name: str, spec: dict[str, Any]) -> tuple[Source | None, list[str]]:
    problems: list[str] = []
    if not NAME_RE.match(name):
        problems.append(f"{name}: names are lowercase identifiers")
    kind = spec.get("kind")
    if kind == "url":
        if not spec.get("url"):
            problems.append(f"{name}: a url source needs `url`")
        if not SHA256_RE.match(str(spec.get("sha256", ""))):
            problems.append(f"{name}: a url source needs a pinned `sha256`")
    elif kind == "hf_dataset":
        if not spec.get("repo_id"):
            problems.append(f"{name}: an hf_dataset source needs `repo_id`")
        if not COMMIT_RE.match(str(spec.get("revision", ""))):
            problems.append(f"{name}: `revision` must be a 40-char commit, not a branch")
    else:
        problems.append(f"{name}: unknown kind {kind!r} (url | hf_dataset)")
    for key in ("license", "purpose"):
        if not spec.get(key):
            problems.append(f"{name}: `{key}` is required (docs/DATA_PROVENANCE.md)")
    if not isinstance(spec.get("verified"), bool):
        problems.append(f"{name}: `verified` must be true or false")
    if problems:
        return None, problems
    patterns = spec.get("allow_patterns")
    return Source(
        name=name,
        kind=cast(Kind, kind),
        license=spec["license"],
        purpose=spec["purpose"],
        verified=spec["verified"],
        url=spec.get("url"),
        sha256=spec.get("sha256"),
        repo_id=spec.get("repo_id"),
        revision=spec.get("revision"),
        allow_patterns=tuple(patterns) if patterns else None,
    ), []


def load_sources(params: dict[str, Any]) -> dict[str, Source]:
    """All sources under `ingest.sources`; one error lists every problem."""
    specs = (params.get("ingest") or {}).get("sources") or {}
    sources: dict[str, Source] = {}
    problems: list[str] = []
    for name, spec in specs.items():
        source, found = _source(name, spec or {})
        problems += found
        if source:
            sources[name] = source
    if problems:
        raise SourceError("invalid ingest sources:\n  " + "\n  ".join(problems))
    return sources
