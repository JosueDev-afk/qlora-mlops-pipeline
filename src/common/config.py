"""Load params.yaml and refuse a config that breaks a hard rule.

Several rules in CLAUDE.md live as values in params.yaml, where one edit can
silently break them: the English Laya checkpoint collapses on Spanish while
staying confident, replay outside 10-20 % trades task accuracy for rigidity,
thinking tokens spend the latency budget, and a hand-set call_rejected
threshold is a magic number H6 cannot defend. Checking them at load time makes
a bad config fail before a training run or a call starts, and reports every
violation at once instead of one per attempt.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path(__file__).resolve().parents[2] / "params.yaml"


class ConfigError(ValueError):
    """params.yaml parses but violates a rule the project depends on."""


def param(params: dict[str, Any], dotted: str) -> Any:
    """Read `a.b.c` from nested params, naming the full path when it is missing."""
    node: Any = params
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"params.yaml has no {dotted!r}")
        node = node[key]
    return node


def _keys(node: Any, prefix: str = "") -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}{key}"
            yield path
            yield from _keys(value, f"{path}.")


def _multilingual_checkpoint(p: dict[str, Any]) -> str | None:
    checkpoint = str(param(p, "laya.checkpoint"))
    if not checkpoint.endswith("laya-multilingual"):
        return f"laya.checkpoint must be the multilingual checkpoint, got {checkpoint!r}"
    return None


def _replay_ratio(p: dict[str, Any]) -> str | None:
    ratio = param(p, "gold.replay_ratio")
    if not 0.10 <= ratio <= 0.20:
        return f"gold.replay_ratio must keep 10-20 % general-instruction replay, got {ratio}"
    return None


def _thinking_disabled(p: dict[str, Any]) -> str | None:
    if param(p, "train.enable_thinking") is not False:
        return "train.enable_thinking must be false: thinking tokens spend the 500 ms budget"
    return None


def _interruption_threshold(p: dict[str, Any]) -> str | None:
    threshold = param(p, "audio.interruption_threshold")
    if threshold >= 0.5:
        return f"audio.interruption_threshold must stay below 0.5, got {threshold}"
    return None


def _no_hand_set_rejection_threshold(p: dict[str, Any]) -> str | None:
    hand_set = [k for k in _keys(p) if k.rsplit(".", 1)[-1] == "rejection_threshold"]
    if hand_set:
        return (
            f"{', '.join(hand_set)}: the call_rejected threshold is derived by "
            "calibrate_laya from calibration.target_recall, never set by hand"
        )
    return None


RULES: tuple[Callable[[dict[str, Any]], str | None], ...] = (
    _multilingual_checkpoint,
    _replay_ratio,
    _thinking_disabled,
    _interruption_threshold,
    _no_hand_set_rejection_threshold,
)


def load_params(path: Path | str = PARAMS_PATH) -> dict[str, Any]:
    """Parse params.yaml and check every rule, raising once with all violations."""
    params = yaml.safe_load(Path(path).read_text())
    if not isinstance(params, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    violations = []
    for rule in RULES:
        try:
            message = rule(params)
        except KeyError as missing:
            message = str(missing.args[0])
        if message:
            violations.append(message)
    if violations:
        raise ConfigError(f"{path}:\n- " + "\n- ".join(violations))
    return params
