"""params.yaml is loaded once and checked against the hard rules it encodes.

A config that breaks a rule (English Laya checkpoint, replay outside 10-20 %,
thinking enabled, a hand-set call_rejected threshold, an interruption threshold
that keeps talking when unsure) must fail at load time, not halfway through a
training run or a live call.
"""

from pathlib import Path

import pytest
import yaml

from src.common.config import PARAMS_PATH, ConfigError, load_params, param


def _write(tmp_path: Path, mutate) -> Path:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    mutate(params)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params, allow_unicode=True))
    return path


def test_committed_params_satisfy_every_rule() -> None:
    params = load_params()
    assert param(params, "gold.tag") == "gold-v1"


def test_param_reads_dotted_paths() -> None:
    params = {"calibration": {"target_recall": 0.95}}
    assert param(params, "calibration.target_recall") == 0.95


def test_param_names_the_missing_key() -> None:
    with pytest.raises(KeyError, match="calibration.max_ece"):
        param({"calibration": {}}, "calibration.max_ece")


def test_english_laya_checkpoint_is_rejected(tmp_path: Path) -> None:
    def mutate(p):
        p["laya"]["checkpoint"] = "convaiinnovations/laya"

    with pytest.raises(ConfigError, match="multilingual"):
        load_params(_write(tmp_path, mutate))


@pytest.mark.parametrize("ratio", [0.05, 0.25])
def test_replay_outside_ten_to_twenty_percent_is_rejected(tmp_path: Path, ratio: float) -> None:
    def mutate(p):
        p["gold"]["replay_ratio"] = ratio

    with pytest.raises(ConfigError, match="replay"):
        load_params(_write(tmp_path, mutate))


def test_thinking_enabled_is_rejected(tmp_path: Path) -> None:
    def mutate(p):
        p["train"]["enable_thinking"] = True

    with pytest.raises(ConfigError, match="thinking"):
        load_params(_write(tmp_path, mutate))


def test_interruption_threshold_at_half_is_rejected(tmp_path: Path) -> None:
    def mutate(p):
        p["audio"]["interruption_threshold"] = 0.5

    with pytest.raises(ConfigError, match="interruption_threshold"):
        load_params(_write(tmp_path, mutate))


def test_hand_set_rejection_threshold_is_rejected(tmp_path: Path) -> None:
    def mutate(p):
        p["calibration"]["rejection_threshold"] = 0.7

    with pytest.raises(ConfigError, match="derived"):
        load_params(_write(tmp_path, mutate))


def test_all_violations_are_reported_together(tmp_path: Path) -> None:
    """One run should surface every broken rule, not one per attempt."""

    def mutate(p):
        p["train"]["enable_thinking"] = True
        p["gold"]["replay_ratio"] = 0.0

    with pytest.raises(ConfigError) as err:
        load_params(_write(tmp_path, mutate))
    assert "thinking" in str(err.value) and "replay" in str(err.value)


def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("- not\n- a mapping\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_params(path)
