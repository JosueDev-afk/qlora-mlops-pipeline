"""The compose stack is checked without Docker, so CI and a laptop without it agree.

These guard the mistakes that only show up when someone runs `make`: a profile
that `make down` forgets to stop, a variable missing from .env.example, a
dependency on a service another profile never starts, a port used twice.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((ROOT / "infra" / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
MAKEFILE = (ROOT / "Makefile").read_text()
ENV_EXAMPLE = (ROOT / ".env.example").read_text()


def _profiles(service: dict) -> set[str]:
    return set(service.get("profiles", []))


def test_make_down_stops_every_profile() -> None:
    all_line = next(line for line in MAKEFILE.splitlines() if line.startswith("ALL "))
    stopped = set(re.findall(r"--profile (\w+)", all_line))
    declared = set().union(*(_profiles(s) for s in SERVICES.values()))
    assert declared == stopped


def test_only_postgres_runs_without_a_profile() -> None:
    """On 16 GB the default `up` must stay small; everything else is opt-in."""
    assert [name for name, s in SERVICES.items() if not _profiles(s)] == ["postgres"]


def test_every_interpolated_variable_is_documented() -> None:
    text = (ROOT / "infra" / "docker-compose.yml").read_text()
    used = set(re.findall(r"\$\{(\w+)", text))
    documented = set(re.findall(r"^(\w+)=", ENV_EXAMPLE, re.M))
    assert used <= documented, used - documented


@pytest.mark.parametrize("name", list(SERVICES))
def test_dependencies_start_with_the_dependent(name: str) -> None:
    """A dependency must be core or share a profile, or `up` fails for that profile."""
    deps = SERVICES[name].get("depends_on", [])
    for dep in deps if isinstance(deps, list) else deps.keys():
        assert dep in SERVICES, dep
        dep_profiles = _profiles(SERVICES[dep])
        assert not dep_profiles or dep_profiles & _profiles(SERVICES[name]), (name, dep)


def test_host_ports_are_unique() -> None:
    ports = [p.split(":")[0] for s in SERVICES.values() for p in s.get("ports", [])]
    assert len(ports) == len(set(ports)), ports


def test_every_service_has_a_memory_limit() -> None:
    missing = [n for n, s in SERVICES.items() if "mem_limit" not in s and n != "airflow-init"]
    assert not missing, missing


@pytest.mark.parametrize(
    "service", [n for n, s in SERVICES.items() if isinstance(s.get("build"), dict)]
)
def test_build_files_exist(service: str) -> None:
    build = SERVICES[service]["build"]
    assert (ROOT / "infra" / build["context"] / build["dockerfile"]).is_file()


def test_services_used_by_make_and_bootstrap_exist() -> None:
    bootstrap = (ROOT / "infra" / "init" / "bootstrap.sh").read_text()
    for name in ("airflow-scheduler", "postgres", "kafka"):
        assert name in SERVICES
        assert name in MAKEFILE or name in bootstrap


def test_no_image_is_left_unpinned() -> None:
    """latest moves under your feet; a pinned tag keeps the stack reproducible."""
    images = [s["image"] for s in SERVICES.values() if "image" in s and "build" not in s]
    assert all(":" in i and not i.endswith(":latest") for i in images), images
