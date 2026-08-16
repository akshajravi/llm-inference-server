"""Shared fixtures. The model loads once per test session, not once per test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_server.config import CONFIG

GOLDENS_DIR = Path(__file__).parent / "goldens"


@pytest.fixture(scope="session")
def model_and_tokenizer():
    from inference_server.model import load

    return load()


@pytest.fixture(scope="session")
def goldens() -> dict:
    path = GOLDENS_DIR / f"{CONFIG.model_id.replace('/', '_')}_greedy.json"
    if not path.exists():
        pytest.skip(f"no goldens at {path} — run `make goldens` first")
    return json.loads(path.read_text())


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "phase(name): the sprint phase this test starts passing in")
