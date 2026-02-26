"""Configuration for manual validation tests."""

from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dir",
        action="store",
        default=None,
        help="Path to the directory organized by the agent.",
    )


@pytest.fixture
def organized_dir(request):
    raw = request.config.getoption("--dir")
    if raw is None:
        pytest.skip("no --dir provided")
    path = Path(raw).resolve()
    assert path.is_dir(), f"{path} is not a directory"
    return path
