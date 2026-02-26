"""Shared test fixtures."""

from io import StringIO

import pytest
from rich.console import Console

from ordnung.security import ToolSecurityPolicy


@pytest.fixture(autouse=True)
def _quiet_tui(monkeypatch):
    """Suppress all Rich console output from TUI functions during tests."""
    monkeypatch.setattr("ordnung.tui.console", Console(file=StringIO()))


@pytest.fixture
def sandbox(tmp_path):
    """A temporary directory that acts as the filesystem sandbox root."""
    return tmp_path


@pytest.fixture
def sec_policy(sandbox):
    """A security policy rooted at the sandbox directory."""
    return ToolSecurityPolicy(fs_root_jail=sandbox)


@pytest.fixture
def populated_sandbox(sandbox):
    """A sandbox with some pre-existing files and directories."""
    (sandbox / "readme.txt").write_text("Hello world!", encoding="utf-8")
    (sandbox / "data.bin").write_bytes(b"\x00\x01\x02\xff")
    (sandbox / "subdir").mkdir()
    (sandbox / "subdir" / "nested.txt").write_text("Nested content", encoding="utf-8")
    return sandbox
