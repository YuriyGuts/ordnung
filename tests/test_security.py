"""Tests for ordnung.security."""

from pathlib import Path

import pytest

from ordnung.security import ToolSecurityPolicy


def test_validate_path_access_within_jail(sec_policy, sandbox):
    # GIVEN a path inside the sandbox
    path = sandbox / "somefile.txt"

    # WHEN validating access
    # THEN no exception is raised
    sec_policy.validate_path_access(path)


def test_validate_path_access_nested_within_jail(sec_policy, sandbox):
    # GIVEN a deeply nested path inside the sandbox
    path = sandbox / "a" / "b" / "c" / "file.txt"

    # WHEN validating access
    # THEN no exception is raised
    sec_policy.validate_path_access(path)


def test_validate_path_access_outside_jail(sec_policy):
    # GIVEN a path outside the sandbox
    path = Path("/etc/passwd")

    # WHEN validating access
    # THEN a RuntimeError is raised
    with pytest.raises(RuntimeError, match="Rejected by security policy"):
        sec_policy.validate_path_access(path)


def test_validate_path_access_parent_traversal(sec_policy, sandbox):
    # GIVEN a path that uses ".." to escape the sandbox
    path = sandbox / ".." / "escape.txt"

    # WHEN validating access
    # THEN a RuntimeError is raised
    with pytest.raises(RuntimeError, match="Rejected by security policy"):
        sec_policy.validate_path_access(path)


def test_validate_path_access_jail_root_itself(sec_policy, sandbox):
    # GIVEN the jail root path itself
    # WHEN validating access
    # THEN no exception is raised
    sec_policy.validate_path_access(sandbox)


def test_approved_tool_names_default_empty():
    # GIVEN a freshly created policy
    policy = ToolSecurityPolicy(fs_root_jail=Path("/tmp"))

    # THEN the approved tool names set is empty
    assert policy.approved_tool_names == set()
