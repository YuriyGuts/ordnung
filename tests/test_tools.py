"""Tests for ordnung.tools."""

from pathlib import Path

import pytest

from ordnung.tools import CreateDirectoryTool
from ordnung.tools import ListDirectoryTool
from ordnung.tools import MoveFileOrDirectoryTool
from ordnung.tools import ReadBinaryFileTool
from ordnung.tools import ReadTextFileTool


class TestToolBase:
    def test_get_name_returns_class_name(self):
        # GIVEN a concrete tool class
        # WHEN getting the name
        name = ListDirectoryTool.get_name()

        # THEN it returns the class name
        assert name == "ListDirectoryTool"

    def test_get_description_returns_docstring(self):
        # GIVEN a concrete tool class
        # WHEN getting the description
        description = ListDirectoryTool.get_description()

        # THEN it returns the docstring
        assert description == "Lists the contents of the specified directory."

    def test_to_schema(self):
        # GIVEN a concrete tool class
        # WHEN converting to a tool schema
        result = ListDirectoryTool.to_schema()

        # THEN it returns name, description, and parameters without API-specific keys
        assert result == {
            "name": "ListDirectoryTool",
            "description": "Lists the contents of the specified directory.",
            "parameters": {
                "properties": {
                    "dir_path": {
                        "description": "The path to the directory to list the contents of.",
                        "format": "path",
                        "type": "string",
                    }
                },
                "required": ["dir_path"],
                "type": "object",
            },
        }


class TestListDirectoryTool:
    def test_with_files_and_dirs(self, sec_policy, populated_sandbox):
        # GIVEN a directory with files and subdirectories
        tool = ListDirectoryTool(dir_path=populated_sandbox)

        # WHEN listing the directory
        result = tool.run(sec_policy)

        # THEN it returns all items with correct types and sizes
        items = sorted(result["output"], key=lambda i: i["name"])
        assert items == [
            {"type": "file", "name": "data.bin", "size_bytes": 4},
            {"type": "file", "name": "readme.txt", "size_bytes": 12},
            {"type": "directory", "name": "subdir", "size_bytes": None},
        ]

    def test_empty(self, sec_policy, sandbox):
        # GIVEN an empty directory
        tool = ListDirectoryTool(dir_path=sandbox)

        # WHEN listing it
        result = tool.run(sec_policy)

        # THEN output is empty
        assert result["output"] == []

    def test_includes_size(self, sec_policy, populated_sandbox):
        # GIVEN a directory with a known file
        tool = ListDirectoryTool(dir_path=populated_sandbox)

        # WHEN listing the directory
        result = tool.run(sec_policy)

        # THEN size_bytes is int for files, null for directories
        for item in result["output"]:
            assert "size_bytes" in item
            if item["type"] == "file":
                assert isinstance(item["size_bytes"], int)
            else:
                assert item["size_bytes"] is None

    def test_outside_jail(self, sec_policy):
        # GIVEN a path outside the jail
        tool = ListDirectoryTool(dir_path=Path("/etc"))

        # WHEN running the tool
        # THEN security policy rejects it
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)


class TestCreateDirectoryTool:
    def test_create(self, sec_policy, sandbox):
        # GIVEN a new directory path
        new_dir = sandbox / "new_folder"
        tool = CreateDirectoryTool(dir_path=new_dir)

        # WHEN creating the directory
        result = tool.run(sec_policy)

        # THEN the directory exists and the result indicates success
        assert result == {"created": True}
        assert new_dir.is_dir()

    def test_nested(self, sec_policy, sandbox):
        # GIVEN a deeply nested new directory path
        nested_dir = sandbox / "a" / "b" / "c"
        tool = CreateDirectoryTool(dir_path=nested_dir)

        # WHEN creating the directory
        result = tool.run(sec_policy)

        # THEN all parent directories are created
        assert result == {"created": True}
        assert nested_dir.is_dir()

    def test_already_exists(self, sec_policy, sandbox):
        # GIVEN an existing directory
        tool = CreateDirectoryTool(dir_path=sandbox)

        # WHEN creating it again
        result = tool.run(sec_policy)

        # THEN it succeeds without error (exist_ok=True)
        assert result == {"created": True}

    def test_outside_jail(self, sec_policy):
        # GIVEN a path outside the jail
        tool = CreateDirectoryTool(dir_path=Path("/tmp/evil"))

        # WHEN running the tool
        # THEN security policy rejects it
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)


class TestMoveFileOrDirectoryTool:
    def test_move_file(self, sec_policy, populated_sandbox):
        # GIVEN a file and a destination path
        src = populated_sandbox / "readme.txt"
        dst = populated_sandbox / "moved.txt"
        tool = MoveFileOrDirectoryTool(source_path=src, destination_path=dst)

        # WHEN moving the file
        result = tool.run(sec_policy)

        # THEN the file is moved
        assert result == {"moved": True}
        assert not src.exists()
        assert dst.read_text(encoding="utf-8") == "Hello world!"

    def test_into_subdirectory(self, sec_policy, populated_sandbox):
        # GIVEN a file and a subdirectory destination
        src = populated_sandbox / "readme.txt"
        dst = populated_sandbox / "subdir" / "readme.txt"
        tool = MoveFileOrDirectoryTool(source_path=src, destination_path=dst)

        # WHEN moving the file
        result = tool.run(sec_policy)

        # THEN the file exists in the new location
        assert result == {"moved": True}
        assert dst.exists()

    def test_source_outside_jail(self, sec_policy, sandbox):
        # GIVEN a source path outside the jail
        tool = MoveFileOrDirectoryTool(
            source_path=Path("/etc/passwd"), destination_path=sandbox / "x"
        )

        # WHEN running the tool
        # THEN security policy rejects the source path
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)

    def test_destination_outside_jail(self, sec_policy, populated_sandbox):
        # GIVEN a destination path outside the jail
        tool = MoveFileOrDirectoryTool(
            source_path=populated_sandbox / "readme.txt",
            destination_path=Path("/tmp/stolen.txt"),
        )

        # WHEN running the tool
        # THEN security policy rejects the destination path
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)


class TestReadTextFileTool:
    def test_read(self, sec_policy, populated_sandbox):
        # GIVEN a text file
        tool = ReadTextFileTool(file_path=populated_sandbox / "readme.txt")

        # WHEN reading it
        result = tool.run(sec_policy)

        # THEN the contents are returned
        assert result == {"contents": "Hello world!"}

    def test_outside_jail(self, sec_policy):
        # GIVEN a file outside the jail
        tool = ReadTextFileTool(file_path=Path("/etc/hostname"))

        # WHEN running the tool
        # THEN security policy rejects it
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)


class TestReadBinaryFileTool:
    def test_read(self, sec_policy, populated_sandbox):
        # GIVEN a binary file with known contents
        tool = ReadBinaryFileTool(file_path=populated_sandbox / "data.bin")

        # WHEN reading it
        result = tool.run(sec_policy)

        # THEN the hex contents are returned
        assert result == {"hex_contents": "00:01:02:ff"}

    def test_with_offset_and_limit(self, sec_policy, populated_sandbox):
        # GIVEN a binary file and an offset/limit
        tool = ReadBinaryFileTool(
            file_path=populated_sandbox / "data.bin",
            offset=1,
            limit=2,
        )

        # WHEN reading it
        result = tool.run(sec_policy)

        # THEN only the specified bytes are returned
        assert result == {"hex_contents": "01:02"}

    def test_limit_exceeds_max(self, sec_policy, populated_sandbox):
        # GIVEN a limit exceeding the maximum
        tool = ReadBinaryFileTool(
            file_path=populated_sandbox / "data.bin",
            limit=2000,
        )

        # WHEN running the tool
        # THEN a RuntimeError is raised
        with pytest.raises(RuntimeError, match="maximum allowed limit"):
            tool.run(sec_policy)

    def test_outside_jail(self, sec_policy):
        # GIVEN a file outside the jail
        tool = ReadBinaryFileTool(file_path=Path("/etc/hostname"))

        # WHEN running the tool
        # THEN security policy rejects it
        with pytest.raises(RuntimeError, match="Rejected by security policy"):
            tool.run(sec_policy)
