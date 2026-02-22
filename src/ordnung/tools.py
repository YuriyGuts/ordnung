from abc import abstractmethod
from pathlib import Path

from pydantic import BaseModel
from pydantic import Field


class Tool(BaseModel):
    @classmethod
    def get_name(cls) -> str:
        return cls.__name__

    @classmethod
    def get_description(cls) -> str:
        return cls.__doc__ or ""

    @abstractmethod
    def run(self) -> dict:
        pass


class ListDirectoryTool(Tool):
    """Lists the contents of the specified directory."""

    dir_path: Path = Field(description="The path to the directory to list the contents of.")

    def run(self) -> dict:
        output_items = []
        for item in self.dir_path.iterdir():
            if item.is_file():
                item_type = "file"
            elif item.is_dir():
                item_type = "directory"
            elif item.is_symlink():
                item_type = "symlink"
            else:
                item_type = "unknown"

            item_output = {
                "type": item_type,
                "name": item.name,
                "size_bytes": item.stat().st_size,
            }
            output_items.append(item_output)

        return {"output": output_items}
