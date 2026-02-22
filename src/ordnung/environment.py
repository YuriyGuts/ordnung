from collections.abc import Sequence

from ordnung.tools import Tool


class Environment:
    def __init__(self, tools: Sequence[type[Tool]]) -> None:
        self.tools = tools
