"""Common entity classes for the application."""

from dataclasses import dataclass
from dataclasses import field
from enum import StrEnum
from pathlib import Path
from typing import Any

# The default maximum number of agentic loop iterations before aborting.
# This exists to safeguard against cases when the LLM gets stuck in a hallucinated repetitive loop.
DEFAULT_MAX_ITERATIONS = 200


class LLMAPIMode(StrEnum):
    """The LLM API mode to use for communication."""

    RESPONSES = "responses"
    COMPLETIONS = "completions"


@dataclass
class LLMReasoning:
    """A reasoning/thinking output from the LLM."""

    text: str


@dataclass
class LLMToolCall:
    """A tool call request from the LLM."""

    call_id: str
    name: str
    # Raw JSON string.
    arguments: str


@dataclass
class LLMContentMessage:
    """A final content message from the LLM."""

    text: str
    is_refusal: bool


# Normalized LLM response types, independent from the API wire format.
LLMOutputItem = LLMReasoning | LLMToolCall | LLMContentMessage


@dataclass
class LLMResponse:
    """A normalized LLM response containing output items and the raw context for re-injection."""

    items: list[LLMOutputItem]
    raw_context: list[Any] = field(default_factory=list)


@dataclass
class OrganizeDirectoryTaskSpec:
    """Represents the input parameters of the task."""

    # The path to the directory to organize.
    dir_path: Path


@dataclass
class OrganizeDirectoryResult:
    """Represents the results of performing the task."""

    # Whether the agent execution was successful.
    is_success: bool

    # If specified, the error message returned by the agent.
    error: str | None = None
