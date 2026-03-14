"""The CLI entry point module of the application."""

import argparse
import os
import sys
from argparse import Namespace
from importlib.metadata import metadata
from pathlib import Path

from ordnung.entities import DEFAULT_MAX_ITERATIONS
from ordnung.entities import LLMAPIMode
from ordnung.organize import organize
from ordnung.tui import print_interrupted_message


def parse_args() -> Namespace:
    """Parse the command line arguments."""
    program_meta = metadata("ordnung")
    parser = argparse.ArgumentParser(
        prog=program_meta["Name"],
        description=program_meta["Summary"],
    )
    parser.add_argument(
        "directory",
        help="The path to the directory to organize.",
        type=Path,
    )
    parser.add_argument(
        "--llm-api-base-url",
        help="The base URL of the OpenAI-compatible LLM API. Defaults to local Ollama.",
        type=str,
        required=False,
        default="http://localhost:11434/v1",
    )
    parser.add_argument(
        "--llm-api-key",
        help="The API key for LLM API authentication. Falls back to OPENAI_API_KEY env var.",
        type=str,
        required=False,
        default=os.environ.get("OPENAI_API_KEY", "ollama"),
    )
    parser.add_argument(
        "--llm-name",
        help="The name of the LLM to use in the LLM API.",
        type=str,
        required=False,
        default="gpt-oss:20b",
    )
    parser.add_argument(
        "--llm-api-mode",
        help="The LLM API mode: 'responses' (default) or 'completions'.",
        type=LLMAPIMode,
        required=False,
        choices=list(LLMAPIMode),
        default=LLMAPIMode.RESPONSES,
    )
    parser.add_argument(
        "--max-iterations",
        help=(
            "The maximum allowed number of agentic loop iterations before aborting"
            f" (default: {DEFAULT_MAX_ITERATIONS})."
        ),
        type=int,
        required=False,
        default=DEFAULT_MAX_ITERATIONS,
    )
    parser.add_argument(
        "--skip-permissions",
        help="Pre-approve all tools so the agent runs without interactive permission prompts.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def main() -> None:
    """Run the entry point of the application."""
    args = parse_args()

    try:
        result = organize(
            dir_path=args.directory,
            llm_api_base_url=args.llm_api_base_url,
            llm_api_key=args.llm_api_key,
            llm_name=args.llm_name,
            llm_api_mode=args.llm_api_mode,
            max_iterations=args.max_iterations,
            skip_permissions=args.skip_permissions,
        )
    except KeyboardInterrupt:
        print_interrupted_message()
        sys.exit(130)

    if not result.is_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
