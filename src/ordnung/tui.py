"""Terminal UI utilities (colorized outputs, spinners, prompts, etc.)."""

from rich.console import Console
from rich.prompt import Prompt
from rich.status import Status

from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec

# A reusable console object for all functions in this module.
console = Console()


def print_task_spec(task_spec: OrganizeDirectoryTaskSpec) -> None:
    """Print the initial task specification."""
    console.print(
        f"✨ Organizing directory: [bold]{task_spec.dir_path}[/bold]",
        style="bright_cyan",
    )


def print_llm_api_details(base_url: str, name: str) -> None:
    """Print the details of the currently used LLM API."""
    console.print(
        f"🤖 Using LLM: [bold]{name}[/bold], API: [bold]{base_url}[/bold]",
        style="bright_blue",
    )


def calling_llm_spinner() -> Status:
    """Show a temporary "loading" spinner while the LLM request is running."""
    return console.status("[red]Calling the LLM...[/red]", spinner_style="red")


def print_reasoning(summary: str) -> None:
    """Print the reasoning summary returned by the LLM."""
    console.print()
    console.print(f"💭 {summary}", style="bright_black")


def print_tool_call_request(name: str, args: dict) -> None:
    """Print the tool call requested by the LLM."""
    console.print()
    console.print(f"🛠️ {name}", style="bright_magenta bold")
    console.print(args)


def approval_prompt() -> str:
    """Ask the user for tool call approval."""
    return Prompt.ask(
        prompt="Approve (yes, no, q=quit, a=always approve this tool, f=your own feedback)?",
        choices=["Y", "n", "q", "a", "f"],
        case_sensitive=False,
        default="Y",
    )


def user_feedback_prompt() -> str:
    """Ask the user for feedback if they chose this option instead of approving the tool call."""
    return Prompt.ask(prompt="Type your feedback")


def print_tool_auto_approval_update(tool_name: str) -> None:
    """Print a security policy update message."""
    console.print(
        f"🔓 Tool [bold]{tool_name}[/bold] added to auto-approval list",
        style="bright_yellow",
    )


def print_tool_result(result: dict) -> None:
    """Print the results of tool execution."""
    console.print("🔨 Tool output:")
    console.print(result)


def print_content_response(content: str) -> None:
    """Print a content response from the LLM."""
    console.print(f"💬 {content}", style="white")


def print_final_result(result: OrganizeDirectoryResult) -> None:
    """Print the final result of agent execution."""
    if result.is_success:
        console.print("✅ Task completed successfully!", style="bright_green")
    else:
        console.print(f"❌ Task failed with error: {result.error}", style="bright_red")


def print_interrupted_message() -> None:
    """Print the message indicating that the user interrupted the program."""
    console.print("🛑 Interrupted by the user", style="bright_red")
