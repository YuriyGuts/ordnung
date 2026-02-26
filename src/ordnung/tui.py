from rich.console import Console
from rich.prompt import Prompt
from rich.status import Status

from ordnung.entities import OrganizeDirectoryResult
from ordnung.entities import OrganizeDirectoryTaskSpec

console = Console()


def print_task_spec(task_spec: OrganizeDirectoryTaskSpec) -> None:
    console.print(f"✨ Organizing directory: {task_spec.dir_path}", style="bright_cyan")


def calling_llm_spinner() -> Status:
    return console.status("[red]Calling the LLM...[/red]", spinner_style="red")


def print_reasoning(summary: str) -> None:
    console.print(f"💭 {summary}", style="bright_black")


def print_tool_call_request(name: str, args: dict) -> None:
    console.print()
    console.print(f"🛠️ {name}", style="bright_magenta bold")
    console.print(args)


def approval_prompt() -> str:
    return Prompt.ask(
        prompt="Approve (Y, n, q=quit, a=approve all, f=your own feedback)?",
        choices=["Y", "n", "q", "a", "f"],
        case_sensitive=False,
        default="Y",
    )


def user_feedback_prompt() -> str:
    return Prompt.ask(prompt="Type your feedback:")


def print_tool_result(result: dict) -> None:
    console.print("🔨 Tool output:")
    console.print(result)


def print_final_result(result: OrganizeDirectoryResult) -> None:
    if result.is_success:
        console.print("✅ Task completed successfully!", style="bright_green")
    else:
        console.print(f"❌ Task failed with error: {result.error}", style="bright_red")
