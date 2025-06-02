from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def warn(message):
    """Prints the warning message in a panel."""
    console = Console()
    console.print(
        Panel(
            Text(message, style="bold yellow"),
            title="⚠️ WARNING",
            title_align="left",
            border_style="yellow",
            expand=False,
        )
    )
