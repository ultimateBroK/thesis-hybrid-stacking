"""Shared Rich UI primitives for the thesis pipeline.

Provides a single Console instance, styled Progress factories,
and helper functions for consistent terminal output across all stages.
"""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel

from rich.text import Text

# ---------------------------------------------------------------------------
# Singleton console — every module imports this for consistent rendering
# ---------------------------------------------------------------------------
console = Console()

# ---------------------------------------------------------------------------
# Stage colour map (used by pipeline + training)
# ---------------------------------------------------------------------------
STAGE_STYLES: dict[int, str] = {
    0: "bold blue",
    1: "bold green",
    2: "bold yellow",
    3: "bold cyan",
    4: "bold magenta",
    5: "bold red",
    6: "bold white",
}

STAGE_LABELS: dict[int, str] = {
    0: "Data Preparation",
    1: "Feature Engineering",
    2: "Triple-Barrier Labeling",
    3: "Data Splitting",
    4: "Model Training",
    5: "Backtest",
    6: "Report Generation",
}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def stage_header(stage: int, total: int = 6) -> None:
    """Print a visually distinct stage banner."""
    style = STAGE_STYLES.get(stage, "bold")
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    console.print()
    console.rule(
        Text(f"  STAGE {stage}/{total}  ·  {label}  ", style=style),
        style=style,
        characters="─",
    )
    console.print()


def stage_skip(stage: int, reason: str) -> None:
    """Print a dim skip line."""
    label = STAGE_LABELS.get(stage, f"Stage {stage}")
    console.print(Text(f"  ⊘ SKIP {label}: {reason}", style="dim"))


def training_progress(label: str, total: float, **fields: float) -> Progress:
    """Create a styled Progress bar for training loops.

    Returns an *unstarted* Progress — caller uses ``with progress:``.
    Extra keyword arguments become live-updating fields in the bar.
    """
    columns: list = [
        SpinnerColumn(spinner_name="dots"),
        TextColumn(f"[bold]{label}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        "•",
    ]
    for key in fields:
        if "loss" in key:
            columns.append(TextColumn(f"[cyan]{key}={{task.fields[{key!r}]:.4f}}"))
        elif "acc" in key or "f1" in key:
            columns.append(TextColumn(f"[green]{key}={{task.fields[{key!r}]:.3f}}"))
        else:
            columns.append(TextColumn(f"{{task.fields[{key!r}]}}"))
    columns.append(TimeElapsedColumn())
    return Progress(
        *columns,
        console=console,
        transient=False,
    )


def result_table(
    title: str, rows: list[dict[str, str]], styles: dict[str, str] | None = None
) -> None:
    """Print a compact results table."""
    if not rows:
        return
    table = Table(
        title=title, show_header=True, header_style="bold", border_style="dim"
    )
    styles = styles or {}
    for col in rows[0]:
        table.add_column(col, style=styles.get(col, ""))
    for row in rows:
        table.add_row(*[row[c] for c in row])
    console.print(table)


def summary_panel(title: str, lines: list[str], style: str = "green") -> None:
    """Print a compact summary panel."""
    console.print(
        Panel(
            "\n".join(lines),
            title=title,
            style=style,
            border_style=style,
            padding=(0, 2),
        )
    )
