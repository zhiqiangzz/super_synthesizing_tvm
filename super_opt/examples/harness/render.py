"""Rich primitives shared by the examples: source panels and result tables.

The harness never inspects a row itself. A table is described by its columns,
and each column carries a callable that pulls its own cell out of whatever row
type the example happens to use -- so an example can report any fields it likes
without this module needing to know about them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


def render_source(console: Console, source: str, lexer: str, title: str, subtitle: str) -> None:
    """Print a syntax-highlighted panel -- unscheduled TIR, scheduled TIR, CUDA."""
    console.print(
        Panel(
            Syntax(source, lexer, theme="ansi_dark", word_wrap=True),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            padding=(1, 2),
        )
    )


@dataclasses.dataclass(frozen=True)
class Column:
    """One table column: a header plus how to get its cell out of a row."""

    header: str
    render: Callable[[Any], str]
    justify: str = "right"


def label(header: str, get: Callable[[Any], str]) -> Column:
    """A left-justified text column -- schedule names, dtypes, output names."""
    return Column(header, get, justify="left")


def verdict(passed: bool) -> str:
    return "[green]PASS[/]" if passed else "[red]FAIL[/]"


def render_table(
    console: Console,
    *,
    title: str,
    caption: str,
    columns: Sequence[Column],
    rows: Iterable[Any],
) -> None:
    table = Table(
        title=title,
        caption=caption,
        box=box.SIMPLE_HEAVY,
        title_style="bold",
        header_style="bold cyan",
    )
    for column in columns:
        table.add_column(column.header, justify=column.justify)
    for row in rows:
        table.add_row(*(column.render(row) for column in columns))
    console.print(table)
