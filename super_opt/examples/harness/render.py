# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
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
