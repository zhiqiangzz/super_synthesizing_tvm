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
"""Does a single pass exist for a chain, and how many states does it take?

This is decided from the program, by algebra, before anything is derived. A
link of a chain is a reduction ``⊕_j f(x_j, κ)`` over the axis whose body
reads contexts ``κ``: results of earlier links, known only once their pass
is over. One statement covers a sum and a maximum alike. Call ``φ`` an
operation that *commutes* with the reduction, ``φ(a ⊕ b) = φ(a) ⊕ φ(b)``:

sum
    multiplying by something that does not move with ``j``, and nothing else;
max / min
    any function that grows with its argument: adding something that does
    not move with ``j``, multiplying by something that cannot be negative,
    ``exp``, ``log``, a root, an odd power. An even power and an absolute
    value grow with ``|u|``, the larger of ``u`` and ``-u``.

If the body is ``f(x, κ) = φ_1(g_1(x)) ⊕ ... ⊕ φ_r(g_r(x))`` with every
``g_k`` free of the contexts and every ``φ_k`` (which may contain them)
commuting with ``⊕``, then::

    ⊕_j f(x_j, κ) = ⊕_k φ_k(⊕_j g_k(x_j))

by regrouping (``⊕`` is associative and commutative) and taking each ``φ_k``
out. The ``r`` reductions ``⊕_j g_k(x_j)`` read no context: they can be
accumulated while the contexts are still unknown, and merged over two parts
of the axis by ``⊕`` itself. ``Σ_j (x_j - mean)²`` is
``Σ x² - 2 mean Σ x + mean² Σ 1``, three sums; ``max_j (x_j - mean)²`` is
the square of the larger of ``max x - mean`` and ``mean + max -x``, two
maxima.

:func:`judge` applies this to every member a chain is required for and to
the members those read. The canonical constructors of
:mod:`..symbolic.canonicalize` are the rules above; a context is handed to
them as a symbol that does not move with ``j``, and what they leave under a
reduction decides the answer:

* nothing but data: the reduction is one of the states of a single pass.
  All of them together, plus the number of elements where the program uses
  the extent of the axis as a value, are enough: a reducer with that many
  states exists, and :mod:`states` goes on to look for one whose arithmetic
  is as sound as the program's;
* a context: the rules do not take it out, and no single pass is known.

The first answer is a proof, by construction. The second is not a proof of
the contrary: ``Σ |x - max x|`` has a single pass (``|x - max x|`` is
``max x - x``, a maximum is never below its elements) that these rules do
not see. Nothing here is sampled or measured.
"""

from __future__ import annotations

import dataclasses

from ..symbolic import ir
from ..symbolic.canonicalize import NONNEG, map_elems, nonneg_alias, nonnegative, walk
from .chain import Chain
from .derive import atoms_in, domains
from .rebase import Candidate, Rebaser
from .source import render


@dataclasses.dataclass(frozen=True)
class Stuck:
    """A reduction that a context does not leave."""

    member: str
    contexts: tuple[str, ...]
    reduction: str  # as text

    def summary(self) -> str:
        return (
            f"{self.member} reads {', '.join(self.contexts)} inside {self.reduction}, "
            "and no rule takes it out of the reduction"
        )


@dataclasses.dataclass(frozen=True)
class Judgement:
    """What the theory says about one chain.

    ``reductions`` are the reductions over the axis, free of every context,
    that the required members are functions of; ``counted`` says that the
    number of elements reduced is needed next to them.
    """

    reductions: tuple[str, ...] = ()  # as text
    counted: bool = False
    stuck: tuple[Stuck, ...] = ()

    @property
    def fusible(self) -> bool:
        """Is a single pass with finitely many states shown to exist?"""
        return not self.stuck

    @property
    def states(self) -> int | None:
        """How many states are enough (``None``: no single pass is known)."""
        return None if self.stuck else len(self.reductions) + int(self.counted)

    def summary(self) -> str:
        if self.stuck:
            return "no single pass is known: " + "; ".join(s.summary() for s in self.stuck)
        kept = list(self.reductions) + (["the count"] if self.counted else [])
        return f"a single pass with {self.states} state(s) exists, keeping {', '.join(kept)}"


def judge(chain: Chain, members: list[Candidate]) -> Judgement:
    """Judge ``chain`` from its ``members`` run on a sub-range (see :func:`states.members_of`)."""
    rebaser = Rebaser(members, chain.axis)
    by_pid = rebaser.by_pid
    _, R, _, _ = domains(chain.axis)
    known = {pid + NONNEG: pid for pid in by_pid}  # a context read as a value that is >= 0

    def member_of(n: ir.Node) -> Candidate | None:
        if not isinstance(n, ir.Elem):
            return None
        return by_pid.get(known.get(n.tensor, n.tensor))

    def free(n: ir.Elem, depth: int):
        """A context as a symbol that does not move with the axis."""
        if not rebaser.pinned(n):
            return None
        out = ir.elem(n.tensor, n.indices[:-1])
        return nonneg_alias(out) if nonnegative(by_pid[n.tensor].state) else out

    def plain(n: ir.Elem, depth: int):
        return ir.elem(known[n.tensor], n.indices) if n.tensor in known else None

    reductions: dict[ir.SymExpr, None] = {}
    stuck: list[Stuck] = []
    counted = False
    done: set[int] = set()
    todo = [by_pid[m.pid] for m in chain.required]
    while todo:
        cand = todo.pop(0)
        if id(cand) in done:
            continue
        done.add(id(cand))
        apart = map_elems(cand.struct, free)
        todo.extend(m for m in map(member_of, walk(apart)) if m is not None)
        for red in atoms_in(apart, R):
            if isinstance(red, ir.Card):
                counted = True
                continue
            inside = dict.fromkeys(m.name for m in map(member_of, walk(red.body)) if m is not None)
            if inside:
                text = render(map_elems(red, plain), chain)
                stuck.append(Stuck(cand.name, tuple(inside), text))
            else:
                reductions[red] = None
    return Judgement(tuple(render(r, chain) for r in reductions), counted, tuple(stuck))


__all__ = ["Judgement", "Stuck", "judge"]
