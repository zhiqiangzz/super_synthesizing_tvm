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
"""Probes: does a single pass with finitely many states exist at all?

The derivation of :mod:`states` either finds a reducer or it does not, and
"not found" says nothing about whether one exists. This module asks the
question the other way round, from the program alone. A later link of a
chain is a reduction whose body reads a *context* ``κ``, the final value of
an earlier link, and what stands in the way of one pass is that ``κ`` is not
known while the elements go by. What has to be carried along depends on the
kind of the reduction:

sum
    ``Σ_j f(x_j, κ)`` can be accumulated without knowing ``κ`` exactly when
    ``f`` separates, ``f(x, κ) = Σ_k g_k(x) h_k(κ)``: then
    ``Σ_j f = Σ_k h_k(κ) Σ_j g_k(x_j)`` and the ``Σ_j g_k`` are states. The
    least number of terms is the *rank* of ``f`` as a table over elements and
    contexts: 1 for ``exp(x - κ)``, 3 for ``(x - κ)²``, unbounded for
    ``|x - κ|`` or ``exp(x / κ)``. A table whose rank keeps up with its size
    rules out every reducer made of that many accumulated sums.
max / min
    A maximum does not commute with sums, and the rank says nothing about
    it. ``max_j f(x_j, κ)`` needs no ``κ`` while the elements go by when the
    element attaining it does not depend on ``κ``: for ``(x - mean) / sigma``
    it is always the largest ``x``, for ``|x - mean|`` the largest or the
    smallest. Then those extremes of the data are the states. When the
    winner moves with ``κ`` through the data (``κ x - x²``), every element
    has to be kept.

Both are decided on *probes*: a handful of elements ``x_i`` and of data sets
that all contain them and differ in what else they hold. Running the chain
on data set ``k`` gives its contexts ``κ_k``, so every pair ``(x_i, κ_k)`` in
the table ``f(x_i, κ_k)`` is one the program can actually meet -- a maximum
is never below the elements it is read with, a variance never negative. On
that table the rank is computed in high-precision arithmetic (in ``float64``
an analytic kernel of unbounded rank looks like one of rank ten), and the
winners are counted.

What a verdict means: a rank ``r`` is evidence that ``r`` accumulated sums
suffice in exact arithmetic (it says nothing about floating point: the sums
it stands for may overflow or cancel); a rank above the budget is a proof,
up to the precision of the probes, that no reducer of that many accumulated
sums exists; the count of winners is evidence either way.
"""

from __future__ import annotations

import dataclasses
import decimal
import random
from decimal import Decimal

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, walk
from .chain import Chain, Member, extent_symbol

PRECISION = 240  # decimal digits of the probe arithmetic
TOLERANCE = Decimal(10) ** -100  # a pivot below this, relative to the table, is zero
SHARED = 16  # probe elements, in every data set
EXTRA = 6  # elements a data set holds besides them
DATASETS = 24  # data sets, i.e. contexts
MAX_EXTREMES = 4  # a maximum attained by more elements than this is not an extreme


@dataclasses.dataclass(frozen=True)
class Verdict:
    """What the probes say about one reduction that reads a context."""

    member: str
    kind: str  # "sum", "max" or "min"
    contexts: tuple[str, ...]  # the members it reads from inside its reduction
    fusible: bool | None  # None: the probes could not decide
    rank: int | None = None  # sum: rank of its kernel; max/min: elements that can attain it
    extremes: tuple[str, ...] = ()  # max/min: the extremes of the data that decide it
    note: str = ""

    def summary(self) -> str:
        reads = ", ".join(self.contexts)
        head = f"{self.member} ({self.kind} reading {reads})"
        if self.fusible is None:
            return f"{head}: undecided ({self.note})"
        if self.kind == "sum":
            if self.fusible:
                return (
                    f"{head}: separates into {self.rank} term(s), "
                    f"a single pass with {self.rank} accumulated sum(s) exists"
                )
            return (
                f"{head}: does not separate ({self.note}), no single pass with finitely many sums"
            )
        if self.fusible:
            what = " or ".join(self.extremes) if self.extremes else f"{self.rank} element(s)"
            return f"{head}: always attained at {what}, a single pass keeping that exists"
        return f"{head}: attained at an element that moves with {reads} ({self.note})"


@dataclasses.dataclass(frozen=True)
class Theory:
    """The verdicts on one chain: one per reduction that reads a context."""

    verdicts: tuple[Verdict, ...]
    budget: int
    note: str = ""  # why nothing could be probed

    @property
    def fusible(self) -> bool | None:
        """Does a single pass with finitely many states exist (``None``: undecided)?"""
        if any(v.fusible is False for v in self.verdicts):
            return False
        if self.note or any(v.fusible is None for v in self.verdicts):
            return None
        return True

    def verdict(self, member: str) -> Verdict | None:
        return next((v for v in self.verdicts if v.member == member), None)

    def summary(self) -> str:
        if self.note:
            return f"not probed ({self.note})"
        return "; ".join(v.summary() for v in self.verdicts)


class _Undefined(Exception):
    """The sample is outside the domain of the program (log of a negative number, 1 / 0)."""


def _moving(n: ir.Elem) -> bool:
    return any(isinstance(i, ir.BIdx) for i in n.indices)


class _Probes:
    """One sampling of a chain: shared elements, data sets around them, the contexts they give."""

    def __init__(self, chain: Chain, rng: random.Random, positive: bool) -> None:
        self.chain = chain
        self.by_pid: dict[int, Member] = {m.pid: m for m in chain.members}
        self.full = ir.dfull(chain.axis)
        self.extent = extent_symbol(chain.axis)
        slots: dict[ir.Elem, None] = {}  # boundary elements that move with j
        fixed: dict[ir.Node, None] = {}  # boundary elements that do not, and other extents
        for m in chain.members:
            for n in walk(m.body):
                if isinstance(n, ir.Elem) and n.tensor not in self.by_pid:
                    (slots if _moving(n) else fixed)[n] = None
                elif isinstance(n, ir.ShapeSym) and n is not self.extent:
                    fixed[n] = None
        self.slots = list(slots)

        def draw(scale: float = 1.0, centre: float = 0.0) -> Decimal:
            v = centre + scale * rng.uniform(-1.0, 1.0)
            return Decimal(abs(v) + 0.1 if positive else v)

        self.params: dict[ir.Node, Decimal] = {}
        for k, n in enumerate(fixed):
            self.params[n] = Decimal(3 + k) if isinstance(n, ir.ShapeSym) else draw()
        self.shared = [{s: draw() for s in self.slots} for _ in range(SHARED)]
        self.count = Decimal(SHARED + EXTRA)
        self.datasets: list[list[dict]] = []
        for _ in range(DATASETS):  # wide and far apart, so that the contexts spread
            scale = 2.0 ** rng.uniform(-1.0, 4.0)
            centre = scale * rng.uniform(-1.0, 1.0)
            extra = [{s: draw(scale, centre) for s in self.slots} for _ in range(EXTRA)]
            self.datasets.append(self.shared + extra)
        # the value of every member on every data set: its contexts
        self.values: list[dict[int, Decimal]] = []
        for data in self.datasets:
            values: dict[int, Decimal] = {}
            for m in chain.members:
                values[m.pid] = self._eval(m.body, values, None, data)
            self.values.append(values)

    # -- evaluation ----------------------------------------------------------------
    def _eval(self, e: ir.SymExpr, values, element, data, memo: dict | None = None) -> Decimal:
        """``e`` with member reads from ``values``; a reduction over the chain axis runs
        over ``data``, or is its body at ``element`` when ``data`` is ``None``."""
        memo = {} if memo is None else memo
        hit = memo.get(e)
        if hit is not None:
            return hit

        def go(x):
            return self._eval(x, values, element, data, memo)

        try:
            if isinstance(e, ir.Const):
                v = e.value
                if isinstance(v, float):
                    out = Decimal("Infinity") if v > 0 else Decimal("-Infinity")
                else:
                    out = Decimal(v.numerator) / Decimal(v.denominator)
            elif isinstance(e, ir.ShapeSym):
                out = self.count if e is self.extent else self.params[e]
            elif isinstance(e, ir.Card):
                out = self.count
            elif isinstance(e, ir.Elem):
                if e.tensor in self.by_pid:
                    out = values[e.tensor]
                elif _moving(e):
                    if element is None:
                        raise Unsupported("an element read outside a reduction over the axis")
                    out = element[e]
                else:
                    out = self.params[e]
            elif isinstance(e, ir.Add):
                out = sum((go(a) for a in e.args), Decimal(0))
            elif isinstance(e, ir.Mul):
                out = Decimal(1)
                for a in e.args:
                    out *= go(a)
            elif isinstance(e, ir.Pow):
                out = _power(go(e.base), e.exponent)
            elif isinstance(e, ir.Exp):
                out = go(e.arg).exp()
            elif isinstance(e, ir.Log):
                arg = go(e.arg)
                if arg <= 0:
                    raise _Undefined
                out = arg.ln()
            elif isinstance(e, ir.Max):
                out = max(go(a) for a in e.args)
            elif isinstance(e, ir.Reduce) and e.domain is self.full:
                if data is None:  # the contribution of one element
                    out = self._eval(e.body, values, element, None, memo)
                else:
                    terms = [self._eval(e.body, values, x, None) for x in data]
                    out = sum(terms, Decimal(0)) if e.kind == "sum" else max(terms)
            else:
                raise Unsupported(f"cannot probe {type(e).__name__}")
        except decimal.DecimalException as err:  # inf - inf, overflow of exp, 0 / 0
            raise _Undefined from err
        memo[e] = out
        return out

    # -- the tables ----------------------------------------------------------------
    def contexts(self, e: ir.SymExpr) -> list[int]:
        """Members read from inside the reductions of ``e`` over the chain axis."""
        inside: set[int] = set()
        for r in self.reductions(e):
            inside |= {n.tensor for n in walk(r.body) if isinstance(n, ir.Elem)}
        return sorted(inside & set(self.by_pid), reverse=True)

    def reductions(self, e: ir.SymExpr) -> list[ir.Reduce]:
        return [n for n in walk(e) if isinstance(n, ir.Reduce) and n.domain is self.full]

    def columns(self, pids: list[int]) -> list[int]:
        """Data sets with pairwise different values of the contexts ``pids``."""
        seen: set[tuple] = set()
        out = []
        for k, values in enumerate(self.values):
            key = tuple(format(values[p], ".40e") for p in pids)
            if key not in seen:
                seen.add(key)
                out.append(k)
        return out

    def table(self, e: ir.SymExpr, columns: list[int]) -> list[list[Decimal]]:
        """``e`` at every shared element (rows) under the contexts of ``columns``."""
        return [[self._eval(e, self.values[k], x, None) for k in columns] for x in self.shared]

    def extreme_of(self, row: int) -> str | None:
        """``"max X"`` / ``"min X"`` when shared element ``row`` is that extreme of an input."""
        for slot in self.slots:
            column = [x[slot] for x in self.shared]
            name = self.chain.lower.placeholder(slot.tensor).op.name
            if self.shared[row][slot] == max(column):
                return f"the largest {name}"
            if self.shared[row][slot] == min(column):
                return f"the smallest {name}"
        return None


def _power(base: Decimal, p) -> Decimal:
    if p.denominator == 1:
        if base == 0 and p < 0:
            raise _Undefined
        return base ** int(p)
    if base < 0 or (base == 0 and p < 0):
        raise _Undefined  # a root of a negative number
    if base == 0:
        return Decimal(0)
    root = base.sqrt() if p.denominator == 2 else base ** (Decimal(1) / Decimal(p.denominator))
    return root ** int(p.numerator)


def _rank(table: list[list[Decimal]]) -> int:
    """Rank by elimination; a pivot ``TOLERANCE`` below the largest entry is zero."""
    rows = [r[:] for r in table]
    scale = max((abs(v) for r in rows for v in r), default=Decimal(0))
    if scale == 0:
        return 0
    tol = scale * TOLERANCE
    rank = 0
    for col in range(len(rows[0])):
        if rank == len(rows):
            break
        pivot = max(range(rank, len(rows)), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) <= tol:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        lead = rows[rank]
        for r in range(rank + 1, len(rows)):
            f = rows[r][col] / lead[col]
            if f:
                rows[r] = [a - f * b for a, b in zip(rows[r], lead)]
        rank += 1
    return rank


def _judge_sum(p: _Probes, m: Member, pids: list[int], names, budget: int) -> Verdict:
    columns = p.columns(pids)
    if len(columns) == 1:
        return Verdict(m.name, m.kind, names, True, 1, note="its contexts do not vary")
    columns = columns[: budget + 2]
    rank = _rank(p.table(m.body, columns))
    if rank <= budget and rank < len(columns):
        return Verdict(m.name, m.kind, names, True, rank)
    if rank > budget:
        return Verdict(m.name, m.kind, names, False, note=f"rank above {budget} on the probes")
    return Verdict(m.name, m.kind, names, None, note=f"only {len(columns)} different contexts")


def _judge_order(p: _Probes, m: Member, pids: list[int], names, reds) -> Verdict:
    columns = p.columns(pids)
    winners: dict[int, None] = {}
    for r in reds:
        table = p.table(r.body, columns)
        for k in range(len(columns)):
            winners[max(range(SHARED), key=lambda i, k=k: table[i][k])] = None
    labels = [p.extreme_of(i) for i in winners]
    count = len(winners)
    if all(labels):
        return Verdict(m.name, m.kind, names, True, count, tuple(dict.fromkeys(labels)))
    if count <= MAX_EXTREMES:
        note = "decided by elements that are not an extreme of one input"
        return Verdict(m.name, m.kind, names, True, count, note=note)
    note = f"{count} of {SHARED} probe elements attain it"
    return Verdict(m.name, m.kind, names, False, note=note)


def judge(chain: Chain, *, budget: int = 8, seed: int = 0) -> Theory:
    """Probe every reduction of ``chain`` that reads a context.

    ``budget`` is the largest number of accumulated sums considered finite:
    a kernel of higher rank on the probes counts as not separable.
    """
    rng = random.Random(seed)
    with decimal.localcontext() as ctx:
        ctx.prec = PRECISION
        probes, why = None, ""
        for positive in (False, True):  # then on positive data: under a log, a root, a divisor
            try:
                probes = _Probes(chain, rng, positive)
                break
            except _Undefined:
                why = "no sample on which the program is defined"
            except Unsupported as err:
                return Theory((), budget, str(err))
        if probes is None:
            return Theory((), budget, why)
        verdicts = []
        for m in chain.members:
            if not m.is_reduction:
                continue
            pids = probes.contexts(m.body)
            if not pids:
                continue  # nothing to wait for: it can be accumulated as it is
            names = tuple(probes.by_pid[q].name for q in pids)
            reds = [r for r in probes.reductions(m.body) if probes.contexts(r)]
            try:
                if all(r.kind == "sum" for r in reds):
                    verdicts.append(_judge_sum(probes, m, pids, names, budget))
                elif all(r.kind == "max" for r in reds):
                    verdicts.append(_judge_order(probes, m, pids, names, reds))
                else:
                    note = "sums and maxima over the axis in one reduction"
                    verdicts.append(Verdict(m.name, m.kind, names, None, note=note))
            except (_Undefined, Unsupported) as err:
                note = str(err) or "the probes leave its domain"
                verdicts.append(Verdict(m.name, m.kind, names, None, note=note))
    return Theory(tuple(verdicts), budget)


__all__ = ["Theory", "Verdict", "judge"]
