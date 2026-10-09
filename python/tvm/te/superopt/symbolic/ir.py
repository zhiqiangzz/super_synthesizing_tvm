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
"""Hash-consed symbolic expression IR: the "shadow semantics" of a TE tensor.

Every node is interned, so structural equality is ``is`` and structural
hashing is ``id``-based and O(1). Nodes are immutable; they are only ever
built through the *canonicalising* constructors in
:mod:`tvm.te.superopt.symbolic.canonicalize`, which guarantee that two
expressions equal under the rewrite theory implemented there are the same
node.

Bound reduction indices use absolute de Bruijn *levels*: the reduction node
with ``d`` binder ancestors is at level ``d`` and binds :class:`BIdx` ``d``.
A :class:`TensorSem` body therefore always starts at level 0 and its free
output indices are :class:`Idx` ``i0 .. i{rank-1}``.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from typing import ClassVar

NEG_INF = float("-inf")
POS_INF = float("inf")

Number = Fraction | float  # float only ever holds +-inf

_INTERN: dict[tuple, Node] = {}
_UID = itertools.count()
_EXTENTS: dict[int, SymExpr] = {}  # DimKey -> extent as a real value (ShapeSym / Const)


def register_extent(key: int, extent: SymExpr) -> None:
    """Record the extent of reduction axis ``key`` (see ``DimTable``)."""
    _EXTENTS[key] = extent


def extent_of(key: int) -> SymExpr | None:
    return _EXTENTS.get(key)


def _intern(cls: type, fields: tuple) -> Node:
    key = (cls, fields)
    node = _INTERN.get(key)
    if node is None:
        node = object.__new__(cls)
        node._fields = fields
        node.uid = next(_UID)
        node._post_init()
        _INTERN[key] = node
    return node


class Node:
    """Base of every IR node. Instances are interned; do not construct directly."""

    __slots__ = ("_fields", "cache", "free_idx", "levels", "size", "uid")
    rank: ClassVar[int] = 0
    field_names: ClassVar[tuple[str, ...]] = ()

    def _post_init(self) -> None:
        children = self.children()
        self.size = 1 + sum(c.size for c in children)
        levels: set[int] = set()
        free: set[str] = set()
        for c in children:
            levels |= c.levels
            free |= c.free_idx
        self.levels = frozenset(levels)
        self.free_idx = frozenset(free)
        self.cache = {}

    def children(self) -> tuple[Node, ...]:
        out = []
        for f in self._fields:
            if isinstance(f, Node):
                out.append(f)
            elif isinstance(f, tuple):
                out.extend(n for n in f if isinstance(n, Node))
        return tuple(out)

    def __getattr__(self, name):
        try:
            i = type(self).field_names.index(name)
        except ValueError as err:
            raise AttributeError(name) from err
        return self._fields[i]

    def __hash__(self) -> int:
        return self.uid

    def __eq__(self, other) -> bool:
        return self is other

    def __lt__(self, other: Node) -> bool:
        return self.key() < other.key()

    def key(self) -> tuple[int, int]:
        return (type(self).rank, self.uid)

    def uses_level(self, level: int) -> bool:
        return level in self.levels

    def __repr__(self) -> str:
        return self.pretty()

    def pretty(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__


# ---------------------------------------------------------------------------
# Index expressions (integer domain)
# ---------------------------------------------------------------------------
class IndexExpr(Node):
    __slots__ = ()


class Idx(IndexExpr):
    """A free output index of the tensor being described (``i0``, ``i1``...)."""

    __slots__ = ()
    rank = 1
    field_names = ("name",)

    def _post_init(self) -> None:
        super()._post_init()
        self.free_idx = frozenset({self.name})

    def pretty(self) -> str:
        return self.name


class BIdx(IndexExpr):
    """A bound reduction index, identified by the absolute level of its binder."""

    __slots__ = ()
    rank = 2
    field_names = ("level",)

    def _post_init(self) -> None:
        super()._post_init()
        self.levels = frozenset({self.level})

    def pretty(self) -> str:
        return f"j{self.level}"


class IConst(IndexExpr):
    __slots__ = ()
    rank = 0
    field_names = ("value",)

    def pretty(self) -> str:
        return str(self.value)


# ---------------------------------------------------------------------------
# Domains of reduction
# ---------------------------------------------------------------------------
class Domain(Node):
    __slots__ = ()

    @property
    def axis(self) -> int:
        return self._fields[0]


class DFull(Domain):
    """The whole ``[0, extent)`` range of reduction axis ``axis`` (a DimKey)."""

    __slots__ = ()
    field_names = ("axis",)

    def pretty(self) -> str:
        return f"ax{self.axis}"


class DSym(Domain):
    """An abstract, non-empty subset of the axis: a sub-range, or one side of a merge."""

    __slots__ = ()
    field_names = ("axis", "name")

    def pretty(self) -> str:
        return f"{self.name}@ax{self.axis}"


class DUnion(Domain):
    """Disjoint union of two sub-domains of the same axis."""

    __slots__ = ()
    field_names = ("axis", "a", "b")

    def pretty(self) -> str:
        return f"({self.a.pretty()}u{self.b.pretty()})"


class DEmpty(Domain):
    __slots__ = ()
    field_names = ("axis",)

    def pretty(self) -> str:
        return f"empty@ax{self.axis}"


# ---------------------------------------------------------------------------
# Scalar (real-valued) expressions
# ---------------------------------------------------------------------------
class SymExpr(Node):
    __slots__ = ()


class Const(SymExpr):
    __slots__ = ()
    rank = 10
    field_names = ("value",)

    def pretty(self) -> str:
        v = self.value
        if isinstance(v, Fraction):
            return str(v.numerator) if v.denominator == 1 else f"({v})"
        return "inf" if v > 0 else "-inf"


class ShapeSym(SymExpr):
    """A symbolic extent (``te.var``) used as a real value, e.g. ``head_dim``."""

    __slots__ = ()
    rank = 11
    field_names = ("name",)

    def pretty(self) -> str:
        return self.name


class Card(SymExpr):
    """Cardinality of a reduction domain (``Σ_D 1``)."""

    __slots__ = ()
    rank = 12
    field_names = ("domain",)

    def pretty(self) -> str:
        return f"|{self.domain.pretty()}|"


class Elem(SymExpr):
    """One element of an input tensor: ``tensor[indices]``."""

    __slots__ = ()
    rank = 13
    field_names = ("tensor", "indices")

    def pretty(self) -> str:
        return f"T{self.tensor}[{','.join(i.pretty() for i in self.indices)}]"


class StateVar(SymExpr):
    """Reducer state slot ``k`` of side ``"a"`` or ``"b"`` inside a merge function."""

    __slots__ = ()
    rank = 14
    field_names = ("side", "k")

    def pretty(self) -> str:
        return f"{self.side}{self.k}"


class Atom(SymExpr):
    """An opaque real-valued symbol (the correction of a re-based context)."""

    __slots__ = ()
    rank = 15
    field_names = ("name",)

    def pretty(self) -> str:
        return self.name


class Add(SymExpr):
    __slots__ = ()
    rank = 20
    field_names = ("args",)

    def pretty(self) -> str:
        return "(" + " + ".join(a.pretty() for a in self.args) + ")"


class Mul(SymExpr):
    __slots__ = ()
    rank = 21
    field_names = ("args",)

    def pretty(self) -> str:
        return "(" + "*".join(a.pretty() for a in self.args) + ")"


class Pow(SymExpr):
    __slots__ = ()
    rank = 22
    field_names = ("base", "exponent")

    def pretty(self) -> str:
        return f"{self.base.pretty()}^{self.exponent}"


class Exp(SymExpr):
    __slots__ = ()
    rank = 23
    field_names = ("arg",)

    def pretty(self) -> str:
        return f"exp({self.arg.pretty()})"


class Log(SymExpr):
    __slots__ = ()
    rank = 25
    field_names = ("arg",)

    def pretty(self) -> str:
        return f"log({self.arg.pretty()})"


class Max(SymExpr):
    __slots__ = ()
    rank = 24
    field_names = ("args",)

    def pretty(self) -> str:
        return "max(" + ", ".join(a.pretty() for a in self.args) + ")"


class Reduce(SymExpr):
    """``kind`` in {"sum", "max"} over ``domain``; binds :class:`BIdx` ``level``."""

    __slots__ = ()
    rank = 30
    field_names = ("kind", "domain", "level", "body")

    def _post_init(self) -> None:
        super()._post_init()
        self.levels = self.levels - {self.level}

    def pretty(self) -> str:
        return f"{self.kind}_{{j{self.level} in {self.domain.pretty()}}}({self.body.pretty()})"


class MonoidReduce(SymExpr):
    """Output ``slot`` of an opaque tuple reducer folded over ``domain``.

    ``leaf`` is evaluated at the bound index (level ``level``); ``merge`` is
    written over :class:`StateVar`; ``identity`` is a tuple of constants.
    """

    __slots__ = ()
    rank = 31
    field_names = ("domain", "level", "leaf", "merge", "identity", "slot")

    def _post_init(self) -> None:
        super()._post_init()
        self.levels = self.levels - {self.level}

    def pretty(self) -> str:
        leaf = ", ".join(x.pretty() for x in self.leaf)
        merge = ", ".join(x.pretty() for x in self.merge)
        dom = self.domain.pretty()
        return f"monoid{self.slot}_{{j{self.level} in {dom}}}[({leaf}) | ({merge})]"


# ---------------------------------------------------------------------------
# Raw (non-canonicalising) constructors. Prefer the ``mk_*`` functions in
# ``canonicalize`` everywhere except inside that module.
# ---------------------------------------------------------------------------
def idx(name: str) -> Idx:
    return _intern(Idx, (name,))


def bidx(level: int) -> BIdx:
    return _intern(BIdx, (level,))


def iconst(value: int) -> IConst:
    return _intern(IConst, (int(value),))


def dfull(axis: int) -> DFull:
    return _intern(DFull, (axis,))


def dsym(axis: int, name: str) -> DSym:
    return _intern(DSym, (axis, name))


def dunion(a: Domain, b: Domain) -> DUnion:
    assert a.axis == b.axis
    return _intern(DUnion, (a.axis, a, b))


def dempty(axis: int) -> DEmpty:
    return _intern(DEmpty, (axis,))


def const(value) -> Const:
    if isinstance(value, float) and value in (NEG_INF, POS_INF):
        return _intern(Const, (value,))
    if isinstance(value, bool):
        value = int(value)
    return _intern(Const, (Fraction(value),))


ZERO = const(0)
ONE = const(1)
MINUS_ONE = const(-1)
NEG_INF_C = const(NEG_INF)
POS_INF_C = const(POS_INF)


def shape_sym(name: str) -> ShapeSym:
    return _intern(ShapeSym, (name,))


def card(domain: Domain) -> Card:
    return _intern(Card, (domain,))


def elem(tensor: int, indices) -> Elem:
    return _intern(Elem, (tensor, tuple(indices)))


def state_var(side: str, k: int) -> StateVar:
    return _intern(StateVar, (side, k))


def atom(name: str) -> Atom:
    return _intern(Atom, (name,))


def raw_add(args) -> Add:
    return _intern(Add, (tuple(args),))


def raw_mul(args) -> Mul:
    return _intern(Mul, (tuple(args),))


def raw_pow(base: SymExpr, exponent) -> Pow:
    return _intern(Pow, (base, Fraction(exponent)))


def raw_exp(arg: SymExpr) -> Exp:
    return _intern(Exp, (arg,))


def raw_log(arg: SymExpr) -> Log:
    return _intern(Log, (arg,))


def raw_max(args) -> Max:
    return _intern(Max, (tuple(args),))


def raw_reduce(kind: str, domain: Domain, level: int, body: SymExpr) -> Reduce:
    assert kind in ("sum", "max")
    return _intern(Reduce, (kind, domain, level, body))


def raw_monoid(domain, level, leaf, merge, identity, slot) -> MonoidReduce:
    return _intern(
        MonoidReduce, (domain, level, tuple(leaf), tuple(merge), tuple(identity), int(slot))
    )


def raw_subst(e: Node, mapping: dict[Node, Node]) -> Node:
    """Replace sub-nodes without canonicalising anything (structure is kept as is)."""
    memo: dict[Node, Node] = {}

    def go(n: Node) -> Node:
        hit = mapping.get(n)
        if hit is not None:
            return hit
        if n in memo:
            return memo[n]
        fields = []
        changed = False
        for f in n._fields:
            if isinstance(f, Node):
                nf = go(f)
            elif isinstance(f, tuple):
                nf = tuple(go(x) if isinstance(x, Node) else x for x in f)
            else:
                nf = f
            changed |= nf is not f and nf != f
            fields.append(nf)
        out = _intern(type(n), tuple(fields)) if changed else n
        memo[n] = out
        return out

    return go(e)


def is_inf(e: Node) -> bool:
    return isinstance(e, Const) and isinstance(e.value, float)


def min_args(e: Node) -> list[SymExpr] | None:
    """``[a, b, ..]`` when ``e`` is ``-max(-a, -b, ..)``, the way a minimum is written."""
    if not (isinstance(e, Mul) and len(e.args) == 2 and e.args[0] is MINUS_ONE):
        return None
    if not isinstance(e.args[1], Max):
        return None
    out = []
    for a in e.args[1].args:
        if not (isinstance(a, Mul) and a.args[0] is MINUS_ONE):
            return None
        rest = a.args[1:]
        out.append(rest[0] if len(rest) == 1 else raw_mul(rest))
    return out
