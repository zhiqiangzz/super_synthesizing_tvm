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
"""A fused chain as text: its definitions in mathematical notation, its reducer as TE source.

:func:`render` prints a definition the way one would write it on paper
(``Σ_j exp(X[i, j] - mx[i])``); it is for reports. :func:`chain_source` prints
the fused reducer as the Python one would type to build it -- the same
reducer :func:`build.build_chain` emits, so the two can be read side by side
with a hand-written version.
"""

from __future__ import annotations

from fractions import Fraction

from ..symbolic import ir
from ..symbolic.canonicalize import Unsupported, recanonicalize, term_view
from ..symbolic.emit import SourceEnv, const_source, hoisted_source, to_source
from .chain import Chain
from .derive import ReducerSpec


def identifier(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    return out if not out[:1].isdigit() else f"_{out}"


# ---------------------------------------------------------------------------
# mathematical notation
# ---------------------------------------------------------------------------
def _number(v) -> str:
    if isinstance(v, float):
        return "inf" if v > 0 else "-inf"
    if v.denominator == 1:
        return str(v.numerator)
    if v.denominator <= 1000:
        return f"{v.numerator}/{v.denominator}"
    return f"{float(v):.6g}"  # a float constant of the program, or a ratio of two


def render(e: ir.Node, chain: Chain) -> str:
    """``e`` over the tensors, loop variables and reduction axis of ``chain``."""
    members = {m.pid: m for m in chain.members}
    j = chain.jname

    def index(i: ir.IndexExpr) -> str:
        if isinstance(i, ir.Idx):
            return chain.coord_names[int(i.name[1:])]
        if isinstance(i, ir.BIdx):
            return j
        return str(i.value)

    def atom(s: str) -> str:
        """Parenthesise a sum when it is used as an operand."""
        depth = 0
        for k, ch in enumerate(s):
            depth += (ch in "([") - (ch in ")]")
            if depth == 0 and s[k : k + 3] in (" + ", " - "):
                return f"({s})"
        return s

    def go(n: ir.Node) -> str:
        if isinstance(n, ir.Const):
            return _number(n.value)
        if isinstance(n, ir.ShapeSym | ir.Atom):
            return n.name
        if isinstance(n, ir.StateVar):
            return f"{n.side}{n.k}"
        if isinstance(n, ir.Card):
            return f"Σ_{j} 1" if isinstance(n.domain, ir.DSym) else f"|{j}|"
        if isinstance(n, ir.Elem):
            if n.tensor in members:  # a read of a member: without the index that pins it
                m = members[n.tensor]
                idx = n.indices[: len(m.coords)]
                return f"{m.name}[{', '.join(index(i) for i in idx)}]" if idx else m.name
            name = chain.lower.placeholder(n.tensor).op.name
            return f"{name}[{', '.join(index(i) for i in n.indices)}]"
        if isinstance(n, ir.Add):
            c, terms = term_view(n)
            parts = []
            for core, coeff in terms.items():
                body = go(core) if abs(coeff) == 1 else f"{_number(abs(coeff))}·{atom(go(core))}"
                parts.append((coeff < 0, body))
            if c != 0:
                parts.append((c < 0, _number(abs(c))))
            parts.sort(key=lambda x: x[0])  # positive terms first
            out = ("-" if parts[0][0] else "") + parts[0][1]
            for neg, body in parts[1:]:
                out += f" {'-' if neg else '+'} {body}"
            return out
        if isinstance(n, ir.Mul):
            if len(n.args) == 2 and n.args[0] is ir.MINUS_ONE and isinstance(n.args[1], ir.Reduce):
                red = n.args[1]  # -max_j -f is how the IR writes min_j f
                body = recanonicalize(ir.raw_mul([ir.MINUS_ONE, red.body]))
                if red.kind == "max":
                    return f"min_{j} {atom(go(body))}"
            smallest = ir.min_args(n)
            if smallest is not None:
                return f"min({', '.join(go(a) for a in smallest)})"
            sign = "-" if any(a is ir.MINUS_ONE for a in n.args) else ""
            args = [a for a in n.args if a is not ir.MINUS_ONE]
            num = [atom(go(a)) for a in args if not (isinstance(a, ir.Pow) and a.exponent < 0)]
            den = [a for a in args if isinstance(a, ir.Pow) and a.exponent < 0]
            out = sign + ("·".join(num) if num else "1")
            for d in den:
                base = atom(go(d.base))
                out += f" / {base}" if d.exponent == -1 else f" / {base}^{_number(-d.exponent)}"
            return out
        if isinstance(n, ir.Pow):
            base = atom(go(n.base))
            if n.exponent == Fraction(1, 2):
                return f"sqrt({go(n.base)})"
            if n.exponent < 0:
                tail = "" if n.exponent == -1 else f"^{_number(-n.exponent)}"
                return f"1 / {base}{tail}"
            return f"{base}^{_number(n.exponent)}"
        if isinstance(n, ir.Exp):
            return f"exp({go(n.arg)})"
        if isinstance(n, ir.Log):
            return f"log({go(n.arg)})"
        if isinstance(n, ir.Max):
            return f"max({', '.join(go(a) for a in n.args)})"
        if isinstance(n, ir.Reduce):
            return f"{'Σ' if n.kind == 'sum' else 'max'}_{j} {atom(go(n.body))}"
        raise Unsupported(f"cannot render {type(n).__name__}")

    return go(e)


def describe_members(chain: Chain) -> list[str]:
    """One line per member: the chain as the program writes it."""
    lines = []
    for m in chain.members:
        idx = ", ".join(chain.coord_names[c] for c in m.coords)
        lines.append(f"{m.name}[{idx}] = {render(m.body, chain)}")
    return lines


# ---------------------------------------------------------------------------
# TE source
# ---------------------------------------------------------------------------
def _extent(e) -> str:
    return str(e.name) if hasattr(e, "name") else str(int(e.value))


def chain_source(
    chain: Chain, spec: ReducerSpec, state_names: list[str], tag: str | None = None
) -> str:
    """Python source building the fused reducer of ``chain`` and its epilogues.

    The names it reads -- boundary tensors, extents -- are the ones the
    original program uses. ``tag`` distinguishes the helper functions of
    several chains printed together (the reduction axis by default).
    """
    tag = chain.jname if tag is None else tag
    dtype = str(chain.reductions[0].tensor.dtype)
    n = spec.arity
    rank = len(chain.coord_extents)
    j = chain.jname
    coords = list(chain.coord_names)
    lower = chain.lower

    def env_for(index: dict) -> SourceEnv:
        env = SourceEnv(dtype=dtype, index=index)
        env.elem = lambda tid, idx: f"{lower.placeholder(tid).op.name}[{', '.join(idx)}]"
        return env

    # a member that needs a compute of its own keeps its name; the state behind it steps aside
    epilogued = set()
    for m, expr in zip(chain.required, spec.epilogues):
        whole = m.coords == tuple(range(rank))
        if not (whole and isinstance(expr, ir.Elem) and expr.tensor < 0):
            epilogued.add(identifier(m.name))
    states = [identifier(s) for s in state_names]
    states = [f"{s}_s" if s in epilogued else s for s in states]

    reducer = f"fused_{tag}"
    env = env_for({})
    for k in range(n):
        env.state[ir.state_var("a", k)] = f"a[{k}]"
        env.state[ir.state_var("b", k)] = f"b[{k}]"
    shared, merged = hoisted_source(spec.merge_code, env, prefix="t")
    lines = [f"def merge_{tag}(a, b):"]
    lines += [f"    {line}" for line in shared]
    lines.append("    return (")
    lines += [f"        {m},  # {states[k]}" for k, m in enumerate(merged)]
    lines.append("    )")
    lines.append("")
    dts = ", ".join(f"t{k}" for k in range(n))
    ident = ", ".join(const_source(spec.identity[k].value, f"t{k}", quoted=False) for k in range(n))
    lines.append(f"def identity_{tag}({dts}):")
    lines.append(f"    return ({ident}{',' if n == 1 else ''})")
    lines.append("")
    lines.append(f'{reducer} = te.comm_reducer(merge_{tag}, identity_{tag}, name="fused_{j}")')
    lines.append(f'{j} = te.reduce_axis((0, {_extent(chain.extent)}), name="{j}")')

    leaf_env = env_for({ir.idx(f"i{c}"): coords[c] for c in range(rank)} | {ir.bidx(0): j})
    leaves = [to_source(leaf, leaf_env) for leaf in spec.leaves]
    shape = ", ".join(_extent(e) for e in chain.coord_extents) + ("," if rank == 1 else "")
    name = "fused_" + "_".join(m.name for m in chain.reductions)
    lines.append(f"{', '.join(states)} = te.compute(")
    lines.append(f"    ({shape}),")
    lines.append(f"    lambda {', '.join(coords)}: {reducer}(")
    lines.append("        (")
    lines += [f"            {leaf},  # {states[k]}" for k, leaf in enumerate(leaves)]
    lines.append("        ),")
    lines.append(f"        axis={j},")
    lines.append("    ),")
    lines.append(f'    name="{name}",')
    lines.append(")")

    for m, expr in zip(chain.required, spec.epilogues):
        if identifier(m.name) not in epilogued:
            continue
        own = [iv.var.name for iv in m.op.axis]
        at = dict(zip(m.coords, own))
        out_env = env_for({ir.idx(f"i{c}"): v for c, v in at.items()})
        boundary = out_env.elem

        def elem(tid, idx, at=at, boundary=boundary):
            if tid >= 0:
                return boundary(tid, idx)
            return f"{states[-tid - 1]}[{', '.join(at.get(c, '0') for c in range(rank))}]"

        out_env.elem = elem
        mshape = ", ".join(_extent(e) for e in m.tensor.shape) + ("," if len(own) == 1 else "")
        lines.append(
            f"{identifier(m.name)} = te.compute(({mshape}), "
            f'lambda {", ".join(own)}: {to_source(expr, out_env)}, name="{m.name}")'
        )
    return "\n".join(lines)


__all__ = ["chain_source", "describe_members", "identifier", "render"]
