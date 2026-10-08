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
"""Derive a tuple ``comm_reducer`` from a set of state definitions.

A state is a function ``I(R)`` of partial reductions over a sub-range ``R``
of the reduction axis ``j``. That single definition fixes the whole reducer;
nothing below is searched:

* ``leaf = I({j})`` and ``identity = I(∅)``;
* the merge, by writing ``I(A u B)`` over the partial reductions of ``A`` and
  ``B`` (``Σ_{A u B} = Σ_A + Σ_B``) and solving those for the state values
  ``I(A)``, ``I(B)``;
* the epilogue of every tensor the program still needs, by writing its closed
  form (``R`` = the whole axis) over the state outputs.

A state set is kept only if every required tensor is covered, every state is
needed, and the derived merge satisfies the monoid laws.
"""

from __future__ import annotations

import dataclasses
import itertools

from ..dims import DimKey
from ..symbolic import ir
from ..symbolic.canonicalize import (
    Unsupported,
    contains_node,
    factor_view,
    is_constant,
    mk_exp,
    mk_mul,
    mk_pow,
    mk_sub,
    shift,
    subst,
    subst_domain,
    term_view,
    transform,
    walk,
)
from . import verify

R_NAME, A_NAME, B_NAME = "R", "A", "B"


@dataclasses.dataclass(frozen=True)
class ReducerSpec:
    """A fully derived and verified tuple reducer over reduction axis ``axis``."""

    axis: DimKey
    states: tuple[ir.SymExpr, ...]  # I_k(R), canonical, over DSym(axis, "R")
    leaves: tuple[ir.SymExpr, ...]  # I_k({j}) at binder depth 1 (BIdx(0) is j)
    merge: tuple[ir.SymExpr, ...]  # over StateVar a_k / b_k
    identity: tuple[ir.Const, ...]
    closed: tuple[ir.SymExpr, ...]  # I_k(full axis)
    proof: str
    # One per required tensor: its closed form written over the state outputs
    # (state ``k`` read as the element ``T-(k+1)[..]`` at its own coordinates).
    epilogues: tuple[ir.SymExpr, ...] = ()
    # The merge as it is evaluated, when it differs from the canonical ``merge``
    # it is proven equal to (see ``rebase``): each side's context re-based.
    merge_print: tuple[ir.SymExpr, ...] | None = None

    @property
    def arity(self) -> int:
        return len(self.states)

    @property
    def merge_code(self) -> tuple[ir.SymExpr, ...]:
        """The merge to generate code from."""
        return self.merge_print if self.merge_print is not None else self.merge

    def pretty(self) -> str:
        lines = [f"comm_reduce over ax{self.axis} ({self.proof}):"]
        for k in range(self.arity):
            lines.append(
                f"  s{k}: leaf={self.leaves[k]}  merge={self.merge_code[k]}  e={self.identity[k]}"
            )
        for k, e in enumerate(self.epilogues):
            lines.append(f"  out{k} = {e}")
        return "\n".join(lines)


def domains(axis: DimKey) -> tuple[ir.DFull, ir.DSym, ir.DSym, ir.DSym]:
    """``(full, R, A, B)``: the whole axis, a sub-range, and the two sides of a merge."""
    return (
        ir.dfull(axis),
        ir.dsym(axis, R_NAME),
        ir.dsym(axis, A_NAME),
        ir.dsym(axis, B_NAME),
    )


def atoms_in(e: ir.Node, domain: ir.Domain) -> list[ir.SymExpr]:
    """Reductions (and cardinalities) over ``domain`` in ``e``: the unknowns of a state."""
    out: dict[ir.SymExpr, None] = {}
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Reduce | ir.Card) and n.domain is domain:
            out[n] = None
            continue
        stack.extend(n.children())
    return list(out)


def state_vars(e: ir.Node) -> set[ir.StateVar]:
    return {n for n in walk(e) if isinstance(n, ir.StateVar)}


# ---------------------------------------------------------------------------
# solving the partial reductions for the state values
# ---------------------------------------------------------------------------
def solve_atoms(states: tuple[ir.SymExpr, ...], R: ir.DSym) -> dict[ir.Reduce, ir.SymExpr] | None:
    """Express partial-reduction atoms as functions of the state values ``a_k``.

    A state with a single unknown atom determines it. A state made of several
    (``Σ_R (c y + d) = c Σ_R y + d |R|``) cannot determine each of them, and
    does not have to: it is solved for one, the others stay *free*, and the
    state is still usable wherever they cancel -- which the callers check
    (a merge or an epilogue with a partial reduction left in it is refused).
    """
    sol: dict[ir.Reduce, ir.SymExpr] = {}
    done: set[int] = set()
    while len(done) < len(states):
        progress = False
        for k, s in enumerate(states):
            if k in done:
                continue
            e = subst(s, sol)
            open_atoms = [a for a in atoms_in(e, R) if a not in sol]
            if not open_atoms:
                done.add(k)
                progress = True
                continue
            if len(open_atoms) != 1:
                continue
            expr = _solve_single(e, open_atoms[0], ir.state_var("a", k))
            if expr is None:
                continue
            sol[open_atoms[0]] = expr
            done.add(k)
            progress = True
        if progress:
            continue
        # every remaining state has several unknowns: solve the first for one of them
        k = next(k for k in range(len(states)) if k not in done)
        e = subst(states[k], sol)
        for alpha in [a for a in atoms_in(e, R) if a not in sol]:
            expr = _solve_single(e, alpha, ir.state_var("a", k))
            if expr is not None:
                sol[alpha] = expr
                done.add(k)
                break
        else:
            return None
    # Close the solution (later solutions may mention earlier atoms).
    for _ in range(len(sol)):
        sol = {a: subst(v, sol) for a, v in sol.items()}
    return sol


def _mentions(e: ir.Node, alpha: ir.Node) -> bool:
    return contains_node(e, lambda n: n is alpha)


def _solve_single(e: ir.SymExpr, alpha: ir.Reduce, value: ir.SymExpr) -> ir.SymExpr | None:
    """Solve ``e == value`` for ``alpha`` when ``e`` is linear in ``alpha``."""
    if e is alpha:
        return value
    if isinstance(e, ir.Log):
        return _solve_single(e.arg, alpha, mk_exp(value))
    c, terms = term_view(e)
    hit = None
    for core, coeff in terms.items():
        if _mentions(core, alpha):
            if hit is not None:
                return None
            hit = (core, coeff)
    if hit is None:
        return None
    core, coeff = hit
    rest = mk_sub(e, mk_mul(ir.const(coeff), core))
    if _mentions(rest, alpha):
        return None
    # core == alpha * F with F alpha-free
    fcoeff, factors, exp_arg = factor_view(core)
    if factors.get(alpha) != 1:
        return None
    others = [mk_pow(b, p) for b, p in factors.items() if b is not alpha]
    if exp_arg is not None:
        others.append(mk_exp(exp_arg))
    F = mk_mul(ir.const(fcoeff), *others)
    if _mentions(F, alpha):
        return None
    return mk_mul(mk_sub(value, rest), mk_pow(mk_mul(ir.const(coeff), F), -1))


def singleton(state: ir.SymExpr, R: ir.DSym) -> ir.SymExpr:
    """``I({j})``: each atom becomes its body (at depth 1); everything else moves to depth 1."""

    def fn(n: ir.Node):
        if isinstance(n, ir.Reduce) and n.domain is R and n.level == 0:
            return n.body
        if isinstance(n, ir.Card) and n.domain is R:
            return ir.ONE
        if isinstance(n, ir.Reduce | ir.MonoidReduce) and n.level == 0:
            return shift(n, 0, 1)
        if not n.children():
            return n
        return None

    return transform(state, fn)


def swap_sides(e: ir.SymExpr, src: str, dst: str) -> ir.SymExpr:
    mapping = {}
    for n in state_vars(e):
        if n.side == src:
            mapping[n] = ir.state_var(dst, n.k)
    return subst(e, mapping)


def merge_of(state, sol, R, A, B) -> ir.SymExpr | None:
    """``I(A u B)`` written over the state values, or None if an atom is left unsolved."""
    atom_map = {}
    for atom, expr in sol.items():  # a solution may mention free atoms: they move with it
        atom_map[subst_domain(atom, {R: A})] = subst_domain(expr, {R: A})
        atom_map[subst_domain(atom, {R: B})] = subst_domain(swap_sides(expr, "a", "b"), {R: B})
    m = subst(subst_domain(state, {R: ir.dunion(A, B)}), atom_map)
    if atoms_in(m, A) or atoms_in(m, B) or contains_node(m, _is_partial):
        return None
    return m


def _is_partial(x: ir.Node) -> bool:
    return isinstance(x, ir.Reduce) or (isinstance(x, ir.Card) and isinstance(x.domain, ir.DSym))


def spec_key(spec: ReducerSpec):
    """Identity of a reducer modulo the order of its slots."""
    order = sorted(range(spec.arity), key=lambda k: (spec.leaves[k].uid, spec.merge[k].uid))
    ren = {}
    for new, old in enumerate(order):
        ren[ir.state_var("a", old)] = ir.state_var("a", new)
        ren[ir.state_var("b", old)] = ir.state_var("b", new)
    return tuple((spec.leaves[k], subst(spec.merge[k], ren), spec.identity[k]) for k in order)


def state_refs(e: ir.Node) -> set[int]:
    """State slots referenced through ``out_k`` pseudo elements (tensor id -(k+1))."""
    return {-n.tensor - 1 for n in walk(e) if isinstance(n, ir.Elem) and n.tensor < 0}


def index_order(name: str):
    return (0, int(name[1:])) if name.startswith("i") else (1, name)


def count(stats, what, n=1):
    if stats is not None:
        stats[what] = stats.get(what, 0) + n


# ---------------------------------------------------------------------------
# identity, laws, invariant: shared by every way of choosing the states
# ---------------------------------------------------------------------------
def needed_states(merge, refs: set[int]) -> set[int]:
    """Slots feeding ``refs`` through the merge functions."""
    needed = set(refs)
    changed = True
    while changed:
        changed = False
        for k in list(needed):
            for sv in state_vars(merge[k]):
                if sv.k not in needed:
                    needed.add(sv.k)
                    changed = True
    return needed


def identity_and_proof(states, merge, axis: DimKey, R):
    """``(identity, proof)`` for the first identity satisfying the monoid laws, else ``None``."""
    options = []
    for s in states:
        try:
            e = subst_domain(s, {R: ir.dempty(axis)})
        except Unsupported:  # e.g. 0/0 for a normalised output: pick by the identity law
            e = None
        if isinstance(e, ir.Const):
            options.append([e])
        else:
            options.append([ir.ZERO, ir.ONE, ir.NEG_INF_C, ir.POS_INF_C])
    for combo in itertools.product(*options):
        try:
            proof = verify.check_laws(merge, combo)
        except Unsupported:
            proof = None
        if proof is not None:
            return tuple(combo), proof
    return None


def invariant_holds(states, merge, R, A, B) -> bool:
    """``M(I(A), I(B)) = I(A u B)`` (true by construction; checked anyway)."""
    inv_map = {}
    for k, s in enumerate(states):
        inv_map[ir.state_var("a", k)] = subst_domain(s, {R: A})
        inv_map[ir.state_var("b", k)] = subst_domain(s, {R: B})
    for k, s in enumerate(states):
        lhs = subst(merge[k], inv_map)
        rhs = subst_domain(s, {R: ir.dunion(A, B)})
        if lhs is not rhs and not verify.equal_modulo_max(lhs, rhs):
            return False
    return True


def out_ref(k: int, closed: ir.SymExpr) -> ir.SymExpr:
    """How an epilogue reads state ``k``: the pseudo element ``T-(k+1)`` at the
    state's own coordinates, or the closed form itself when that reads no data
    (the count over the whole axis is the extent)."""
    if is_constant(closed):
        return closed
    names = sorted(closed.free_idx, key=index_order)
    return ir.elem(-(k + 1), tuple(ir.idx(nm) for nm in names))


def derive(
    states: tuple[ir.SymExpr, ...],
    required: tuple[ir.SymExpr, ...],
    axis: DimKey,
    stats: dict | None = None,
) -> ReducerSpec | None:
    """The reducer with these ``states``, if it reproduces every ``required`` closed form.

    ``states`` are canonical partial definitions over ``DSym(axis, "R")``;
    ``required`` the closed forms (over the whole axis) of the tensors the rest
    of the program reads.
    """
    full, R, A, B = domains(axis)
    states = tuple(states)
    sol = solve_atoms(states, R)
    if sol is None:
        count(stats, "derive:unsolvable")
        return None
    n = len(states)
    closed = tuple(subst_domain(s, {R: full}) for s in states)
    refs = {ir.state_var("a", k): out_ref(k, closed[k]) for k in range(n)}
    out_map = {}
    for atom, expr in sol.items():
        atom_full = subst_domain(atom, {R: full})
        if not is_constant(atom_full):
            out_map[atom_full] = subst_domain(subst(expr, refs), {R: full})
    try:
        epilogues = tuple(subst(t, out_map) for t in required)
    except Unsupported:
        count(stats, "derive:uncovered")
        return None
    if any(atoms_in(e, full) for e in epilogues):
        count(stats, "derive:uncovered")
        return None
    merge = []
    for st in states:
        m = merge_of(st, sol, R, A, B)
        if m is None:
            count(stats, "derive:missing_context")
            return None
        merge.append(m)
    merge = tuple(merge)
    used: set[int] = set()
    for e in epilogues:
        used |= state_refs(e)
    if needed_states(merge, used) != set(range(n)):
        count(stats, "derive:irrelevant_state")
        return None
    found = identity_and_proof(states, merge, axis, R)
    if found is None:
        count(stats, "derive:law_failed")
        return None
    identity, proof = found
    if not invariant_holds(states, merge, R, A, B):
        count(stats, "derive:invariant_failed")
        return None
    leaves = tuple(singleton(s, R) for s in states)
    count(stats, "derive:specs")
    return ReducerSpec(axis, states, leaves, merge, identity, closed, proof, epilogues)


__all__ = [
    "ReducerSpec",
    "atoms_in",
    "derive",
    "domains",
    "merge_of",
    "singleton",
    "solve_atoms",
    "spec_key",
    "state_refs",
    "state_vars",
    "swap_sides",
]
