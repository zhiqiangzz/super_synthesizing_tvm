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
  ``I(A)``, ``I(B)``. Where the caller gives it (``rebase.Printer.kept``), each
  context is kept as its merged value ``m_k`` instead of being written over
  the partial reductions too: the merge reads the merged contexts, evaluated
  first (:func:`closed_merges`);
* the epilogue of every tensor the program still needs, by writing its closed
  form (``R`` = the whole axis) over the state outputs.

A state set is kept only if every required tensor is covered and every state
is needed. The laws are not tested one by one: the merge satisfies
``M(I(A), I(B)) = I(A u B)`` by construction (proven, see
:func:`invariant_holds`, and :func:`psi_inverts` for the contexts kept), and
commutativity and associativity on the states a part of the axis can have
follow from that of the union. The identity is ``I(∅)`` read off the
definitions (:func:`empty_value`) and the identity law is proven on those
states as well (:func:`identity_law`).
"""

from __future__ import annotations

import dataclasses

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
    # over StateVar a_k / b_k, and m_k (slot k merged) where a context is kept
    merge: tuple[ir.SymExpr, ...]
    identity: tuple[ir.Const, ...]
    closed: tuple[ir.SymExpr, ...]  # I_k(full axis)
    proof: str
    # One per required tensor: its closed form written over the state outputs
    # (state ``k`` read as the element ``T-(k+1)[..]`` at its own coordinates).
    epilogues: tuple[ir.SymExpr, ...] = ()
    # The merge as it is evaluated, when it differs from the canonical ``merge``
    # it is proven equal to (see ``rebase``): each side's context re-based.
    merge_print: tuple[ir.SymExpr, ...] | None = None
    # The epilogues as they are evaluated, when they differ from the canonical
    # ones: a function taken out of a maximum kept as the program writes it.
    epilogue_print: tuple[ir.SymExpr, ...] | None = None

    @property
    def arity(self) -> int:
        return len(self.states)

    @property
    def merge_code(self) -> tuple[ir.SymExpr, ...]:
        """The merge to generate code from, over the two sides alone."""
        return self.merge_print if self.merge_print is not None else closed_merges(self.merge)

    @property
    def epilogue_code(self) -> tuple[ir.SymExpr, ...]:
        """The epilogues to generate code from."""
        return self.epilogue_print if self.epilogue_print is not None else self.epilogues

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
@dataclasses.dataclass(frozen=True)
class Solved:
    """How an atom was solved for: from which state, dividing by what."""

    state: int
    divisors: tuple[ir.SymExpr, ...] = ()  # over the state values and other atoms


def solve_atoms(
    states: tuple[ir.SymExpr, ...], R: ir.DSym, trace: dict | None = None
) -> dict[ir.Reduce, ir.SymExpr] | None:
    """Express partial-reduction atoms as functions of the state values ``a_k``.

    A state with a single unknown atom determines it. A state made of several
    (``Σ_R (c y + d) = c Σ_R y + d |R|``) cannot determine each of them, and
    does not have to: it is solved for one, the others stay *free*, and the
    state is still usable wherever they cancel -- which the callers check
    (a merge or an epilogue with a partial reduction left in it is refused).

    ``trace`` receives, per atom solved, a :class:`Solved`: the state it was
    read from and the factors that were divided by to read it. Where one of
    those is zero the state does not determine the atom.
    """
    sol: dict[ir.Reduce, ir.SymExpr] = {}
    how: dict[ir.Reduce, Solved] = {}
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
            divisors: list[ir.SymExpr] = []
            expr = _solve_single(e, open_atoms[0], ir.state_var("a", k), divisors)
            if expr is None:
                continue
            sol[open_atoms[0]] = expr
            how[open_atoms[0]] = Solved(k, tuple(divisors))
            done.add(k)
            progress = True
        if progress:
            continue
        # every remaining state has several unknowns: solve the first for one of them
        k = next(k for k in range(len(states)) if k not in done)
        e = subst(states[k], sol)
        for alpha in [a for a in atoms_in(e, R) if a not in sol]:
            divisors = []
            expr = _solve_single(e, alpha, ir.state_var("a", k), divisors)
            if expr is not None:
                sol[alpha] = expr
                how[alpha] = Solved(k, tuple(divisors))
                done.add(k)
                break
        else:
            if trace is not None:
                trace["unsolvable"] = k
            return None
    # Close the solution (later solutions may mention earlier atoms).
    for _ in range(len(sol)):
        sol = {a: subst(v, sol) for a, v in sol.items()}
    if trace is not None:
        for atom, h in how.items():
            trace[atom] = Solved(h.state, tuple(subst(d, sol) for d in h.divisors))
    return sol


def _mentions(e: ir.Node, alpha: ir.Node) -> bool:
    return contains_node(e, lambda n: n is alpha)


def _solve_single(
    e: ir.SymExpr, alpha: ir.Reduce, value: ir.SymExpr, divisors: list | None = None
) -> ir.SymExpr | None:
    """Solve ``e == value`` for ``alpha`` when ``e`` is linear in ``alpha``.

    What is divided by on the way is appended to ``divisors`` (an exponential,
    never zero, is not).
    """
    if e is alpha:
        return value
    if isinstance(e, ir.Log):
        return _solve_single(e.arg, alpha, mk_exp(value), divisors)
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
    scale = mk_mul(ir.const(coeff), F)
    if divisors is not None:  # a factor under a negative power is multiplied by instead
        divisors.extend(mk_pow(b, p) for b, p in factors.items() if b is not alpha and p > 0)
    return mk_mul(mk_sub(value, rest), mk_pow(scale, -1))


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


def leftover(state, sol, R, A, B) -> list[ir.SymExpr]:
    """The partial reductions (over ``R``) ``I(A u B)`` still needs once written over
    the state values: what the states lack to merge this one."""
    atom_map = {}
    for atom, expr in sol.items():
        atom_map[subst_domain(atom, {R: A})] = subst_domain(expr, {R: A})
        atom_map[subst_domain(atom, {R: B})] = subst_domain(swap_sides(expr, "a", "b"), {R: B})
    m = subst(subst_domain(state, {R: ir.dunion(A, B)}), atom_map)
    out: dict[ir.SymExpr, None] = {}
    for side in (A, B):
        for atom in atoms_in(m, side):
            out[subst_domain(atom, {side: R})] = None
    return list(out)


def _is_partial(x: ir.Node) -> bool:
    return isinstance(x, ir.Reduce) or (isinstance(x, ir.Card) and isinstance(x.domain, ir.DSym))


def spec_key(spec: ReducerSpec):
    """Identity of a reducer modulo the order of its slots."""
    order = sorted(range(spec.arity), key=lambda k: (spec.leaves[k].uid, spec.merge[k].uid))
    ren = {}
    for new, old in enumerate(order):
        for side in "abm":
            ren[ir.state_var(side, old)] = ir.state_var(side, new)
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


def empty_value(state: ir.SymExpr, R: ir.DSym, struct: ir.SymExpr | None = None) -> ir.Const | None:
    """``I(∅)``, the value of a state on an empty part, where it is a constant.

    Read off the reduction as the program writes it (``struct``, its contexts
    still symbols): an empty sum is 0 and an empty maximum -inf whatever they
    would have been computed against. Then off the definition: an empty count
    is 0; where that leaves ``0 / 0`` (a mean, ``Σ x / |R|``) the count is kept
    as an unknown positive number instead, so that the mean of nothing is
    ``0 / |R| = 0`` and a maximum shifted by it still -inf. ``None`` when no
    constant comes out.
    """
    for e in (struct, state):
        if e is None:
            continue
        try:
            e = subst_domain(e, {R: ir.dempty(R.axis)})
        except Unsupported:
            continue
        if isinstance(e, ir.Const):
            return e

    def fn(n: ir.Node):
        if isinstance(n, ir.Reduce) and n.domain is R:
            return ir.ZERO if n.kind == "sum" else ir.NEG_INF_C
        return None

    try:
        e = transform(state, fn)
    except Unsupported:
        return None
    return e if isinstance(e, ir.Const) else None


def identity_of(states, merge, R, A, B, structs=None) -> tuple[tuple[ir.Const, ...], str] | None:
    """The identity element and how its law is proven, or ``None``.

    A state with a value on an empty part (:func:`empty_value`) takes it. One
    without (a weighted mean ``Σ w x / Σ w``) is left as an unknown ``t`` and
    the law is solved for it: where the merge multiplies ``t`` by an empty
    weight the law holds whatever ``t`` is, and 0 is taken; where it does not,
    the law fixes it (it must come out a constant).
    """
    structs = structs or (None,) * len(states)
    values = [empty_value(s, R, st) for s, st in zip(states, structs)]
    unknown = {ir.atom(f"t{k}"): k for k, v in enumerate(values) if v is None}
    trial = [ir.atom(f"t{k}") if v is None else v for k, v in enumerate(values)]
    for _ in range(len(unknown) + 1):
        residues = identity_residues(states, merge, tuple(trial), R, A, B)
        if residues is None:
            return None
        fixed = False
        for diff in residues:
            for t, k in unknown.items():
                if trial[k] is not t or not contains_node(diff, lambda x, t=t: x is t):
                    continue
                v = _solve_single(diff, t, ir.ZERO)
                if isinstance(v, ir.Const):
                    trial[k], fixed = v, True
        if not fixed:
            break
    method = identity_law(states, merge, tuple(trial), R, A, B)
    if method is None:
        return None
    return tuple(ir.ZERO if v in unknown else v for v in trial), method


def identity_residues(states, merge, identity, R, A, B) -> list[ir.SymExpr] | None:
    """``M(s, e) - s`` slot by slot (both orders), canonical; ``None`` if undefined."""
    out = []
    for mine, other, dom in (("a", "b", A), ("b", "a", B)):
        env = {}
        for k, s in enumerate(states):
            env[ir.state_var(mine, k)] = subst_domain(s, {R: dom})
            env[ir.state_var(other, k)] = identity[k]
        for k in range(len(states)):
            try:
                out.append(mk_sub(subst(merge[k], env), env[ir.state_var(mine, k)]))
            except Unsupported:
                return None
    return out


def identity_law(states, merge, identity, R, A, B) -> str | None:
    """``M(s, e) = M(e, s) = s`` for every state ``s`` a non-empty part can have:
    how it is proven, or ``None``.

    ``s`` is written as the definitions over a part (``I(A)``), so the law is
    asked of reachable states only; a state that is a function of others then
    agrees with them.
    """
    n = len(states)
    method = "canonical"
    for mine, other, dom in (("a", "b", A), ("b", "a", B)):
        env = {}
        for k, s in enumerate(states):
            env[ir.state_var(mine, k)] = subst_domain(s, {R: dom})
            env[ir.state_var(other, k)] = identity[k]
        for k in range(n):
            try:
                got = subst(merge[k], env)
            except Unsupported:
                return None
            how = verify.prove_equal(got, env[ir.state_var(mine, k)])
            if how is None:
                return None
            method = verify.worst(method, how)
    return method


def invariant_holds(states, merge, R, A, B) -> str | None:
    """``M(I(A), I(B)) = I(A u B)`` as an identity in the reductions over ``A`` and
    ``B``: how it is proven, or ``None``.

    The merge is built to satisfy it, so it holds canonically unless a ``max``
    or a common denominator is in the way. Commutativity and associativity of
    ``M`` on reachable states follow from it, because the union is commutative
    and associative::

        M(I(A), I(B)) = I(A u B) = I(B u A) = M(I(B), I(A))
        M(M(I(A), I(B)), I(C)) = I((A u B) u C) = I(A u (B u C)) = M(I(A), M(I(B), I(C)))
    """
    inv_map = {}
    for k, s in enumerate(states):
        inv_map[ir.state_var("a", k)] = subst_domain(s, {R: A})
        inv_map[ir.state_var("b", k)] = subst_domain(s, {R: B})
    method = "canonical"
    for k, s in enumerate(states):
        lhs = subst(merge[k], inv_map)
        rhs = subst_domain(s, {R: ir.dunion(A, B)})
        how = verify.prove_equal(lhs, rhs)
        if how is None:
            return None
        method = verify.worst(method, how)
    return method


def closed_merges(merge) -> tuple[ir.SymExpr, ...] | None:
    """``merge`` over the two sides alone: each merged context ``m_k`` it reads
    replaced by slot ``k``'s own merge, recursively. ``None`` if slots read each
    other in a cycle."""
    done: dict[int, ir.SymExpr] = {}

    def get(k: int, busy: frozenset[int]) -> ir.SymExpr | None:
        if k in done:
            return done[k]
        if k in busy:
            return None
        mapping = {}
        for r in {n for n in walk(merge[k]) if isinstance(n, ir.StateVar) and n.side == "m"}:
            v = get(r.k, busy | {k})
            if v is None:
                return None
            mapping[r] = v
        done[k] = subst(merge[k], mapping)
        return done[k]

    out = tuple(get(k, frozenset()) for k in range(len(merge)))
    return None if any(x is None for x in out) else out


def reads_merged(e: ir.Node) -> bool:
    """Does ``e`` read a merged context ``m_k``?"""
    return any(isinstance(n, ir.StateVar) and n.side == "m" for n in walk(e))


def psi_inverts(states, sol, R, A) -> str | None:
    """``Ψ(I(A))`` is the reduction over ``A`` itself, for every reduction solved for:
    how it is proven, or ``None``.

    What a merge with its contexts kept rests on, besides its construction: it
    is the definition over the union with each context read as its merged value
    and each side's reductions as ``Ψ`` of that side's states. Where ``Ψ``
    returns the reductions of a part from the states the part has, it is
    ``I(A u B)`` once the contexts' own merges are -- slot by slot in the order
    the chain reads them.
    """
    env = {ir.state_var("a", k): subst_domain(s, {R: A}) for k, s in enumerate(states)}
    method = "canonical"
    for atom, expr in sol.items():
        try:
            got = subst_domain(subst(expr, env), {R: A})
        except Unsupported:
            return None
        how = verify.prove_equal(got, subst_domain(atom, {R: A}))
        if how is None:
            return None
        method = verify.worst(method, how)
    return method


def out_ref(k: int, closed: ir.SymExpr) -> ir.SymExpr:
    """How an epilogue reads state ``k``: the pseudo element ``T-(k+1)`` at the
    state's own coordinates, or the closed form itself when that reads no data
    (the count over the whole axis is the extent)."""
    if is_constant(closed):
        return closed
    names = sorted(closed.free_idx, key=index_order)
    return ir.elem(-(k + 1), tuple(ir.idx(nm) for nm in names))


def state_outputs(states, sol, axis: DimKey) -> dict[ir.SymExpr, ir.SymExpr]:
    """The reductions over the whole axis that ``states`` determine, over the state outputs."""
    full, R, _, _ = domains(axis)
    closed = tuple(subst_domain(s, {R: full}) for s in states)
    refs = {ir.state_var("a", k): out_ref(k, closed[k]) for k in range(len(states))}
    out = {}
    for atom, expr in sol.items():
        atom_full = subst_domain(atom, {R: full})
        if not is_constant(atom_full):
            out[atom_full] = subst_domain(subst(expr, refs), {R: full})
    return out


def derive(
    states: tuple[ir.SymExpr, ...],
    required: tuple[ir.SymExpr, ...],
    axis: DimKey,
    stats: dict | None = None,
    structs: tuple[ir.SymExpr, ...] | None = None,
    kept: tuple[ir.SymExpr | None, ...] | None = None,
) -> ReducerSpec | None:
    """The reducer with these ``states``, if it reproduces every ``required`` closed form.

    ``states`` are canonical partial definitions over ``DSym(axis, "R")``;
    ``required`` the closed forms (over the whole axis) of the tensors the rest
    of the program reads. ``kept``, one per state or ``None``, is its merge with
    the contexts kept as merged values (``m_k``): the merge wherever given, when
    ``Ψ`` inverts the states (:func:`psi_inverts`) and the identity law holds;
    the merge written over the partial reductions otherwise.
    """
    full, R, A, B = domains(axis)
    states = tuple(states)
    try:
        sol = solve_atoms(states, R)
    except Unsupported:  # solving one state for a reduction divides by zero under another
        sol = None
    if sol is None:
        count(stats, "derive:unsolvable")
        return None
    n = len(states)
    closed = tuple(subst_domain(s, {R: full}) for s in states)
    try:
        out_map = state_outputs(states, sol, axis)
        epilogues = tuple(subst(t, out_map) for t in required)
    except Unsupported:  # the closed form is not defined over these states (0 / 0)
        count(stats, "derive:uncovered")
        return None
    if any(atoms_in(e, full) for e in epilogues):
        count(stats, "derive:uncovered")
        return None
    merge = []
    for st in states:
        try:
            m = merge_of(st, sol, R, A, B)
        except Unsupported:
            m = None
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
    first = invariant_holds(states, merge, R, A, B)
    if first is None:
        count(stats, "derive:invariant_failed")
        return None
    found = None
    if kept is not None and any(k is not None for k in kept):
        swapped = tuple(m if k is None else k for m, k in zip(merge, kept))
        flat = closed_merges(swapped)
        how = None if flat is None else psi_inverts(states, sol, R, A)
        if how is not None:
            found = identity_of(states, flat, R, A, B, structs)
            if found is not None:
                merge, first = swapped, verify.worst(first, how)
    if found is None:
        found = identity_of(states, merge, R, A, B, structs)
    if found is None:
        count(stats, "derive:law_failed")
        return None
    identity, second = found
    proof = verify.worst(first, second)
    try:
        leaves = tuple(singleton(s, R) for s in states)
    except Unsupported:  # a state not defined on a single element (0 / 0)
        count(stats, "derive:undefined_leaf")
        return None
    count(stats, "derive:specs")
    return ReducerSpec(axis, states, leaves, merge, identity, closed, proof, epilogues)


__all__ = [
    "ReducerSpec",
    "Solved",
    "atoms_in",
    "closed_merges",
    "derive",
    "domains",
    "leftover",
    "merge_of",
    "psi_inverts",
    "reads_merged",
    "singleton",
    "solve_atoms",
    "spec_key",
    "state_outputs",
    "state_refs",
    "state_vars",
    "swap_sides",
]
