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
"""Result-directed synthesis of tuple ``comm_reducer`` reductions.

Given the target's semantics and a reduction axis ``j``, the synthesiser:

1. extracts the *goals* -- the reductions over ``j`` the target needs, each
   un-nested to a tensor of its own;
2. enumerates candidate state sets over a bounded, prior-free grammar
   (:mod:`grammar`): the goals' partial reductions plus a few latent ones
   (``max``/``sum``/count of the goals' own signals), each state being its
   atom combined with a small expression over earlier atoms (the
   *triangular* structure that keeps the set invertible);
3. for each set derives -- rather than searches -- the reducer:
   ``leaf = I({j})``, ``identity = I(∅)`` and the merge by solving
   ``I(A u B)`` for the state values ``I(A)``, ``I(B)``;
4. keeps a set only if it covers every goal, every state is needed to finish
   the target, and the derived reducer satisfies the monoid laws.
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
    mk_exp,
    mk_mul,
    mk_neg,
    mk_pow,
    mk_sub,
    positive,
    shift,
    subst,
    subst_domain,
    term_view,
)
from ..symbolic.realize import node_count
from . import grammar, verify

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
    # Free indices of the closed forms / leaves, in output order, with their extents.
    index_space: tuple[tuple[str, DimKey], ...] = ()
    # The merge as it is evaluated, when it differs from the canonical ``merge``
    # it is proven equal to (see ``partial``): each side's context re-based.
    merge_print: tuple[ir.SymExpr, ...] | None = None
    origin: str = "grammar"  # "grammar" or "partial" (states taken from the original)
    # The target written over the state outputs (state k read as the element
    # ``T-(k+1)[..]`` at its own indices), when it needs nothing else: the
    # epilogue a reducer op may compute right after the reduction.
    finalize: ir.SymExpr | None = None

    @property
    def arity(self) -> int:
        return len(self.states)

    @property
    def merge_code(self) -> tuple[ir.SymExpr, ...]:
        """The merge to generate code from."""
        return self.merge_print if self.merge_print is not None else self.merge

    def pretty(self) -> str:
        lines = [f"comm_reduce over ax{self.axis} ({self.origin}, {self.proof}):"]
        for k in range(self.arity):
            lines.append(
                f"  s{k}: leaf={self.leaves[k]}  merge={self.merge_code[k]}  e={self.identity[k]}"
            )
        return "\n".join(lines)


def atoms_over(e: ir.Node, domain: ir.Domain) -> list[ir.SymExpr]:
    """Reductions / cardinalities of ``e`` over ``domain`` (the unknowns of the merge equations)."""
    return grammar.atoms_in(e, domain)


# ---------------------------------------------------------------------------
# goals and closure
# ---------------------------------------------------------------------------
def nested_index_name(level: int, axis: DimKey) -> str:
    """Output-index name given to an enclosing reduction binder when a goal is un-nested."""
    return f"n{level}a{axis}"


def collect_goals(target: ir.SymExpr, axis: DimKey):
    """Reductions over ``axis`` anywhere in ``target``, un-nested to level 0.

    A goal sitting inside other reductions (e.g. ``Σ_j exp(s) v_e`` under a
    later ``Σ_e``) is a legitimate tensor of its own: the enclosing binders
    become extra output indices of the reducer. Returns ``(goals, occurrences,
    index_axes)`` where ``occurrences`` pairs each node as it appears in the
    target with its un-nested form and ``index_axes`` maps the new index names
    to their extents.
    """
    full = ir.dfull(axis)
    occurrences: list[tuple[ir.Reduce, ir.Reduce]] = []
    index_axes: dict[str, DimKey] = {}

    def visit(n: ir.Node, enclosing: list[DimKey]) -> None:
        if isinstance(n, ir.Reduce) and n.domain is full:
            depth = len(enclosing)
            un = n
            if depth:
                mapping = {}
                for k, ax in enumerate(enclosing):
                    name = nested_index_name(k, ax)
                    index_axes[name] = ax
                    mapping[ir.bidx(k)] = ir.idx(name)
                un = shift(subst(n, mapping), depth, -depth)
            occurrences.append((n, un))
            return
        if isinstance(n, ir.Reduce):
            visit(n.body, [*enclosing, n.domain.axis])
            return
        if isinstance(n, ir.MonoidReduce):
            return
        for c in n.children():
            visit(c, enclosing)

    visit(target, [])
    goals = list(dict.fromkeys(un for _, un in occurrences))
    return goals, occurrences, index_axes


def extract_goals(target: ir.SymExpr, axis: DimKey) -> list[ir.Reduce]:
    return collect_goals(target, axis)[0]


def solve_atoms(states: tuple[ir.SymExpr, ...], R: ir.DSym) -> dict[ir.Reduce, ir.SymExpr] | None:
    """Express every partial-reduction atom as a function of the state values ``a_k``."""
    unknowns: dict[ir.Reduce, None] = {}
    for s in states:
        for a in atoms_over(s, R):
            unknowns[a] = None
    sol: dict[ir.Reduce, ir.SymExpr] = {}
    done: set[int] = set()
    progress = True
    while progress and len(sol) < len(unknowns):
        progress = False
        for k, s in enumerate(states):
            if k in done:
                continue
            e = subst(s, sol)
            open_atoms = [a for a in atoms_over(e, R) if a not in sol]
            if not open_atoms:
                done.add(k)
                continue
            if len(open_atoms) != 1:
                continue
            alpha = open_atoms[0]
            expr = _solve_single(e, alpha, ir.state_var("a", k))
            if expr is None:
                continue
            sol[alpha] = expr
            done.add(k)
            progress = True
    if len(sol) < len(unknowns):
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
    from ..symbolic.canonicalize import transform

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
    for n in _state_vars(e):
        if n.side == src:
            mapping[n] = ir.state_var(dst, n.k)
    return subst(e, mapping)


def _state_vars(e: ir.Node) -> set[ir.StateVar]:
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.StateVar):
            out.add(n)
        stack.extend(n.children())
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SynthesisProblem:
    target: ir.SymExpr  # target body over its own output indices i0..
    axis: DimKey
    max_states: int
    max_merge_nodes: int
    target_keys: tuple[DimKey, ...] = ()  # extent of each target output index
    min_states: int = 1
    consts: tuple[ir.SymExpr, ...] = ()  # the target's constant pool
    max_leaf_nodes: int = 4
    max_state_expr_nodes: int = 4
    max_latent_atoms: int = 1


def synthesize(problem: SynthesisProblem, stats: dict | None = None) -> list[ReducerSpec]:
    axis = problem.axis
    full, R = ir.dfull(axis), ir.dsym(axis, R_NAME)
    A, B = ir.dsym(axis, A_NAME), ir.dsym(axis, B_NAME)
    goals, occurrences, index_axes = collect_goals(problem.target, axis)
    if not goals:
        return []
    for t, key in enumerate(problem.target_keys):
        index_axes.setdefault(f"i{t}", key)
    goal_atoms = [subst_domain(g, {full: R}) for g in goals]
    sigs = grammar.signals(goals)
    latent = [a for a in grammar.atom_candidates(sigs, R) if a not in goal_atoms]
    _count(stats, "reducer:latent_candidates", len(latent))
    consts = _state_consts(problem.consts)
    enum = _StateEnumerator(problem, full, R, A, B, occurrences, index_axes, sigs, consts, stats)
    enum.goal_atoms = goal_atoms
    n_goals = len(goal_atoms)
    lo = max(1, problem.min_states)
    hi = problem.max_states
    for n_latent in range(0, min(problem.max_latent_atoms, hi - n_goals) + 1):
        if n_goals + n_latent < lo:
            continue
        for lat in itertools.combinations(latent, n_latent):
            # latent atoms are context: defined first, then the goals in any order
            for lat_order in itertools.permutations(lat):
                for goal_order in itertools.permutations(goal_atoms):
                    enum.extend([*lat_order, *goal_order], [])
    return enum.specs


def _state_consts(pool: tuple[ir.SymExpr, ...]) -> list[ir.SymExpr]:
    out: dict[ir.SymExpr, None] = {ir.ONE: None, ir.MINUS_ONE: None}
    for c in pool:
        out[c] = None
        out[mk_neg(c)] = None
    return list(out)


class _StateEnumerator:
    """Triangular enumeration of state sets with per-slot pruning."""

    def __init__(self, problem, full, R, A, B, occurrences, index_axes, sigs, consts, stats):
        self.problem = problem
        self.full, self.R, self.A, self.B = full, R, A, B
        self.occurrences, self.index_axes = occurrences, index_axes
        self.opaque: set[ir.Node] = grammar.primitive_signals(sigs)
        self.goal_atoms: list[ir.SymExpr] = []
        self.consts = consts
        self.tables: dict[tuple, dict[ir.SymExpr, ir.SymExpr]] = {}
        self.stats = stats
        self.specs: list[ReducerSpec] = []
        self.seen_specs: set = set()
        self.seen_sets: set = set()

    def extend(self, order: list[ir.SymExpr], states: list[ir.SymExpr]) -> None:
        k = len(states)
        if k == len(order):
            key = frozenset(states)
            if key in self.seen_sets:
                return
            self.seen_sets.add(key)
            if not self._latent_justified(order, states):
                _count(self.stats, "reducer:prune_latent_unjustified")
                return
            _count(self.stats, "reducer:state_sets")
            spec = _derive(
                tuple(states),
                self.problem,
                self.full,
                self.R,
                self.A,
                self.B,
                self.occurrences,
                self.index_axes,
                self.stats,
            )
            if spec is not None:
                sk = _spec_key(spec)
                if sk not in self.seen_specs:
                    self.seen_specs.add(sk)
                    self.specs.append(spec)
            return
        atom = order[k]
        table = self._table(tuple(order[:k]))
        leaf_a = singleton(atom, self.R)
        bare = grammar.complexity(leaf_a, self.opaque)
        # Candidate states A, A*F, A+G, A-G, log A. The leaf of each is computed
        # from leaf(A) and the singleton form of F/G before the state itself is
        # built, so that the (cheap) leaf bounds prune most of the table first.
        earlier = set(order[:k])
        cands: list[tuple[ir.SymExpr, ir.SymExpr, bool]] = [(atom, leaf_a, True)]
        for f, fj in table.items():
            if grammar.is_data_free(f):
                continue  # combining with a constant only reparametrises the state
            # ``A / B`` and ``A * B`` with ``B`` an earlier atom are the only
            # combinations allowed to leave the leaf as it is (a running mean is
            # ``S1 / n``); any other combination has to simplify the leaf.
            simple = f in earlier or (isinstance(f, ir.Pow) and f.base in earlier)
            try:
                if fj is not ir.ZERO:  # a factor vanishing per element degenerates the leaf
                    cands.append((grammar.mk_mul(atom, f), grammar.mk_mul(leaf_a, fj), simple))
                if not grammar.is_data_free(fj):  # shifting by a constant is a reparametrisation
                    cands.append((grammar.mk_add(atom, f), grammar.mk_add(leaf_a, fj), False))
                    cands.append((grammar.mk_sub(atom, f), grammar.mk_sub(leaf_a, fj), False))
            except Unsupported:
                continue
        if positive(atom):
            cands.append((grammar.mk_log(atom), grammar.mk_log(leaf_a), False))
        seen: set[ir.SymExpr] = set()
        for cand, leaf, simple in cands:
            # a constant factor or term on a state is a reparametrisation: drop it
            cand, leaf = _strip_constants(cand), _strip_constants(leaf)
            if cand in seen or isinstance(cand, ir.Const):
                continue
            seen.add(cand)
            _count(self.stats, "reducer:grammar_states")
            size = grammar.complexity(leaf, self.opaque)
            if size > self.problem.max_leaf_nodes:
                _count(self.stats, "reducer:prune_leaf")
                continue
            # Combining an atom with earlier ones must make its leaf strictly
            # simpler (exp(s - max) -> 1, x^2 - x^2/1 -> 0), except for the
            # plain normalisations flagged ``simple`` above, which keep it.
            if cand is not atom and (size > bare or (size == bare and not simple)):
                _count(self.stats, "reducer:prune_leaf_growth")
                continue
            if not self._slot_ok(states, cand):
                continue
            self.extend(order, [*states, cand])

    def _table(self, atoms: tuple[ir.SymExpr, ...]) -> dict[ir.SymExpr, ir.SymExpr]:
        """Expression table over ``atoms`` with each entry's singleton (``R = {j}``) form."""
        hit = self.tables.get(atoms)
        if hit is None:
            raw = grammar.expr_table(atoms, self.consts, self.problem.max_state_expr_nodes)
            hit = {}
            for f in raw:
                try:
                    hit[f] = singleton(f, self.R)
                except Unsupported:
                    continue
            self.tables[atoms] = hit
        return hit

    def _latent_justified(self, order: list[ir.SymExpr], states: list[ir.SymExpr]) -> bool:
        """Every latent atom must make some other state's leaf strictly simpler.

        A latent quantity earns its place only by simplifying what the reducer
        has to read per element (the running max turns ``exp(s)`` into ``1``,
        the count turns ``x²`` into ``0``); otherwise the set is a mere
        reparametrisation of a smaller one.
        """
        goal_set = set(self.goal_atoms)
        for i, atom in enumerate(order):
            if atom in goal_set:
                continue
            justified = False
            for k, st in enumerate(states):
                if k == i or atom not in grammar.atoms_in(st, self.R):
                    continue
                bare = grammar.complexity(singleton(order[k], self.R), self.opaque)
                got = grammar.complexity(singleton(st, self.R), self.opaque)
                if got < bare:
                    justified = True
                    break
            if not justified:
                return False
        return True

    def _slot_ok(self, states: list[ir.SymExpr], cand: ir.SymExpr) -> bool:
        """Per-slot checks after the leaf bounds: solvability and merge size."""
        prefix = (*states, cand)
        try:
            sol = solve_atoms(prefix, self.R)
            m = _merge_of(cand, sol, self.R, self.A, self.B) if sol is not None else None
        except Unsupported:
            sol, m = None, None
        if sol is None:
            _count(self.stats, "reducer:prune_unsolvable")
            return False
        if m is None:
            _count(self.stats, "reducer:prune_context")
            return False
        if node_count([m]) > self.problem.max_merge_nodes:
            _count(self.stats, "reducer:prune_merge")
            return False
        if not self._identity_ok(prefix, m):
            _count(self.stats, "reducer:prune_identity")
            return False
        return True

    def _identity_ok(self, prefix: tuple[ir.SymExpr, ...], merge_k: ir.SymExpr) -> bool:
        """Some identity element makes ``M_k(a, e) = a_k`` hold for this slot.

        Rejects early the states whose merge is undefined against an empty
        partition (``b1 / b0`` with a zero count), which the full derivation
        would only discover after the whole set is built.
        """
        k = len(prefix) - 1
        options: list[list[ir.SymExpr]] = []
        for st in prefix:
            try:
                e = subst_domain(st, {self.R: ir.dempty(self.problem.axis)})
            except Unsupported:
                e = None
            options.append(
                [e] if isinstance(e, ir.Const) else [ir.ZERO, ir.ONE, ir.NEG_INF_C, ir.POS_INF_C]
            )
        target = ir.state_var("a", k)
        for combo in itertools.product(*options):
            try:
                got = subst(merge_k, {ir.state_var("b", i): combo[i] for i in range(len(prefix))})
            except Unsupported:
                continue
            if got is target or verify.equal_modulo_max(got, target):
                return True
        return False


def _strip_constants(e: ir.SymExpr) -> ir.SymExpr:
    """Remove data-free factors, additive constants and a common data-free factor."""
    if isinstance(e, ir.Mul):
        keep = [a for a in e.args if not grammar.is_data_free(a)]
        if len(keep) != len(e.args):
            return _strip_constants(grammar.mk_mul(*keep)) if keep else ir.ONE
    if isinstance(e, ir.Add):
        keep = [a for a in e.args if not grammar.is_data_free(a)]
        if len(keep) != len(e.args):
            return _strip_constants(grammar.mk_add(*keep)) if keep else ir.ZERO
        parts = [_split_data_free(a) for a in e.args]
        common = parts[0][0]
        if common is not ir.ONE and all(p[0] is common for p in parts):
            return _strip_constants(grammar.mk_add(*[p[1] for p in parts]))
    return e


def _split_data_free(term: ir.SymExpr) -> tuple[ir.SymExpr, ir.SymExpr]:
    """``term = coeff * rest`` with ``coeff`` the product of its data-free factors."""
    if not isinstance(term, ir.Mul):
        return ir.ONE, term
    coeff = [a for a in term.args if grammar.is_data_free(a)]
    rest = [a for a in term.args if not grammar.is_data_free(a)]
    if not coeff:
        return ir.ONE, term
    return grammar.mk_mul(*coeff), (grammar.mk_mul(*rest) if rest else ir.ONE)


def _merge_of(state, sol, R, A, B) -> ir.SymExpr | None:
    """``I(A u B)`` written over the state values, or None if an atom is left unsolved."""
    atom_map = {}
    for atom, expr in sol.items():
        atom_map[subst_domain(atom, {R: A})] = expr
        atom_map[subst_domain(atom, {R: B})] = swap_sides(expr, "a", "b")
    m = subst(subst_domain(state, {R: ir.dunion(A, B)}), atom_map)
    if atoms_over(m, A) or atoms_over(m, B) or contains_node(m, _is_partial):
        return None
    return m


def _is_partial(x: ir.Node) -> bool:
    return isinstance(x, ir.Reduce) or (isinstance(x, ir.Card) and isinstance(x.domain, ir.DSym))


def _spec_key(spec: ReducerSpec):
    """Identity of a reducer modulo the order of its slots."""
    order = sorted(range(spec.arity), key=lambda k: (spec.leaves[k].uid, spec.merge[k].uid))
    ren = {}
    for new, old in enumerate(order):
        ren[ir.state_var("a", old)] = ir.state_var("a", new)
        ren[ir.state_var("b", old)] = ir.state_var("b", new)
    return tuple((spec.leaves[k], subst(spec.merge[k], ren), spec.identity[k]) for k in order)


def _index_space(exprs, index_axes: dict[str, DimKey]) -> tuple[tuple[str, DimKey], ...]:
    names: set[str] = set()
    for e in exprs:
        names |= e.free_idx

    return tuple((nm, index_axes[nm]) for nm in sorted(names, key=_index_order))


def _count(stats, what, n=1):
    if stats is not None:
        stats[what] = stats.get(what, 0) + n


def _derive(states, problem, full, R, A, B, occurrences, index_axes, stats) -> ReducerSpec | None:
    sol = solve_atoms(states, R)
    if sol is None:
        _count(stats, "reducer:unsolvable")
        return None
    n = len(states)
    # ---- finalize / relevance -------------------------------------------
    # Each state output is stood in for by a pseudo tensor element ``out_k[...]``
    # (tensor id -(k+1)) that keeps the state's free indices, so that a goal
    # nested inside another reduction still depends on that reduction's index
    # and cannot be canonicalised away.
    state_names = [sorted(subst_domain(st, {R: full}).free_idx, key=_index_order) for st in states]

    def out_ref(k: int, depth: int) -> ir.Elem:
        idxs = []
        for nm in state_names[k]:
            level = _nested_level(nm)
            idxs.append(ir.bidx(level) if level is not None and level < depth else ir.idx(nm))
        return ir.elem(-(k + 1), tuple(idxs))

    out_map = {}
    for atom, expr in sol.items():
        atom_full = subst_domain(atom, {R: full})
        out_map[atom_full] = subst(expr, {ir.state_var("a", k): out_ref(k, 0) for k in range(n)})
    for nested, un in occurrences:  # goals inside other reductions: same output, nested
        if un is nested or un not in out_map:
            continue
        atom = subst_domain(un, {full: R})
        if atom not in sol:
            continue
        out_map[nested] = subst(
            sol[atom], {ir.state_var("a", k): out_ref(k, nested.level) for k in range(n)}
        )
    fin = subst(problem.target, out_map)
    if atoms_over(fin, full):
        _count(stats, "reducer:goal_uncovered")
        return None
    # ---- merge by solving I(A u B) ----------------------------------------
    merge = []
    for st in states:
        m = _merge_of(st, sol, R, A, B)
        if m is None:
            _count(stats, "reducer:missing_context")
            return None
        merge.append(m)
    merge = tuple(merge)
    # relevance: every state must feed the target or another needed state
    needed = _state_refs(fin)
    changed = True
    while changed:
        changed = False
        for k in list(needed):
            for sv in _state_vars(merge[k]):
                if sv.k not in needed:
                    needed.add(sv.k)
                    changed = True
    if needed != set(range(n)):
        _count(stats, "reducer:irrelevant_state")
        return None
    if node_count(merge) > problem.max_merge_nodes:
        _count(stats, "reducer:merge_too_big")
        return None
    # ---- identity, leaves -------------------------------------------------
    candidates = []
    for s in states:
        try:
            e = subst_domain(s, {R: ir.dempty(problem.axis)})
        except Unsupported:  # e.g. 0/0 for a normalised output: pick by the identity law
            e = None
        if isinstance(e, ir.Const):
            candidates.append([e])
        else:
            candidates.append([ir.ZERO, ir.ONE, ir.NEG_INF_C, ir.POS_INF_C])
    leaves = tuple(singleton(s, R) for s in states)
    # ---- laws --------------------------------------------------------------
    proof = None
    identity = None
    for combo in itertools.product(*candidates):
        try:
            proof = verify.check_laws(merge, combo)
        except Unsupported:
            proof = None
        if proof is not None:
            identity = tuple(combo)
            break
    if proof is None:
        _count(stats, "reducer:law_failed")
        return None
    # ---- reduction invariant (holds by construction; checked anyway) -------
    inv_map = {}
    for k, s in enumerate(states):
        inv_map[ir.state_var("a", k)] = subst_domain(s, {R: A})
        inv_map[ir.state_var("b", k)] = subst_domain(s, {R: B})
    for k, s in enumerate(states):
        lhs = subst(merge[k], inv_map)
        rhs = subst_domain(s, {R: ir.dunion(A, B)})
        if lhs is not rhs and not verify.equal_modulo_max(lhs, rhs):
            _count(stats, "reducer:invariant_failed")
            return None
    closed = tuple(subst_domain(s, {R: full}) for s in states)
    _count(stats, "reducer:specs")
    space = _index_space([*closed, *leaves], index_axes)
    finalize = fin if _epilogue_only(fin) else None
    return ReducerSpec(
        problem.axis,
        tuple(states),
        leaves,
        merge,
        identity,
        closed,
        proof,
        space,
        finalize=finalize,
    )


def _epilogue_only(fin: ir.SymExpr) -> bool:
    """``fin`` combines state outputs elementwise (no inputs, no reductions, not a bare state)."""
    if isinstance(fin, ir.Elem):
        return False
    stack = [fin]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Reduce | ir.MonoidReduce | ir.BIdx):
            return False
        if isinstance(n, ir.Elem) and n.tensor >= 0:
            return False
        stack.extend(n.children())
    return True


def _state_refs(e: ir.Node) -> set[int]:
    """State slots referenced through ``out_k`` pseudo elements (tensor id -(k+1))."""
    out = set()
    stack = [e]
    while stack:
        n = stack.pop()
        if isinstance(n, ir.Elem) and n.tensor < 0:
            out.add(-n.tensor - 1)
        stack.extend(n.children())
    return out


def _index_order(name: str):
    return (0, int(name[1:])) if name.startswith("i") else (1, name)


def _nested_level(name: str) -> int | None:
    """Binder level of an un-nesting index name ``n{level}a{axis}`` (None for others)."""
    if name.startswith("n") and "a" in name:
        return int(name[1 : name.index("a")])
    return None


__all__ = [
    "ReducerSpec",
    "SynthesisProblem",
    "atoms_over",
    "collect_goals",
    "extract_goals",
    "singleton",
    "solve_atoms",
    "synthesize",
]
