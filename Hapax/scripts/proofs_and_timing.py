"""How each fused chain is proven, and how long each stage takes.

For every fusible registry operator and each of its chains:

* the method that proved the merge invariant M(I(A), I(B)) = I(A u B)
  (``"canonical"``: the two sides are one hash-consed node; ``"case-split"``: on
  the arguments of an open max; ``"sympy"``: ``cancel(together(l - r)) == 0``);
* for the printed (re-based) merge, slot by slot, how it was proven equal to the
  canonical merge (the method documentation's ``printed_vs_canonical.py``,
  adjusted to HEAD's ``printed_merges``);
* wall-clock time of the judgement alone, of state selection (which derives the
  reducer and discharges every proof), and of the whole ``fuse`` (which adds the
  floating-point checks and two LLVM compilations), each the median of REPEATS
  runs after a warm-up.

    Hapax/scripts/py Hapax/scripts/proofs_and_timing.py Hapax/data/proofs_timing.json
"""

import statistics
import sys
import time

import common  # noqa: F401
from chain_ops import OPERATORS

from tvm.te.superopt import AccuracyConfig, fuse
from tvm.te.superopt.reducer import rebase, states, verify
from tvm.te.superopt.reducer.chain import discover_chains
from tvm.te.superopt.reducer.derive import closed_merges, reads_merged
from tvm.te.superopt.reducer.domain import TeSigns
from tvm.te.superopt.reducer.states import members_of
from tvm.te.superopt.reducer.theory import judge as judge_chain

REPEATS = 5
_orig = states.printed_merges
_log: list = []


def _spy(printer, merge):
    prints = [printer.print(c) for c in printer.chosen]
    if any(p is None for p in prints):
        _log.append({"printed": False, "slots": []})
        return _orig(printer, merge)
    expanded = rebase._expand(prints)
    closed = closed_merges(merge)
    if expanded is None or closed is None:
        _log.append({"printed": False, "slots": []})
        return _orig(printer, merge)
    result = _orig(printer, merge)
    how = []
    for p, e, m, c in zip(prints, expanded, merge, closed):
        method = verify.proves(p, m) if reads_merged(m) else verify.proves(e, c)
        how.append(method or ("smt-reachable" if result is not None else None))
    _log.append({"printed": result is not None, "slots": how})
    return result


states.printed_merges = _spy


def median_time(fn) -> dict:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return {"median": statistics.median(times), "min": min(times), "max": max(times)}


def prepare(outputs):
    """Use the same index-view specialisation as fuse before timing its stages."""
    chains, skipped = discover_chains(outputs)
    if any("two different indices" in item.reason for item in skipped):
        from tvm.te.superopt.reducer.views import specialize

        outputs = specialize(outputs)
        chains, _ = discover_chains(outputs)
    return outputs, chains


def one(name: str) -> dict:
    op = OPERATORS[name]
    ins, outs = op.unfused()
    prepared_outputs, chains = prepare(outs)
    signs = TeSigns(prepared_outputs, op.positive)
    row = {"name": name, "chains": []}
    for chain in chains:
        _log.clear()
        sel = states.select(chain, members_of(chain), signs, max_states=op.max_states)
        sol = sel.solution
        row["chains"].append(
            {
                "members": [r.tensor.op.name for r in chain.reductions],
                "selected": sol is not None,
                "invariant_proof": None if sol is None else sol.spec.proof,
                "printed": _log[-1] if _log else None,
            }
        )

    def judge_all():
        _, cs = prepare(outs)
        for c in cs:
            judge_chain(c, members_of(c))

    def select_all():
        prepared, cs = prepare(outs)
        for c in cs:
            states.select(c, members_of(c), TeSigns(prepared, op.positive), max_states=op.max_states)

    def fuse_all():
        fuse(outs, ins, max_states=op.max_states, accuracy_config=AccuracyConfig(positive=op.positive))

    # Proof classification is collected above, outside the timed calls.
    states.printed_merges = _orig
    row["t_judge"] = median_time(judge_all)
    row["t_select"] = median_time(select_all)
    row["t_fuse"] = median_time(fuse_all)
    states.printed_merges = _spy
    return row


def main() -> None:
    names = sys.argv[2:] or [n for n, op in OPERATORS.items() if op.fusible]
    ins, outs = OPERATORS["softmax"].unfused()
    fuse(outs, ins)  # warm-up: imports, LLVM start-up
    rows = []
    for n in names:
        rows.append(one(n))
        r = rows[-1]
        print(n, r["t_judge"]["median"], r["t_select"]["median"], r["t_fuse"]["median"], flush=True)
    common.dump(
        sys.argv[1],
        {"meta": common.meta("proofs_and_timing.py", repeats=REPEATS), "rows": rows},
    )


if __name__ == "__main__":
    main()
