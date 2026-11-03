"""Measure every fusible operator of the registry and write the facts to JSON.

Adapted from the method documentation's ``collect_fused.py`` (super_opt worktree,
reports/归约链融合说明文档/scripts). For each operator: the program before fusion as
TE text, what the judgement says before anything is derived, the synthesised
reducer as TE text, its states and how each was obtained, the time ``fuse`` took
(judgement, derivation and every check), the passes over the reduction axis
before and after, the accuracy gate's per-family numbers, and the error of the
three forms (unfused, fused by hand, synthesised) against a float64 reference on
centred data and on data shifted by 1e4.

    Hapax/scripts/py Hapax/scripts/collect_fused.py Hapax/data/fused_ops.json [op ...]
"""

import sys
import time

import common  # noqa: F401  (sets sys.path)
import numpy as np
from chain_ops import OPERATORS
from chain_ops.run import extents, passes, rel_err, run_llvm, sample

from tvm.te.superopt import AccuracyConfig, fuse

OFFSET = 1.0e4
EXTENT = 256
SEED = 0


def derived_comments(chain):
    """Render the new supplement origin missing from HEAD's report formatter.

    This affects report comments only; selection, emitted code and proofs are
    those of the unmodified compiler revision recorded in the data.
    """
    from tvm.te.superopt.reducer.source import render

    try:
        return chain.derived()
    except KeyError as error:
        if error.args != ("trivial",):
            raise
        return [
            f"{state.name} = {render(state.struct, chain.chain)} ({state.origin})"
            for state in chain.solution.auxiliaries
        ]


def report_source(result):
    parts = []
    axes = [chain.axis for chain in result.chains if chain.fused]
    for index, chain in enumerate(result.chains):
        if not chain.fused:
            parts.append(f"# {chain.summary()}")
            continue
        tag = chain.axis if axes.count(chain.axis) == 1 else f"{chain.axis}{index}"
        comments = [f"# {chain.summary()}"]
        comments += [f"#   {line}" for line in derived_comments(chain)]
        parts.append("\n".join([*comments, chain.source(tag)]))
    return "\n\n".join(parts)


def measure(ins, outs, op, values):
    errs = []
    for shift in (0.0, OFFSET):
        arrays = sample(ins, outs, values, SEED, shift=shift, positive=op.positive)
        with np.errstate(all="ignore"):
            errs.append(rel_err(run_llvm(ins, outs, arrays, values), op.reference(*arrays)))
    return errs


def collect(name: str) -> dict:
    op = OPERATORS[name]
    ins, outs = op.unfused()
    config = AccuracyConfig(positive=op.positive)
    start = time.perf_counter()
    res = fuse(outs, ins, max_states=op.max_states, accuracy_config=config)
    seconds = time.perf_counter() - start
    row = {
        "name": name,
        "group": op.group,
        "doc": op.doc,
        "showcase": None
        if op.showcase is None
        else {"domain": op.showcase.domain, "baselines": list(op.showcase.baselines)},
        "max_states": op.max_states,
        "positive": list(op.positive),
        "fused": res.fused,
        "seconds": seconds,
        "before_te": common.program_text(ins, outs),
        "summary": res.summary(),
        "chains": [],
    }
    if not res.fused:
        return row
    row["after_te"] = report_source(res)
    values = extents(ins, outs, reduce=EXTENT)
    row["extents"] = values
    axes = list({str(c.chain.extent): c.chain.extent for c in res.chains}.values())
    row["passes_before"] = [passes(outs, e) for e in axes]
    row["passes_after"] = [passes(res.outputs, e) for e in axes]
    hand_ins, hand_outs = op.fused()
    row["passes_hand"] = [passes(hand_outs, e) for e in axes]
    row["err_unfused"] = measure(ins, outs, op, values)
    row["err_hand"] = measure(hand_ins, hand_outs, op, values)
    row["err_synth"] = measure(res.inputs, res.outputs, op, values)
    row["gate"] = [
        {
            "family": f.family,
            "original": f.original,
            "inherent": f.inherent,
            "candidate": f.candidate,
            "ok": f.ok,
        }
        for f in res.accuracy.families
    ]
    for c in res.chains:
        row["chains"].append(
            {
                "axis": c.axis,
                "members": list(c.members),
                "original": c.original(),
                "required": list(c.required),
                "states": list(c.states),
                "auxiliaries": list(c.auxiliaries),
                "derived": derived_comments(c),
                "origins": [s.origin for s in c.solution.states],
                "proof": c.solution.spec.proof,
                "rejected": len(c.rejected),
                "selection": [st.summary() for st in c.selection.steps if st.action != "keep"],
                "route": c.selection.route,
                "search": [
                    {"stage": attempt.stage, "status": attempt.status,
                     "candidates": attempt.candidates, "detail": attempt.detail}
                    for attempt in c.selection.search
                ],
                "actions": [st.action for st in c.selection.steps if st.action != "keep"],
                "theory": c.theory.summary(),
                "theory_states": c.theory.states,
                "theory_reductions": list(c.theory.reductions),
                "theory_counted": c.theory.counted,
            }
        )
    return row


def main() -> None:
    out = sys.argv[1]
    names = sys.argv[2:] or [n for n, op in OPERATORS.items() if op.fusible]
    # one fusion first, so that imports and the LLVM start-up are not charged to the first row
    warm_ins, warm_outs = OPERATORS["softmax"].unfused()
    fuse(warm_outs, warm_ins)
    rows = []
    for name in names:
        row = collect(name)
        rows.append(row)
        print(f"{name}: fused={row['fused']} {row['seconds']:.2f}s", flush=True)
    meta = common.meta(
        "collect_fused.py",
        extent=EXTENT,
        seed=SEED,
        offset=OFFSET,
        dtype="float32",
        registry=len(OPERATORS),
        registry_fusible=sum(op.fusible for op in OPERATORS.values()),
    )
    common.dump(out, {"meta": meta, "rows": rows})


if __name__ == "__main__":
    main()
