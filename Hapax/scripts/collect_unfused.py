"""What the tool says about programs it does not fuse, measured on the current code.

Adapted from the method documentation's ``collect_unfused.py``. Three sources: the
negative operators of the registry, the survey candidates that were not fused
when the survey was taken (``programs/survey.py``, a copy of the survey's
``cands.py``), and three programs written for the documentation. For each: what
the judgement says before anything is derived, whether ``fuse`` then fuses it
within that many states and a time limit, the reason it reports, and what
selection did.

Every program runs in a process of its own and is killed at the time limit.

    Hapax/scripts/py Hapax/scripts/collect_unfused.py Hapax/data/unfused.json
"""

import json
import subprocess
import sys
import time

import common
import survey as cands
from chain_ops import OPERATORS

from tvm import te
from tvm_tirx import tirx as tir
from tvm.te.superopt import fuse, judge

LIMIT = 300
F = "float32"


def _rows():
    return te.placeholder((te.var("n"), te.var("m")), name="X", dtype=F)


def _mean(X):
    n, m = X.shape
    j = te.reduce_axis((0, m), "j")
    total = te.compute((n,), lambda i: te.sum(X[i, j], axis=j), name="total")
    return te.compute((n,), lambda i: total[i] / m.astype(F), name="mean")


def nearest_to_mean():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    out = te.compute((n,), lambda i: te.min(tir.abs(X[i, j] - mean[i]), axis=j), name="out")
    return [X], [out]


def moving_maximum():
    X = _rows()
    n, m = X.shape
    mean = _mean(X)
    j = te.reduce_axis((0, m), "j")
    out = te.compute(
        (n,), lambda i: te.max(mean[i] * X[i, j] - X[i, j] * X[i, j], axis=j), name="out"
    )
    return [X], [out]


def scale_of_unknown_sign():
    n, m = te.var("n"), te.var("m")
    X = te.placeholder((n, m), name="X", dtype=F)
    G = te.placeholder((n, m), name="G", dtype=F)
    j1, j2 = te.reduce_axis((0, m), "j"), te.reduce_axis((0, m), "j")
    g = te.compute((n,), lambda i: te.sum(G[i, j1], axis=j1), name="g")
    out = te.compute((n,), lambda i: te.max(g[i] * X[i, j2], axis=j2), name="out")
    return [X, G], [out]


EXTRA = {
    "nearest_to_mean": ("min_j |x - mean|", nearest_to_mean),
    "moving_maximum": ("max_j (mean x - x^2)", moving_maximum),
    "scale_of_unknown_sign": ("max_j (g x), g = sum G of either sign", scale_of_unknown_sign),
}
SURVEY = ["05", "10", "11a", "11b", "13", "14", "15", "16", "19", "21", "22", "28", "29a"]
SURVEY += ["33", "37a", "37b", "38"]


def examine(key: str, doc: str, build, max_states=None) -> dict:
    row = {"key": key, "doc": doc, "max_states": max_states}
    try:
        ins, outs = build()
    except Exception as err:  # noqa: BLE001
        row["raised"] = f"building the program: {type(err).__name__}: {err}"
        return row
    try:
        row["before_te"] = common.program_text(ins, outs)
    except Exception:  # noqa: BLE001
        pass
    try:
        judgements = judge(outs)
        row["judged"] = [j.fusible for j in judgements]
        row["judged_states"] = [j.states for j in judgements]
        row["theory"] = [j.summary() for j in judgements]
    except Exception as err:  # noqa: BLE001
        row["judge_raised"] = f"{type(err).__name__}: {str(err)[:300]}"
    print("JUDGED " + json.dumps(row, ensure_ascii=False), flush=True)
    start = time.perf_counter()
    try:
        res = fuse(outs, ins, max_states=max_states)
    except Exception as err:  # noqa: BLE001
        row["raised"] = f"{type(err).__name__}: {str(err)[:300]}"
        return row
    row["seconds"] = time.perf_counter() - start
    row["fused"] = res.fused
    row["summary"] = res.summary()
    row["chains"] = [
        {
            "members": list(c.members),
            "fused": c.fused,
            "states": list(c.states),
            "reason": c.reason,
            "rejected": [{"states": list(r.states), "why": r.why} for r in c.rejected[:6]],
            "rejected_total": len(c.rejected),
            "selection": []
            if c.selection is None
            else [st.summary() for st in c.selection.steps if st.action != "keep"],
        }
        for c in res.chains
    ]
    row["skipped"] = [{"members": s.members, "reason": s.reason} for s in res.skipped]
    return row


def programs() -> dict:
    """key -> (doc, build, max_states, source)."""
    out = {}
    for name, op in OPERATORS.items():
        if not op.fusible:
            out[name] = (op.doc, op.unfused, op.max_states, "registry")
    for key in SURVEY:
        c = cands.CANDS[key]
        out[f"{key} {c.name}"] = (c.doc, c.build, None, "survey")
    for name, (doc, build) in EXTRA.items():
        out[name] = (doc, build, None, "extra")
    return out


def one(key: str) -> None:
    doc, build, max_states, source = programs()[key]
    row = examine(key, doc, build, max_states) | {"source": source}
    print("RESULT " + json.dumps(row, ensure_ascii=False, default=str), flush=True)


def main() -> None:
    rows = []
    for key, (doc, _, max_states, source) in programs().items():
        row = {"key": key, "doc": doc, "max_states": max_states, "source": source}
        try:
            run = subprocess.run(
                [sys.executable, __file__, "--one", key],
                capture_output=True,
                text=True,
                timeout=LIMIT,
                check=False,
            )
            lines = run.stdout.splitlines()
            if run.returncode != 0 and not any(ln.startswith("RESULT ") for ln in lines):
                row["crashed"] = run.stderr.strip().splitlines()[-1:] if run.stderr else []
        except subprocess.TimeoutExpired as err:
            row["timeout"] = LIMIT
            text = err.stdout or ""
            lines = (text.decode() if isinstance(text, bytes) else text).splitlines()
        for tag in ("JUDGED ", "RESULT "):
            hit = [ln for ln in lines if ln.startswith(tag)]
            if hit:
                row |= json.loads(hit[-1][len(tag) :])
        row["source"] = source
        rows.append(row)
        print(key, row.get("fused"), row.get("timeout"), row.get("raised"), flush=True)
    common.dump(sys.argv[1], {"meta": common.meta("collect_unfused.py", limit=LIMIT), "rows": rows})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        one(sys.argv[2])
    else:
        main()
