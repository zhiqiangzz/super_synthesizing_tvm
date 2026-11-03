"""Run the prior papers' own programs (programs/prior.py) through Hapax.

For each program: whether the frontend reads it, what the judgement says, whether
``fuse`` fuses it, with how many states and passes, and the reason it gives
otherwise. The RedFuser Appendix A.1 program is also run, fused, on the data of
our counterexample (u = (1, -1 | 3, 0), v = (1, 1 | 1, 0); true y = 9).

Each program runs in its own process (killed at LIMIT seconds), like collect_unfused.

    Hapax/scripts/py Hapax/scripts/collect_prior.py Hapax/data/prior.json
"""

import json
import subprocess
import sys
import time

import common
import numpy as np
from prior import PRIORS
from collect_fused import report_source

from tvm.te.superopt import AccuracyConfig, fuse, judge

LIMIT = 300
BY_KEY = {p.key: p for p in PRIORS}


def a1_numbers(res) -> dict:
    """The fused A.1 program on the counterexample data, in float32 and float64."""
    from chain_ops.run import run_llvm

    u = np.array([[1.0, -1.0, 3.0, 0.0]], dtype="float32")
    v = np.array([[1.0, 1.0, 1.0, 0.0]], dtype="float32")
    values = {"n": 1, "m": 4}
    got = run_llvm(res.inputs, res.outputs, [u, v], values)[0]
    return {"u": u.tolist(), "v": v.tolist(), "truth": float(u.sum() * v.sum()), "fused": got.tolist()}


def examine(key: str) -> dict:
    p = BY_KEY[key]
    row = {"key": key, "paper": p.paper, "where": p.where, "what": p.what, "spelling": p.spelling}
    ins, outs = p.build()
    try:
        row["before_te"] = common.program_text(ins, outs)
    except Exception as err:  # noqa: BLE001
        row["text_raised"] = f"{type(err).__name__}: {err}"
    try:
        js = judge(outs)
        row["judged"] = [j.fusible for j in js]
        row["judged_states"] = [j.states for j in js]
        row["theory"] = [j.summary() for j in js]
    except Exception as err:  # noqa: BLE001
        row["judge_raised"] = f"{type(err).__name__}: {str(err)[:300]}"
    print("JUDGED " + json.dumps(row, ensure_ascii=False, default=str), flush=True)
    start = time.perf_counter()
    try:
        res = fuse(outs, ins, accuracy_config=AccuracyConfig(positive=p.positive))
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
            "auxiliaries": list(c.auxiliaries) if c.fused else [],
            "reason": c.reason,
            "selection": []
            if c.selection is None
            else [st.summary() for st in c.selection.steps if st.action != "keep"],
        }
        for c in res.chains
    ]
    row["skipped"] = [{"members": s.members, "reason": s.reason} for s in res.skipped]
    if res.fused:
        row["after_te"] = report_source(res)
    if key == "a1_program" and res.fused:
        row["a1_numbers"] = a1_numbers(res)
    return row


def main() -> None:
    rows = []
    for p in PRIORS:
        row = {"key": p.key, "paper": p.paper, "where": p.where, "what": p.what, "spelling": p.spelling}
        try:
            run = subprocess.run(
                [sys.executable, __file__, "--one", p.key],
                capture_output=True,
                text=True,
                timeout=LIMIT,
                check=False,
            )
            lines = run.stdout.splitlines()
            if run.returncode != 0 and not any(ln.startswith("RESULT ") for ln in lines):
                row["crashed"] = run.stderr.strip().splitlines()[-3:] if run.stderr else []
        except subprocess.TimeoutExpired as err:
            row["timeout"] = LIMIT
            text = err.stdout or ""
            lines = (text.decode() if isinstance(text, bytes) else text).splitlines()
        for tag in ("JUDGED ", "RESULT "):
            hit = [ln for ln in lines if ln.startswith(tag)]
            if hit:
                row |= json.loads(hit[-1][len(tag) :])
        rows.append(row)
        print(p.key, row.get("fused"), row.get("raised") or row.get("crashed") or "", flush=True)
    common.dump(sys.argv[1], {"meta": common.meta("collect_prior.py", limit=LIMIT), "rows": rows})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        print("RESULT " + json.dumps(examine(sys.argv[2]), ensure_ascii=False, default=str), flush=True)
    else:
        main()
