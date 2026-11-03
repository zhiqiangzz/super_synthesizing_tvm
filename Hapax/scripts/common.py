"""Shared helpers for the Hapax data scripts.

Every script is run through ``Hapax/scripts/py`` (this worktree's python/tvm and
compiled libraries) from the worktree root. Each JSON a script writes carries a
``meta`` block saying which commit of the tool produced it, so that the paper's
generated macros can refuse stale data.
"""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

HAPAX = Path(__file__).resolve().parents[1]
ROOT = HAPAX.parent  # the worktree
sys.path.insert(0, str(ROOT / "super_opt" / "examples"))
sys.path.insert(0, str(HAPAX / "scripts" / "programs"))

# the tool's own sources; a snapshot is valid only if these are unchanged at HEAD
TOOL_PATHS = ["python/tvm/te/superopt", "super_opt/examples", "tests/python/te/superopt"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def meta(script: str, **extra) -> dict:
    """Provenance of a data file: commit, cleanliness of the tool tree, versions."""
    import numpy
    import sympy
    import z3
    import cvc5

    dirty = bool(_git("status", "--porcelain", "--", *TOOL_PATHS))
    return {
        "commit": _git("rev-parse", "HEAD"),
        "commit_date": _git("show", "-s", "--format=%cI", "HEAD"),
        "dirty": dirty,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "script": script,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "sympy": sympy.__version__,
        "z3": z3.get_version_string(),
        "cvc5": cvc5.__version__,
        "machine": platform.machine(),
        "cpu": _cpu(),
        **extra,
    }


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=_default)
        f.write("\n")


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ---------------------------------------------------------------------------
# TE programs as text (the form a reader would type)
# ---------------------------------------------------------------------------
def program_text(inputs, outputs) -> str:
    from tvm import te
    from tvm.te.superopt.loads import is_load, load_indices, load_tensor
    from tvm.te.superopt.symbolic.lower import classify_combiner, compute_ops
    from tvm_tirx import tirx as tir
    from tvm_tirx.ir import Call, FloatImm, IntImm

    prec = {tir.Add: 1, tir.Sub: 1, tir.Mul: 2, tir.Div: 2}
    infix = {tir.Add: "+", tir.Sub: "-", tir.Mul: "*", tir.Div: "/"}
    calls = {"pow": "tir.power", "fabs": "tir.abs"}

    def text(e, parent: int = 0) -> str:
        if isinstance(e, FloatImm):
            v = float(e.value)
            return repr(int(v)) + ".0" if v == int(v) and abs(v) < 1e6 else repr(v)
        if isinstance(e, IntImm):
            return str(int(e.value))
        if isinstance(e, tir.Var):
            return e.name
        if isinstance(e, tir.Cast):
            inner = text(e.value, 9)
            dtype = getattr(getattr(e, "ty", None), "dtype", None) or str(getattr(e, "dtype", "float32"))
            return f'{inner}.astype("{dtype}")' if isinstance(e.value, tir.Var) else inner
        for cls, sym in infix.items():
            if isinstance(e, cls):
                p = prec[cls]
                out = f"{text(e.a, p)} {sym} {text(e.b, p + 1)}"
                return f"({out})" if p < parent else out
        if isinstance(e, tir.Max):
            return f"tir.max({text(e.a)}, {text(e.b)})"
        if isinstance(e, tir.Min):
            return f"tir.min({text(e.a)}, {text(e.b)})"
        if is_load(e):
            idx = ", ".join(text(i) for i in load_indices(e))
            return f"{load_tensor(e).op.name}[{idx}]"
        if isinstance(e, Call):
            short = e.op.name.split(".", 1)[-1]
            name = calls.get(short, f"tir.{short}")
            return f"{name}({', '.join(text(a) for a in e.args)})"
        if isinstance(e, te.Reduce):
            kind = classify_combiner(e.combiner)
            axis = ", ".join(iv.var.name for iv in e.axis)
            if kind is not None:
                return f"te.{kind}({text(e.source[0])}, axis={axis})"
            srcs = ", ".join(text(s) for s in e.source)
            return f"<tuple reducer>(({srcs}), axis={axis})[{int(e.value_index)}]"
        return str(e)

    def shape(sh) -> str:
        parts = [text(s) for s in sh]
        return "(" + ", ".join(parts) + ("," if len(parts) == 1 else "") + ")"

    lines = []
    for t in inputs:
        lines.append(f'{t.op.name} = te.placeholder({shape(t.shape)}, name="{t.op.name}")')
    for op in compute_ops(outputs):
        axes = ", ".join(iv.var.name for iv in op.axis)
        body = op.body[0]
        if isinstance(body, te.Reduce):
            for iv in body.axis:
                lines.append(
                    f'{iv.var.name} = te.reduce_axis((0, {text(iv.dom.extent)}), "{iv.var.name}")'
                )
        lines.append(
            f"{op.name} = te.compute({shape([iv.dom.extent for iv in op.axis])}, "
            f'lambda {axes}: {text(body)}, name="{op.name}")'
        )
    lines.append("# outputs: " + ", ".join(t.op.name for t in outputs))
    return "\n".join(lines)
