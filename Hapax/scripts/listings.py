"""Write the paper's code listings from the data, never by hand.

From data/fused_ops.json: the TE program before fusion and the synthesised TE
after it, for the operators the paper shows. Long ``te.compute`` lines are broken
at their arguments; non-ASCII symbols in comments are spelled in ASCII so that
``listings`` can typeset them. The unscheduled TVMScript of the fused variance
is produced by fusing again (``Fused.prim_func().script()``, which calls ``te.create_function``).

    Hapax/scripts/py Hapax/scripts/listings.py
"""

import json
import re

import common
from chain_ops import OPERATORS

from tvm.te.superopt import AccuracyConfig, fuse

SHOW = ["softmax", "variance", "layernorm", "attention", "kurtosis", "max_abs_deviation"]
WIDTH = 92
ASCII = {"Σ": "sum", "²": "^2", "·": "*", "−": "-", "√": "sqrt", "≤": "<=", "≥": ">="}


def ascii_only(s: str) -> str:
    for k, v in ASCII.items():
        s = s.replace(k, v)
    return s


def split_args(s: str) -> list[str]:
    """Top-level comma split of the argument list of a call."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def wrap(line: str) -> str:
    if len(line) <= WIDTH:
        return line
    m = re.match(r"^(\s*)(\w+ = te\.compute)\((.*)\)$", line)
    if not m:
        return line
    indent, head, args = m.groups()
    # the commas between a lambda's parameters are not argument separators
    protected = re.sub(r"lambda [^:]*:", lambda mm: mm.group(0).replace(",", "\0"), args)
    parts = [p.replace("\0", ",") for p in split_args(protected)]
    inner = indent + "    "
    return f"{indent}{head}(\n" + ",\n".join(inner + p for p in parts) + f",\n{indent})"


def before_text(s: str) -> str:
    lines = [ln for ln in s.splitlines() if not ln.startswith("# outputs")]
    return "\n".join(wrap(ln) for ln in lines) + "\n"


def main() -> None:
    rows = {r["name"]: r for r in json.load(open(common.HAPAX / "data" / "fused_ops.json"))["rows"]}
    out_dir = common.HAPAX / "listings"
    out_dir.mkdir(exist_ok=True)
    manifest = {}
    for name in SHOW:
        r = rows[name]
        if not r.get("fused"):
            for suffix in ("before", "after"):
                (out_dir / f"{name}-{suffix}.py").unlink(missing_ok=True)
            continue
        before = ascii_only(before_text(r["before_te"]))
        after = ascii_only(r["after_te"].rstrip() + "\n")
        (out_dir / f"{name}-before.py").write_text(before, encoding="utf-8")
        (out_dir / f"{name}-after.py").write_text(after, encoding="utf-8")
        manifest[name] = {
            "before_lines": before.count("\n"),
            "after_lines": after.count("\n"),
            "before_width": max(len(x) for x in before.splitlines()),
            "after_width": max(len(x) for x in after.splitlines()),
        }
    op = OPERATORS["variance"]
    ins, outs = op.unfused()
    res = fuse(outs, ins, accuracy_config=AccuracyConfig(positive=op.positive))
    script = res.prim_func().script()
    (out_dir / "variance-fused.tvmscript.py").write_text(ascii_only(script) + "\n", encoding="utf-8")
    manifest["variance_tvmscript_lines"] = script.count("\n") + 1
    common.dump(out_dir / "MANIFEST.json", {"meta": common.meta("listings.py"), "files": manifest})
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
