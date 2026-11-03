"""Compare the paper's macros between two data snapshots and write data/CHANGES.md.

    python3 -I Hapax/scripts/changes.py data/archive-aec28bdb85

Generates the macros of the archived snapshot into a temporary directory (stale
data allowed there), reads the current gen/stats.tex, and lists every macro whose
value differs, with its source line.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HAPAX = Path(__file__).resolve().parents[1]
MACRO = re.compile(r"^\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$")


def read(path: Path) -> dict:
    out, src, last = {}, {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = MACRO.match(line)
        if m:
            last = m.group(1)
            out[last] = m.group(2)
        elif line.startswith("%   src:") and last:
            src[last] = line[len("%   src:") :].strip()
    return out, src


def main() -> None:
    old_dir = (HAPAX / sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, HAPAX_DATA=str(old_dir), HAPAX_GEN=tmp)
        subprocess.run(
            [sys.executable, "-I", str(HAPAX / "scripts" / "gen_tex.py"), "--allow-stale"],
            env=env, check=True, capture_output=True,
        )
        old, _ = read(Path(tmp) / "stats.tex")
    new, src = read(HAPAX / "gen" / "stats.tex")
    rows = []
    for k in sorted(set(old) | set(new)):
        if old.get(k) != new.get(k) and k not in ("dataCommit", "dataStale"):
            rows.append((k, old.get(k, "--"), new.get(k, "--"), src.get(k, "")))
    lines = [
        f"# Changes between `{old_dir.name}` and the current data",
        "",
        f"Old commit: `{old.get('dataCommit')}`; new commit: `{new.get('dataCommit')}`.",
        f"{len(rows)} of {len(new)} macros differ.",
        "",
        "| macro | old | new | source |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{k}` | {o} | {n} | {s} |" for k, o, n, s in rows]
    (HAPAX / "data" / "CHANGES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
