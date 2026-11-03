"""Every cited key must appear in bib/VERIFIED.md (and in bib/refs.bib).

    python3 -I Hapax/scripts/check_cites.py build/main.aux [more.aux ...]
"""

import re
import sys
from pathlib import Path

HAPAX = Path(__file__).resolve().parents[1]
verified = set(re.findall(r"^\| (\w+) \|", (HAPAX / "bib" / "VERIFIED.md").read_text(), re.M))
bib = set(re.findall(r"^@\w+\{(\w+),", (HAPAX / "bib" / "refs.bib").read_text(), re.M))
cited = set()
for aux in sys.argv[1:]:
    p = HAPAX / aux if not Path(aux).is_absolute() else Path(aux)
    if p.exists():
        for m in re.findall(r"\\citation\{([^}]*)\}", p.read_text()):
            cited.update(k.strip() for k in m.split(","))
missing_v = sorted(cited - verified)
missing_b = sorted(cited - bib)
unused = sorted(bib - cited)
print(f"  cited {len(cited)} keys; {len(bib)} in refs.bib; {len(unused)} unused")
if missing_b:
    print("  FAIL: cited but not in refs.bib:", ", ".join(missing_b))
if missing_v:
    print("  FAIL: cited but not in VERIFIED.md:", ", ".join(missing_v))
sys.exit(1 if (missing_v or missing_b) else 0)
