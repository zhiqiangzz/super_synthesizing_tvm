"""Check the prose-only Chinese translation against its English source.

Headings, floats, algorithms, listings, equations, inline math, footnotes,
references and generated numerical macros must retain their English content.
A source fingerprint also detects later English prose changes needing translation.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = re.compile(
    r"\\begin\{(paperdiagram|wrapfigure|figure|table|algorithm|equation|align\*?|longtable)\}"
    r"[\s\S]*?\\end\{\1\}|\\\[[\s\S]*?\\\]|^\\lstinputlisting[^\n]*",
    re.M,
)
MATH = re.compile(r"(?<!\\)\$[^$]*(?<!\\)\$")
REFS = re.compile(r"\\(?:[cC]ref|ref|label|citep|citet)(?:\[[^\]]*\])*\{[^}]*\}")
HEADINGS = re.compile(r"\\(?:section\*?|subsection|subsubsection|paragraph|textbf|footnote)\{")


def commands(pattern, text):
    """Read balanced TeX arguments, including nested commands in headings."""
    result = []
    for match in pattern.finditer(text):
        depth = 1
        pos = match.end()
        while depth and pos < len(text):
            if text[pos] in "{}" and text[pos - 1] != "\\":
                depth += 1 if text[pos] == "{" else -1
            pos += 1
        if depth:
            raise ValueError(f"Unbalanced argument near {match.group()}")
        result.append(text[match.start():pos])
    return result


def main():
    failures = []
    fingerprints = json.loads((ROOT / "zh/source-manifest.json").read_text())
    for relative, digest in fingerprints.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != digest:
            failures.append(f"English source changed; review translation: {relative}")
    numerical = re.findall(r"\\newcommand\{(\\[A-Za-z]+)\}", (ROOT / "gen/stats.tex").read_text())
    macro_pattern = re.compile("(?:" + "|".join(re.escape(x) for x in numerical) + r")(?![A-Za-z])")
    for english in sorted((ROOT / "sections").glob("*.tex")):
        original = english.read_text()
        chinese = (ROOT / "zh/sections" / english.name).read_text()
        checks = {
            "figures, tables, algorithms, listings and displayed equations":
                ([m.group() for m in BLOCKS.finditer(original)], [m.group() for m in BLOCKS.finditer(chinese)]),
            "inline math": (MATH.findall(original), MATH.findall(chinese)),
            "headings and footnotes": (commands(HEADINGS, original), commands(HEADINGS, chinese)),
            "references and citations": (REFS.findall(original), REFS.findall(chinese)),
            "numerical macros": (macro_pattern.findall(original), macro_pattern.findall(chinese)),
            "theorem/proof titles": (
                re.findall(r"\\begin\{(?:theorem|lemma|definition|example|proposition|proof)\}(?:\[[^\]]*\])?", original),
                re.findall(r"\\begin\{(?:theorem|lemma|definition|example|proposition|proof)\}(?:\[[^\]]*\])?", chinese)),
        }
        for kind, (left, right) in checks.items():
            if Counter(left) != Counter(right):
                failures.append(f"{english.name}: changed {kind}: {Counter(left) - Counter(right)} / {Counter(right) - Counter(left)}")
        if not re.search(r"[\u4e00-\u9fff]", chinese):
            failures.append(f"{english.name}: no Chinese prose")
    original = (ROOT / "main.tex").read_text()
    chinese = (ROOT / "main-zh.tex").read_text()
    restored = chinese.replace("\\input{preamble/chinese}\n", "").replace("\\input{zh/sections/", "\\input{sections/")
    if restored != original:
        failures.append("Main document changed beyond Chinese font support and translated section paths")
    if "--pdf" in sys.argv:
        log = (ROOT / "build/main-zh.log").read_text()
        for pattern in [r"(?:Reference|Citation) .* undefined", r"Missing character:", r"^!", r"Font .* not found"]:
            if re.search(pattern, log, re.M):
                failures.append(f"PDF log contains {pattern}")
        over = re.findall(r"Overfull \\hbox \(([0-9.]+)pt", log)
        if any(float(x) > 1 for x in over):
            failures.append(f"Overfull boxes > 1pt: {over}")
        pdf = ROOT / "build/main-zh.pdf"
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        if not re.search(r"Page size:\s+486 x 720 pts", info):
            failures.append("PDF page geometry differs from acmsmall")
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        for row in fonts.splitlines()[2:]:
            if "Type 3" in row or row.split()[-5] != "yes":
                failures.append(f"Unembedded or Type 3 font: {row}")
        text = subprocess.check_output(["pdftotext", str(pdf), "-"], text=True)
        if len(re.findall(r"[\u4e00-\u9fff]", text)) < 5000:
            failures.append("Extracted PDF has insufficient Chinese body text")
        if "\ufffd" in text:
            failures.append("Extracted PDF contains replacement characters")
        print(re.search(r"Pages:.*", info).group())
    if failures:
        sys.exit("\n".join(failures))
    print("Chinese translation checks passed: headings, protected content, math, references and numerical macros match English.")


if __name__ == "__main__":
    main()
