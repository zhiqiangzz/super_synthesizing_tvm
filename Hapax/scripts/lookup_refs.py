"""Look up bibliographic metadata for the paper's references (Crossref, arXiv).

Writes bib/lookup.json: for each key, the best Crossref match (title, authors,
venue, year, volume, issue, pages, DOI) or the arXiv record. refs.bib is written
from these records, never from memory; bib/VERIFIED.md records, per key, the
record used and whether the work itself was read.

    python3 -I Hapax/scripts/lookup_refs.py
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HAPAX = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "hapax-paper-refcheck/0.1 (mailto:anonymous@example.org)"}

CROSSREF = {
    "neptune": "Neptune: Advanced ML Operator Fusion for Locality and Parallelism on GPUs",
    "redfuser": "RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators",
    "mirage": "Mirage: A Multi-Level Superoptimizer for Tensor Programs",
    "stenso": "Tensor Program Superoptimization through Cost-Guided Symbolic Program Synthesis",
    "halide": "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines",
    "rfactor": "Parallel associative reductions in Halide",
    "tensorir": "TensorIR: An Abstraction for Automatic Tensorized Program Optimization",
    "triton": "Triton: an intermediate language and compiler for tiled neural network computations",
    "welford": "Note on a Method for Calculating Corrected Sums of Squares and Products",
    "west": "Updating mean and variance estimates: an improved method",
    "chan": "Algorithms for Computing the Sample Variance: Analysis and Recommendations",
    "pebay": "Numerically stable, scalable formulas for parallel and online computation of higher-order multivariate central moments with arbitrary weights",
    "higham": "Accuracy and Stability of Numerical Algorithms Higham",
    "blanchard": "Accurately computing the log-sum-exp and softmax functions",
    "goldberg": "What every computer scientist should know about floating-point arithmetic",
    "herbie": "Automatically improving accuracy for floating point expressions",
    "blue": "A Portable Fortran Program to Find the Euclidean Norm of a Vector",
    "bird87": "An Introduction to the Theory of Lists Bird",
    "gibbons96": "The third homomorphism theorem Gibbons",
    "meijer91": "Functional programming with bananas, lenses, envelopes and barbed wire",
    "hu96": "Construction of list homomorphisms by tupling and fusion",
    "hu97": "Tupling calculation eliminates multiple data traversals",
    "bird84": "Using circular programs to eliminate multiple traversals of data",
    "pettorossi87": "Program development using lambda abstraction",
    "morita07": "Automatic inversion generates divide-and-conquer parallel programs",
    "farzan17": "Synthesis of divide and conquer parallelism for loops",
    "farzan19": "Modular divide-and-conquer parallelization of nested loops",
    "farzan21": "Phased synthesis of divide and conquer programs",
    "morihata21": "Reverse engineering for reduction parallelization via semiring polynomials",
    "autolifter": "Decomposition-based synthesis for applying divide-and-conquer-like algorithmic paradigms",
    "sufu": "Superfusion: Eliminating Intermediate Data Structures via Inductive Synthesis",
    "opera": "From Batch to Stream: Automatic Generation of Online Algorithms",
    "datacube": "Data Cube: A Relational Aggregation Operator Generalizing Group-By, Cross-Tab, and Sub-Totals",
    "jesus": "A Survey of Distributed Data Aggregation Algorithms",
    "kuczma": "An Introduction to the Theory of Functional Equations and Inequalities Cauchy's Equation and Jensen's Inequality",
    "sympy": "SymPy: symbolic computing in Python",
    "bpr": "Algorithms in Real Algebraic Geometry Basu Pollack Roy",
    "ieee754": "IEEE Standard for Floating-Point Arithmetic 754-2019",
    "flashattention": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
    "attention": "Attention is all you need Vaswani",
    "cole95": "Parallel programming with list homomorphisms Cole",
    "cuturi": "Sinkhorn Distances: Lightspeed Computation of Optimal Transport",
    "tvm": "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning",
    "fptaylor": "Rigorous Estimation of Floating-Point Round-Off Errors with Symbolic Taylor Expansions",
    "cce": "Cut Your Losses in Large-Vocabulary Language Models",
    "flashlight": "FlashLight: PyTorch Compiler Extensions to Accelerate Attention Variants",
}
ARXIV = {
    "milakov": "1805.02867",
    "redfuser_arxiv": "2603.10026",
    "flashattention2": "2307.08691",
    "tilelang": "2504.17577",
    "almira": "1612.03756",
    "prism": "2604.15272",
    "flashsinkhorn": "",  # resolved by title search below
    "flashlight_arxiv": "",
    "cce_arxiv": "2411.09009",
    "nautilus": "2604.14825",
}
ARXIV_TITLE = {
    "flashsinkhorn": "FlashSinkhorn IO-Aware Entropic Optimal Transport",
    "flashlight_arxiv": "FlashLight PyTorch Compiler Extensions to Accelerate Attention Variants",
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


def crossref(query: str) -> list[dict]:
    q = urllib.parse.urlencode(
        {
            "query.bibliographic": query,
            "rows": 3,
            "select": "title,author,container-title,issued,DOI,volume,issue,page,article-number,type,publisher,event",
        }
    )
    d = json.loads(get(f"https://api.crossref.org/works?{q}"))
    out = []
    for it in d["message"]["items"]:
        out.append(
            {
                "title": (it.get("title") or [""])[0],
                "authors": [
                    f"{a.get('given', '')} {a.get('family', '')}".strip() for a in it.get("author", [])
                ],
                "venue": (it.get("container-title") or [""])[0],
                "year": it.get("issued", {}).get("date-parts", [[None]])[0][0],
                "volume": it.get("volume"),
                "issue": it.get("issue"),
                "pages": it.get("page") or it.get("article-number"),
                "doi": it.get("DOI"),
                "type": it.get("type"),
                "publisher": it.get("publisher"),
            }
        )
    return out


def arxiv(ids: str = "", title: str = "") -> list[dict]:
    if ids:
        q = urllib.parse.urlencode({"id_list": ids})
    else:
        q = urllib.parse.urlencode({"search_query": f'ti:"{title}"', "max_results": 3})
    xml = get(f"http://export.arxiv.org/api/query?{q}").decode()
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", entry, re.S)
        pid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", entry)
        pub = re.search(r"<published>(\d{4})", entry)
        authors = re.findall(r"<name>(.*?)</name>", entry)
        out.append(
            {
                "title": " ".join(t.group(1).split()) if t else "",
                "authors": authors,
                "arxiv": pid.group(1) if pid else "",
                "year": int(pub.group(1)) if pub else None,
            }
        )
    return out


def main() -> None:
    result = {}
    for key, q in CROSSREF.items():
        try:
            result[key] = {"source": "crossref", "query": q, "matches": crossref(q)}
        except Exception as err:  # noqa: BLE001
            result[key] = {"source": "crossref", "query": q, "error": str(err)}
        print(key, (result[key].get("matches") or [{}])[0].get("title", result[key].get("error")), flush=True)
        time.sleep(1.0)
    for key, ids in ARXIV.items():
        try:
            recs = arxiv(ids=ids) if ids else arxiv(title=ARXIV_TITLE[key])
            result[key] = {"source": "arxiv", "query": ids or ARXIV_TITLE[key], "matches": recs}
        except Exception as err:  # noqa: BLE001
            result[key] = {"source": "arxiv", "query": ids, "error": str(err)}
        print(key, (result[key].get("matches") or [{}])[0].get("title", result[key].get("error")), flush=True)
        time.sleep(3.0)
    (HAPAX / "bib").mkdir(exist_ok=True)
    with open(HAPAX / "bib" / "lookup.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
