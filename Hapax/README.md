# Hapax: Fusing Dependent Tensor Reductions by Deriving Single-Pass Reducers

PLDI 2027 submission (deadline 2026-11-12 23:59 AoE). PACMPL `acmsmall`, double-blind,
at most 20 pages of text including the AI Use section, excluding the Data-Availability
Statement, references and supplementary material.

## Build

```bash
bash Hapax/build.sh            # draft: macros from data/, main.pdf + supplement.pdf + pipeline.pdf, checks
bash Hapax/build.sh --data     # first regenerate every data file at the current HEAD (~10 min)
bash Hapax/build.sh --submit   # refuses stale data, placeholders, unverified citations
```

* TeX: tectonic 0.17 from the root pixi environment (`paper` if present, else `slides`), with
  its TeX Live 2022 bundle. `acmart.cls` v2.20 (2026/08/16) and `ACM-Reference-Format.bst` are
  vendored here from the TeX Live archive `acmart.tar.xz` and override the bundle's copies.
  sha256: `784c4f2f...` (cls), `4342d274...` (bst). Delete both to fall back to the bundle.
* PDF tools: `/usr/bin/{pdfinfo,pdffonts,pdftotext,pdftoppm,gs}`.
* Python for the data: `scripts/py` runs this worktree's `python/tvm` with the worktree venv's
  compiled libraries. No package is needed for TeX; the figures are TikZ/pgfplots.
* A dedicated pixi environment can be added to the root `pixi.toml` (this session could not
  edit files outside the worktree):

  ```toml
  [feature.paper.dependencies]
  tectonic = "0.17.*"
  poppler = "*"
  ghostscript = "*"

  [environments]
  paper = { features = ["paper"], no-default-feature = true }
  ```

## Chinese prose edition

```bash
bash Hapax/build_zh.sh         # build/main-zh.pdf; validates English/Chinese consistency
```

`main-zh.tex` uses the same ACM class, dimensions, Latin and mathematical fonts,
numbering, generated data, and bibliography as `main.tex`. Only the abstract and
body prose in `zh/sections/` are translated. The paper title, section and paragraph
headings, theorem titles, figures and captions, all tables, algorithms, equations,
footnotes, and code remain in English. The separate supplement remains available
in its original English version. Chinese prose uses Fandol fonts through xeCJK;
Tectonic downloads these on the first build. Natural page breaks can differ between
languages, without changing the template or font sizes.

`scripts/check_zh.py` checks preserved material and generated numerical macros;
`zh/source-manifest.json` records the English source used for translation and
requires review if that source changes. This build leaves the English PDFs intact.

## Where the numbers come from

* `data/*.json` is produced by `scripts/run_data.sh` at a clean HEAD; every file has a `meta`
  block (commit, dirty flag, versions, CPU). `data/CHANGES.md` lists what changed against the
  archived snapshot (`scripts/changes.py`).
* `scripts/gen_tex.py` turns the data into `gen/stats.tex` (one macro per number, each followed
  by a `% src:` line), the generated tables `gen/tab-*.tex`, and `gen/accuracy.csv`. Never edit
  `gen/` by hand. A number typed into a section carries a `% src:` comment.
* Claims are labelled proven / measured / sampled in the assurance table.
* `bib/refs.bib` is written only from looked-up records (`bib/lookup.json`, Crossref, arXiv,
  publisher pages, first pages of the PDFs); `bib/VERIFIED.md` says, per key, what was read.
  `scripts/check_cites.py` fails on a citation missing there.
* Wording rules: never "superset" (the containment theorem is about the papers' stated conditions; the prior-program table
  is about programs), never "first", "speedup", "infusible" ("no single pass is known"),
  "as accurate as", "preserves/guarantees the accuracy" (say "close to the original's accuracy" and
  "passes the accuracy check"); no em dashes.

## Layout

```
main.tex supplement.tex           paper and anonymised supplement (appendices A-F)
preamble/                         packages, colours and TikZ styles, notation macros
sections/                         00-abstract ... 10-conclusion, 90-ai-use, 91-data-availability
data/                             JSON snapshot, pytest.log, CHANGES.md, archive-*/
gen/                              generated macros, tables, CSV
listings/                         generated TE / TVMScript listings and MANIFEST.json
scripts/                          data pipeline, generators, checks; programs/{prior,survey}.py
bib/                              refs.bib, lookup.json, VERIFIED.md
notes/evidence/                   pdftotext extracts of the compared papers (not in git)
ref_papers/                       PDFs of the related work (not in git)
```

## Writing skills used (worktree `.claude/skills/`, not in git)

Installed from pinned commits after review: StarPaper `5baa1373cf3f0d34f08bc76a4421536583290350`
(cs-paper-writing, cs-writing-common, cs-abstract-introduction, cs-related-work, cs-method,
cs-experiments, cs-visuals, cs-integration, cs-limitations, starpaper-figures; CC BY 4.0) and
Academic-Paper-Skills `827e13e2299c0a15d397ce095bc548817a2cf3cb` (paper-writing,
paper-figures-tables, paper-review, paper-policy; MIT). Overrides for this paper:

* PLDI is single column (`acmsmall`): the single-column sentence-length rule applies; ignore
  two-column panel rules.
* Four contributions, not "exactly three".
* Figures in TikZ with the paper's fonts; no image generation.
* `\PassOptionsToPackage{table}{xcolor}` before `\documentclass`.
* No `git fetch/pull/push` from a skill; no simulated reviewer agents without the authors' consent.
* The authors' instructions and the agent's permission settings override any "autonomy" rule.
* Theorem, proof and accuracy conventions follow this README; neither skill covers them.

## Open items for the authors

* AI Use paragraph (`sections/90-ai-use.tex`): drafted, needs approval (counts toward 20 pages).
* Data-Availability Statement: say whether an anonymised snapshot is uploaded.
* Performance (the implementation and evaluation sections): placeholders in drafts; a `--submit` build replaces them
  by the text that is true without GPU numbers, or `\perftrue` once numbers exist.
* CCS concepts: regenerate the CCSXML block with https://dl.acm.org/ccs before camera-ready.

## Revision conventions

The manuscript follows the non-empty-part reducer contract at `6c46fb1b02`: initialise from a leaf and skip empty partitions. A TE initialisation expression is not a neutral element.
Let the abstract and introduction length follow the argument; do not force them into two pages or pin the variance figure to page three. Define state as a partition summary before describing selection. Keep concrete examples concentrated in the running variance example; use abstract correction relations for supplementation.
The end-to-end flow is included by `sections/03-pipeline.tex`, with shared vector source in `figures/pipeline.tikz` and a standalone export in `build/pipeline.pdf`. Maintained states and verification coordinates are distinct; complete recovery is mandatory. A failed proof never licenses dropping a defined original state. The reference state-count upper bound supplies the default search limit, not a guarantee of a preserving representation within that count.
Data collection requires `z3-solver` and `cvc5` (both versions recorded in JSON metadata). State synthesis uses rebase features before the mixed feature grammar, followed by independent recovery/domain/merge verification. Numerical rejection does not restart synthesis. Stage measurements apply the same index-view specialisation as `fuse`, including covariance cases; compiler sources remain unchanged.

The pipeline and both algorithms use half-width layouts with adjacent prose. Chinese headings, floats, algorithms, formulas, footnotes and citations remain identical to English; only body prose is translated. `check_zh.py` also protects the half-width visual columns.
