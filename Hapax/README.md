# Hapax: Fusing Dependent Tensor Reductions by Deriving Single-Pass Reducers

PLDI 2027 submission (deadline 2026-11-12 23:59 AoE). PACMPL `acmsmall`, double-blind,
at most 20 pages of text including the AI Use section, excluding the Data-Availability
Statement, references and supplementary material.

## Build

```bash
bash Hapax/build.sh            # draft: macros from data/, main.pdf + supplement.pdf + pipeline.pdf, checks
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
bash Hapax/build_zh.sh         # draft build/main-zh.pdf; validates English/Chinese consistency
bash Hapax/build_zh.sh --submit # requires current data and no placeholders
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
  It also exports `gen/state-definitions.txt`, which maps every maintained state identifier
  in the registry table to the recorded source or derived expression without rerunning synthesis.
* Claims are labelled proven / measured / sampled in the assurance table.
* `bib/refs.bib` is written only from looked-up records (`bib/lookup.json`, Crossref, arXiv,
  publisher pages, first pages of the PDFs); `bib/VERIFIED.md` says, per key, what was read.
  `scripts/check_cites.py` fails on a citation missing there.
* Wording rules: never "superset" (the containment theorem is about the papers' stated conditions; the prior-program table
  is about programs), never "first", "speedup", "infusible" ("no single pass is known"),
  "as accurate as", "preserves/guarantees the accuracy" (say "close to the original's accuracy" and
  "passes the accuracy check"); no em dashes.

## Numerical-method update and development handoff

This worktree is for the manuscript, bibliography, imported results and typesetting.
New compiler code, numerical tests, reference-state code generation and experimental
collection belong in a development worktree. The complete task and result-package
contract are in [the numerical evaluation handoff](notes/numerical-evaluation-handoff.md).
The manuscript follows `main@440b06a2ac`: state selection collects its reference
reductions from the retained or repaired initial state definitions. Required outputs
do not add reference components. The previous proposal to reuse the existence
construction's entire collection has been superseded and requires no implementation.
The numerical audit entry point, its supporting modules and reference-state tests
are already in the development tree. The handoff specifies the measurements still
to be delivered and the additional diagnostics and controlled synthesis experiments.
Their presence in the source tree does not establish that current results have been
collected or imported. The paper claims no unmeasured coverage or cost improvements.
Do not run `build.sh --data` or the legacy collectors here during this transition:
they have not been adapted to the new accuracy-report interface.

The synthesis account distinguishes two mechanisms: reference recovery eliminates
a separate join-synthesis task, while fixed source states and context-correction
candidates restrict and guide auxiliary-state search. These structural properties
do not establish faster search or better numerical results without the controlled
comparisons in the handoff. Nonlinear reference expressions remain possible
alternative representations; their execution formulas determine their numerical
behaviour.

The current implementation retains a source chain when the selected reducer fails
numerical acceptance; it does not resume state synthesis. The handoff separately
specifies an opt-in `retry_rejected_candidates=False` extension for a development
agent, including candidate continuation, isolated rewrites, resource accounting,
tests and a controlled comparison. Until that extension is implemented and its
results imported, the manuscript and pipeline diagram describe the current flow.

Sections 6--7 and Appendix E now describe the MPFR/Hypothesis implementation from
`d2b0653238`. The checked-in data still records the synthesis snapshot `6c46fb1b02`.
Its coverage/state inventory and algebraic derivation measurements are explicitly
identified as that snapshot and must be rerun before attributing them to the current
implementation. Old numerical error ratios, acceptance margins and
complete-fusion timings are not presented as measurements of the new protocol.
The draft marks the replacement results as pending; the old JSON files remain
unchanged for provenance. Existing hand-calculated examples illustrate numerical
mechanisms and are not measurements of generated reference-state programs.

The registry table prints state information only until new numerical evidence is
imported. Draft builds retain the stale-data notice. Submission builds must pass
the normal freshness checks and contain no pending-results placeholders.

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

* Numerical evaluation: import a versioned result package from the current development
  baseline, including actual reference-state execution and updated full-fusion timings.
* State synthesis: complete the handoff's diagnostics and controlled comparisons before
  making claims about their effect on coverage, search cost or numerical accuracy.
* AI Use paragraph (`sections/90-ai-use.tex`): drafted, needs approval (counts toward 20 pages).
* Data-Availability Statement: say whether an anonymised snapshot is uploaded.
* Performance (the implementation and evaluation sections): placeholders in drafts; a `--submit` build replaces them
  by the text that is true without GPU numbers, or `\perftrue` once numbers exist.
* CCS concepts: regenerate the CCSXML block with https://dl.acm.org/ccs before camera-ready.

## Revision conventions

The manuscript follows the non-empty-part reducer contract at `6c46fb1b02`: initialise from a leaf and skip empty parts. A TE initialisation expression is not a neutral element. Use **part** for one subset of the axis and **partition** for a division of the axis into disjoint parts. The Chinese equivalents are 分块 and 划分.

Use **state tuple** for the information associated with a non-empty part and **state component** for one scalar value in that tuple; the **state map** supplies these values as functions of the part. State counts refer to the number of components. A tuple summarises a part, but **summary** does not introduce a separate object type. Define the tuple and its join contract before describing synthesis. Use **reference state tuple/components**, **maintained state tuple**, **auxiliary states**, **join**, and **context-independent** consistently for Hapax. Context-independent means that a reduction body does not read a context as defined in the manuscript. Preserve the terminology of cited work, such as lifting, compression function, or merge operator, when describing that work; shared terminology does not imply identical synthesis problems.

Use `I_src` for the source-derived initial state tuple, including necessary domain repairs; `I` for the maintained state tuple; and `I_ref` for the reference tuple. These instantiate the state map `I` already introduced in the reducer contract. The known map `G_src` expresses `I_src` in `I_ref`. Current implementation names `A0`, `A` and `B` correspond to these roles; diagnostic field names need not be renamed.

State selection derives `I_ref` from context-independent reduction atoms in the canonical definitions of `I_src`, after domain repair. This tuple stays fixed throughout auxiliary-state search. The fourth chapter's `Q` is the collection constructed to establish single-pass existence; its size supplies the default state limit, but it need not equal the selection procedure's reference collection. Required outputs are reconstructed during reducer derivation without extending `I_ref`. Synthesis searches only for auxiliary states `C` such that the augmented tuple recovers `I_ref`; a separate procedure constructs `H`, and the state definitions give `G`. Verified recovery constructs the join as `G(H(a) ⊕ H(b))`, eliminating a separate join-synthesis problem. Retain the independent recovery, domain and complete-reducer checks. A recovery map reconstructs reference components, not input elements; any inverse claim must state its domain and direction.

Describe the expressions supplied to synthesis as **candidate expressions** and, when discussing their grammar role, as **grammar terminals**. They are not all maintained states. The search uses expressions from context-correction analysis before adding reference components to the grammar, followed by independent recovery, domain and join verification. Describe augmentation with all admissible correction coefficients directly; do not call it a closure without a corresponding closure property. Describe constructing context-correction formulas from source definitions and verifying them against the join obtained through recovery. A component without a supported correction formula uses its algebraically derived join. Complete recovery is the selection procedure's sufficient contract, not a necessary condition for every correct join. These are manuscript terminology conventions, not claims that every phrase is a standard term from prior work.

Let the abstract and introduction length follow the argument; do not force them into two pages or pin the variance figure to page three. Keep variance as the motivating example in Section 2 and use Householder-type norm to connect the steps of state synthesis in Section 5. Introduce each example's quantities where needed and keep the general construction symbolic. The end-to-end flow is included by `sections/03-pipeline.tex`, with shared vector source in `figures/pipeline.tikz` and a standalone export in `build/pipeline.pdf`.

Development-side data collection requires `z3-solver`, `cvc5` and `gmpy2`; property tests additionally require `hypothesis`. The result package records their versions and the MPFR version. Numerical rejection does not restart synthesis. Stage measurements apply the same index-view specialisation as `fuse`, including covariance cases; compiler sources remain unchanged in this paper worktree.

The pipeline, algorithms and compact state-relation figure use half-width layouts with adjacent prose. Chinese headings, floats, algorithms, formulas, footnotes and citations remain identical to English; only body prose is translated. `check_zh.py` also protects the half-width visual columns.
