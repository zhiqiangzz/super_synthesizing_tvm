# Bibliography verification

Metadata source: **CR** = Crossref record (`lookup.json`, or a DOI lookup on 2026-10-28/29 or 2026-11-03),
**AX** = arXiv API record, **PDF** = the first page / reference list of a paper in `../ref_papers` or the linked author copy,
**WEB** = the named publisher, author-institution, or official project citation page. New WEB records were checked on 2026-11-04 and recorded in `lookup.json`; they are not Crossref responses. Nothing was typed from memory.

Read status: **full** = the work was read for this paper (or for the method documentation it
builds on); **part** = the sections cited were read; **bib** = only the record was checked; the
paper cites it for its standard, well-known content only. `check_cites.py` refuses a citation
whose key is missing here.

| key | metadata | read | cited for |
|---|---|---|---|
| neptune | CR 10.1145/3808298 | full (ref_papers PDF, 37 pp.) | repair-function analysis (Eq. 3-5, p.7), Welford failure (p.10-11), accuracy only for attention (App. F), schedule template (Fig. 13) |
| redfuser | CR 10.1145/3779212.3790209 (ASPLOS'26); AX 2603.10026 | full (ref_papers PDF, 22 pp.) | decomposability F = G (x) H (Eq. 4, Tab. 1, Eq. 23), invertibility (p.5), A.1 repair (p.15), variance/inertia A.6 |
| flashlight | AX 2511.02043 | full (notes/evidence/flashlight.txt) | attention variants by exp homomorphism |
| mirage | PDF + Neptune ref. [47] (OSDI'25, pp. 21-38) | part (LAX fragment, Def. 5.1; floating-point filter, Sec. 5.2) | superoptimizer; verification fragment excludes max |
| stenso | CR 10.1109/CGO68049.2026.11395198 | full (notes/evidence/stenso.txt) | symbolic superoptimization |
| prism | AX 2604.15272 | bib | superoptimization (named only) |
| nautilus | AX 2604.14825 | bib | Neptune's follow-up that automates scheduling, as Neptune's conclusion states (Neptune PDF p.22) |
| tvm | WEB usenix.org/conference/osdi18/presentation/chen; pp. from refs of flashlight/stenso | bib | TVM, TE, comm_reducer |
| halide | CR 10.1145/2491956.2462176 | bib | schedule languages |
| rfactor | CR 10.1109/CGO.2017.7863747 | part (method doc. verified text) | associative reductions in Halide |
| tensorir | CR 10.1145/3575693.3576933 | bib | TensorIR (TVM's tensor-program IR) |
| triton | CR 10.1145/3315508.3329973 | bib | tile compiler |
| tilelang | AX 2504.17577 | bib | RedFuser's backend |
| welford | CR 10.1080/00401706.1962.10490022 | bib | the one-pass variance update |
| west | CR 10.1145/359146.359153 | bib | weighted update |
| chan | CR 10.1080/00031305.1983.10483115; [paper scan](https://math.pku.edu.cn/teachers/litj/notes/numer_anal/AmerStat_37_242_Chan_Variance.pdf) | part (Eqs. 1.5-1.6, Secs. 2-3, Table 1; checked 2026-11-04) | centred variance joins; error analysis of two-pass, updating, and shifted formulas. The pairwise error bound in Table 1 is explicitly a conjecture; the paper does not prove that a centred join preserves two-pass accuracy. |
| pebay | CR 10.1007/s00180-015-0637-z; [author PDF](https://people.xiph.org/~tterribe/pubs/P%C3%A9bayTerriberryKolla%2B16-_Numerically_Stable,_Scalable_Formulas_for_parallel_and_Online_Computation_of_Higher-Order_Multivariate_Central_Moments_with_Arbitrary_Weights.pdf) | part (Sec. 2.3, Props. 3.1-3.2, Sec. 4.2/Table 3; checked 2026-11-04) | algebraic parallel/online central-moment formulas and their numerical evaluation. Table 3 labels the pairwise variance bound as conjectured; local compensated evaluation does not prevent error accumulation under recursive use. |
| milakov | AX 1805.02867 | full | online softmax normaliser |
| flashattention | CR 10.52202/068431-1189 | full | FlashAttention |
| flashattention2 | AX 2307.08691 | full | deferred normalisation |
| cce | AX 2411.09009 | full | cut cross-entropy |
| flashsinkhorn | AX 2602.03067 | part | single-pass Sinkhorn |
| higham | CR 10.1137/1.9780898718027 | bib | rounding error analysis background |
| blanchard | CR 10.1093/imanum/draa038 (online 2020, vol. 41(4) 2021); [author PDF](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P149.pdf) | part (Sec. 4, Algorithm 4.1, Theorems 4.2-4.4, Sec. 6; checked 2026-11-04) | overflow avoidance and error bounds for max-shifted batch log-sum-exp/softmax; division-free softmax can be less accurate. These results concern a fixed global shift, not online joins or arbitrary reduction trees. |
| goldberg | CR 10.1145/103162.103163; [authorized reprint](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) | part (Systems Aspects: Languages and Compilers, Optimizers; Theorem 8; checked 2026-11-04) | reassociation can destroy the correction in compensated summation; source evaluation structure can encode numerical reasoning. This is not a guarantee for newly synthesized joins. |
| boldo_fp | WEB [Springer](https://link.springer.com/article/10.1007/s10817-014-9317-x), [author citation](https://xavierleroy.org/bibrefs/Boldo-Jourdan-Leroy-Melquiond-JAR.html); [author PDF](https://xavierleroy.org/publi/floating-point-compcert.pdf) | part (Sec. 1 Fast-Two-Sum example, Sec. 3.3, Sec. 6.2; checked 2026-11-04) | floating-point semantics depend on expression evaluation, precision, and permitted transformations; semantics-preserving compilation protects those choices. It does not establish accuracy preservation for a changed state representation or join. Journal issue year 2015; published online December 2014. |
| higham_decomposition | WEB [University of Manchester](https://research.manchester.ac.uk/en/publications/the-sensitivity-of-computational-control-problems/); [author PDF](https://eprints.maths.manchester.ac.uk/652/01/covered/MIMS_ep2006_405.pdf) | part (Computational Algorithms, pp. 32-33, Eqs. 7-9, Examples 6-7; checked 2026-11-04) | a decomposition into subproblems can amplify intermediate errors even when individual stages are computed accurately; rationale for avoiding ill-conditioned intermediate representations, not a theorem about Hapax state selection. |
| mpfr | WEB [official MPFR citation](https://www.mpfr.org/credit.html), [manual reference (Article 13)](https://www.mpfr.org/mpfr-current/mpfr.html#References), DOI 10.1145/1236463.1236468 | bib (official recommended citation checked 2026-11-04) | the MPFR arbitrary-precision library and correctly rounded operations; no claim that stabilized precision alone certifies a whole-program result. |
| hypothesis | WEB [JOSS](https://joss.theoj.org/papers/10.21105/joss.01891); [paper PDF](https://joss.theoj.org/papers/10.21105/joss.01891.pdf) | full (3 pp.; checked 2026-11-04) | property-based test generation, test-case reduction, and targeted search; the paper makes no exhaustive-coverage guarantee. |
| ieee754 | CR 10.1109/IEEESTD.2019.8766229 | bib | IEEE semantics of special values |
| herbie | CR 10.1145/2737924.2737959 | bib | accuracy-improving rewriting (related work) |
| fptaylor | CR 10.1145/3230733 | bib | rigorous round-off bounds (related work) |
| blue | CR 10.1145/355769.355771 | bib | scaled Euclidean norm |
| bird87 | CR 10.1007/978-3-642-87374-4_1 | bib | list homomorphisms |
| gibbons96 | CR 10.1017/S0956796800001908; title from PDF | full | third homomorphism theorem |
| meijer91 | CR 10.1007/3540543961_7 | bib (in ref_papers) | catamorphisms (related work) |
| hu96 | CR 10.1007/3-540-61550-4_166 | full | tupling and fusion |
| hu97 | CR 10.1145/258948.258964 | full | tupling eliminates multiple traversals |
| bird84 | CR 10.1007/BF00264249 | full | circular programs, multiple traversals |
| pettorossi87 | CR 10.1007/3-540-18625-5_65 | full | tupling by lambda abstraction |
| morita07 | CR 10.1145/1250734.1250752 | full | weak right inverse, third homomorphism theorem |
| farzan17 | CR 10.1145/3062341.3062355 | full | auxiliary accumulators by unfolding/normalising |
| farzan19 | CR 10.1145/3314221.3314612 | part | nested loops |
| farzan21 | CR 10.1145/3453483.3454089 | part | phased synthesis |
| morihata21 | CR 10.1145/3453483.3454079 | part | semiring polynomials |
| autolifter | CR 10.1145/3648440 | full (Sec. 2.1, Def. 3.1, Sec. 4.2 rechecked 2026-11-03) | auxiliary values and combinators through decomposition of lifting problems |
| sufu | CR 10.1145/3656415; [author PDF](https://cseweb.ucsd.edu/~npolikarpova/publications/pldi24-extended.pdf) | part (Secs. 1-2, 4.5; rechecked 2026-11-03) | compression functions and scalar computations eliminate intermediate data structures; compression need not recover the original structure |
| opera | CR 10.1145/3656418 | full (notes/evidence/opera.txt) | online algorithms; kurtosis the only failure of 51 |
| ink | [CR 10.1145/3763072](https://api.crossref.org/works/10.1145/3763072), lookup 2026-11-03; [author PDF](https://www.cs.utexas.edu/~isil/ink.pdf); [arXiv journal reference](https://arxiv.org/abs/2508.15109) | part (Secs. 1, 3-4, 7-8; Defs. 3.4-3.5) | homomorphism verification/refutation and merge construction for given dataframe aggregations, distinct from discovering auxiliary states; merge need not commute. Formal publication: PACMPL 9(OOPSLA2), Article 294, pp. 615-641 (27 pp.); author extended PDF: 35 pp. |
| datacube | CR 10.1109/ICDE.1996.492099; authors from PDF (MSR-TR-95-22) | full | distributive / algebraic aggregates |
| jesus | CR 10.1109/COMST.2014.2354398 | part | distributed aggregation survey |
| levicivita | PDF (Opere matematiche reprint: Rend. Acc. Lincei ser. 5, vol. XXII, 1913, pp. 181-183) | full | finite addition formulas are exponential polynomials |
| almira | AX 1612.03756 | full | modern statement of Levi-Civita's theorem |
| kuczma | CR 10.1007/978-3-7643-8749-5 | bib | Cauchy's equation under regularity |
| sympy | CR 10.7717/peerj-cs.103 | bib | the computer algebra used for one proof rule |
| bpr | CR 10.1007/3-540-33099-2 | bib | Sturm sequences |

| cvc5 | Official citation: https://cvc5.github.io/publications.html (2026-11-02) | metadata + API | Native checkSynth/checkSynthNext and grammar enumeration checked against https://cvc5.github.io/docs/latest/api/python/base/solver.html; no claim about internal solver algorithms |
