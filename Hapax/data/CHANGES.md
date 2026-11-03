# Changes between `archive-d07be6a460` and the current data

Old commit: `d07be6a460`; new commit: `6c46fb1b02`.
82 of 146 macros differ.

| macro | old | new | source |
|---|---|---|---|
| `cvcFiveVersion` | unrecorded | 1.4.2 | fused_ops.json meta.cvc5 |
| `failedSelectMaxMs` | 34 | 1702 | unfused registry candidates: max median selection time, ms |
| `nAddedCount` | 15 | 0 | auxiliaries named count |
| `nAddedH` | 5 | 6 | auxiliaries ending .h<k> (hoisted coefficients) |
| `nAddedR` | 11 | 34 | auxiliaries ending .r<k> (supplemented coefficients) |
| `nChains` | 38 | 45 | fused_ops.json: chains over fused rows |
| `nCountOnlyOps` | 7 | 0 | rows whose only auxiliary is count |
| `nDerived` | 32 | 49 | sum of len(chain.auxiliaries) |
| `nDerivedOps` | 19 | 26 | rows with a non-empty auxiliaries list |
| `nDomains` | 6 | 7 | info_en.py domains of fused rows, excluding Constructed |
| `nFused` | 35 | 42 | fused_ops.json: rows with fused = true |
| `nFusible` | 40 | 44 | fused_ops.json meta.registry_fusible |
| `nGateHuge` | 9 | 10 | fused rows gated on the huge family |
| `nGateTiny` | 21 | 22 | fused rows gated on the tiny family |
| `nHoist` | 5 | 6 | selection steps with action hoist |
| `nInvCanonical` | 37 | 40 | proofs_timing.json invariant_proof == canonical |
| `nMechHoist` | 5 | 6 | fused rows whose heaviest step is hoist |
| `nMechSupplement` | 14 | 20 | fused rows whose heaviest step is supplement |
| `nMoreAuxOps` | 12 | 26 | derived-ops minus count-only |
| `nNegative` | 8 | 7 | registry minus fusible |
| `nNeptuneRefused` | 0 | 1 | Neptune: refused by the checks |
| `nNotPrintedChains` | 0 | 3 | chains printed in canonical form |
| `nPassesAfter` | 38 | 45 | sum of passes_after |
| `nPassesBefore` | 99 | 117 | sum of passes_before |
| `nPassesEqualHand` | 35 | 42 | rows with passes_after == passes_hand |
| `nPreservingFailures` | 5 | 2 | fused_ops.json rows not fused |
| `nPrintedChains` | 38 | 42 | chains whose re-based merge printed |
| `nPriorRefused` | 0 | 2 | prior.json rows refused |
| `nRatioCentred` | 35 | 42 | operators with a centred ratio |
| `nRatioCentredLeFive` | 34 | 40 | count ratio_c <= 5 |
| `nRatioCentredLeOne` | 15 | 20 | count ratio_c <= 1 |
| `nRatioCentredLeTwo` | 23 | 30 | count ratio_c <= 2 |
| `nRatioOffset` | 33 | 40 | operators with an offset ratio |
| `nRatioOffsetLeTen` | 30 | 36 | count ratio_o <= 10 |
| `nRedFuserRefused` | 0 | 1 | RedFuser: refused by the checks |
| `nRegNoSelect` | 1 | 0 | registry negatives where selection failed |
| `nRegNotRead` | 1 | 0 | registry negatives with no chain read |
| `nRegRefused` | 1 | 2 | registry negatives refused by the checks |
| `nRegistry` | 48 | 51 | fused_ops.json meta.registry (len(OPERATORS)) |
| `nSameAsHand` | 5 | 6 | rows whose synthesised errors equal the hand-written ones (both families) |
| `nShowFused` | 13 | 17 | successfully fused showcase rows |
| `nShowcase` | 14 | 17 | all selected showcase rows, including failures |
| `nShowcaseDomains` | 5 | 6 | info_en.py domains of the showcase rows |
| `nSlots` | 126 | 149 | printed slots |
| `nSlotsCanonical` | 125 | 146 | printed slots proven by canonical identity |
| `nStates` | 126 | 162 | sum of len(chain.states) |
| `nStatesEqualJudged` | 38 | 45 | chains with len(states) == theory_states |
| `nSupplement` | 27 | 43 | selection steps with action supplement |
| `nTests` | 285 | 350 | data/pytest.log 'N passed' |
| `nTimeRest` | 38 | 42 | operators other than the two Sinkhorn iterations |
| `ratioCentredMax` | 7.3 | 9.7 | max of the same |
| `ratioCentredMedian` | 1.33 | 1.20 | median err_synth[0]/err_unfused[0] |
| `ratioOffsetMax` | 60.2 | 60.0 | max of the same |
| `ratioOffsetMedian` | 1.09 | 1.06 | median err_synth[1]/err_unfused[1] (both > 0) |
| `ratioOffsetMin` | 0.30 | 0.00 | min of the same |
| `showChains` | 15 | 19 | showcase: chains |
| `showDerived` | 7 | 19 | showcase: auxiliaries |
| `showDerivedOps` | 5 | 9 | showcase rows with auxiliaries |
| `showFuseMax` | 4.26 | 13.08 | accepted showcase: max median fuse time, s |
| `showFuseMedian` | 0.24 | 0.25 | accepted showcase: median fuse time, s |
| `showFuseSinkhorn` | 4.3 | 4.2 | showcase: sinkhorn_iteration median fuse time, s |
| `showInvCanonical` | 14 | 16 | showcase: invariants proven by canonical identity |
| `showJudgeMaxMs` | 7.7 | 13.4 | showcase: max median judge time, ms |
| `showOriginalOps` | unrecorded | 8 | fused_ops.json showcase operators with every chain route = original |
| `showPassesAfter` | 15 | 19 | showcase: sum of passes_after |
| `showPassesBefore` | 42 | 54 | showcase: sum of passes_before |
| `showRatioLeOne` | 6 | 9 | showcase: centred ratio <= 1 |
| `showRebaseOps` | unrecorded | 7 | fused_ops.json showcase operators with every chain route = rebase |
| `showSelectMaxMs` | 158 | 4045 | showcase: max median select time, ms |
| `showSelectMedianMs` | 32 | 65 | showcase: median of median select times, ms |
| `showStates` | 48 | 72 | showcase: states |
| `showSygusOps` | unrecorded | 2 | fused_ops.json showcase operators with every chain route = sygus |
| `timeFuseKurtosis` | 0.39 | 0.38 | kurtosis median fuse time, s |
| `timeFuseMax` | 0.90 | 13.08 | max fuse time excluding the two Sinkhorn iterations, s |
| `timeFuseMedian` | 0.18 | 0.23 | median fuse time, s |
| `timeFuseMin` | 0.06 | 0.11 | min fuse time excluding the two Sinkhorn iterations, s |
| `timeFuseSinkhorn` | 4.3 | 4.2 | max fuse time of the two Sinkhorn iterations, s |
| `timeJudgeMaxMs` | 8.0 | 13.4 | max median judge time, ms |
| `timeJudgeMedianMs` | 2.9 | 3.2 | median of median judge times, ms |
| `timeSelectMaxMs` | 570 | 4045 | max median select time, ms |
| `timeSelectMaxOp` | moment5 | papr | operator with the slowest selection |
| `timeSelectMedianMs` | 25 | 49 | median of median select times, ms |
