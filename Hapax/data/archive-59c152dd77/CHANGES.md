# Changes between `archive-58ffb2e0d8` and the current data

Old commit: `58ffb2e0d8`; new commit: `59c152dd77`.
46 of 136 macros differ.

| macro | old | new | source |
|---|---|---|---|
| `nAddedH` | 13 | 7 | auxiliaries ending .h<k> (hoisted coefficients) |
| `nAddedR` | 8 | 10 | auxiliaries ending .r<k> (supplemented coefficients) |
| `nCountOnlyOps` | 6 | 7 | rows whose only auxiliary is count |
| `nDerived` | 48 | 44 | sum of len(chain.auxiliaries) |
| `nDerivedOps` | 25 | 24 | rows with a non-empty auxiliaries list |
| `nHoist` | 10 | 6 | selection steps with action hoist |
| `nMechHoist` | 10 | 6 | fused rows whose heaviest step is hoist |
| `nMechPromote` | 11 | 0 | fused rows whose heaviest step is promote |
| `nMechRebase` | 13 | 16 | fused rows whose heaviest step is rebase |
| `nMechSupplement` | 1 | 13 | fused rows whose heaviest step is supplement |
| `nMoreAuxOps` | 19 | 17 | derived-ops minus count-only |
| `nNoAuxOps` | 15 | 16 | fused rows without auxiliaries |
| `nNoStep` | 16 | 19 | chains with no selection step |
| `nOtherNowFused` | 2 | 3 | of those, fused at HEAD |
| `nOtherRefused` | 5 | 4 | survey/extra refused |
| `nPromote` | 22 | 0 | selection steps with action promote |
| `nRatioCentredLeFive` | 38 | 37 | count ratio_c <= 5 |
| `nRatioCentredLeOne` | 16 | 15 | count ratio_c <= 1 |
| `nRatioCentredLeTwo` | 25 | 24 | count ratio_c <= 2 |
| `nRatioOffsetLeTen` | 34 | 35 | count ratio_o <= 10 |
| `nSameAsHand` | 7 | 6 | rows whose synthesised errors equal the hand-written ones (both families) |
| `nSlotsCanonical` | 139 | 140 | printed slots proven by canonical identity |
| `nSlotsSympy` | 1 | 0 | printed slots proven by SymPy |
| `nSupplement` | 27 | 30 | selection steps with action supplement |
| `nTests` | 265 | 271 | data/pytest.log 'N passed' |
| `ratioOffsetMax` | 74.2 | 60.2 | max of the same |
| `ratioOffsetMedian` | 1.27 | 1.01 | median err_synth[1]/err_unfused[1] (both > 0) |
| `showDerived` | 13 | 11 | showcase: auxiliaries |
| `showDerivedOps` | 7 | 6 | showcase rows with auxiliaries |
| `showFuseMax` | 0.80 | 0.86 | showcase except sinkhorn_iteration: max median fuse time, s |
| `showFuseMedian` | 0.19 | 0.23 | showcase except sinkhorn_iteration: median fuse time, s |
| `showJudgeMaxMs` | 6.8 | 7.6 | showcase: max median judge time, ms |
| `showNoStep` | 6 | 8 | showcase rows without a selection step |
| `showRatioLeOne` | 7 | 6 | showcase: centred ratio <= 1 |
| `showSelectMaxMs` | 225 | 264 | showcase: max median select time, ms |
| `showSelectMedianMs` | 30 | 31 | showcase: median of median select times, ms |
| `timeFuseKurtosis` | 0.42 | 0.49 | kurtosis median fuse time, s |
| `timeFuseMax` | 0.81 | 0.90 | max fuse time excluding the two Sinkhorn iterations, s |
| `timeFuseMedian` | 0.16 | 0.18 | median fuse time, s |
| `timeFuseMin` | 0.10 | 0.12 | min fuse time excluding the two Sinkhorn iterations, s |
| `timeJudgeMaxMs` | 6.8 | 8.2 | max median judge time, ms |
| `timeJudgeMedianMs` | 2.4 | 3.0 | median of median judge times, ms |
| `timeSelectKurtosisMs` | 225 | 264 | kurtosis median select time, ms |
| `timeSelectMaxMs` | 351 | 372 | max median select time, ms |
| `timeSelectMedianMs` | 25 | 29 | median of median select times, ms |
| `varianceOffsetSynth` | $8.0\times10^{-4}$ | $2.9\times10^{-4}$ | variance gate offset candidate |
