> Historical record, frozen before the curated handoff. Commands and missing-feature
> claims below may be superseded. Start with [README.md](README.md).

# Reviewer follow-up and FireSim preparation

Baseline: `1a8e98149870ba9e89dc74c18d5feda48f3d04b4`.
The external review passed all 842 tests but found performance and attribution
problems. Passing numerical checks does not establish useful placement or tuning.
Historical results remain immutable; replacement evidence uses fresh paths.

| Requirement | Implementation / evidence required | Status |
| --- | --- | --- |
| 1. Correct tuning comparisons | Measure CPU and the untuned default first; retain exhaustive reviewer results; correct speedup and timeout claims | Corrected historical claims; default/CPU-first sweep reproduced 462/9578; final six-trial framework run completes all baselines |
| 2. Cost-aware placement | Copy offload opt-in; no implicit scalar DMA; per-function CPU comparisons and whole-graph ablations | Implemented; five actual CNN functions and actual decode score measured; CPU winners replay into packages; final cost-gated CNN/decode packages pass Spike/RTL positive and wrong-oracle checks |
| 3. Constant packing | Use framework layout rewriting/constant folding; demonstrate no repeated constant packing in runtime K loops | Implemented using standard layout/preprocessing split/FoldConstant; whole-CNN Spike/RTL controls pass; no runtime constant-weight packing |
| 4. Reliable bounded search | Durable trials, continuation after runner failure, resume, cycle-derived deadlines, isolated builds | Implemented and tested; real failed CPU build/timeout retained and native trials continue; corrected campaigns complete |
| 5. Standard MetaSchedule | Sampled trace decisions, framework mutation/search/cost model and external runner; small-space exhaustive sweep | Six final native tune_tasks/EvolutionarySearch/XGBModel trials pass; CPU 40209/default 1178/winner 987; fresh-process replay passes; learned predictions identified after warmup |
| 6. Exact CNN epilogue | Prove eligible quantize/ReLU readout equivalence and validate on Spike/RTL; retain precise fallback | Both CNN sites integrated; full-INT32 threshold proof and native boundary fixtures pass Spike/RTL positive and wrong-oracle controls |
| 7. Accurate reporting | Separate observed dispatch from maintained semantic explanations; record weight initializer honestly | Corrected; report_followup binds exported buffer ABIs, dispatch, execution results and costs; LIMITS prose explicitly maintained |
| 8. Meaningful tuning tests | Detect constant/tied predictions, default regressions, redundant schedules and failed trials | Implemented; diagnostic framework and measured native prediction diversity recorded |
| 9. CPU rounding | Prove finite clamped Q/DQ conversion and use RV64 hardware RNE; preserve general round semantics | Guarded bounded Q/DQ lowering tested at all half ties, saturation and NaN boundaries; RV64 fcvt verified; full CNN/MLP/decode controls pass |
| Export portability | Restore serialized-export metadata and own aligned constants; fresh-process tests | Fresh-process archive tests pass; actual saved CNN archive imports and extracts eight functions |
| Large learned reductions | Preserve exact source semantics with proven frozen-weight bounds or explicit integer input; test boundaries | Frozen-weight L1 proof extends eligible reductions; unproved cases retain exact CPU math; tests pass |
| FireSim compiler package | Reproducible source/toolchain/ISA/ABI/capability binding, memory plan, loader contract, validation and bounded run commands | Portable archives and guarded boot ELFs implemented, including CPU-only/measured placement; default archive passes both simulators; cost-gated CNN/decode archive boot ELFs pass all four controls |
| Qualification | Re-run affected suite; Spike positive/negative and bounded whole-slice RTL; publish exact receipts, timing scope and remaining hardware-only gates | Full suite875 pass; after CPU bias temporary repair41 affected checks pass; receipt-checked report-final-v2 covers 135 execution receipts across 43 bundles, preserving incomplete earlier runs |

FireSim hardware execution is outside the requested local proof: the user asked
for Spike and Verilator validation without FireSim. A deployment package must
reject incompatible hardware and distinguish locally verified contracts from
bitstream/loader/DRAM facts that require deployment qualification.

Current follow-up artifacts are under `../tensorir-dev/review-followup-v1`.
`source-freeze-v1` and `source-freeze-v2` are immutable experiment snapshots,
not final-source qualification. The first native framework run exposed a cloned-
Var admission bug after its two completed baselines; its journal is retained.
The second native run completed all six trials (32x4x8): CPU 9144 cycles,
default 676, two-row reuse 599. These are isolated RTL-model measurements,
not FireSim or whole-model speedups. The initial broad follow-up run passed
849/859 and found ten placement/negative-control test failures; corrected
targeted runs are retained. The complete follow-up rerun passes 875; the subsequent small CPU temporary-scope repair passes 41 affected export/bias/packing/harness checks.

Closing software evidence: `report-final-v2.json/.md`,
`final-record-replay-v1.json`, `final-package-binary-equivalence-v1.json`,
`portable-cnn-cost-gated-v1`, and `portable-decode-cost-gated-v1`.
The native CNN archive remains byte-identical after the CPU-only temporary fix.
The final cost-gated CNN median is 98897 target cycles versus 127016 for the matched
ordinary CPU graph; the default native median was99010, so no graph tuning gain is
claimed from that small difference. MLP medians are2734/3411 Gemmini/CPU.
The actual decode score function selects CPU 5544 versus 6957 native cycles, and
the CNN classifier selects CPU 519 versus 563 native. Both replay into packages.
All five corrected CNN function campaigns complete both required baselines.
Standard XGB uses random warmup until four observations; later predictions in
the final six-trial run have 2–4 distinct values. No random-warmup diversity is
reported as a learned ranking.
