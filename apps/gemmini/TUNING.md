# TensorIR scheduling and MetaSchedule

The active path is `tune_models.py` with stock extraction, EvolutionarySearch,
XGBModel, JSONDatabase and MetaScheduleApplyDatabase. The companion Merlin
OPTIMIZATION_KNOBS.md records the current audit, trial-budget research and approved
median-timeout policy. No custom bounded-search controller is used here.

## Preparation and schedule choices

Start the simplified graph at fusion=2/CSE off; stock fusion levels 0–3 and CSE
are the remaining graph alternatives. Automatic packing/bias folding are removed.
Every changed prepared graph needs fresh extraction and structural matching;
old 47-workload/26-owner counts belong to the predecessor configuration.

A contraction schedule exposes these ordinary TensorIR decisions:

| Choice | Implementation |
| --- | --- |
| Tile rectangle and tails | Legal 1..16 tile dimensions; split, blockize, pad/partition and tensorize. |
| M/N/K reuse | Three sample_perfect_tile decisions, with exact SRAM capacity checks. |
| Traversal and OS/WS | Structural sketches using primitive commands; hardware loop FSMs forbidden. |
| Operand hoisting/staging | cache_read, compute_at and explicit root/tile packing work. |
| Pipeline and buffering | Stage annotations, stock InjectSoftwarePipeline and checked physical buffers. |
| Epilogue placement and unroll | Traced placement and stock categorical/mutation rules. |

Secondary families currently offer rectangular/scalar structural alternatives.
The implementation inside an intrinsic is fixed; XGB does not independently tune
its command sequence. CPU/device alternatives share prepared mathematics, but
CPU-only unsupported tasks are reported and excluded by the current campaign
selector. Remote eligible contractions must retain required device placement.

`VerifyGemminiMemory` is a legality postprocessor, not a performance optimizer.
Allocation, host-DMA dependencies and completion waits remain required.
Command coalescing and zero-init elimination have been removed.

## Cost model and feedback

`MathematicalFeatures` replays a trace without postprocessing and calls stock
PerStoreFeature. This preserves loop/store features hidden by late tensorization.
Eager tensorization can still hide part of a secondary kernel. There is no explicit
Gemmini command/bank/latency model; finite features do not prove predictive power.

Stock XGB trains on workload-normalized costs. The helper currently sets warmup=4
instead of upstream's 100 and disables adaptive training, with one training core.
The threshold is total feature-bearing samples in that model object, not per task;
failed runner results can count. Learned predictions being enabled is not proof
that subsequent proposals were better ranked. Model save/load includes cached
training data; bind reuse to compiler, features, hardware and measurement policy.

Compact EvolutionarySearch uses population 32, two genetic iterations and four
initial unmeasured candidates; the full profile uses 64, three and sixteen.
Both use mutation probability .85 and random exploration .1. Mutator weights are
tile .9 / location .05 / unroll .05. These are integration settings, not a new
algorithm. Keep the chosen profile explicit; do not silently alter it mid-campaign.

The generic local runner historically stores cycles in run_secs. The current
Merlin FireSim adapter converts raw cycles to seconds using the documented target
counter frequency. Keep raw cycles in receipts, and never mix those database
conventions or substitute emulator wall throughput for counter frequency.

Stock runner errors have no timing; XGB uses a large failure sentinel rather than
censoring-aware estimation. Preserve error categories and never report that value
as measured cycles. The approved median wall-watchdog policy retains completed correct slow runs
as training data. Timeout samples do not update successful-duration history.
Remote wall-cancellation/correctness qualification replaces the live-observer gate.

## Record reuse and verification

- Register workload-dependent intrinsics before opening a portable database;
  TVM validates every stored trace, including unmatched records.
- Apply records with stock structural matching. No model-name or node-ID lookup.
- Keep actual hardware measurements separate from historical trace seeds.
- Check default, measured winner and connected chunk with the same arithmetic,
  inputs and timing interval. Never sum operator minima into claimed E2E cycles.
- Report unique successes, errors, training samples per workload, post-training
  proposals and measured ranking quality. A saved booster alone proves none of these.
- One legal schedule with no remaining sampled choices needs a measurement,
  not a cost-model search. Recheck this for each shape rather than assuming it.

Historical presets can have zero or partial structural coverage after this
cleanup. Re-extract and report misses; new measurements require fresh source-bound
records. The predecessor experiments and commands remain in
[the pinned tuning history](https://github.com/grammy117/tvm/blob/42b3c1ef27abe878f930e48b335b24956beb0e56/apps/gemmini/TUNING.md).
