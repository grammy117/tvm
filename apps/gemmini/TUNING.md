# TensorIR tuning defaults

Use stock MetaSchedule **ReplayTrace** and ordinary TensorIR traces. The common
tuner and tune_models.py default to replay; the FireSim runner selects replay
explicitly. There is no XGBoost import, feature extraction, model training/load/
save, UpdateCostModel callback or learned ranking in this path. A stock RandomModel
is an unused API placeholder; ReplayTrace samples trace decisions directly.
Round-robin assigns proposals across extracted tasks. Evolutionary/XGB remains
an explicit legacy option, not part of this campaign.

## Deterministic starting schedule

The common contraction baseline uses the largest exact tile divisor up to 16 for
M/N/K, respecting the PE bound and the existing exact-tail path. M=1 remains 1.
It ranks legal reuse groups (bounded 64 tiles per axis) by logical operand traffic,
then output-group size and K batch size. M grouping reuses B, N grouping reuses A;
K batching combines loads without claiming additional arithmetic reuse.
Accumulator and scratchpad estimates prune choices; actual memory lowering checks
aligned allocations/lifetimes. Try at most eight ranked choices, then the safe
one-group baseline. Explicit group/order/chunk settings are preserved.
No model names, expected answers, measured historical costs or custom graph
rewrites select the default. OS/no-hoist/no-pipeline is the conservative structural
baseline; ReplayTrace can explore the existing legal WS/hoist/pipeline/staging/
tail alternatives. This heuristic is not a latency predictor or a measured optimum.

The same helper feeds direct compilation, baseline measurement and fallback records.
Keep a correct measured baseline in JSONDatabase. Promote only actual faster correct
measurements, then validate connected chunks/complete calls; operator minima are
not an E2E result. Capacity, hazards and completion remain correctness lowering.

## Campaign

Active: ResNet50 plus 16 kernel cases in the companion Merlin
KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md. Full TinyLlama/SmolVLA are
backlogged. Default 25 total ReplayTrace proposals per chunk/case across internal
tasks and graph choices, not 25 per internal contraction/head. No ML warmup or
five-guided phase. Baseline/watchdog calibrations and timing repeats are separate
execution counts charged to the same wall budget. Failed/duplicate proposals count;
report successful unique schedules and stop reason. Stochastic failure to find
another candidate is not proof of complete space exhaustion.

ResNet 4h DSE/30min per E2E; the 16-case TVM kernel sweep has a conservative 24
occupied-slot-hour aggregate ceiling. Stop admissions at time or candidate cap;
finish active jobs under frozen median watchdogs. The selected graph starts at
fusion=2/CSE off without automatic prepacking or bias folding.

Raw measured completed target cycles are evidence. The FireSim adapter converts
them to seconds at the Runner boundary using the target counter frequency.
Never mix old cycle-valued databases with seconds-valued records. Wall duration
is for watchdogs/occupancy, not the optimization label. No timeout becomes cycles.

Current FireSim implementation still measures extracted contractions; grouped
complete-call builders/checks remain required for the new kernel campaign.
Bind actual source, toolchain, hardware, placement, complete outputs and timing
scope. ReplayTrace does not remove these requirements. See Merlin's current
FIRESIM_HANDOFF.md for restore/build commands, receipts and readiness limitations.
