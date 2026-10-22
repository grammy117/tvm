# Gemmini TensorIR and MetaSchedule

The current TensorIR backend tunes individual prepared TIR operators
for the pinned GemminiRocketConfig. It uses mathematical TensorIR, registered
intrinsic families, stock MetaSchedule search, and the stock Relax database pass.
The primary entrypoint is `tune_models.py`; `tune_tensorir.py` forwards to it.
The previous custom cycle controller and custom tuning-record replay format have
been removed from this path. The older BYOC/template backend is retained for
regression comparisons and is not the tuning implementation described here.

The design follows the separation of tensor semantics, schedule transformations,
and tensorization in [TensorIR](https://arxiv.org/abs/2207.04296), and the use of
intrinsic families and probabilistic schedule decisions illustrated by the
[RVV MetaSchedule work](https://arxiv.org/abs/2507.01457). Their published speedups
are not evidence for Gemmini performance.

## What is stock TVM and what is target-specific

| Responsibility | Implementation |
| --- | --- |
| Relax legalization, fusion, extraction and call-count weights | Stock `LegalizeOps`, `FuseOps`, `FuseTIR`, `ms.relax_integration.extract_tasks` |
| Mathematical loop transformations | Stock TensorIR `split`, `reorder`, `cache_read`, `cache_write`, `compute_at`, `transform_layout`, `blockize`, `decompose_reduction`, `loop_partition` and `pad_einsum` |
| Sampling and mutation | `sample_perfect_tile`, stock unroll categorical decisions, `MutateTileSize`, `MutateComputeLocation`, `MutateUnroll` |
| Search and task allocation | Stock `tune_tasks`, `EvolutionarySearch`, gradient task scheduler; `ReplayTrace` for the random comparison |
| Cost model | Stock `XGBModel` and `PerStoreFeature`; a target adapter replays traces without postprocessing to expose mathematical stores to the extractor |
| Persistent records and application | Stock `JSONDatabase`, `TuningRecord`, `MetaScheduleApplyDatabase` |
| CPU placement | Stock LLVM schedule-rule classes with parallel launch and vectorization disabled for one RV64GC hart |
| Accelerator placement | `GemminiScheduleRule`, `gemmini_sketch.py`, shape-family intrinsics, bias/readout normalization and capacity verification |
| Physical SRAM and host synchronization | `gemmini_memory.py` lifetime allocation, dependence checks and command lowering |
| Device evaluation | Target builder/runner, guarded mathematical oracle, Spike verification and Verilator cycle measurement |

The compiler-core changes are general TensorIR/MetaSchedule fixes:
subspace division moves tile-aligned partition offsets outside the tensorized
inner iterator (including its single-term fast path), and region coverage
analysis handles an empty intersection before subtracting its infinity sentinels.
Trace replay now rejects a resampled/mutated candidate when a later schedule
primitive raises a legality error. The rendered C++ exception retains a distinct
type, so unrelated internal failures still propagate. This repairs the pinned
stock search implementation; it does not replace its algorithm or mutators.
Arith, blockize, loop-partition and both search-strategy regressions exercise
these changes.

After a contraction sketch is applied, its padding/reindexing stages remain
part of that sketch. They are not reconsidered as standalone DMA-copy kernels
when reduction decomposition removes the original initialization block.
Standalone native copy scheduling also rejects residual block predicates:
an im2col boundary condition cannot be replaced by an unconditional rectangle.

## Search space and fixed lowering

`GemminiScheduleRule` operates on the supplied schedule and block. The space is a
stock union of Gemmini sketches and CPU sketches over **the same prepared
function**. Bias folding, layout preparation and scale specialization occur
before both placements are considered. The CPU baseline reported as
`cpu_untuned` is distinguished from the stock-scheduled CPU candidates.

A contraction sketch chooses legal intrinsic rectangles, traversal, operand
hoisting, primitive OS/WS dataflow, software pipelining and producer/epilogue
placement. Within it, three `sample_perfect_tile` decisions split the M, N and K
tile counts into outer and reuse groups. Ordinary stock mutators change these
decisions. Structural alternatives are separate sketches, not Python branches
hidden behind unrecorded decisions. Candidate traces retain their sampled values.

* Reuse is checked against 16,384 scratchpad rows and 1,024 accumulator rows
  (1,024 and 64 full 16x16 tiles). Operand hoisting can retain a complete operand
  when it fits; a product of sampled factors that exceeds capacity is rejected.
  A factor is bounded at 64 tiles to avoid obviously excessive groups; exact
  allocation, including two pipeline slots, is the admission test.
* Intrinsics support rectangles from 1 through 16 in each dimension. Divisor
  rectangles avoid padding. Interior/tail sketches use `loop_partition`; a tail
  rectangle divides both its offset and extent, preserving aligned tile indexing.
  For example, a 50-row tail uses two rows, while a 19-column tail uses three
  one-column rectangles. These output tails need neither host padding nor a wait
  after each DMA. Separate potentially overlapping output groups are ordered.
* K16 is the default when K is divisible by 16. Smaller K chunks are considered
  when they reduce padding. A padded full-depth alternative can materialize its
  padding once at function root; tile-local staging remains an alternative with
  the synchronization its host-buffer reuse requires. No claim is made that
  arbitrary K padding is free.
* `InjectSoftwarePipeline` creates alternate operand-buffer slots. The physical
  allocator accounts for both slots; overlapping operations do not silently
  alias one SRAM range.
* Bias initialization preloads the accumulator. Supported FP32 nearest-even
  scalar readout chains, including `clip(round_even(x / 2^k), 0, 127)`, are proved
  against the full signed INT32 input domain before INT8 scaled/ReLU readout is
  selected. Per-channel scales and different rounding recipes remain CPU work.
* Primitive WS uses preload/compute commands and retains a compatible weight
  tile when legal. The instruction audit forbids hardware loop FSM opcodes.

The following are fixed lowering choices, not tuning knobs: load/store
configuration is issued when its shadow changes; a zero-initialized first K
chunk overwrites the accumulator instead of issuing a separate zero fill;
compatible full-width adjacent loads are coalesced into multi-block mvin;
non-overlapping buffer lifetimes reuse SRAM rows; and host-memory hazards get
completion waits. Bias preload is preserved. The final function completion wait
makes returned data available to the caller.

## Measurement and record contracts

The builder exports one scheduled operator into a freestanding RV64 ELF with
fixed inputs and a host mathematical oracle. Spike must accept the positive ELF
and reject its wrong-oracle variant. Verilator executes the same positive ELF,
checks its outputs, and supplies the cycle label. No Spike wall time is a cost.
RTL verification of a wrong-oracle ELF is retained separately for representative
slices. Input immutability and guard checks run on Spike; they are intentionally
outside the RTL timing interval.

`qualify_tuning_package.py` links a representative operator directly from the
exported `libmodel.a`, using the campaign's guarded inputs and mathematical
oracle. It recompiles only the small calling wrapper and checks the operator
workspace ABI. Both positive and wrong-oracle ELFs run on Spike and RTL. This
binds slice qualification to the delivered archive rather than only to a
separately compiled search candidate.

The timer surrounds `model_run`, including its CPU epilogue and completion fence.
A first call shorter than 400,000 cycles is repeated, and the second call is the
reported warm interval; longer operators report the first interval. Both values
and the timing policy are retained. Comparisons that cross this threshold are
not presented as a controlled warm-cache experiment.

TVM's `run_secs` field stores **target cycle counts** in these databases. It is a
ranking field here, not seconds and not a conversion using an assumed frequency.
Stock scheduler log headings may consequently say microseconds; use the retained
cycle fields for interpretation. Failed measurements use TVM's invalid-record
sentinel and cannot become queried winners.

Baselines are measured outside the search database, and the measurement cache
serializes concurrent requests for identical workload/guest-ELF pairs. This also
binds the input blobs, wrapper and adapter. Older pilot caches used LLVM hashes;
the current driver migrates their entries through the retained ELF identities.
A failed capped run may be retried only when its cycle allowance is increased.
A repeated proposal uses the retained result and is marked `reused`. Once search finishes,
the verified default trace is committed through the standard database API so a
deployment can retain it if search found nothing faster. No custom winner replay
is used. Fresh compilation registers the workload's intrinsic families before
loading the database; the stock Relax pass performs matching and application.
After both comparisons finish, `merge_comparison_records.py` can retain verified
ReplayTrace records in the deployment JSONDatabase using its standard commit
API. Search-only minima and matched budgets still come from the separate trial
logs. This aggregation neither feeds measurements back into a running search nor
implements a separate schedule replay path.

When a completed candidate reached its cycle cap, `retry_capped.py` can execute
that exact retained ELF with a larger allowance. It verifies the executable and
hardware hashes, preserves the original outcome, and commits a successful label
with the existing trace through `JSONDatabase.commit_tuning_record`. It neither
builds a new schedule nor applies a trace. These extra measurements are listed
separately and excluded from the original search minima and matched budgets.
The bounded ResNet campaign retries the first 3x3 proposal from each strategy
after all original 3x3 proposals reached their 20-million-cycle cap.

The XGB warmup is four feature-bearing samples for these bounded experiments, rather
than its upstream default of 100. The model and `cost-model.json` retain its
training state. Early pilot campaigns that used the upstream warmup are labeled
as such; they do not establish learned ranking. The bounded ResNet campaign
submitted all eight proposals before its first RTL result, so its final trained
model is reusable but is not evidence of learned ranking during that run. The
16-trial smoke campaign includes proposals after training and retains the
corresponding model state. A small budget does not establish a global optimum.
ReplayTrace comparisons and actual successful-trial counts are reported
alongside results.

## Workloads and interpretation

`model_workloads.py` emits semantic Relax operator programs with real projection
or convolution dimensions and repeated calls. Stock extraction deduplicates
identical tasks and supplies their call counts. Synthetic constants are used;
this is a compiler/operator experiment, not pretrained quality validation.

* TinyLlama uses the integer projection recipe, with one-row decode and 32-row
  prefill chunks. Selected attention projections retain K=2048 and N=256/2048.
* SmolVLA includes vision, prefix and action-expert inventories. The bounded
  campaign selects action input/output projections with 50 action rows,
  including the M50 tail. Counts reflect ten denoising steps.
* ResNet50 uses NHWC/OHWI INT8 convolutions, INT32 bias and its integer rounding
  recipe. The selected 1x1 and 3x3 tasks retain their model spatial/channel sizes.
  The integer recipe differs from the FP32 nearest-even/ReLU readout pattern,
  so its CPU requantization is not mislabeled as native INT8 readout.

A weighted sum covers only the selected operators. It excludes attention,
normalization, nonlinear activations, scheduling between operators, and other
unselected work; it is not an end-to-end model latency. TinyLlama and SmolVLA
packages contain these representative operator programs, not complete inference
sessions. Full ResNet execution is optional and is not required by this bounded
per-operator campaign.

Cycle and wall-time caps bound simulator use. An unfinished CPU/default trial is
reported as unavailable with its failure artifact; it is never assigned a fake
cycle value or used to calculate a speedup. Differences measured on separate
operators, phases or baseline types are not added together.
The RTL cap covers the complete simulation, including startup and validation,
not just the timed operator interval. An extended retry can therefore pass with
an operator cycle count below the original simulation cap.

## Reproduction and delivery

Source `../metaschedule-dse-env.sh` in this workspace. The campaign launcher uses
`merlin.common.artifacts.new_measurement` and freezes Python/app sources, the
worktree diff and compiler-library hashes under the measurement directory.
`hardware-provenance.json` verifies actual simulator/plugin/adapter bytes against
the pinned receipts before a run begins.

```sh
../metaschedule-dse-campaign.sh tinyllama-decode --operators 0 1 \
  --trials-per-task 8 --parallel 4 --builders 2 --baseline-workers 2
$PY apps/gemmini/report_tuning.py /absolute/path/to/campaign
$PY apps/gemmini/compile_package.py --program /path/to/campaign/program.json \
  --tuning-database /path/to/campaign/database --output-dir /path/to/package \
  --riscv-gcc /path/to/riscv64-unknown-elf-gcc --adapter-receipt /path/to/receipt.json
$PY apps/gemmini/qualify_tuning_package.py /path/to/campaign /path/to/package WORKLOAD_KEY
```

Each finished campaign carries `results.json`, `results.csv`, sampled best traces,
standard reusable databases, baseline and candidate ELFs/logs, source identities
and deployment packages. Package manifests distinguish measured database
placements from provisional defaults and include a relocatable record copy.
The final archive is checked for unresolved symbols and forbidden FSM opcodes.

A FireSim-ready package is an offline RV64 library with a documented ABI and
local Spike/RTL evidence. It is **not a FireSim deployment qualification**. The
actual bitstream, driver, loader, memory map and deployed timing still require
binding and validation on the selected FireSim system. LOOP_WS/LOOP_CONV remain
out of scope by explicit policy, even though the pinned RTL contains those FSMs.

## Real programs and complete extraction

`--model` uses synthetic weights and representative operator shapes. It does not
load a checkpoint or execute a model session. For an actual exported EL4 stage,
use the same semantic input that `compile_package.py` will compile:

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$PROGRAM" \
  --work-dir "$RUN/extraction" --extract-only
```

`PROGRAM` is semantic Relax JSON or a `torch.export` `.pt2` archive. Both tuning
and compilation use `program_io.import_program`. `extraction.json` includes every
stock extracted task, its call-count weight and its eligibility. Unsupported
Gemmini tasks remain visible as CPU fallback; the current driver does not tune
those CPU-only tasks. Eligible functions have both CPU and Gemmini placements
unless `--no-cpu` is selected. `--tasks` selects eligible indices, not all-task
indices. There is no hardware access or toolchain requirement in extraction-only
mode. `prepared.json` records the common mathematical pipeline. A changed
semantic program is refused before existing campaign evidence is overwritten.
Use separate fresh roots for different stages, recipes and compiler revisions.

Remove `--extract-only` and supply the four receipt/toolchain flags documented
above to measure. An extraction from a real program still uses generated,
independent per-operator inputs for tuning; it is not whole-model correctness.
Compare extracted weights with actual session invocation counts. A decode stage
called 32 times and a denoiser called ten times need external session weighting;
`extract_tasks` only sees calls inside the imported IRModule.

## Further tuning work, in priority order

1. Extract every actual model stage and account for all retained CPU work. Freeze
   checkpoint, preprocessing, quantization, sequence length, state and output
   contracts first. A faster synthetic projection says little about a session
   dominated by host softmax, conversion or cache copies.
2. Use several search batches after XGB has trained; compare multiple seeds with
   ReplayTrace under equal successful-measurement and wall-time budgets. Report
   censored/failed baselines separately. A trained booster alone does not prove
   that any later candidate was ranked by it.
3. Add a FireSim implementation of the existing Builder/Runner contract, backed
   by Merlin's queue/receipt APIs. Preserve ELF/input identity, independent output
   checks, negative controls and one documented timing window. Keep FireSim and
   Verilator labels in separate databases. Do not feed Spike wall time to XGB.
4. Tune remaining CPU-only tasks with the same one-hart ISA policy. Expand the
   current measurement oracle only where its input/output contracts can be
   checked independently; do not silently drop unsupported tasks.
5. Add numerical-contract-driven fusion and producer placement, packing amortized
   over full sessions, and cross-call weight/cache lifetime planning. The present
   SRAM allocator is per function; it is not a model-global resident-weight plan.
6. Expand legal intrinsic/sketch families and resource checks for more layouts,
   tails and proven readout chains. A new hardware ABI needs a new capability
   fingerprint, adapter, instruction audit and qualification. The field named
   `dataflow` in the current fingerprint is the historical default OS profile;
   explicit primitive WS sketches are already supported. Changing that schema
   invalidates records and must be versioned deliberately.
7. Add traffic/host-staging features to the mathematical feature adapter, compare
   against stock PerStoreFeature, and validate transfer on withheld shapes. Reuse
   measurements only when program, inputs, compiler, adapter, hardware and timing
   contracts agree. Do not infer cost-model transfer from shared model names.
8. Only after kernels pass, optimize full-session latency and peak memory on the
   selected FireSim image. Call-count weights are a starting heuristic; measured
   end-to-end bottlenecks decide priorities. Keep quality gates unchanged.
