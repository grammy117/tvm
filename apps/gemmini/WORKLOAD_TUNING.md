# Per-workload Verilator chunks

Use `workload_tuning.py` to prepare small, independently tunable pieces and inspect
current knobs **before** spending search time. This is an experiment driver;
compiler decisions still derive from IR and capability facts, not model names.
The output directory is immutable. Each case owns a program hash, capability
fingerprint, seed database, placement/resource inventory, and later campaign.
Changing a program, phase, numerical recipe or hardware profile requires a fresh
case. Decode and prefill never share a campaign by default.

## Compiler-controlled choices

This driver does not assign tile sizes by model name or edit winner traces.
`current` replays unchanged measured records through TVM structural matching;
`tune` delegates to the existing compiler search. The eleven graph definitions
are experiment inputs, not hand-written kernel schedules or new fusion passes.

The existing `gemmini_sketch.py` / `gemmini_meta_schedule.py` search space includes
M/N/K reuse, rectangles/tails, OS/WS, traversal, hoisting, software pipelining,
producer/epilogue placement and unroll. Structural sketch alternatives and
sampled tile/location/unroll decisions are distinct parts of that search. The
CPU/device union covers eligible functions; unsupported CPU-only tasks are
reported but are not currently tuned.

`tune_chunk.py` now searches graph preparation choices before task extraction.
The generic surface exposes fusion levels 0/1/2/3, constant weight packing,
bias preload canonicalization, and common-subexpression elimination, each as
explicit compiler options. These are stock Relax
`Knob`/`Choice`/`Trace` alternatives, replayed through `graph_plans.py`. Each plan
is compiled and measured as a complete connected chunk, including CPU work and
host intermediates. Whole-chunk timing selects the plan. Inner TensorIR campaigns
remain separate so their exact-matching records can be shared across models.
Use `--tuning-preset none` for an unseeded/default connected control; the default
`rtl-v1` supplies current matching traces. An explicit validated database takes
precedence over the preset. Historical costs never score an outer graph choice.
For a partial fresh library, `--fallback-tuning-database DIR` retains frozen
current traces on primary-library misses through stock `OrderedUnionDatabase`.
It requires a primary match, records both database identities and selection
sources, and never imports fallback costs into the fresh measurement library.
Without an explicit fallback, misses use reported defaults. The validated
16-record library from the FASED campaign ships as preset `fased-v1`; static
compilation selects it with `--tuning-preset fased-v1 --fallback-tuning-preset rtl-v1`. Measure the whole
configuration: replacing a partial library can otherwise change unrelated
operator schedules as well as the newly tuned operator.

The bounded experiment initially measures four combinations (fusion 2/0 ×
packing on/off), with a separate CSE alternative motivated by repeated producers.
This is an experiment budget, not a hardcoded model policy. `--plans` accepts any
of the 32 canonical `fuse{0,1,2,3}-pack{0,1}[-bias0][-cse1]` configurations.
The same choices are available through `GraphPlan` and compile/tune CLI flags
`--graph-fusion-level`, `--no-prepack-weights`, `--no-bias-preload`, and `--graph-cse`.
All 32 combinations preserve outputs on an independent integer two-contraction
test; a separate floating projection test verifies shared-producer CSE and trace
replay. Performance still requires measurement on the selected workload.

CSE uses TVM's existing `EliminateCommonSubexpr` before legalization; it is not a
model-specific quantizer rewrite. Shared producers can change both work and live
storage, so the choice stays optional and measured. Model names occur in fixtures
and experiment profiles, never in these compiler decisions. Semantic legality,
capacity and synchronization remain mandatory; search chooses profitability.

Cross-function and cross-call SRAM residency are not implemented. A legal future
residency choice needs a boundary representation, alias/lifetime and capacity
proofs, supported intervening arithmetic and completion handling. Existing fixed
passes still perform normalization and legality checks. Searching their legal
alternatives changes profitability decisions; it does not weaken their proofs.
The companion Merlin `PERFORMANCE_AUDIT.md` records the original audit and full
PyTorch-export-to-ELF flow; `FASED_SWEEPS.md` records this execution follow-up.

The current outer knob sets a whole-module fusion policy; it does not enumerate
per-edge partitions. Inner search still uses two-level perfect-tile sampling, a
64-group sampling guard, divisor-based rectangles/tails and legal boolean pipeline
alternatives. CPU-only extracted tasks are excluded. Generalizing those search
policies and adding proven resident-region representations are future compiler
work; a model-name switch would not address these limits.

## FASED latency-pipe campaigns

Use the new opt-in memory variant for current tuning. Keep the default simulator
and its old measurements unchanged. The runner reads the selected pin registry
through `mem_perturb.pinned_latency_pipe`, verifies the variant's control relink
against the qualified base simulator, clears inherited perturbation settings,
and requires the console to confirm the exact model and all four parameters.
The current pinned policy is read/write latency 30/30 and outstanding limits
16/16. This is not yet a same-ELF FireSim calibration.

Use a fresh campaign per phase/graph plan and explicit limits. For example:

```bash
"$PY" "$TVM/apps/gemmini/tune_models.py" --program "$CASE/program.json" \
  --work-dir "$RUN/operators/phase-name" --baseline-database "$CASE/seed" \
  --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --spike-receipt "$SPIKE_RECEIPT" --hardware-receipt "$RTL_RECEIPT" \
  --memory-variant-receipt "$MEMORY_VARIANT_RECEIPT" \
  --firesim-memory-artifact "$FIRESIM_MEMORY_ARTIFACT" \
  --memory-pin-registry "$MEMORY_PIN_REGISTRY" --timing-policy fixed-warm \
  --graph-fusion-level 2 --trials-per-task 4 --trials-per-iteration 2 \
  --parallel 1 --builders 1 --baseline-workers 1 \
  --max-cycles 6000000 --cpu-max-cycles 500000 --rtl-seconds 1200
```

These bounds are illustrative: the retained study records separate limits for
each phase. `--tasks` selects extracted task indices for a limited inner campaign;
report untuned functions too. Current matching traces are replayed and measured
first, then defaults/CPU controls and new search candidates. Only fresh passing
costs enter the database. A four-trial campaign is an initial search, not evidence
of convergence or post-training XGB ranking.

Operator fixtures now preserve immutable arguments from the prepared graph.
`--input-fixture FILE.npz` additionally supplies `input_0`, etc., and propagates
them through unscheduled mathematical host TIR to bind consumer activations.
This reference is independent of the candidate schedule, not an independent model
oracle. `operator-input-bindings.json` records every occurrence and the chosen
representative; differing constant sets are explicit. Unbound activations use
the fixed synthetic generator. Shape/dtype mismatches, output overrides and
known zero direct-input integer divisors are refused. These mechanisms derive
from IR and array bytes, without model-specific constants or seed substitutions.

The simulator limit covers warmup, measured work, startup and **all output
validation**. `--validation-overhead-cycles` sets the additive allowance used
with adaptive caps; it is never added to the reported inference cost. Its legacy
default is 300,000, which can be too small for large floating outputs. The hard
`--max-cycles`, CPU cap and wall limit still apply. A timed line followed by a
validation timeout is a failed candidate, not a ranking label.

The connected driver consumes a serialized `measure.Toolchain` JSON with fields
`gcc`, `adapter_receipt`, `spike_receipt`, `hardware_receipt`,
`memory_variant_receipt`, `firesim_memory_artifact`, `memory_pin_registry`, and
`timing_policy: "fixed-warm"`. The retained study's `memory/toolchain.json` is a
concrete example. All file paths identify the actual receipts on the current host.

```bash
"$PY" "$TVM/apps/gemmini/workload_reference.py" --program "$CASE/program.json" \
  --output-dir "$RUN/fixtures/phase-name" --checkpoints
"$PY" "$TVM/apps/gemmini/tune_chunk.py" --program "$CASE/program.json" \
  --fixture "$RUN/fixtures/phase-name/fixture.npz" \
  --fixture-provenance "$RUN/fixtures/phase-name/fixture-provenance.json" \
  --toolchain "$RUN/memory/toolchain.json" --output-dir "$RUN/connected/phase-name" \
  --max-cycles 60000000 --rtl-seconds 3600
```

The independent reference interpreter supports the prepared surrogate subset;
it refuses unsupported arithmetic. Actual exports need independently admitted
boundary fixtures with their own numerical contract. The command above generates
synthetic inputs, not actual-model capture evidence. ResNet's post-ReLU input
uses `--nonnegative-int8`; actual fixtures can be supplied with `--inputs`.

Both drivers use one unconditional warmup and one measured invocation of the
same buffers. The connected guest links the complete static package and reuses
the existing measurement runtime. It checks every output with unchanged
tolerances, plus Spike input/canary and wrong-reference controls. No graph timing
is inferred from operator labels. `sweep.json`, graph traces, packages, ELF
audits, logs and failed/capped trials remain alongside the results.

After campaigns complete, validate and share only structurally applicable records:

```bash
"$PY" "$TVM/apps/gemmini/tuning_library.py" merge \
  --campaigns "$RUN/operators/phase-one" "$RUN/operators/phase-two" \
  --output-dir "$RUN/operator-library"
"$PY" "$TVM/apps/gemmini/tuning_library.py" coverage \
  --database "$RUN/operator-library" --program "$CASE/program.json" \
  --output "$RUN/library-coverage.json"
# Recompile and remeasure a graph with matching inner records in a NEW output:
"$PY" "$TVM/apps/gemmini/tune_chunk.py" --program "$CASE/program.json" \
  --fixture "$RUN/fixtures/phase-name/fixture.npz" \
  --fixture-provenance "$RUN/fixtures/phase-name/fixture-provenance.json" \
  --toolchain "$RUN/memory/toolchain.json" --output-dir "$RUN/combined/phase-name" \
  --tuning-database "$RUN/operator-library" --plans fuse2-pack1 \
  --max-cycles 60000000 --rtl-seconds 3600
```

The library checks workload, target, exact trace replay against retained scheduled
IR, cost, fixed timing/memory identity, ELF/LLVM/runner and execution evidence.
It never imports historical costs. A zero-match explicit database is rejected by
compilation. Partial coverage still uses compiler defaults for misses; inspect
that inventory before interpreting connected results. A measured graph winner
belongs to its exact program and phase, not every stage with the same model name.

Use `replay_candidate.py --source-campaign DIR --artifact FILE --output-dir NEW
--max-cycles N --rtl-seconds N` to retry a retained capped search candidate with
more total validation allowance. It verifies the exact trace/module/ELF and
numerical blobs, copies them into a fresh campaign, reruns controls and commits a
standard record only after a complete pass. The original failure remains intact.
The older `retry_capped.py` predates these memory/input bindings and must not be
used on the new FASED campaigns.

## Prepare now

Use the established TVM environment, with `PYTHONPATH=$TVM/python`, the matching
native `TVM_LIBRARY_PATH`, required dependencies, and `OMP_NUM_THREADS=1`.
`$PY` is that environment's Python. No C++ changes are required for these tools.
The known compiler-v5 library is usable only because its native source matches
the inspected handoff; record actual source/library identities on another host.
Use Merlin's product helper for generated output:

```bash
export MERLIN_REPO_ROOT="$MERLIN"
RUN=$(PYTHONPATH="$MERLIN/src" "$PY" -c \
  'from merlin.common.artifacts import new_product; p=new_product("audits", version=1, target="gemmini-workload-tuning"); p.write_manifest(); print(p.path)')
"$PY" "$TVM/apps/gemmini/workload_tuning.py" prepare --output-dir "$RUN/chunks"
```

The default eleven cases cover seven current measured operator shapes and four
connected chunks. Use `--cases smolvla-action-out tinyllama-decode-kv` for a smaller
preparation. No simulator or search runs during preparation. Full K/N dimensions,
32-row prefill, one-row decode, M50 action tails and full 7×7 ResNet spatial
extents are retained. A single operator executes one site rather than all the
layer copies. Architectural site count and outer stage count are separate
metadata; these counts are not measured full-model latency.

Read each case's:

- `case.json`: source kind, shape/scope relationship, phase/site counts, program,
  source/library and capability identities; qualification/search initially not run.
- `extraction.json`: all tasks, including unsupported CPU work. A connected chunk
  includes its quantization, activation, concatenation, residual or requant work.
- `current-knobs.json`: exact stock traces that structurally match this program,
  historical operator cycle labels, and explicit misses. Historical cycles are
  not connected-chunk timings. It is valid for a new arithmetic graph to miss.
- `placement-{default,current}.json` and `resources-{default,current}.json`:
  actual lowered placement and SRAM allocation, staging waits and extra fences.
  The final completion fence is additional to the reported internal fence count.
- `seed/`: integrity-checked `rtl-v1` stock JSONDatabase. Compiler matching is
  structural; it does not assign a schedule by case name or M/N/K alone.

The bootstrap weights and inputs are synthetic and use legacy pilot arithmetic.
They are useful for schedule/traffic experiments, not an EL4 numerical release.
The TinyLlama MLP chunk takes normalized hidden and pre-normalization residual
as distinct inputs. It omits attention, norm production and KV updates. SmolVLA's
chunk takes the time embedding as input and omits its production and the expert
transformer. The ResNet pair omits the third bottleneck convolution/residual.
Thus none is a complete layer, model or recurrence. No camera-dependent prefix
pilot is silently imported from the older three-camera model.

## Prefer exact prepared program chunks when available

```bash
"$PY" "$TVM/apps/gemmini/workload_tuning.py" inventory \
  --program "$PROGRAM" --output "$RUN/bindings.json"
# Inspect binding indices, dependencies and tensor types; choose a connected interval.
"$PY" "$TVM/apps/gemmini/workload_tuning.py" slice \
  --program "$PROGRAM" --start "$START" --stop "$STOP" \
  --workload tinyllama-prefill-layer0-ffn --stage-calls 1 --sites 22 \
  --output-dir "$RUN/exact-prefill-ffn"
```

The interval is half-open `[start, stop)`. It preserves every selected expression
and static dimension. Every selected value used later is a live-out, including
residual/cache branches; external tensor dependencies become live-ins. External
literal constant bindings are retained. Constant-producing computation outside
the interval can instead become a live-in; inspect the boundary and include that
producer in the interval if its cost belongs to the experiment. Opaque/global
calls and non-VarBinding bodies are refused rather than changing semantics.
Inspect the full stage's inventory too: a chunk cannot account for excluded CPU
work, distant consumers or a different recurrence. Supply real boundary fixtures
from an independently admitted capture before claiming actual-model coverage.

## Compile current choices and use the general package qualifier

```bash
CASE="$RUN/chunks/smolvla-action-out"
"$PY" "$TVM/apps/gemmini/workload_tuning.py" compile --case-dir "$CASE" \
  --variant current --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT"
# Repeat with --variant default and --variant cpu for separately retained baselines.
"$PY" "$TVM/apps/gemmini/compile_package.py" --verify "$CASE/package-current"
```

`current` uses the frozen per-case seed. With zero matches it compiles reported
unmeasured defaults. `tuned` uses only this case's `campaign/database` and rejects
zero matches. Separate package directories prevent overwriting baselines.
`memory-audit.json` reports actual graph call counts, per-function SRAM use and
direct host tensor edges. `graph/graph.json` contains constant/workspace/liveness
plans. Edge bytes are logical tensor extents, not measured DRAM or DMA traffic;
views/aliases are not expanded. Inspect every CPU fallback and miss.

Use the existing independent-fixture qualifier for the **whole connected graph**:

```bash
"$PY" "$TVM/apps/gemmini/qualify_package.py" \
  --package "$CASE/package-current" --fixture "$FIXTURE" \
  --fixture-provenance "$REFERENCE_PROVENANCE" --output-dir "$CASE/qualified-current" \
  --riscv-gcc "$GCC" --rtol 1e-5 --atol 2e-6 --timing-repeats 3
"$PY" "$TVM/apps/gemmini/verify_tensorir.py" run \
  --bundle "$CASE/qualified-current" --backend spike --name spike-valid \
  --simulator-receipt "$SPIKE_RECEIPT" --seconds 120
"$PY" "$TVM/apps/gemmini/verify_tensorir.py" run \
  --bundle "$CASE/qualified-current" --backend spike --name spike-wrong \
  --simulator-receipt "$SPIKE_RECEIPT" --negative --seconds 120
"$PY" "$TVM/apps/gemmini/verify_tensorir.py" run \
  --bundle "$CASE/qualified-current" --backend verilator --name rtl-valid \
  --hardware-receipt "$RTL_RECEIPT" --cycles 40000000 --seconds 1800
"$PY" "$TVM/apps/gemmini/verify_tensorir.py" run \
  --bundle "$CASE/qualified-current" --backend verilator --name rtl-wrong \
  --hardware-receipt "$RTL_RECEIPT" --negative --cycles 40000000 --seconds 1800
```

The fixture contains `input_0`, etc., and `expected` (or `expected_0`, etc.) with
the exact exported shapes/dtypes. Provenance records its SHA256, reference contract
and independently evaluated reference source hashes. Never use candidate outputs
as their own reference. Integer outputs remain exact and floats finite. Keep the
selected recipe's actual tolerance if it differs from the example. The qualifier
has one warmup and three fixed measured graph invocations, including CPU glue,
admission and completion, excluding startup/checks/HTIF. Capped runs are failures,
not timing labels. The CLI accepts explicit bounds through 100 million cycles and
3,600 seconds; its default remains 300,000 cycles / 60 seconds. Size cycle/wall
caps per case and record them; larger connected chunks can exceed this illustrative
bound. The earlier setup's SmolVLA action-output positive control with three measured
repeats timed out at 900 seconds, while its wrong-oracle RTL control and both
Spike controls passed. Thus this setup has no completed positive RTL timing;
the larger example allowance was not then validated. These historical commands
use stock magic memory. The FASED campaign above has a separate fixed-warm timing
policy and results; do not combine their labels. No paid FPGA runs are launched here.

## Tune later, independently for each prepared case

```bash
"$PY" "$TVM/apps/gemmini/workload_tuning.py" tune --case-dir "$CASE" \
  --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT" \
  --spike-receipt "$SPIKE_RECEIPT" --hardware-receipt "$RTL_RECEIPT" \
  --trials-per-task 16 --parallel 1 --max-cycles 40000000
"$PY" "$TVM/apps/gemmini/workload_tuning.py" compile --case-dir "$CASE" \
  --variant tuned --gcc "$GCC" --adapter-receipt "$ADAPTER_RECEIPT"
```

This delegates to the existing stock MetaSchedule path, including positive and
wrong-oracle candidate controls and CPU/default baselines. It does **not** turn
operator search into graph-level search. Current records are a baseline seed for
compilation, not unmeasured search proposals. The old default 14 runner processes
is deliberately not inherited: this wrapper defaults to one runner, one builder
and one baseline worker; a four-trial batch permits later post-warmup proposals.
The budget is an initial setting, not convergence evidence. Search remains over
M/N/K groups, rectangular tiles/tails, OS/WS, traversal, operand hoisting,
pipelining, staging/epilogue placement, and unroll choices within existing proofs.

Requalify the delivered tuned **chunk**, then compare its whole-graph cycles,
working set and CPU/device mix against default/current using the same fixture,
hardware, warmup policy and bounds. A sum of operator minima cannot promote a
chunk winner. Keep phase-specific profiles and use whole-session FireSim runs to
confirm that improvements transfer. The new Merlin session `--profile-stages`
option preserves aggregate timing and prints buffered warm/measured per-stage
counters after validation; its instrumentation overhead is explicitly recorded.

The package qualifier identifies array work from compiled operator primitive
references, so a database schedule does not need legacy M/N/K metadata to pass
its activity control. Both primitive OS and WS compute calls are counted.
