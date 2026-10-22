# Lowering, optimization ownership and upstream audit

## Audit boundary

The upstream reference is Apache TVM v0.19.0,
`c4dc0c29ff81ddae688da24625a603da3b4a2c0e`. The inherited ONNX-fix base
`fb78e0e3adcee44948a1542cd394710b3a6f9dba` is **not** the upstream boundary:
five earlier frontend fixes must be included. The original published bringup tip
is `6b376cc8895a83387ec38cc06ef8f7b09827b071`; the completed TensorIR/MetaSchedule
snapshot is `87f07883b65ee55dbd1b93f665aaab87cd8a1e50`. The latter spans 136
commits from the Apache base. The handoff artifact contains the original patch,
commit list, complete per-file inventory, source hashes, reconstruction proof,
curated commit mapping and validation logs. It also inventories inherited EL4
changes separately from the TVM comparison addition to Merlin.

“Stock TVM” below means its general machinery, with the listed fork correctness
fixes. It does not mean an unmodified upstream binary. “Supplied” means this
integration provides target semantics, policy or adapters to TVM. A custom TVM
pass still runs **inside** TVM; it is not an external postprocessor.

## Exact lowering boundary

```text
checkpoint + data + session + numerical/quantization contract [Merlin recipes/operator]
  -> torch.export or ONNX, explicit state inputs/outputs [frontend producer]
  -> Relax import/type/shape normalization [TVM frontend + fork fixes]
  -> LegalizeOps / pattern annotation / folding / graph and TIR fusion [TVM]
  -> scale specialization / weight packing / bias / readout normalization [supplied TVM passes]
  -> extracted mathematical tasks + occurrence weights [TVM]
  -> CPU/Gemmini union of legal sketches [supplied policy using TVM schedules]
  -> stock search + cost model + JSONDatabase [TVM, supplied feature/measurement adapters]
  -> MetaScheduleApplyDatabase, or default checked scheduling [TVM + supplied rules]
  -> tensorization, software pipeline [TVM against supplied intrinsic semantics]
  -> SRAM address assignment and host-DMA waits [supplied TIR lowering]
  -> LLVM RV64 object code [TVM/LLVM]
  -> static call_tir graph wrapper/constants/workspace [custom exporter]
  -> primitive adapter + math/runtime archives + startup/linker [external toolchain/platform]
  -> complete model session, independent outputs and queued FireSim receipt [Merlin/operator]
```

TVM is responsible for:

1. IR construction/import infrastructure, shape/type inference, binding and
   constant representation. The producer still defines model semantics.
2. Mathematical Relax-to-TE/TIR legalization, operator-pattern annotation,
   constant folding, binding canonicalization, graph fusion, TIR fusion,
   reshape rewriting and dead-code elimination.
3. Buffer-region/dependency analysis and schedule legality; split, reorder,
   cache read/write, placement, layout transform, blockization, reduction
   decomposition, padding/partitioning, tensorization and software pipelining.
4. Deduplicated task extraction and IR-local call-count weights; trace recording,
   sample decisions, stock mutators, ReplayTrace/EvolutionarySearch, gradient
   task scheduling, XGB training/prediction, standard database serialization,
   workload matching and `MetaScheduleApplyDatabase` replay.
5. General CPU loop scheduling, storage flattening/compaction, dtype legalization,
   LLVM lowering, arithmetic instructions and object emission. The CPU rules here
   explicitly disable unsupported vector/parallel execution for one RV64GC hart.
6. Semantic/structural rejection when a schedule cannot be justified. Runtime
   mathematical equivalence and hardware fidelity are additional external gates.

We provide to TVM:

1. Imported program, constants, static shapes/layouts, explicit state ABI and
   numerical contracts; frontend extensions when standard import cannot express them.
2. `GemminiCapabilities`: 16x16 array, signed INT8/INT32 interface, 20-bit PE
   intermediate bounds, scratchpad/accumulator capacities, full-width readout,
   prohibited hardware loops, header/config digests, one-hart LLVM target and
   fingerprint. This is a pinned supported profile, not automatic RTL discovery.
3. TensorIntrin descriptions **and implementations** for movement, initialization,
   contraction, bias, reduction, copy/fill/add/ReLU and proved scaled readout;
   matching rules, legal shape families and explicit overflow/rounding proofs.
4. Schedule sketches, CPU/Gemmini placement policy, structural alternatives,
   sampled tile factors, SRAM checks and postprocessors. TVM does not derive
   Gemmini command semantics, legal schedules or memory limits from hardware.
5. The custom graph/TIR passes enumerated below, physical-memory lowering and
   primitive external-call ABI. `matmul.c` implements those primitive calls;
   schedules remain visible in TensorIR rather than hidden in a matmul FSM.
6. A feature adapter that exposes mathematical stores before postprocessing, and
   Builder/Runner/callback implementations that bind compilation and measurement.
7. Measured labels, independent oracles, receipt identities, budgets and failure
   policy. TVM supplies no bitstream, FireSim queue, calibration data or quality gate.

Work outside stock TVM (current implementation, not an intrinsic impossibility):

1. Acquire and hash checkpoints, tokenizers/processors, calibration/development/
   held-out data; choose and freeze quantization, source precision and quality gates.
2. Build external-runtime sessions; preserve full prefill/decode KV state,
   observation streams, denoising recurrence, resets and final-output meaning.
3. Generate independent references and negative controls, enforce resource bounds,
   and separate compiler agreement from model accuracy or policy quality.
4. `apps/gemmini/baremetal.py` walks the lowered static graph, plans explicit host
   buffers/constants, emits the C `model_run` ABI and static manifests. It uses
   TVM to compile operators but is **not stock TVM VM/AOT graph code generation**.
5. Cross-compile the primitive C adapter and runtime, assemble immutable constants,
   select libm/libgcc, link startup/stack/loader/session orchestration, and audit the
   entire archive and final ELF. Compiler output alone does not prove loadability.
6. Execute Spike/Verilator/FireSim, validate outputs and guards, verify source and
   deployed hardware identities, queue ownership, elapsed-cycle scope and receipts.
7. Interpret results, keep cap failures and negative evidence, qualify complete
   sessions and report E2E latency/memory/quality. No stock TVM pass performs these.

## All custom compiler passes and their location

| Implementation | Purpose and boundary | Current path |
| --- | --- | --- |
| `gemmini_tensorir.prepare_gemmini_tensorir_workloads` | Sequences stock passes and the supplied semantic preparations | Current, inside TVM |
| `SpecializeGemminiReadoutScales` (`gemmini_epilogue.py`) | Specializes constant readout scales before common CPU/device consideration | Current, Relax pass |
| `prepack_constants` (`gemmini_layout.py`) | Packs constant weight layouts and folds them; no runtime answer substitution | Current, Relax transform |
| `CanonicalizeGemminiBias` (`gemmini_bias.py`) | Moves proved integer bias into reduction initialization; keeps mathematical reference | Current, TIR module pass |
| `NormalizeGemminiReadout` / `normalize_readout_layout` | Normalizes supported output layout/scale chains | Current, TIR module pass |
| `AnnotateGemminiWorkloads` (`gemmini_search.py`) | Binds structural workload identity before extraction/application | Current, metadata pass |
| `ScheduleGemminiTensorIR` / contraction normalization | Selects matching database trace or legal default; reports CPU fallbacks | Current, TIR scheduling pass |
| `gemmini_sketch`, `gemmini_regions` | Semantic contraction recognition; rectangular/interior/tail placement and copy-region partitioning | Current, TVM Schedule API |
| `gemmini_elementwise`, `gemmini_reduction`, bias/readout schedule helpers | Recognize precisely supported formulas and tensorize their blocks | Current, TVM Schedule API |
| `VerifyGemminiMemory` / `gemmini_memory.lower_memory` | SRAM lifetime allocation, bounds/alias checks, command addresses, host-DMA fences | Current, TVM postprocessor/TIR lowering |
| `_inline_cpu_producers` (`gemmini.py`) | Legal CPU producer inlining after placement | Current, TVM Schedule API |
| `tir.transform.rounding.lower_bounded_rne` | Preserve NaN guard and bounded FP32 nearest-even byte quantization via LLVM lrint | Exporter invokes custom TIR rewrite; requires RNE environment |
| `DecomposeGemminiLinear`, `DecomposeGemminiConv2D`, `LowerGemminiMatmul`, `LowerGemminiScheduledMatmul` | Earlier graph matching, im2col/template/extern lowering | Explicit legacy path, not current DSE |
| Merlin `static_buffer_abi.py`, `cpu_row_schedule.py`, `block_tiling.py` | Opt-in compact-buffer ABI specialization, row scratch placement/serial loops, bounded block study | Comparison drivers using TVM APIs; not automatic global passes |
| Merlin `bf16_export.py` | Explicit source/fused export recipes and custom ONNX operators | Before TVM import; opt-in numerical contract |

## Optimization and ownership chart

“Implemented” describes code availability, not full-model FireSim qualification.
Independent formulas, tails, refusal cases and runtime checks bound applicability.

| Optimization/mechanism | Decides applicability | Executes transform | State / important limit |
| --- | --- | --- | --- |
| W8A8 weight/activation quantization | Merlin model recipe + operator policy | PyTorch/reference preparation | Explicit recipe; TVM does not auto-quantize |
| BN folding, residual accumulator recipe | Merlin ResNet recipe | Recipe/Relax construction | Opt-in, unchanged development/held-out gates |
| Shape specialization, constant folding, dead-code removal | TVM + supplied static inputs | Stock Relax | Dynamic/data-dependent cases may refuse static export |
| Relax graph fusion / FuseTIR | Stock TVM patterns and legality | Stock TVM | Not an automatic whole-session fusion optimizer |
| Integer-carrier/QDQ recognition | Fork frontend semantic checks | Torch/ONNX frontend | Proven dtype/bias/range checks; unsupported cases retain semantics |
| General matmul/conv/bmm normalization | Supplied recognizers | TensorIR schedule preparation | Generic semantics/shapes; no model-name dispatch |
| Constant weight prepacking | Supplied layout proof | Custom Relax + stock folding | Per imported graph; packed bytes included in memory accounting |
| Bias preload | Supplied equality/overflow proof | TIR init + intrinsic + adapter | CPU fallback when proof absent |
| Scaled/ReLU INT8 readout fusion | Supplied full-INT32-domain proof | TensorIntrin + adapter | Supported FP32 RNE chains; no arbitrary per-channel approximation |
| CPU/device placement | Supplied union; stock cost/search | Stock trace replay | Same prepared IR; CPU-only unsupported tasks currently untuned |
| M/N/K tiling and reuse factors | Supplied legal structure; stock sampling/mutation | Stock split/reorder/cache/placement | Capacity checked, failed candidates rejected |
| Tile traversal/load order | Supplied sketch family | Stock schedule primitives | Explicit trace, not hidden runtime autotuning |
| Operand hoisting/full-operand reuse | Supplied capacity/placement checks | Stock compute/cache placement | Must fit scratchpad and accumulator including pipeline slots |
| Primitive OS/WS | Supplied sketch/intrinsic family | Tensorization + primitive adapter | WS exists; capability field OS names historical default profile |
| Rectangular 1..16 intrinsic shapes | Supplied semantic families | Stock Tensorize | Handles divisible dimensions and supported tails |
| Interior/tail partitioning | Supplied partition choice | Fixed general TVM loop_partition/blockize | Offset alignment and predicates preserved |
| Pad once versus tile-local staging | Supplied sketches | pad_einsum/placement | Host scratch and reuse fences counted |
| Double buffering/software pipeline | Supplied stage annotations and capacity | Stock InjectSoftwarePipeline + custom allocation | Legal local overlap; not globally asynchronous sessions |
| SRAM lifetime reuse | Custom liveness/alias policy | `gemmini_memory.py` | Per function; no cross-call residency claim |
| Multi-block mvin coalescing | Custom adjacency/width checks | Memory lowering + C adapter | Fixed lowering, not an XGB knob |
| Load/store/execute config shadow | Primitive runtime state | `matmul.c` | Avoids redundant config; not model graph optimization |
| First-K overwrite / zero-fill elision | Lowering knows zero init and first chunk | Adapter commands | Bias preloads preserved |
| Host-DMA hazard waits and completion fence | Custom dependence analysis | TIR command lowering | Required correctness; wait removal needs proof |
| Native copy/pad/fill/add/ReLU regions | Supplied semantic/region recognizers | TVM schedule + primitive intrinsics | Residual predicates refused; no unconditional im2col boundary DMA |
| Native reductions | Supplied reduction recognizers/proofs | TVM schedule + intrinsic family | Unsupported reductions use CPU |
| CPU inlining/unrolling/row compaction | Stock rules + supplied one-hart policy | TensorIR/LLVM; optional Merlin row helper | No RVV or guest thread-pool assumption |
| Bounded nearest-even quantization lowering | Custom guarded-pattern proof | TIR rewrite/LLVM lrint | FP environment must be RNE; preserves NaN behavior |
| Exact source arithmetic recipes | Explicit versioned ONNX domains | `_source_fidelity.py` | Host/source-specific opt-in; not general accelerator speedups |
| BF16 narrowing preservation | General fork correctness fix | TIR dtype legalizer | Logical rounding boundaries retained |
| Strict float division | General fork correctness fix | Arith simplifier/LLVM | Removes unsafe reciprocal rewrite, may cost performance |
| Static host-buffer reuse/constants | Custom exporter lifetime plan | Generated C/static objects | Separate from accelerator SRAM allocation |
| Multi-stage/session memory reuse | Session contract + Merlin planners | External orchestration | Full deployed chain still needs qualification |
| Evolutionary proposal/mutation/task budget | Stock MetaSchedule | Stock EvolutionarySearch/gradient scheduler | Supplied legal search space and measurement budgets |
| XGB ranking | Stock XGBModel | Stock model + mathematical feature adapter | Prove post-training proposals; do not infer from saved booster |
| ELF measurement cache/dedup | Campaign identity policy | External runner/cache | Workload + actual guest bytes and bound toolchain; not a search heuristic |
| Non-regressing default record fallback | Integration policy after search | Standard database commit | Preserve pure-search minima separately |

## Upstream modifications by subsystem

| Area | Full purpose of fork delta | Review/ownership disposition |
| --- | --- | --- |
| ONNX standard frontend | Inherited opset18 reducer axes, scalar/nonscalar binary values, Gather negative/cast handling, ScatterND dtype, Expand/ConstantOfShape; BF16 tensor/constant/cast fidelity, MatMulInteger, CastLike, typed broadcasting/min/max, shape folding, Range, normalization, rounding, newer Gelu and convolution schemas | General frontend fixes; separate upstream candidates with tests, not Gemmini ISA work |
| ONNX extension domains | Fixed-order dot/mean-square and source SiLU/mean/softmax/layernorm/conv/SGEMM/BF16 attention recipes | Moved into `_source_fidelity.py`; explicit domain/version/shape refusal remains in importer. Experimental source contracts, not standard ONNX coverage |
| Torch frontend | Captured dtype/alpha semantics, modular integer arithmetic/matmul/convolution, int_mm and QDQ, frozen linear/bias proofs, norm/creation/views/scatter/pad, pooling/overlap/backward formulas, conv transpose/gradients, parameter/buffer/output handling | General semantics; detailed tests by operator family. No automatic pretrained quality claim |
| Relax/TOPI | Dilated transposed convolution and convolution gradients, edge/reflect padding bounds, empty/no-axis reductions, pooling divisor/ceil-mode and 1-D shape fixes, transform handling | Generic compiler fixes; affects non-Gemmini users too |
| Arithmetic/TIR/TE | Strict FP division; nonrectangular reductions and zero-based loops; correct min/extent construction; index-map integer constants; predicate preservation; BF16/FP8 logical narrowing | Generic semantics. Rebuild native library when these change |
| TIR scheduling | Correct partition cloning/regions/offsets/init/private-buffer identity; copy-on-write shared containers; reduction init-read/update-read regions; unique blockize names and scope metadata; literal bit equality; set_dtype and padding trace rendering | Generic schedule correctness, with regression coverage |
| MetaSchedule core | Preserve rendered legality-error type and reject illegal mutated/resampled traces; unrelated internal failures propagate | Stock algorithm repair, not replacement search |
| Gemmini backend modules | Capabilities, primitive semantics, legacy references, current normalization/scheduling/memory/MetaSchedule integration | Target-owned extension in TVM contrib; upstream inclusion would require separate review |
| Apps/runtime/export/evaluation | Primitive adapter, host-only auto-copy guard, static graph/archive exporter, runners, independent fixtures, reports and repair tools | Integration-owned tooling, not stock TVM runtime guarantees |
| LICENSE / tests / docs | Third-party arithmetic notices, behavioral/refusal tests, command and evidence descriptions | Preserve attribution and historical evidence; no benchmark result manufactured by tests |

The original code also retains legacy `gemmini_tuning.py` and
`gemmini_schedule.py` APIs because existing regression tools reference them.
They are explicitly outside the new tuning path. Removing their consumers and
then deleting them is a future compatibility change, not a prerequisite for
running the current path. The saved dirty `tvm-tensorir-source` work contains an
older `ObservedXGBModel`/custom-record campaign and `case_studies.py`; it is archived
with hashes and not merged into this standard JSONDatabase implementation.

## Known limits to carry forward

No complete actual-model FireSim result, driver binding, loader qualification or
16-GiB capacity probe is supplied by this handoff. The static archive does not
implement a recurrent session. Unsupported arithmetic remains CPU work; neither
all operators nor all CPU fallbacks are tuned. The feature model cannot price
cross-call residency or full-session bottlenecks. Whole-model SmolVLA numerical
closure remains required. Historical timing baselines that capped or failed are
unavailable comparisons, not speedup denominators. These are concrete remaining
qualification/optimization tasks, not reasons to replace independent references,
relax tolerances or enable prohibited hardware loops.
