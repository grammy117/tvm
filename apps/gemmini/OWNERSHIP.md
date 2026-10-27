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
  -> complete model session, independent outputs and FireSim evidence [Merlin/operator]
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
   deployed hardware identities, manager ownership, elapsed-cycle scope and receipts.
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
| Absolute constant placement | Exporter declares optional wrapper address macro; Merlin plans weights/references/arena and checks final link | Original operators/constants, rebuilt pointer wrapper, existing Merlin loader layout | Small high-address Spike controls pass; actual full-model sizes and FireSim remain unqualified; not a tuning optimization |
| Multi-stage/session memory reuse | Session contract + Merlin planners | External orchestration | Full deployed chain still needs qualification |
| Evolutionary proposal/mutation/task budget | Stock MetaSchedule | Stock EvolutionarySearch/gradient scheduler | Supplied legal search space and measurement budgets |
| XGB ranking | Stock XGBModel | Stock model + mathematical feature adapter | Prove post-training proposals; do not infer from saved booster |
| ELF measurement cache/dedup | Campaign identity policy | External runner/cache | Workload + actual guest bytes and bound toolchain; not a search heuristic |
| Non-regressing default record fallback | Integration policy after search | Standard database commit | Preserve pure-search minima separately |

## Complete current device-operation inventory

`Prev` means available before the device-operation extension, at
`3423bd6818ab752ddd9cdb5d5773743e48c65f1d`; `New` means added in
`gemmini/device-ops`. These are mathematical pattern/dtype capabilities, not
unconditional ATen-overload coverage or measured performance claims. The
[reviewed baseline](https://github.com/ucb-bar/merlin/blob/608f3939b9ccadf552c0be6737859fdc64b7d1d6/examples/gemmini/target/software-spec.yaml)
defines the source/numerical admission boundary. No model-name dispatch, CPU
loop/packing search knobs, new runtime, or TVM C++ change is introduced here.

| Device operation | Status | Handling and limits |
| --- | --- | --- |
| INT8 x INT8 -> INT32 contraction: `mm`, `bmm`, eligible convolutions/linear/attention matmuls | Prev | Existing TensorIR contraction sketches and primitive OS/WS; static layouts/tails and PE/accumulator bounds remain mandatory. Conv indexing/packing may be host work. |
| Contraction with INT32 bias: eligible `addmm`/linear/conv bias | Prev | Proved bias normalization, full-width seed load once, existing reduction tensorization. Arbitrary FP32 bias is not this operation. |
| Accumulator zero/init; INT8 A/B operand loads; multi-block loads | Prev | Existing primitive zero/mvin commands and SRAM lifetime allocation. |
| Full-width INT32 accumulator overwrite/add | Prev | Unscaled full-width DMA; addition uses the existing modulo-32 accumulator semantics. |
| Signed INT8 -> INT32 widening | Prev | Shrunk/unit-scaled accumulator load. |
| INT32 tensor/scalar broadcast addition | Prev | Full-width overwrite followed by accumulator-add DMA. |
| INT8 ReLU | Prev | Widen plus native identity-scale ReLU readout; fused producer readout also supported. |
| INT8/INT32/FP32 copy, repeat/broadcast, indexed copy, constant/scalar fill | Prev | Existing DMA intrinsics and region partitioning. FP32 copies preserve bits and do no floating arithmetic. |
| Static padding, concatenation, slice/select/scatter and materialized permutation | Prev machinery; New dtype reach | Existing static copy-region rules, with native byte-copy implementations for the additional storage types below. Region predicates/aliasing must be proved. Residual producer work can remain CPU. |
| Opaque bool/INT16/FP16/BF16/INT64/FP64 copies, broadcasts and materialized permutations | New | Unscaled scratchpad byte DMA, explicit byte widths/strides/pointers. No conversion or arithmetic. Physical tiles stay within 16 rows and 16 bytes per row. |
| Bounded affine `a*x + b*y + c`, including subtract, negation and scalar coefficients | New | One/two signed-INT8 sources widened to INT32; signed byte coefficients; constant INT32 seed; full-domain prefix bounds. Unit terms use accumulator loads; other coefficients execute on the WS mesh. Raw INT32 or proved RNE/saturated INT8 output. Wrapping INT8 arithmetic, general INT32 coefficient arithmetic and FP32 affine operations are not silently reinterpreted. |
| Single-rounding byte residual sum | New | Unit input loads, exact INT32 addition, one proved readout rounding/saturation, optional ReLU. |
| Per-operand-rounding byte residual sum | New | Each positive input scale in `(0,1]` scales/RNE/saturates its byte before INT32 addition, then a proved narrow readout. Two terms, no bias, no raw accumulator exit for this family. Independent exhaustive 65,536-pair fixtures exercise two scale pairs. |
| Scalar scaled/RNE/saturated/ReLU INT8 readout fused with contraction | Prev | Existing full-INT32-domain output-threshold proof; no tolerance widening or scale averaging. |
| Standalone INT32 -> scaled/RNE/saturated/ReLU INT8 readout | New integration | Same proof and existing readout command, now recognized by normal placement, including INT32 extremes. |
| Signed-INT8 sums, including static window sums | Prev | Existing reduction-against-ones kernel; bounded INT32 accumulation, selected exact INT64 exits. Arbitrary FP32/wide-integer sums remain CPU. |
| Quantized window average / mean epilogue | New | Existing integer sum kernel plus proved scaled narrow readout, including non-power-of-two reciprocal factors already present in the prepared program. Layout reconstruction remains visible. This does not convert arbitrary FP32 average pooling to a different quantized program. |
| Values-only INT8 max, complete 2x2/3x3 windows | New | Integer maximum executes in pooled accumulator readout. Index gathering and literal padding remain explicit host TensorIR producers. Every physical window is complete, so the hardware's implicit zero padding is unused; negative-valued source padding is preserved correctly. No indices or FP32 maximum. |
| Legal reshape/view/alias/squeeze/unsqueeze/split | Prev | Metadata/view when possible; materialized results use the copy paths above. A metadata-only operation is not advertised as native arithmetic. |

Copy offload keeps its existing explicit opt-in or measured-record policy;
it is not globally enabled by this extension. New arithmetic/window rules use
the same default placement and stock MetaSchedule union as existing operators.
Pool Tensorize is recorded before stock reduction decomposition because its
intrinsic includes initialization; all schedule operations remain replayable.
This adds coverage, not a measured winner or an automatic expansion of the
contraction-only remote campaign's task selection.

Frozen scales shared by groups of 16 output channels retain their exact values
and short-tail handling. **General fusion of a nonuniform grouped-scale vector
is still partial**: raw INT32 output followed by the original CPU FP32 rescaling
remains supported and is checked by `group16_preserved`. This extension never
replaces a vector by one tensor scale. General fused conv-plus-pool residency,
arbitrary/adaptive floating averages, wider-integer limb constructions,
comparisons, arg reductions, sort, nonlinear floating arithmetic and complex
storage are outside this minimal extension. RMSNorm/RoPE/softmax/SiLU remain CPU
work under the selected policy.

All primitive adapter entrypoints are accounted for below; the new C entrypoints
only issue configuration/transfer commands and contain no CPU output arithmetic.

| Primitive ABI entrypoint(s) | Status | Role |
| --- | --- | --- |
| `tvm_gemmini_begin`, `tvm_gemmini_end` | Prev | Exclusive command session, configuration reset, completion/coherence fences. |
| `tvm_gemmini_load_a`, `tvm_gemmini_load_b` | Prev | Primitive scratchpad loads using current load configuration. |
| `tvm_gemmini_dma_a`, `tvm_gemmini_dma_b`, `tvm_gemmini_dma_a_blocks`, `tvm_gemmini_dma_b_blocks` | Prev | Configured scratchpad DMA, including byte-copy payloads and legal coalesced blocks. |
| `tvm_gemmini_compute`, `tvm_gemmini_compute_ws` | Prev | Primitive OS/WS mesh contractions; weighted affine uses the same compute ABI. |
| `tvm_gemmini_zero_acc` | Prev | Hardware zero load. |
| `tvm_gemmini_store`, `tvm_gemmini_dma_out` | Prev | Full-width INT32 result readout. |
| `tvm_gemmini_dma_acc_write`, `tvm_gemmini_dma_acc_add` | Prev | Full-width accumulator overwrite/add without a scaler. |
| `tvm_gemmini_dma_widen` | Prev | Unit-scale signed-byte accumulator load. |
| `tvm_gemmini_dma_spad_out` | Prev | Uninterpreted scratchpad bytes to host. |
| `tvm_gemmini_dma_relu_out`, `tvm_gemmini_dma_scaled_out` | Prev | Native narrow readout with the proved activation/scale/rounding contract. |
| `tvm_gemmini_dma_scaled_in` | New | Byte load scale plus overwrite/add. Load-configuration caching now includes scale bits, so subsequent unit loads reset correctly. |
| `tvm_gemmini_dma_pool_out` | New | One complete values-only 2x2/3x3 signed-byte max window. Invalidates cached store configuration so subsequent ordinary stores clear pooling. |
| `tvm_gemmini_matmul_i8_i32` | Prev, legacy API | Retained whole-operation interface; current TensorIR schedules issue primitive calls instead. |
| `tvm_gemmini_validate_matmul_i8_i32`, `tvm_gemmini_validate_tensor_buffers`, `tvm_gemmini_validate_tensor_output` | Prev, CPU admission | Range/alias/ABI checks, not accelerator operations. |

`verify_device_ops.py` builds guarded original-TVM-operator guests with independent
NumPy references, primitive-use checks and a wrong-oracle ELF. Execute through
`verify_tensorir.py run --backend spike`; Spike establishes functional semantics,
not FireSim cycles or a full-model quality result. Existing local/remote model
artifacts remain bound to their original source and numerical identities.
The new affine tiles use one column and max pooling uses one output window per
command group. These conservative implementations establish coverage; their
throughput and full-model benefit have not been measured.

With this TVM source/Python/native library selected and the normal adapter and
functional simulator receipts built, reproduce the qualification with:

```sh
"$PY" apps/gemmini/verify_device_ops.py --output-dir "$OUT/device-ops" \
  --adapter-receipt "$ADAPTER_RECEIPT" --riscv-gcc "$GCC"
"$PY" apps/gemmini/verify_tensorir.py run --bundle "$OUT/device-ops" \
  --backend spike --simulator-receipt "$SPIKE_RECEIPT" --name positive
"$PY" apps/gemmini/verify_tensorir.py run --bundle "$OUT/device-ops" \
  --backend spike --simulator-receipt "$SPIKE_RECEIPT" --name negative --negative
```

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
