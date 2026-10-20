# Gemmini through mathematical TensorIR

Development branch: `gemmini/tensorir`, based on `6b376cc8895a83387ec38cc06ef8f7b09827b071`.
The original checkout and branch remain unchanged. The old template backend is
available as `prepare_gemmini_graph(..., route="legacy")` and
`prepare_gemmini_graph_legacy`; its tests explicitly select that route.

## Required end state

This migration is complete only after the framework integration, requested
operator inventory, regression checks and bounded Spike/Verilator checks are
verified. A passing small matmul does not complete the migration. The user also
requires exact explanations for requested operations that cannot map to the
selected hardware and numerical contract.

## Implemented architecture

1. Ordinary Relax `LegalizeOps`, pattern annotation, constant folding and
   supported `FuseOps`/`FuseTIR` retain mathematical computations.
2. `ScheduleGemminiTensorIR` recognizes contractions by their buffer accesses,
   arithmetic and reduction axes. Names do not select implementations. Unmatched
   functions retain CPU lowering; schedule refusals retain a reason.
3. TensorIR `split`, `reorder`, caches, producer placement, standard padding and
   reduction decomposition construct the execution schedule. The full trace
   replays from the untiled workload with all scheduling debug checks enabled.
4. `TensorIntrin` descriptions/implementations model scoped DMA, accumulator
   initialization, update and readout. Intrinsics are authored with typed
   TVMScript; no generated source strings are parsed.
5. Standard allocation location planning and compaction precede physical SRAM
   row allocation. CPU accesses to SRAM, unaligned rows and resource excesses
   are rejected. CPU-private DMA staging buffers require completion before reuse.
6. Static export uses ordinary LLVM RV64 code generation. TVM USMP assigns an
   explicit caller-owned pool when private allocations need workspace; the pool
   belongs in preflight accounting. Guest allocation callbacks remain forbidden.

The hardware target is LLVM/RV64 with a `gemmini` dispatch key and a capability
fingerprint in its model. Rocket executes the generated code and issues RoCC
commands, so this route does not need a new C++ `TargetKind` or LLVM codegen.
Intrinsics and target scheduling rules are registered independently of graph
matching. Registration describes an explicitly selected device; it does not
discover a connected accelerator or make arbitrary operators tensorizable.

Importing `tvm.relax.backend.contrib.gemmini` registers two standard Relax
pipelines. `relax.get_pipeline("gemmini", **options)(mod)` prepares mathematical
graphs for static export, retaining `call_tir`. For ordinary VM compilation,
`relax.build(mod, target, pipeline="gemmini_build")` composes that preparation
with TVM's stock `default_build_pipeline`. Options can be passed through
`relax.get_pipeline("gemmini_build", **options)`. The implementation does not
replace VM lowering or change the default pipeline for other targets.

This remains a TVM v0.19 fork rather than an unmodified upstream baseline. The
base commit already contains ONNX fidelity changes, FP/BF16 storage and strict
arithmetic fixes, source-preserving CPU recipes and the host-only `LowerAutoCopy`
guard described in README.md. The migration adds no C++ core patch. The guard
rejects unsupported automatic-copy annotations and is not a DMA optimization.
The handwritten static deployment packaging is retained, with standard USMP
used for operator pools; this is not a claim of stock microTVM/AOT deployment.

`gemmini_space_generator()` is a standard MetaSchedule `PostOrderApply` space
generator with a target-specific `PyScheduleRule`, standard `RewriteTensorize`
and a memory legality postprocessor. Its 72 choices combine matrix traversal,
A/B issue order, PE-safe K chunks of 8/16 and row/column tile groups of 1/2/4.
Blocked B and accumulator layouts let one DMA load serve multiple tile products.
Independent command-count checks confirm reuse while preserving full and tail
results. This does not implement DMA software pipelining or double buffering.

`ScheduleGemminiTensorIR` uses a matching current MetaSchedule database record
when present, otherwise a deterministic untuned schedule. The v0.19 default
database query ignores its target argument, so the backend explicitly compares
the returned record's target/profile before replay. The standard database test
uses a diagnostic invented cost solely to prove selection/replay; it is never a
target timing label. A target performance builder/runner and measured tuning
campaign remain future work.

Noncanonical exact contractions use standard `transform_block_layout`,
`reindex`, `transform_layout`, producer inlining and `compute_at` to form matrix
regions from their access maps. Convolution therefore keeps ordinary Relax/TIR
semantics until scheduling. CPU window gathering remains necessary for layouts
that primitive rectangular DMA cannot express, but the placed staging regions
replace the old full im2col graph tensor. Quantization and pooling semantics
are not changed by this layout schedule.

## Hardware evidence and numerical boundaries

Facts were inspected in the selected local Chipyard/Gemmini source and matching
generated parameter header. `GemminiCapabilities` binds the target model to its
profile digest; it is an explicit compiler profile, not hardware discovery.

| Source | SHA256 | Relevant facts |
| --- | --- | --- |
| `gemmini/Configs.scala` | `178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b` | INT8 inputs, INT32 accumulator, 20-bit PE output, 16x16 array, both dataflows, 256-KiB scratchpad, 64-KiB accumulator; no accumulator-load scaler |
| `gemmini/GemminiConfigs.scala` | `7294a082a1df4ae7aa7d81adc5bde75b7d5118683c083f6a4b998c9ce64d102e` | Configuration fields and optional units |
| selected `gemmini/LoadController.scala` | `780ee0a292d8653fa8a822aee28e2fb2cffbc25301431b4332182008d34fcf6a` | Null DMA address produces zeros; stride zero broadcasts a row |
| `include/gemmini_params.h` | `3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924` | Parameter/ABI geometry and float scaling/narrowing definitions |

The implemented instruction subset is primitive-only OS; hardware loops remain
forbidden. The selected existing RTL source has a previously disabled debug
assertion in LoadController; this migration does not edit RTL. Simulator and
software source digests must accompany each execution result.

Scaling an INT8 load rounds and saturates back to INT8. INT32 accumulator loads
have no configured scaler. Narrow readout rounds/scales/saturates to INT8; full
INT32 readout is a different path. These features are not arbitrary FP32 or
INT32 multiply/divide operations. In particular, negating INT8 -128 during a
scaled INT8 load saturates, whereas widening then negating produces INT32 128.
Any lowering must preserve the admitted operator's actual dtype and rounding.

## Full requested operator inventory — completion audit pending

The following 51 operator spellings retain the user's full requested scope.
Statuses here are development state, not a completed hardware-support table.

| Family | Requested operations | Current migration state |
| --- | --- | --- |
| Matrix | `mm`, `bmm` | Structural scheduling, queued-DMA checks and bounded Spike/RTL checks implemented; full operator inventory remains pending |
| Biased matrix | `addmm` | Standard fused INT32 bias/ReLU CPU epilogue tested with accelerator matmul; bias preload not yet implemented |
| Convolution | `convolution`, `convolution_backward` | Exact INT8/INT32 sliding-window contraction scheduling implemented; generic FP training gradients do not fit this numeric unit. Frontend/decomposition and grouped/transpose admission audit remains pending |
| Arithmetic | `add.Tensor`, `sub.Tensor`, `add.Scalar`, `sub.Scalar`, `mul.Scalar`, `div.Scalar`, `neg`, `relu` | INT32 tensor addition uses accumulator overwrite/add loads and full readout. Ordinary signed TIR has a no-overflow domain; explicit widen/add/narrow TIR also maps natively and defines modulo-2^32 output. Other mappings remain under audit; generic CPU fallback retained |
| Reduction | `mean.dim`, `mean.default`, `sum.dim_IntList` | Bounded signed INT8-to-INT32 sums use the array. The original CPU conversion/division can follow that sum. Default INT64 and arbitrary floating reductions remain CPU work; full frontend and rounding audit still required |
| Pooling | `avg_pool1d`, `avg_pool2d`, `avg_pool3d`, `adaptive_avg_pool1d`, `_adaptive_avg_pool2d`, `_adaptive_avg_pool3d` | Window/divisor/padding semantics and native admission still required |
| Overlap accumulation | `avg_pool2d_backward`, `_adaptive_avg_pool2d_backward`, `col2im` | Gather/scatter accumulation and numeric semantics still require investigation |
| Movement | `clone`, `repeat`, `cat`, `slice_scatter`, `select_scatter`, `fill.Scalar`, `full`, `full_like`, `constant_pad_nd`, `reflection_pad1d`, `reflection_pad2d`, `reflection_pad3d`, `replication_pad2d`, `replication_pad3d`, `_to_copy` | INT8/INT32 copies, indexed repeats/slices, signed widening and runtime scalar fills use native DMA where Tensorize proves the access map. Multiple-source selection, immediate-value admission and overlapping updates remain under audit |
| Views/metadata | `as_strided`, `diagonal`, `permute`, `scalar_tensor`, `select.int`, `slice.Tensor`, `split_with_sizes`, `squeeze.dim`, `squeeze.dims`, `unsqueeze`, `view` | Audit view versus materialization and static-export ABI; do not count host metadata as accelerator arithmetic |

### Corrections to the proposed native-unit mappings

An operator name alone is not a numerical contract. The native array here has
signed INT8 operands and INT32 accumulation; ordinary FP32 `mm`, training
convolutions and gradients cannot be silently quantized. The native statuses
above describe admitted mathematical TIR after an explicit integer recipe, not
complete acceptance of every ATen overload or frontend graph.

| Requested mapping | Evidence-backed boundary |
| --- | --- |
| `add.Tensor` / `sub.Tensor` through load scaling | Full-width accumulator loads have no configured scaler. Unscaled INT32 addition is supported and the RTL SRAM adder wraps to 32 bits. General alpha scaling or subtraction does not follow from that. INT8 scaled loads round/saturate; `-(-128)` demonstrates why they cannot replace arbitrary wrapping negation or widened subtraction. |
| `add.Scalar`, `sub.Scalar` | A zero load stride repeats a scalar down a one-column DMA rectangle. Runtime INT32 scalar-tensor addition is now supported on either side, including an explicit wrapping variant. Immediate constants and native subtraction remain separate admission work; there is no full-width negative load scaler. |
| `mul.Scalar`, `div.Scalar`, `neg` | The available scaled small readout narrows to INT8. It is not an arbitrary FP32 or full-INT32 multiply/divide unit. Any narrower integer recipe must specify its rounding, saturation and value range before admission. |
| `relu` | Full-width INT32 readout bypasses the activation/scaling result. CPU INT32/FP ReLU therefore remains explicit. An INT8 input/output ReLU mapping is feasible but still needs a dedicated native rule. |
| `mean.dim`, `mean.default`, `sum.dim_IntList` | A registered sum intrinsic uses a ones-vector contraction for signed INT8-to-INT32 sums with K <= 16,777,216. This is not the default INT64 or floating reduction contract. Mean retains its specified CPU conversion/division; a narrowing readout scale cannot stand in for arbitrary floating mean. |
| `avg_pool1d/2d/3d`, adaptive average pooling | Ordinary floating inputs do not fit the integer arithmetic unit. An integer recipe must specify windows, divisors, exclude-padding/ceil behavior and rounding. The max-pool readout feature does not implement average pooling. |
| Average-pool backward and `col2im` | These require the actual gradient dtype, scatter/overlap semantics and window divisors. Accumulate-on-load updates local accumulator rows; it does not scatter-add into arbitrary host output addresses. No generic backward mapping is claimed. |
| Generic data movement | Primitive DMA handles rectangular regions with contiguous columns and one row stride. Copies preserving bits and signed widening are admitted. Index remapping, multiple sources, fill broadcasting and overlapping updates need appropriate schedules or remain CPU work. They must not be described as absent silicon features merely because the current schedule does not handle them. |
| Views and metadata | Reshape-like operations may disappear or become host metadata when consumers can accept the layout. Permute, diagonal, slicing and `as_strided` are not automatically free under the static exporter's compact-buffer ABI. Materialization and alias semantics must be audited; host metadata is not accelerator arithmetic. |

The existing v0.19 Torch exported-program translator also has an incomplete
direct ATen conversion table. Prior frontend decompositions may remove an
unsupported spelling, but a missing conversion is not a Gemmini CPU-fallback
success. This backend accepts ordinary Relax/mathematical TIR; frontend gaps
must be reported independently of native-unit legality.

`gemmini_elementwise.py` registers typed copy, widening and INT32 add intrinsics
and uses standard `split`, `reorder`, `cache_write`, placement and `tensorize`.
Prime extents can require small divisor tiles; they do not imply padded DMA or
an unmeasured speedup. Extent-one descriptor axes are canonicalized with typed
TIR construction to match v0.19's blockize behavior. Standard MetaSchedule
postprocessing and feature extraction also cover this DMA rule.

The LLVM code generator's `DEFINE_CODEGEN_BINARY_OP` macro marks ordinary
signed >=32-bit arithmetic `nsw` and unsigned >=32-bit arithmetic `nuw`.
Consequently a hardware overflow result alone cannot prove full ATen wrapping
semantics for arbitrary ordinary TVM integer expressions. The wrapping-add
intrinsic has a separate reference: widen both INT32 inputs to INT64, add within
its proven range, then narrow to INT32. Target overflow/underflow fixtures use
that explicit mathematical reference. The plain-add intrinsic preserves the
defined no-overflow domain; it does not silently change the global TVM policy.

Indexed INT8/INT32 copies are also matched by the framework: rectangular tiles
are tried first, then single-element DMA when Tensorize cannot prove a larger
access map. This covers materialized slices, diagonals, permutations and repeats
without CPU value copying. Such schedules can issue many commands and are not
performance winners without measurements. Both options are visible to standard
MetaSchedule, whose tensorization postprocessor rejects illegal candidates.
In-place copies remain refused. DMA completion also protects external output
buffers when CPU code in the same function subsequently accesses them.

Runtime scalar fills support INT8-to-INT8, INT32-to-INT32 and signed
INT8-to-INT32 output. They use one-column, zero-stride DMA and preserve the
ordinary output layout through standard cache placement. Constant-only fills
may disappear through normal FoldConstant. Non-integer output conversion is
not implied. The same broadcast layout supports runtime INT32 scalar addition;
the ordinary no-overflow and explicit widen/add/narrow variants remain distinct.

Scalar descriptors omit stride metadata because v0.19's Python match-buffer
builder indexes `shape[0]` when a stride list is supplied. There is also a
target-local compatibility bridge before LowerMatchBuffer: Buffer.ElemOffset
returns an INT32 zero for scalar regions while Tensorize may promote the match
offset to INT64. The bridge binds an INT32 zero-offset variable and casts its
pointer use back, preserving address, dtype and external ABI. It accepts only
zero-offset scalar sources and access-pointer uses; it is not a core C++ patch
or an arithmetic rewrite. Non-buffer primitive scalar ABIs retain CPU fallback.

`gemmini_reduction.py` leaves zero-initialized sums as mathematics and matches
direct signed INT8 casts or a fused pointwise widening producer. Every prefix
is bounded by `[-128*K,127*K]`; K <= 2^24 therefore fits INT32 exactly. Array
chunks contain at most 16 values. Standard block/index maps, reindex caches,
placement, splitting and reduction decomposition construct the schedule. A
sum intrinsic broadcasts a CPU-created scalar one into a scoped scratchpad
vector, then calls the same array primitive as matmul. There is no C reduction
kernel, hidden CPU reduction or hardware-loop command. Owned implementation
temporaries live in a nested TIR block, which standard Tensorize preserves.

Scalar, multiple-axis and keep-dimension sums restore the original output
layout with ordinary CPU copies. Input reindexing currently stages each DMA
tile on the CPU; even contiguous input can incur that copy. Divisor tile choices
avoid padded DMA but prime extents can require scalar chunks. MetaSchedule
exposes both divisor and scalar candidates. These are functional capabilities,
not measured performance wins. The original CPU cast/divide can consume a
native INT32 sum without changing its specified floating division. General
floating inputs and INT64 accumulation still retain CPU fallback.
Converting a proven bounded INT32 sum to INT64 output is feasible; direct
admission of the equivalent INT64 sum remains an implementation gap. Likewise,
known INT8 values widened to floating point may admit additional range proofs.
Neither case is evidence that arbitrary INT64 or floating reductions fit this
integer array.

The shared finalizer retains a TensorIR root block around admission and device
ownership. This lets stock Relax reshape analysis inspect prepared single-input
functions during normal `relax.build`, instead of requiring a special pipeline
to avoid that analysis. No core compiler modification is needed.

## Verification tools

`tests/python/relax/test_backend_contrib_gemmini_tensorir.py` executes independent
queued-DMA simulation, INT64 oracles, multiple shapes, tails and traversal orders,
trace replay, altered-primitive controls, fused/batched Relax execution and CPU
fallback. The queue deliberately delays DMA until fences to expose premature
CPU reuse of staging storage.

`verify_tensorir.py prepare` compiles fresh full/small/tail/batched fixtures and
positive/wrong-oracle ELFs. `run` executes a selected ELF on explicit Spike or
Verilator artifacts with a wall deadline, cycle limit for RTL, one CPU and a
2-GiB virtual-memory cap. Final ELF instruction audits reject hardware loops.
The original workload and numerical reference remain identical between the
positive and wrong-oracle builds; only the verifier define changes.

Bounded target evidence now includes small/full/tail/batched matmul with fused
CPU bias/ReLU epilogues (four cases) and a separate new convolution/reuse pair.
The convolution/reuse pair passes Spike and the selected Verilator artifact;
the reuse case exercises a two-column cache group with K=8, including B tiles
beginning at non-16-multiple row addresses. RTL row addresses are element-row
addresses, not mandatory 16-row tile boundaries. The four-case target check was
also refreshed after the reuse/layout changes, with positive and wrong-oracle
checks on both simulators. A final source freeze must refresh the bound evidence
after any further code changes. Functional runs do not supply timing labels or
a measured winner.

Development evidence under the adjacent `tensorir-dev/` directory is immutable:
`device-conv-reuse-v2` and `device-core-v6` contain source/object/ELF bindings,
instruction audits, simulator receipts and logs. Host logs include
`unit-structural-v3.log` (10 checks, including 72 postprocessed designs),
`unit-conv-nd-v1.log` (independent 1D/2D/3D convolution checks),
`unit-legacy-full-v1.log` (32 legacy checks), `unit-export-v2.log` (18 static
export checks) and `unit-classifier-v2.log` (two result-decoding checks).
Subsequent cleanup checks are `unit-migration-v4.log` (30 checks),
`unit-scalar-bool-v1.log` (scalar-input/boolean-output epilogue) and
`unit-export-v3.log` (including byte-sized boolean operator workspace).

Native DMA evidence is recorded in `device-native-v4`: INT32 wrapping addition
uses explicit INT64-add/INT32-narrow reference mathematics, INT8/INT32 copies
preserve bits, widening covers all 256 signed-INT8 values, and materialized
permutation/repeat fixtures exercise scalar-DMA fallback. Fixtures verify actual
DMA/accumulate calls, every output, unchanged inputs and surrounding guards.
`unit-elementwise-v6.log` covers mathematical CPU/device equivalence, standard
MetaSchedule postprocessing, indexed layouts, tails and delayed DMA. The
CPU-consumer test deliberately removes a required output wait and fails its
oracle. `unit-structural-native-v2.log` retains 11 contraction regressions and
`unit-export-native-v1.log` retains 21 exporter/classifier checks.

`device-scalar-v3` adds bounded runtime-scalar fixtures: INT8 and INT32 fills,
INT8-to-INT32 fill widening and explicit wrapping INT32 scalar addition.
Positive and wrong-oracle checks pass on both Spike and the selected RTL.
`unit-scalar-fill-v3.log` covers eight host checks including both scalar operand
positions, vectors/batches/tails, standard MetaSchedule processing, the named
Relax preparation/build pipelines and preservation of CPU fallback for a hidden
primitive scalar ABI. `unit-scalar-structural-v1.log` retains 11 contraction
regressions after scalar support.

`device-sum-v4` adds five array-reduction fixtures: a prime reduction extent,
two full 16-value chunks, scalar output, multiple reduction axes and retained
dimensions. Positive and wrong-oracle checks pass on Spike and the selected
RTL; instruction audits reject hardware-loop commands. `adapter-sum-v1`
rebuilds the source-pinned receipt after documenting the sum-specific bound in
the header; its adapter object is byte-identical to `adapter-native-v1`.
`unit-reduction-v9.log` covers four host checks with independent INT64 oracles,
extremes, trace replay, standard MetaSchedule postprocessing/features on a fused
widen/sum, non-integral CPU FP32 mean readout and numeric/index-width refusal.
`unit-default-vm-v1.log` covers prepared single-input permutation and widening
through the default VM pipeline. Contraction, DMA, exporter and classifier
regressions are in `unit-reduction-structural-v1.log`,
`unit-reduction-elementwise-v1.log`, `unit-reduction-export-v1.log` and
`unit-reduction-classifier-v1.log`. These are functional checks, not timings.

The migration remains incomplete until the inventory's remaining native
admission, CPU/frontend explanations and final regression/target checks are
resolved. Do not infer those from passing contraction fixtures.
