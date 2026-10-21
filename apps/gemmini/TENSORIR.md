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
   Standard `RewriteDataflowReshape` and `DeadCodeElimination` then remove
   eligible internal reshape copies and unused functions before device matching.
   `CanonicalizeGemminiBias` can then expose a proven bias-initialized integer
   reduction, separately from scheduling, retaining both mathematical forms.
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
guard described in README.md. That guard rejects unsupported automatic-copy
annotations and is not a DMA optimization.

The migration fixes nine generic C++ scheduler/IR issues.
In `decompose_reduction`, initialization reads were omitted and retained
on the update block instead. Separate read-region analysis preserves an
input-dependent initialization and lets subsequent cache placement see its
producer. The regression tests cover read dependencies, cache placement, trace
replay and CPU execution, independently of Gemmini.
In schedule-state replacement, shared statement containers such as `SeqStmt`
must trigger copy-on-write even when their child loop appears uniquely owned.
Otherwise splitting an inner loop can mutate the original workload and break
trace replay. General split/reorder tests retain and check the original input.
The handwritten static deployment packaging is retained, with standard USMP
used for operator pools; this is not a claim of stock microTVM/AOT deployment.

Constant-padding scheduling exposed three further general issues, each with
target-independent regressions. `decompose_padding`'s Python trace printer must
accept the loop's printed name rather than a LoopRV object. Repeated blockization
must give generated outer blocks unique names so ordinary `GetBlock` and
MetaSchedule tensorization can address them. Tensorization must compare floating
literal bits: numerical equality rejects an identical NaN and accepts opposite
zero signs. The matcher now preserves NaN payloads and signed zero. These are
explicit fork patches, not changes to the accelerator's arithmetic contract.

Copy-region scheduling exposed two more generic defects. `loop_partition` now
uses zero-based physical loops while preserving logical block domains and
access maps. The old implementation narrowed domains and rewrote regions by
display names, which could change a reduction's initialization or make a later
blockization simplify a copy selector incorrectly. Partitioning now renews
private definitions, derives wrapper dependencies and also handles a nonzero
input loop origin or a single partition. `RenewDefs` preserves a domain's
minimum/extent and the identities of externally defined buffers, including
metadata-only references. General regressions check CPU execution, private
allocations, dependencies and trace replay independently of the target.

Exact INT64-result sum scheduling adds two composability fixes. `unsafe_set_dtype`
uses the usual folding cast constructor for buffer stores, so an INT64 zero
becomes an INT32 zero rather than a redundant cast that Tensorize cannot match
after simplifying its descriptor. Tensorize refreshes the complete enclosing
scope's cached block information and preserves its existing affine binding;
starting at the scope's body can misclassify a tensorized block as a root.
Target-independent constant-initialization tests exercise both the INT32 and
INT64 producer cases, CPU execution and full trace replay. Python IndexMap
construction also converts integer literals to the requested index dtype;
otherwise a printed map containing a constant zero cannot replay in Python.

`gemmini_space_generator()` is a standard MetaSchedule `PostOrderApply` space
generator with a target-specific `PyScheduleRule`, standard `RewriteTensorize`
and a memory legality postprocessor. Up to 72 choices combine matrix traversal,
A/B issue order, PE-safe K chunks of 8/16 and row/column tile groups of 1/2/4.
Blocked B and accumulator layouts let one DMA load serve multiple tile products.
Independent command-count checks confirm reuse while preserving full and tail
results. This does not implement DMA software pipelining or double buffering.
Alignment is attached to the explicit blockized region consumed by Tensorize.
This matters for short grouped tiles: aligning only the original inner block
can lose the physical row pitch when that block is replaced. Memory lowering
also checks the sixteen-element row pitch (unless the row axis has extent one)
and contiguous columns. The sum intrinsic's private ones matrix explicitly
includes all sixteen physical columns, even though only column zero is used.

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

Bias preload applies to a bounded, private, unshared INT8/INT32 product followed
by a bijective pointwise addition of a read-only INT32 tensor. The post-fusion
mathematical pass exposes bias initialization and explicit modulo-32 reduction
updates. This preserves the original defined results even when moving bias
earlier makes an intermediate prefix wrap; it also preserves explicitly wrapping
source additions. It does not make ordinary TVM integer arithmetic globally
wrapping. Reports retain the original/normalized IR and proof conditions; a
scheduling refusal restores the original CPU function.

Standard schedules inline vector/matrix bias or place a necessary scalar/column
broadcast gather near initialization, then tensorize full-width accumulator DMA.
The final product copy can inline into device writeback. CPU INT32/FP ReLU and
quantization remain separate. The wrapped reduction currently uses divisor
tiles because v0.19 `pad_einsum` does not recognize that expression; prime
extents may therefore require many tiny commands. Cache groups that do not
divide those extents are rejected. Bias fusion is currently a fixed policy,
not a measured profitability decision, and no speedup is claimed.

## Hardware evidence and numerical boundaries

Facts were inspected in the selected local Chipyard/Gemmini source and matching
generated parameter header. `GemminiCapabilities` binds the target model to its
profile digest; it is an explicit compiler profile, not hardware discovery.

| Source | SHA256 | Relevant facts |
| --- | --- | --- |
| `gemmini/Configs.scala` | `178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b` | INT8 inputs, INT32 accumulator, 20-bit PE output, 16x16 array, both dataflows, 256-KiB scratchpad, 64-KiB accumulator; no accumulator-load scaler |
| `gemmini/GemminiConfigs.scala` | `7294a082a1df4ae7aa7d81adc5bde75b7d5118683c083f6a4b998c9ce64d102e` | Configuration fields and optional units |
| selected `gemmini/LoadController.scala` | `780ee0a292d8653fa8a822aee28e2fb2cffbc25301431b4332182008d34fcf6a` | Null DMA address produces zeros; stride zero broadcasts a row |
| selected `gemmini/AccumulatorScale.scala` | `bb7c65bdf112e5ab93da52278e5c30152fb2f84a8ed990d23008439cf41281d4` | Narrow output applies activation/scale/clipping; full-width data follows a separate unchanged path |
| selected `gemmini/StoreController.scala` | `a0b556d109196c3c9d9ac5ed4dffc1bcad334c0f1f87780d948b84bde7402c6d` | Store configuration supplies activation and accumulator scale |
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

The target profile explicitly records full-width readout and narrow-readout
ReLU, both enabled in this pinned configuration. Native INT8 ReLU widens each
tile into a private INT32 accumulator, then uses identity-scale narrow ReLU
readout. Every INT8 value is exactly representable at both steps; all 256 values
are checked on the target. One typed tile intrinsic describes the entire
sequence, with its owned accumulator allocation retained in a nested block.
TVM chooses tile loops and rectangular/scalar access strategies; the temporary
and its two-command sequence are fixed intrinsic implementation details. No
arbitrary INT32 narrowing or floating ReLU is implied.

FP32 copies use full-width DMA as raw 32-bit transport, with no scale or
accumulation. The floating buffer type remains visible in mathematical TIR;
the physical words preserve NaN payloads, signed zero, infinities and subnormals.
Runtime scalar FP32 fills and indexed repeat/copy layouts use the same typed
copy rules. This adds data movement, not floating accelerator arithmetic.

## Full requested operator inventory — completion audit pending

The following 51 operator spellings retain the user's full requested scope.
Statuses here are development state, not a completed hardware-support table.

| Family | Requested operations | Current migration state |
| --- | --- | --- |
| Matrix | `mm`, `bmm` | Mathematical integer contractions have structural scheduling, queued-DMA checks and bounded Spike/RTL checks. The direct ATen audit still finds missing `mm.default` import and incorrect FP32 output for integer `bmm`; these are separate frontend defects |
| Biased matrix | `addmm` | Exact integer matmul with eligible INT32 tensor bias uses accumulator preload; shared products and unsupported epilogues retain their original mathematics. CPU activation/scaling and complete ATen/frontend admission remain separate |
| Convolution | `convolution`, `convolution_backward` | Exact INT8/INT32 sliding-window contraction scheduling implemented; generic FP training gradients do not fit this numeric unit. Frontend/decomposition and grouped/transpose admission audit remains pending |
| Arithmetic | `add.Tensor`, `sub.Tensor`, `add.Scalar`, `sub.Scalar`, `mul.Scalar`, `div.Scalar`, `neg`, `relu` | INT32 addition uses accumulator overwrite/add loads and full readout, with separate ordinary and explicit modulo-32 contracts. INT8 ReLU uses exact widen/readout tile intrinsics. ATen add/sub currently ignore nondefault alpha and direct scalar overload conversions are missing; the next fidelity work must repair these before claiming complete frontend/CPU fallback support |
| Reduction | `mean.dim`, `mean.default`, `sum.dim_IntList` | Bounded signed INT8 sums use INT32 array accumulation, including exact INT64 results with CPU widening. The original CPU conversion/division can follow a sum. Ordinary floating reductions and floating-input means retain CPU work; wider frontend and rounding audit remains required |
| Pooling | `avg_pool1d`, `avg_pool2d`, `avg_pool3d`, `adaptive_avg_pool1d`, `_adaptive_avg_pool2d`, `_adaptive_avg_pool3d` | Window/divisor/padding semantics and native admission still required |
| Overlap accumulation | `avg_pool2d_backward`, `_adaptive_avg_pool2d_backward`, `col2im` | Gather/scatter accumulation and numeric semantics still require investigation |
| Movement | `clone`, `repeat`, `cat`, `slice_scatter`, `select_scatter`, `fill.Scalar`, `full`, `full_like`, `constant_pad_nd`, `reflection_pad1d`, `reflection_pad2d`, `reflection_pad3d`, `replication_pad2d`, `replication_pad3d`, `_to_copy` | INT8/INT32/FP32 copies, indexed repeats/slices, constant/reflection/edge padding, signed widening and runtime scalar fills use native DMA where Tensorize proves the access map. Constant padding includes admitted cropping. Index-selected copies expose concatenation and static positive-stride scatter regions through standard scheduling. Constant full/full_like import preserves the captured dtype; normal constant folding can eliminate them, and unfolded literal fills use the existing DMA intrinsic. Broader dtype/frontend/aliasing admission remains under audit |
| Views/metadata | `as_strided`, `diagonal`, `permute`, `scalar_tensor`, `select.int`, `slice.Tensor`, `split_with_sizes`, `squeeze.dim`, `squeeze.dims`, `unsqueeze`, `view` | Admitted static frontend forms execute as standard metadata or INT8/INT32/FP32 mathematical copies. Internal reshapes can disappear; indexed or boundary materializations use DMA when Tensorize proves the access map. Constants and nonunit squeeze can require no device calls. Hidden storage aliases, dynamic contracts and zero-extent static-export buffers remain explicit limitations |

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
| `relu` | Full-width INT32 readout bypasses activation/scaling, so INT32/FP ReLU remains CPU work. Native INT8 ReLU uses a typed tile intrinsic with exact widening, identity scale and narrow readout. Its private INT32 temporary contains only widened INT8 values. |
| `mean.dim`, `mean.default`, `sum.dim_IntList` | A registered sum intrinsic uses a ones-vector contraction for signed INT8 sums with K <= 16,777,216. The prefix bound permits exact internal INT32 accumulation and an ordinary CPU widening for INT64 results. Mean retains its specified CPU conversion/division; a narrowing readout scale cannot stand in for arbitrary floating mean. |
| `avg_pool1d/2d/3d`, adaptive average pooling | Ordinary floating inputs do not fit the integer arithmetic unit. An integer recipe must specify windows, divisors, exclude-padding/ceil behavior and rounding. The max-pool readout feature does not implement average pooling. |
| Average-pool backward and `col2im` | These require the actual gradient dtype, scatter/overlap semantics and window divisors. Accumulate-on-load updates local accumulator rows; it does not scatter-add into arbitrary host output addresses. No generic backward mapping is claimed. |
| Generic data movement | Primitive DMA handles rectangular regions with contiguous columns and one row stride. Copies preserving bits and signed widening are admitted. Index remapping, multiple sources, fill broadcasting and overlapping updates need appropriate schedules or remain CPU work. They must not be described as absent silicon features merely because the current schedule does not handle them. |
| Views and metadata | Reshape-like operations may disappear or become host metadata when consumers can accept the layout. Permute, diagonal, slicing and `as_strided` may need materialization under the compact-buffer ABI. `as_strided` admits contiguous sources and rebases storage offsets only when all reads remain within the logical input. Unsupported hidden storage access is an importer representation limit, not a missing DMA feature. Host metadata is not accelerator arithmetic. |

The existing v0.19 Torch exported-program translator also has an incomplete
direct ATen conversion table. Prior frontend decompositions may remove an
unsupported spelling, but a missing conversion is not a Gemmini CPU-fallback
success. This backend accepts ordinary Relax/mathematical TIR; frontend gaps
must be reported independently of native-unit legality.

Padding import now handles direct `reflection_pad1d/2d/3d`,
`replication_pad1d/2d/3d`, `constant_pad_nd` and generic `pad` by translating
PyTorch's reversed-axis width pairs into ordinary Relax `nn.pad`. The generic
legalizer previously ignored `pad_mode`, producing constant padding even for
reflection/edge requests. It now uses mathematical TOPI reflection or a clamped
TE load for edge padding. These are target-independent correctness changes,
not Gemmini kernel selection. Reflection requires proven positive input extents
and widths smaller than the corresponding input dimension, matching the
single-reflection TOPI implementation. Unproved bounds retain the Relax call;
they are an explicit legalization limitation, not a successful CPU fallback.
Constant padding follows its generic mathematical legalization before the
region schedule described below. Import alone does not establish native support
for other dtypes or an arbitrary frontend graph.

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

Indexed INT8/INT32/FP32 copies are also matched by the framework: rectangular tiles
are tried first, then single-element DMA when Tensorize cannot prove a larger
access map. This covers materialized slices, diagonals, permutations and repeats
without CPU value copying. Such schedules can issue many commands and are not
performance winners without measurements. Both options are visible to standard
MetaSchedule, whose tensorization postprocessor rejects illegal candidates.
In-place copies remain refused. DMA completion also protects external output
buffers when CPU code in the same function subsequently accesses them.

For a block that performs a single indexed load per input, the DMA rule can use
standard `Schedule.annotate_buffer_access` to state the exact point read where
automatic region inference was conservative. All actual reads, including index
buffer reads, are retained; differing access maps to the same buffer are never
collapsed. The annotation and subsequent scheduling replay from the original
mathematical body. The v0.19 Python annotation wrapper currently requires equal
block/buffer ranks, so rank-changing accesses needing this refinement retain
fallback. This enables reflection and edge padding through the existing typed
copy intrinsics, with scalar DMA where larger rectangles cannot tensorize.
There is no padding-specific C implementation, source-template generator,
hardware-loop command or new C++ patch.

Runtime scalar fills support INT8-to-INT8, INT32-to-INT32 and signed
INT8-to-INT32 output. They use one-column, zero-stride DMA and preserve the
ordinary output layout through standard cache placement. Constant-only fills
may disappear through normal FoldConstant. Non-integer output conversion is
not implied. The same broadcast layout supports runtime INT32 scalar addition;
the ordinary no-overflow and explicit widen/add/narrow variants remain distinct.

Constant padding uses standard `decompose_padding` to form a full-output fill
and an interior copy. Standard blockization gives each region its own scope;
ordinary caches can then serve one local writer even though the external output
has two writers. Translated output index maps are admitted only when each is a
constant, in-bounds translation of the corresponding block axis. The entire
decomposition, scoping, tiling, placement and tensorization trace replays from
the original mathematical workload, in Python and JSON.

A typed literal intrinsic materializes one host scalar and broadcasts it by
zero-stride DMA; the CPU does not fill the output tensor. Constant identity uses
its dtype and exact scalar bits, keeping positive/negative zero distinct.
Floating implementation constants use typed integer-bit reinterpretation:
v0.19's internal Tensorize implementation copy cannot JSON-load a nonfinite
FloatImm. This does not fix general JSON loading of nonfinite mathematical
workloads, so persistence of such workloads through the JSON tuning database
remains unqualified. Ordinary trace replay and static device execution are
checked separately.

The default DMA schedule now uses the same standard `RewriteTensorize`
postprocessor as design-space candidates. v0.19 logs a failed replacement but
can return true, so target memory legality must also reject residual CPU SRAM
accesses before accepting a candidate or trying scalar DMA. Multiple regions
currently share the rectangular/scalar strategy choice; independent per-region
search and alternative padding algorithms are not implemented.

Concatenation and functional scatter updates remain mathematical lazy selections
between buffer loads. `gemmini_regions.py` proves index-only predicates with
TVM's arithmetic analyzer, then exposes constant-selection regions through
`loop_partition`, `cache_write`, `compute_inline` and `blockize`. Both temporary
global caches disappear through ordinary inlining; the surviving region is a
pure mathematical copy before scoped DMA caches and Tensorize are applied.
Unit iterators are eliminated during normalization so singleton regions do not
retain a selector depending on an outer unit binding. Block references are
resolved structurally through ordinary child-block queries, including when
generated display names are duplicated. Conservative ancestor read declarations
can still mention unreachable branches; executable leaf copies and their DMA
regions have the selected source only.

Region analysis has explicit compiler-work budgets: at most 128 regions, and
at most 65,536 index positions for a nonlinear one-axis predicate. These are
engineering limits, not hardware limitations. Runtime data predicates, excessive
partitioning or unproved access maps retain the original CPU mathematics.
MetaSchedule also preserves an unchanged CPU candidate if no region schedule
is admitted. Memory-resource checks can impose a tighter bound because region
SRAM allocations currently do not share storage by lifetime.

The PyTorch ExportedProgram importer now admits INT8 tensors and functional
`slice_scatter`/`select_scatter` with static shapes and bounds, positive slice
steps, negative dimensions/indices and clipped Python slice bounds. It emits
ordinary TOPI `strided_set` mathematics; it makes no hardware or scheduling
decisions. Empty slices produce an identity copy. TOPI's index expressions are
promoted to the shape's index dtype, correcting its INT32-bound/INT64-shape
mismatch independently of the target. Internal constant index-vector producers
are removed through standard `compute_inline`. A separate expand-dimensions
materialization may remain CPU work in the static exporter, particularly when
the copy changes rank; the ordinary VM pipeline can apply its own view lowering.

DMA writes complete when an output has multiple DMA writers or is subsequently
read through DMA, as well as for the existing CPU/staging dependencies. This
conservatively orders full fill before interior overwrite. A host control can
reverse captured store completion order; correct waits preserve the padding,
while deliberately removing them exposes an incorrect overlapping result.
This is a dependency check, not a hardware timing model or measured speedup.

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
floating inputs and genuinely wider accumulation still retain CPU fallback.
For an INT64 sum of signed INT8 values, the same prefix bound proves that the
private reduction allocation can use INT32. Standard `reindex_cache_write` and
`unsafe_set_dtype` introduce that storage choice and preserve the INT64 output
conversion; the trace records both. The numerical proof is retained in schedule
metadata, including database replay. Typed update descriptions retain the
actual pure signed-widening cast chain, including explicit INT32 sum requests.
CPU final casts preserve explicitly requested narrower output wrapping. This
does not admit INT64 input arithmetic to the array. Known INT8 values widened
to floating point may admit additional range proofs; those are not implemented.

The PyTorch importer preserves positional/keyword keepdim, default integer
promotion to INT64, explicit dtype conversion, global sum/mean overloads, empty
dimension lists and scalar reductions. Explicit narrow integer sums accumulate
in INT64 under a static prefix bound before converting to the requested output
type. Unproved or excessive bounds are explicit import limitations. Generic
Relax legalization expresses empty sums as zero fills and empty floating means
as NaN fills, rather than leaving initialization inside a zero-trip loop.
Zero-axis and scalar reductions retain identity mathematics. These frontend
and legalization corrections are independent of accelerator selection.

The shared finalizer retains a TensorIR root block around admission and device
ownership. This lets stock Relax reshape analysis inspect prepared single-input
functions during normal `relax.build`, instead of requiring a special pipeline
to avoid that analysis. No core compiler modification is needed.

View and metadata admission now uses the ordinary frontend and Relax pipeline.
Nonunit explicit squeeze axes retain the input, negative select indices are
normalized and statically checked, and select uses ordinary slice/squeeze
operators without a runtime index tensor. Split sizes use their actual cuts,
including integer split sizes that do not evenly divide the extent. Diagonal
and admitted as-strided expressions are target-independent TE index maps;
their frontend definitions contain no scheduling or hardware decisions.
Constant scalar_tensor uses its captured dtype. Existing exporter-era expected
IR for select/slice is refreshed for current PyTorch's removal of no-op slices.

Standard RewriteDataflowReshape can turn internal copy loops into reshape
metadata. DeadCodeElimination removes their unused PrimFuncs before scheduling,
so an unused function cannot be counted as accelerator placement. Static export
accepts same-byte-extent, same-dtype compact reshape aliases and extends each
backing allocation's liveness through all view consumers and returned aliases.
Views do not allocate intermediate storage. Caller-owned return slots still
require copies when returning an input, a constant, or an aliased workspace
value. Programs with only metadata/constants therefore need no operator calls.
The stock build driver rejects an empty IRModule; for these zero-operator
programs only, packaging asks the standard LLVM target codegen for an empty
object container. There is no replacement math or scheduling in that bridge.

Vector-to-matrix copy/widen descriptors expose zero-stride row broadcast to
Tensorize. Ordinary rectangular or scalar candidates remain available through
MetaSchedule; indexed/permuted shapes can use scalar DMA when a rectangle does
not match. This removes a previous rank-changing copy limitation without a
whole-operation library call or new C++ patch. As-strided sources must be
contiguous, their storage offsets are rebased to the logical Relax input, and
all actual reads must stay within that input. Hidden storage access and dynamic
shape/offset contracts are refused explicitly. Zero-extent tensors can execute
in ordinary VM tests but remain outside the static exporter's positive-size ABI.
The scalar_tensor importer currently admits constant scalars only.

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

`device-bias-v2` adds five native bias-preload fixtures: vector, matrix, column
and scalar broadcasts, plus explicit modulo-32 final output. The ordinary
addition fixtures deliberately cross INT32 limits after preload and return to
an in-range final result. Positive and wrong-oracle runs pass on Spike and RTL,
with 60-second / 300,000-cycle RTL bounds. The guest requires array compute and
accumulator overwrite DMA, and refuses a zero-initialized substitute. Source,
library, adapter, ELF and simulator bindings are retained with each run.

The generic core checks in `core-sequence-after-v1.log` pass 80 tests, with one
existing affine-analysis skip. `unit-bias-full-v1.log` passes 125 backend/app
checks; its one failure and one skip are missing optional XGBoost. Both pass in
`unit-bias-xgb-v1.log` with isolated `xgboost-cpu==2.1.4`, completing all 127
Gemmini checks without changing the legacy search. That learned-search test
uses explicitly synthetic diagnostic labels, not Gemmini performance data.
The exact rebuilt libraries are preserved in `compiler-core-v1`, independently
of the mutable CMake build directory. The new tests include original-input
preservation, full bias trace replay, grouped short tiles and rejection of a
deliberately misaligned SRAM program. The 51-operator audit remains pending.

`device-relu-copy-v1` adds exhaustive INT8 ReLU, a prime-sized ReLU tail and
FP32 copy/repeat/scalar-fill fixtures. Positive and wrong-oracle checks pass on
Spike and RTL within the same short-run bounds. FP32 guest data and references
are emitted and compared as raw unsigned words, including signaling/quiet NaN
payloads, signed zero, infinities and subnormals. The guest also requires actual
ReLU readout calls. `adapter-relu-v1` binds the new primitive and audits the
object against hardware-loop instructions.

`unit-relu-copy-regression-v1.log` passes all 130 collected Gemmini backend/app
checks with the isolated optional XGBoost dependency. The additional
`unit-relu-wait-v1.log` check verifies a CPU consumer after device ReLU, then
deliberately removes completion and detects the wrong result. No additional
C++ patch is introduced for these operations. None of these checks supplies a
performance label or completes the remaining operator inventory.

The migration remains incomplete until the inventory's remaining native
admission, CPU/frontend explanations and final regression/target checks are
resolved. Do not infer those from passing contraction fixtures.

`device-pad-v1` adds reflection 1D/2D/3D and replication 2D/3D fixtures through
normal Relax legalization and the structural DMA rule. Positive and wrong-oracle
checks pass on Spike and RTL within 60 seconds and 300,000 RTL cycles. Fixtures
mix INT8, INT32 and raw-word FP32 movement, including NaN payloads and signed
zero, and require actual DMA commands. The existing adapter is unchanged.
`unit-pad-native-v2.log` checks ranks one through five, all three admitted
movement dtypes, source immutability, delayed DMA, full trace replay, standard
MetaSchedule postprocessing and the registered VM build pipeline.
`unit-pad-frontend-v2.log` passes 37 checks for ordinary CPU legalization,
cropping, wide edge padding, unsafe-bound refusal and eight actual PyTorch
exported padding forms with unequal axis widths. `unit-pad-full-v1.log` passes
170 collected Gemmini backend/app and padding checks; the final 37-check run
refreshes the generic/frontend tests after strengthening those axis widths.

The broader `test_transform_legalize_ops_nn.py` run reports nine structural
expectation failures, 47 passes and three skips. The same nine tests fail with
the unchanged baseline Python legalizers and the same compiled library;
`unit-pad-nn-regression-v1.log` and `unit-pad-nn-baseline-v1.log` retain both
runs. These failures are not reported as passing regressions or resolved by
this padding change.

`device-constant-v2` adds INT8/INT32/FP32 constant padding, negative zero,
negative-NaN literal padding and cropping, plus small contraction and sum
regressions. All eight cases pass positive and wrong-oracle checks on Spike
and RTL within 60 seconds and 300,000 RTL cycles. The adapter is unchanged;
ELF audits continue to reject hardware loops. `unit-regions-full-v2.log`
passes all 173 collected Gemmini/backend/app and generic padding checks with
the preserved `compiler-core-v4` libraries. Region checks include integer
extremes, both zero signs, NaNs/infinities, raw input payload preservation,
cropping, independent references, Python/JSON traces, standard MetaSchedule
and deliberately missing completion.

`unit-core-regions-v4.log` passes 88 general schedule tests, with one existing
affine-analysis skip. The literal-only tests pass all nine cases after the
matcher fix; the prior immutable library fails the two identical-NaN cases
and the opposite-zero-sign refusal. `qualification-core-regions-v1.json`
binds those checks, source patches and immutable compiled libraries. These
checks qualify the disclosed scheduler fixes and functional region support;
the full requested operator inventory and final coverage-wide source freeze
remain unfinished.

`device-copy-v1` adds eight copy-region fixtures: INT8/INT32/FP32 concatenation,
contiguous and positive-stride slice updates, and select updates, plus small
contraction and sum regressions. All ten cases pass Spike and the pinned RTL;
wrong-oracle controls fail as intended on both substrates. The RTL positive
run completes within the 60-second, 300,000-cycle bounds. The primitive adapter
is unchanged and both ELF instruction audits contain no prohibited instructions.
`unit-copy-full-pinned-v1.log` passes all 195 collected backend/app, padding and
scatter-frontend checks against immutable `compiler-core-v5` libraries.
`unit-core-partition-pinned-v2.log` passes 230 generic scheduler/TOPI checks,
with one existing affine-analysis skip. The previous immutable compiler fails
five focused partition/definition regressions. The new checks include duplicate
display names, singleton regions, all three copy dtypes, NaN payloads, input
preservation, Python/JSON traces, ordinary VM compilation, MetaSchedule and
CPU retention for runtime predicates or excessive partitioning.

This checkpoint does not complete the operator inventory. A subsequent frontend
audit at that checkpoint reproduced incorrect default integer `sum` output dtypes and lost
`keepdim=True` shape in the existing `_sum` importer. Correct INT64 semantics
must be restored independently of native INT8-to-INT32 reduction scheduling;
passing the latter does not qualify the current ATen importer for those sums.

The following reduction change resolves that audit for the admitted PyTorch
dtype/shape contracts. `device-wide-sum-v1` passes seven fixtures on Spike and
the pinned RTL, including default INT64 row sums, keepdims over multiple axes,
a scalar INT64 result, explicit INT32/INT8 output requests, and small ordinary
contraction/sum regressions. Wrong-oracle controls fail as intended on both
substrates within the 60-second/300,000-cycle bounds. The primitive adapter is
unchanged and both ELF audits contain no prohibited instructions.

`unit-wide-sum-full-pinned-v1.log` passes 248 backend/app/frontend checks with
immutable `compiler-core-v6` libraries. The final Boolean empty-reduction
adjustment and stronger Boolean inputs are separately checked by all 49 cases
in `unit-frontend-reduction-v5.log`; those overlap the broader suite and are
not added to its count. `unit-core-dtype-pinned-v1.log` passes 250 general
scheduler/index-map checks with one existing affine-analysis skip. Both
constant-initialization cases fail on the previous immutable compiler, one
on its stale scope flags and one on its redundant constant cast.

The broader statistical legalization tests retain four older IR expectation
failures concerning division versus reciprocal multiplication; all other 17
checks pass. The same four fail with the unchanged prior legalizer and the same
compiler, as recorded in `unit-statistical-baseline-v1.log`; the modified run
is `unit-frontend-reduction-v4.log`. They are not counted as passing or repaired
by this milestone. The full requested operator inventory remains unfinished.

`device-views-indexed-v1` and `device-views-metadata-v1` qualify thirteen new
view/frontend fixtures plus small contraction and INT64-sum regressions. The
fixtures exercise all eleven requested view/metadata spellings in their admitted
static forms. Diagonal, as-strided, negative-index select, permutation, slicing,
split, unit squeeze, unsqueeze and boundary reshape materializations require
actual DMA. Nonunit squeeze and constant scalar output issue no operator calls;
an internal reshape alias consumed by a runtime-index CPU gather also issues no
device commands. Positive and wrong-oracle checks pass on Spike and the pinned
RTL, with positive RTL runs of approximately 31 and 44 seconds under the
60-second/300,000-cycle bounds. The adapter and immutable `compiler-core-v6`
libraries are unchanged; final ELF audits contain no prohibited instructions.

The guest verifier now checks every output independently, including both split
results, and checks guards surrounding each result. FP32 comparisons preserve
raw payloads. Separate host fault injection proves that a wrong second output
and a corrupted second-output guard are detected, independently of the general
wrong-oracle control. These are functional checks, not performance labels.

`unit-views-full-pinned-v2.log` passes 297 backend/app/frontend checks, and the
disjoint `unit-views-exported-graph-v1.log` passes fifteen exported-graph checks.
The preceding broader run had one duplicate-intrinsic registration failure;
canonicalizing optional registration arguments fixes its cache key, and the
complete final run passes. The checks include independent PyTorch references,
input bit preservation, Python/JSON trace replay, standard MetaSchedule
postprocessing, CPU fallback after reshape elimination, returned-view backing
lifetimes and zero-operator/static-constant export. The existing baseline-only
NN/statistical expectation failures described above remain outside this suite.

This does not complete every view/alias contract: dynamic shapes/offsets,
hidden storage reads, noncontiguous as-strided sources and zero-sized static
export buffers remain explicit limitations. Constant scalar_tensor is admitted;
runtime scalar construction still needs a representable frontend contract.
The next creation-op audit reproduces unsupported `aten.full.default` and
`aten.full_like.default` import. Those are frontend gaps, not missing hardware
fill capabilities, and do not count as successful CPU fallback.

The following creation change resolves that `full`/`full_like` audit for constant
scalar arguments. Import emits ordinary Relax `full`/`full_like` with the captured
output dtype, preserving default integer/bool/float inference and explicit dtype
overrides. Standard constant folding may remove the computation. Without that
optimization, standard legalization and the existing literal-fill scheduling
rule tensorize the mathematical fill; there is no new creation kernel or
operator-name hardware dispatch. The shared frontend dtype converter also admits
FP64, including a changed PyTorch floating default. Native movement remains
limited to the explicitly admitted INT8/INT32/FP32 contracts.

TensorIR admission now accepts output-only functions. Their output pointer,
alignment, positive span and address overflow are checked before any device
command. `adapter-creation-v1` retains all nineteen preceding executable section
byte sequences and adds only the six-line output-buffer validator. This is an
ABI admission helper, not a mathematical implementation. The guest directly
checks null, nonpositive, misaligned and wrapping output spans against the
cross-compiled helper; target positive runs require these rejection checks.

`device-creation-native-v1` qualifies both creation ops with INT8/INT32/FP32
literal DMA and a small contraction regression. These explicitly unoptimized
fixtures demonstrate hardware mapping, not performance. The optimized fixtures
in `device-creation-folded-v1` require all six creation computations to fold to
constant outputs with zero device commands; split and INT64 sum regressions
still exercise the array/DMA path. All fifteen fixture executions pass Spike
and RTL with wrong-oracle controls. Positive RTL runs take approximately 33 and
40 seconds within the existing bounds; ELF audits contain no prohibited loop
instructions. The immutable compiler-core-v6 libraries are unchanged.

`unit-creation-full-pinned-v1.log` passes all 339 collected checks, including
the preceding view/export regressions and 27 creation checks. Creation checks
preserve output shape/dtype/bits for defaults, overrides, integer extremes,
fractional-to-integer conversion, NaN, signed zero, scalar and empty shapes.
They check native DMA and constant folding separately. Empty tensors remain
outside the static export ABI, and runtime scalar arguments to these ATen
creation overloads are not admitted by this importer; runtime tensor fills
continue through the separately tested fill.Scalar path.

The independent `arithmetic-next-audit-v1.json` probes thirty small ATen
operator/dtype combinations rather than certifying whole operators. Seventeen
are refused or fail, eight have semantic mismatches, and five match independent
CPU references. It finds missing direct mm/scalar-overload conversion, forced
FP32 integer bmm output, mixed-dtype integer addmm failure and ignored tensor
add/sub alpha. A twelve-case diagnostic with unchanged legacy Python importers
reproduces five INT32/FP32 mismatches and six refusals; that older importer also
refuses INT8 input. These defects are not concealed by the passing mathematical
TensorIR tests. Frontend fidelity work and the remaining pooling/backward
inventory are still required before completing the migration.
