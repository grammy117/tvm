> Historical record, frozen before the curated handoff. Commands and missing-feature
> claims below may be superseded. Start with [README.md](README.md).

<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# Gemmini integration organization

The `gemmini/tensorir` migration changes the default `prepare_gemmini_graph`
route to mathematical legalization/fusion followed by structural TensorIR
scheduling. [EL4_STATUS.md](EL4_STATUS.md) records current placement, scheduling, numerical
contracts and reviewer corrections. [FIRESIM.md](FIRESIM.md) documents the offline
compiler, portable archive, guarded boot ELFs and deployment bindings.
[TENSORIR.md](TENSORIR.md) retains the earlier functional migration audit. The original template path is
preserved as `route="legacy"` and `prepare_gemmini_graph_legacy`; the template
scheduling/search recipes below describe that preserved route.

The current compiler branch is `gemmini/metaschedule-dse`; the older setup and template recipes below refer to `gemmini/bringup`. The isolated host build, synthetic frontend checks and ResNet50 v1.5 host diagnostics are verified on CPU. The Gemmini C-library adapter, bounded TVM-scheduled graphs and a complete random integer ResNet graph pass numerical checks in the pinned functional simulator. [TUNING.md](TUNING.md) describes the per-operator RTL measurements and package-slice qualification. Pretrained quality, full-model RTL timing and deployed FireSim hardware remain unqualified.

| Location | Integration responsibility |
| --- | --- |
| `python/tvm/relax/backend/contrib/gemmini.py` | Matmul admission, lowering and graph optimization around the device boundary. |
| `python/tvm/relax/backend/contrib/gemmini_conv.py` | Semantic convolution packing, matmul decomposition and layout restoration. |
| `python/tvm/relax/backend/contrib/gemmini_schedule.py` | Semantic tiled TIR, primitive tensorization and resource contracts. |
| `python/tvm/relax/backend/contrib/gemmini_tuning.py` | Preserved legacy template search. |
| `gemmini_tensorir.py`, `gemmini_intrin.py`, `gemmini_target.py` in the same contrib directory | Current mathematical scheduling, TensorIntrins, scopes and capability identity. |
| `gemmini_meta_schedule.py`, `gemmini_search.py`, `gemmini_sketch.py` | Sampled sketches, stock MetaSchedule tuning and JSONDatabase/Relax integration; see [TUNING.md](TUNING.md). |
| `gemmini_layout.py`, `gemmini_epilogue.py` | Frozen layout folding and exact scaled-readout proof/normalization. |
| `apps/gemmini/` | C primitives, static graph export and host/simulator verification recipes. |
| `tests/python/relax/test_backend_contrib_gemmini*.py` | Focused admission, scheduling, graph, export and search regressions. |

`verify_host.py` verifies checkout identity, LLVM code generation and CPU execution against NumPy, and records loaded-library/build identity. Use `examples/gemmini/comparisons/tvm/HOST_SETUP.md` in the parent comparison repository for the pinned environment, host verification commands and provenance.

The host recipe opts into `USE_HOST_ONLY_AUTO_COPY_GUARD` (OFF by default) because this checkout lacks the `LowerAutoCopy` implementation required by existing driver/MetaSchedule callers. `auto_copy_guard.cc` preserves unannotated IR and rejects automatic-copy markers; it is a validation-only pass, supplies no optimization, and must not coexist with the full implementation. The host verifier checks preservation and rejection through direct calls and ordinary `tvm.build`; record this local patch and enabled mode with the base commit.

`verify_onnx.py --output-dir DIR` checks matmul, batched matmul, convolution, LayerNorm and RMSNorm at opsets 17 and 18 against PyTorch and ONNX ReferenceEvaluator. `--suite importer` instead checks shape arithmetic, constant/runtime negative Gather indices, lower-rank Expand and scalar ConstantOfShape against independent NumPy expectations. Integer outputs require exact equality, including an int64 value beyond the int32 range. Both suites retain graphs, imported IR, arrays and dependency/library provenance.

Optional `--case`, `--opset` and `--importer-source` narrow checks or isolate a historical Python importer on the current runtime. Historical negative-Gather overrides retain imported IR but do not execute potentially unchecked indices; their status is `not_executed`, never a pass. The original v0.19 importer fails the opset-18 RMSNorm case numerically and fails the shape-array, lower-rank Expand and scalar ConstantOfShape cases during import. A local follow-up preserves the NumPy dtype when folding shape arithmetic, preventing the inherited binary fix from silently narrowing int64 output. These cases do not exercise every inherited change or qualify model/device behavior.

Reshape, Expand and NonZero can fold closed, static producer calculations through TVM's constant folder, bounded to 32 levels, 128 nodes and 64 KiB of aggregate tensor payload. NonZero also bounds its worst-case index bytes and preserves coordinate order and scalar semantics. Unknown inputs, unsupported operations and oversized intermediates retain conservative refusal/dynamic behavior; the original producer graph remains unchanged. Same-dtype Cast preserves its input directly. Independent ONNX/LLVM regressions prove closed ScatterND lengths; empty index payload semantics are checked without qualifying this runtime's empty-constant VM loader. This does not provide general dynamic-shape support.

[Range schema 11](https://onnx.ai/onnx/operators/onnx__Range.html#range-11) folds only proven closed scalar, exact integral progressions within the same payload budget. Real scalar casts execute before folding; literal primitive endpoints are supported. Exact count arithmetic and typed output must be established before materialization. Fractional, oversized, symbolic or otherwise unproved endpoints retain the runtime path. Focused ONNX/LLVM checks and saved full-prefix import/optimization pass; neither establishes complete-model numerical or device qualification.

Variadic Min/Max/Mean/Sum use schema-versioned broadcasting and declared-dtype arithmetic instead of stacking incompatible operand ranks. Legacy versions require proven equal shapes. Floating Min/Max preserve NaNs and second-operand zero ties; Sum/Mean use ordered additions with each dtype boundary retained. Focused constant/runtime LLVM checks include BF16 rounding and symbolic shapes. Mean may retain the standard TIR reciprocal-multiply rounding difference; this does not change a model's numerical gate or promise bitwise equivalence to every runtime.

The explicit `merlin.fixed_order` version-1 operators `FixedTreeDot` and `FixedTreeMeanSquares` round FP32 products separately, then add adjacent pairs at each tree level without an initial zero addition, contraction or reassociation. Dot accepts `[B,H,M,K]` and `[B,H,N,K]`; mean squares retains the final singleton dimension and multiplies by the exact power-of-two reciprocal. Both require positive static FP32 shapes and power-of-two `K<=4096`, with finite, non-overflowing operands as a caller assumption. Missing/duplicate/wrong-domain imports, attributes, incompatible shapes and byte-span overflow are refused; standard ONNX arithmetic is unchanged.

Opaque typed CPU helpers use int64 indices and explicit private scratch outputs, so serial static planning accounts for at most `4*K` scratch bytes per call without hidden allocators. Independent staged-FP32, shape/order, admission and LLVM checks pass; both graph modes also pass a bounded raw-byte matched-Spike fixture with a separate INT8 primitive and explicit libc binding for generated copies. These are opt-in arithmetic contracts, not complete-model acceptance or performance results; changing a model reference still requires fresh quality/cache/compiler/reset gates.

For original BF16 host models, the importer preserves tensor types and initializer/Cast payload bits through `ml_dtypes`, including constant folding. BF16 MatMul/Gemm use FP32 accumulation, Conv widens already-rounded BF16 operands before padding/contraction, and LayerNormalization uses centered FP32 statistics/affine with `stash_type=1`; their declared outputs remain BF16. This preserves the source CPU arithmetic contract without converting a checkpoint or its session ABI. Unsupported mixed BF16 contracts are refused. Native ONNX BF16 Conv requires opset 22.

```sh
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite importer --output-dir "$BUILD_ROOT/importer-new-run"
python "$TVM_ROOT/apps/gemmini/verify_onnx.py" --suite bfloat16 --opset 22 --output-dir "$BUILD_ROOT/bfloat16-new-run"
```

The importer suite includes BF16 initializer/Constant/Cast regressions. The eleven BF16 arithmetic cases cover MatMul, scaled/bias Gemm, Conv, centered LayerNorm, asymmetric SAME_UPPER/LOWER convolution, softmax at reductions 17/241, both GELU approximations and sigmoid. Softmax/activation math runs internally in FP32 before the declared BF16 result boundary; GELU follows the ONNX `approximate` attribute rather than always choosing erf. The importer accepts both ONNX 1.17's legacy type map and ONNX 1.19's public dtype helper. LLVM results match independent PyTorch CPU BF16 outputs at the unchanged strict tolerances. ONNX 1.17's BF16 reference arithmetic is unavailable for those contractions, so the report explicitly identifies PyTorch as the oracle. These small cases do not qualify a complete trained policy, accelerator execution or application quality.

The selected backend route calls the Gemmini C operator library, following the comparison team's fairness guidance. Preserve shapes, layouts, quantization scales, rounding and output semantics; make supported shapes/layouts, argument contracts, numerical rules, workspace requirements and instruction policy explicit. Agustin requires every implementation, including this baseline, to avoid FSM/hardware-loop instructions. The handwritten compiler is the evaluated Phase-1/2 implementation, not a baseline dependency or configuration authority. The library owns instruction encoding; the intended optimized backend gives TVM control over tiling, transfers and reuse through library primitives. The whole-operation adapter remains a reference path. Pin headers to the actual hardware; a local `gemmini_params.h` must not be assumed to describe signed-int8 arithmetic.

Agustin permits baremetal or RTOS; this integration chooses baremetal. Host export/reload alone proves neither RISC-V nor baremetal deployment. The independent CPU probe establishes a bounded static generated-graph route under generic Spike, not general Relax runtime support. Confirm actual deployment ISA/ABI, startup and addressability; hardware revision, generated headers, quantization and timing boundaries remain explicit inputs. MX Gemmini requires a different hardware and numerical contract. Expose CPU work, data conversion, transfer and completion costs to the comparison harness.

## Model tuning orchestration

`collect_gemmini_workloads` inventories admitted semantic contraction shapes after convolution/projection decomposition and separately reports remaining host contractions. Sites with the same shape share candidates; this inventory is not execution coverage. `GemminiModelSearch` issues reproducible candidate requests under one finite model-wide budget, ranks each shape through the existing TVM feature/XGB machinery, and ingests repeated `TimingRecord` samples only after candidate-specific correctness and build/platform/protocol checks. Failed ranking or ingestion leaves the previous ledger unchanged. Requests are proposals, not measured winners; synthetic labels require explicit opt-in.

Pass a request's `(M,N,K): (tile_i,tile_j)` map as `tile_choices` to `prepare_gemmini_graph` to apply different schedules to different shapes. Unspecified shapes retain the supplied default tiles. Retain untuned, baseline-graph and optimized-graph ablations. Qualified end-to-end measurements include required host, packing and conversion work; operator measurements do not substitute for complete-session timing. Final FireSim measurements and source/adapter bindings remain external inputs.

## Integer learned-weight projections

`prepare_gemmini_graph` flattens static vector or batched signed-INT8 activations multiplying a shared rank-two signed-INT8 weight, schedules the resulting INT32 matmul, and restores the original output shape. The existing reduction, resource and byte bounds apply. Quantization, bias and dequantization stay explicit; batched weights and unsupported arithmetic stay on the host. The ONNX frontend preserves `MatMulInteger` signed operands with absent or scalar constant zero points equal to zero; unsigned operands, nonzero points and per-axis points are currently refused. This supports projection integration, not complete transformer device or quality qualification.

The existing schedule suite covers vector/batched shapes, both graph modes, repeated calls, retained outputs, immutable inputs, independent INT64 references and primitive bounds. It also checks ONNX import and unsupported-case refusal. Run its focused entry point using the selected host environment:

```sh
python "$TVM_ROOT/tests/python/relax/test_backend_contrib_gemmini_schedule.py"
```

## External matmul compiler boundary

Apply `tvm.relax.backend.contrib.gemmini.LowerGemminiMatmul()` before `LegalizeOps` and build for a CPU target. The pass admits positive, static, rank-two signed-int8 `[M,K] @ [K,N]` matmul with signed-int32 `[M,N]` output and `K <= 131071`. That bound makes every product and partial sum representable for all signed-int8 inputs. Dynamic shapes, batches, other dtypes and explicitly non-CPU tensor placement remain ordinary Relax operations. Existing `Codegen`/`Composite` functions are preserved. No quantization, bias, activation, scale or transpose fusion is implied, and a float ResNet graph receives no acceleration from this pass.

The generated TIR wrapper calls the synchronous ABI in [matmul.h](matmul.h), using compact row-major element strides `K,N,N`. Link an implementation when exporting the executable. Zero status means complete visible output; nonzero status raises a TVM runtime error. The call and status check remain active when optional TIR assertions are disabled, although callers should retain the normal shape/stride checks. Inputs must be compact zero-offset buffers, and the implementation must preserve them. The wrapper uses the ordinary TVM runtime error API; it does not establish a freestanding entrypoint.

The tests export/reload both bytecode and compiled Relax executables and explicitly link a C implementation embedded in the test file. That implementation accumulates independently in int64, records call counts and injects failures. It is a host oracle, not the Gemmini library's CPU fallback or a production device implementation. Missing external symbols do not provide an automatic CPU substitute for admitted calls. Unsupported operations retain TVM's normal CPU path. In an activated host environment, run:

```sh
python tests/python/relax/test_backend_contrib_gemmini.py -v
```

The test requires LLVM-enabled TVM, NumPy and `cc` on PATH. It covers rectangular shapes, negative/extreme inputs, the admitted reduction bound, repeated calls, frozen constants, wrapper reuse, CPU fallback, function/control-flow preservation and failure propagation. No accelerator headers, simulator or FPGA are used.

## Pinned C-library adapter

[matmul.c](matmul.c) implements the external ABI through `tiled_matmul` with primitive OS and fixed one-array I/J/K tiles, null bias, identity input/output scales, no activation and full int32 output. It rejects invalid dimensions, strides, unrepresentable byte spans and output/input overlap before issuing accelerator instructions. Callers must supply mapped, DMA-accessible buffers and exclusive accelerator ownership. CPU/compiler barriers surround the library's device call and fence; platform coherence remains a separate qualification.

Hardware loops are forbidden, with explicit `TVM_GEMMINI_FORBID_HW_LOOPS=1` and `TVM_GEMMINI_PE_OUTPUT_BITS=20`. The provisional Scala configuration has 20-bit PE output, which the C parameter header does not expose. Fixed DIM=16 and tile_K=1 keep each PE reduction representable for arbitrary int8 inputs, while outer K tiles accumulate in int32. An arbitrary automatic OS tile choice does not preserve this guarantee. The canonical integer header lacks `NORM_STAT_IDS`, which the operator header references even for no-activation calls. An integration-only fallback allows those unused branches to parse while rejecting `HAS_NORMALIZATIONS`. The canonical parameter header is unchanged and normalization is not supported.

[verify_matmul.py](verify_matmul.py) extracts only the selected Git blobs from existing local repositories. Its default source set is Gemmini `6ad65b90b1eb270c20ce4f04109e3dde0180b36f`, gemmini-rocc-tests `7c540b3adf1b86ad93d07f893abe3a73489b568e` and libgemmini `ea8f7ed7afd68e001fb06ddccad9a023c990961d`. It validates parameter/operator digests, stages the exact nested includes, cross-compiles RV64GC/lp64d and records resolved header, source, compiler, object and disassembly identities. It checks primitive-only instruction policy as well as rejection of incompatible headers. The final object must have no FSM instructions in any executable section; the same audit must be applied again to the final linked ELF. This verifies compilation, not a simulator/plugin build or numerical device execution.

From the comparison checkout, set the actual reference checkout and RISC-V toolchain paths:

```sh
GEMMINI_ROOT=/path/to/chipyard/generators/gemmini
RISCV_BIN=/path/to/riscv-tools/bin
TVM_ROOT="$PWD/third_party/baselines/tvm-gemmini"
BUILD_ROOT="$PWD/out/build/baselines/tvm-gemmini"
python "$TVM_ROOT/apps/gemmini/verify_matmul.py" --gemmini-repo "$GEMMINI_ROOT" --build-root "$BUILD_ROOT" \
  --cc "$RISCV_BIN/riscv64-unknown-elf-gcc" --objdump "$RISCV_BIN/riscv64-unknown-elf-objdump"
```

All required Git objects, including the pinned nested rocc-software commit, must already exist locally. No fetch, checkout, reference-header edit or TVM/RTL rebuild occurs. Use `--rocc-tests-repo` or `--rocc-software-repo` for separate checkouts and `--output-dir` for a fresh child of the selected build root. Link only the audited `matmul.o`: a relocatable link keeps the ABI entrypoint and its relocation dependencies while removing unused vendor WS code from `matmul.raw.o`. The raw object is diagnostic and must not be linked. Pass `--audit-elf /path/to/final.elf` to check all executable sections again after final linking. The no-FSM policy is confirmed. Jack identifies default Gemmini with a Rocket host; the exact FireSim build, matching generated headers and guest capacity remain unbound, and these source pins do not adopt gemmini-mlir as a baseline.

## TVM-owned primitive schedules

`LowerGemminiScheduledMatmul(tile_i=1, tile_j=1)` is a separate opt-in pass for the same static signed-int8/int32 matmul contract. Each macro dimension admits 1, 2 or 4 array tiles. The mathematical TensorIR template retains operand tiles across multiple outputs; genuine `TensorIntrin` substitutions replace copies, first-chunk overwrite, subsequent accumulation and stores with the C primitives. TVM emits the surrounding loops and row addresses. Reductions stay within 16 products per signed20 PE operation, with global accumulation in int32 SRAM.

`gemmini_schedule.make_semantic_matmul`, `tensorize_gemmini_matmul` and `make_gemmini_matmul` expose semantic IR, tensorized IR, the actual substitution trace and resource/call metadata. The trace replays tensorization of the generated tiled template; it is not a split/cache/reorder trace from an untiled TE matmul. Fixed tail descriptors preserve exact extents. `LowerMatchBuffer` must run before default buffer compaction to retain physical 16×16 slot strides. The pinned compiler's optional cached-region debug verifier fails on a partial-update case; normal structural matching, sref checks, IR well-formed checks and numerical tests remain active.

Eleven native tests cover all nine tile choices, full/partial shapes, the maximum exact reduction, reuse, replay, rejection controls and surrounding graph optimization. Their independent C primitive emulator checks row bounds, initialization, operand rectangles and PE/accumulator overflow. For a 64×64×33 matmul, A and B loads each fall from 48 to 24 to 12 for 1×1, 2×2 and 4×4 choices, with identical computation and outputs. Counts are structural evidence, not timing. Run `python tests/python/relax/test_backend_contrib_gemmini_schedule.py` in the activated LLVM host environment.

Apply the pass before `LegalizeOps` and link the audited primitive adapter. Unsupported calls retain ordinary host lowering. The whole-call `LowerGemminiMatmul` remains the reference path. Arbitrary fused contractions, transfer/compute overlap and target-measured tuning remain future work.

## Graph optimization around Gemmini

`prepare_gemmini_graph(mod, tile_i=1, tile_j=1, optimize=True)` accepts semantic Relax IR. It decomposes eligible integer convolutions, folds constants and canonicalizes bindings before device substitution, then legalizes and fuses surrounding CPU operations. Gemmini functions remain opaque; their admission checks and primitive schedule are preserved. Legal TIR `compute_inline` removes internal pointwise temporaries while retaining returned buffers. The exporter rejects remaining TVM allocator dependencies rather than assuming a guest runtime.

`optimize=False` provides the scheduled/legalized baseline. Tests compare both paths against independent integer references through casts, bias, ReLU, clipping, shifts and tuple outputs, including extreme inputs and failure recovery. Frozen constants fold before accelerator IR exists; the optimized path rejects already device-lowered input because generic constant folding can CPU-evaluate `call_tir`. No affine quantization correction or through-device fusion is implied. The tested CPU graph shrinks from nine functions to two without changing its Gemmini body.

## Primitive convolution decomposition

`gemmini_conv.DecomposeGemminiConv2D()` admits positive static groups=1 NCHW/OIHW signed-int8 convolution with int32 output, supported CPU placement, nonnegative padding and positive stride/dilation. It creates compact CPU im2col and weight-packing operations, semantic Relax matmul and NCHW restoration. For `P=N*OH*OW`, `K=C*KH*KW` and `O=output_channels`, these tensors are `[P,K]`, `[K,O]` and `[P,O]`. The existing reduction and byte-span bounds apply. Unsupported operations and externally owned functions retain their original lowering.

The graph pipeline applies decomposition before constant folding so fixed weight packing can fold before device scheduling. The resulting matmul uses the existing TVM primitive schedule and no-FSM C interface. Raw integer padding is zero; bias, affine zero-point corrections, activation and requantization remain explicit operations. Full im2col/product materialization needs whole-graph memory admission. `get_conv2d_contract(call)` supplies admitted shapes and individual tensor sizes, not a peak-memory or model-quantization claim.

Run `python tests/python/relax/test_backend_contrib_gemmini_conv.py -v` in the activated environment. Six direct-script tests compare compiled semantics with independent int64 convolution over kernels 1/3/7, batches, rectangular/tail shapes, asymmetric padding, stride/dilation and extreme values. They also cover constant folding, limits and fallback. No separate convolution runtime or opaque vendor convolution is introduced.

## Bounded learned search

`gemmini_tuning.BoundedGemminiSearch(M, N, K, seed=0, compiler_id=..., adapter_id=...)` enumerates the nine legal macro-tile choices. TVM MetaSchedule's `PerStoreFeature` observes mathematical TIR before tensorization, and `XGBModel` learns rankings from accepted repeated measurements. Candidate identities bind semantic/device IR, trace, metadata and caller-supplied build identities; mutation invalidates the binding. This is a bounded template search, not a general schedule explorer.

Use `rank()` to obtain proposals and `schedule_for(candidate_id)` to obtain the bound schedule. Pass its tile metadata to `prepare_gemmini_graph`; the verifier checks the resulting device body matches exactly. `validate_semantics` checks independent CPU mathematics. `record_device_check` separately records the candidate's functional result on an identified platform. `update([TimingRecord(...)])` accepts a list of records with at least two finite positive seconds, consistent workload/build/platform/protocol identities and both correctness gates. These caller declarations enforce consistency, not authenticity. Updates retrain cumulatively and preserve the prior model/ledger if fitting or prediction fails; `manifest()` records provenance and checks.

Without measurements, rankings are seeded untrained proposals. Synthetic training requires `allow_diagnostic=True` and labels every resulting proposal diagnostic. Neither predictions nor functional Spike evidence establish a performance winner. Tested optional dependencies are `xgboost==1.7.6` and `pytest==8.3.5`; ordinary compilation and untrained proposals do not require them. Run `python -m pytest -c /dev/null -p no:cacheprovider tests/python/relax/test_backend_contrib_gemmini_tuning.py -q` in the activated environment. Eight tests cover distinct features, semantic/tail correctness, reproducible actual XGB updates, graph application and invalid provenance/mutations/failures. Genuine target measurements and selection remain pending.

## Primitive adapter numerical verification

The header also exposes CPU-only whole-operation validation and `begin`, `load_a`, `load_b`, `compute`, `store` and `end` primitives for compiler-generated schedules. TVM supplies plain scratchpad/accumulator row allocations, tile extents and reduction ordering; these wrappers only configure the device and emit vendor-library primitive instructions. Validate all user buffers before `begin`, keep A/B immutable through `end`, and prove the row/dimension contract in [matmul.h](matmul.h). Each compute restarts a reduction of at most 16 products in the signed20 PE; later chunks add into int32 accumulator SRAM. First-chunk overwrite and subsequent accumulation are distinct operations. The adapter verifier retains and audits every public primitive, including uncalled ones; never link the raw object containing unused vendor FSM code.

[verify_matmul_runtime.py](verify_matmul_runtime.py) links the audited adapter object into baremetal programs and executes them with an explicitly selected, source-bound Gemmini Spike plugin. It requires passing adapter and simulator build receipts, verifies their source/tool/header bindings before and after execution, and audits every executable section of the final ELFs for prohibited FSM instructions. The simulator builder lives in the comparison repository at `examples/gemmini/comparisons/tvm/build_simulator.py`.

```sh
python "$TVM_ROOT/apps/gemmini/verify_matmul_runtime.py" --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$SIMULATOR_BUILD/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$SIMULATOR_BUILD/libgemmini.so" --output-dir "$BUILD_ROOT/matmul-runtime-check"
```

Set each path to the selected build/tool installation and use a fresh output directory. The checked suite covers 13 valid calls including K=131071, partial tiles, padded strides, input preservation, output guards, overwrite and retained outputs, plus 18 invalid calls and recovery. Missing-extension and deliberately wrong-oracle controls must fail. The numerical oracle uses independent int64 arithmetic. This is functional qualification of the separate C-library adapter; it establishes neither TVM graph integration, RTL bit-exactness, platform coherence nor timing.

## Bounded baremetal CPU proof

[baremetal.py](baremetal.py) provides the reusable `plan_graph` and `export_graph` interfaces for straight-line, static Relax `call_tir` graphs. Optimize and legalize the graph before export. It derives raw-pointer calls, a single constants blob, caller-owned outputs and an aligned workspace reused after each tensor's last use. Integer, FP32, FP64, raw BF16 and bool tensors, multiple inputs and tuple outputs are supported; dynamic shapes, control flow and hidden scalar ABIs are rejected. Export writes the operator object, C orchestration, constant assembly/blob, IR and `graph.json` into the supplied output directory.

The generated `model_run(inputs, outputs, workspace, workspace_bytes)` validates alignment, byte spans and writable-buffer overlap before dispatch. The manifest's memory limit covers explicit tensors only; ELF sections, operator stack allocations and platform reservations must be added for deployment. Export rejects remaining TVM workspace allocation/free dependencies, including indirect callbacks that would otherwise link successfully but trap without runtime initialization. Fifteen native tests cover mixed dtypes, repeated calls, tuple/residual lifetimes, workspace reuse, fusion and ABI/allocator/budget rejection. BF16 uses raw uint16 buffers/constants with two-byte accounting; FP64 uses double buffers. Native checks cover BF16 payloads and ties-even cast boundaries, and FP64 values that distinguish narrowing to FP32. A bounded matched-Spike Gemmini/BF16/FP64 graph also passes these dtype boundaries, preservation and failure controls; complete-model qualification remains pending. In the activated host environment, run `python tests/python/relax/test_backend_contrib_gemmini_baremetal.py`.

`preflight_graph(mod, memory_limit_bytes=...)` reports tensor capacity before compilation or artifact creation. Its per-call logical liveness is separate from resident inputs/constants, caller output slots and the allocated aligned workspace. Duplicate returns share a logical value but require disjoint caller output storage. The export manifest includes these counts, output-copy bindings and the largest intermediates. Packing buffers and residual branches participate in ordinary graph lifetimes; the report does not account for hidden operator scratch, ELF contents or deployed DRAM.

`export_graphs({"prefill": prefill_mod, "decode": decode_mod}, output_dir, memory_limit_bytes=...)` packages several static stages with distinct C/operator symbols, one error runtime and one deduplicated constant pool. It writes a shared header and `graphs.json`; the caller owns state routing and disjoint stage input/output buffers. Serial calls may reuse workspace. The bound covers shared constants plus one invocation, so retained session state, ELF, stack and fixtures remain additional. Native tests link and alternate stages containing shared and private constants, verify source IR preservation and retained outputs, and reject insufficient budgets before artifact creation. This is packaging support, not full-session device qualification.

[verify_baremetal_cpu.py](verify_baremetal_cpu.py) compiles a fixed FP32 matmul/add/ReLU Relax graph into RISC-V CPU operations and derives static calls/constants from its legalized bindings. It links machine-mode startup, HTIF console/exit and caller-owned buffers without guest libc, libtvm, a C++ runtime or heap. Eight calls across five inputs must match an independent scalar oracle exactly, preserve input/constant/previous-output bytes and buffer canaries, and recover from a null-input rejection. A deliberately incorrect oracle must produce a failing target exit. Dynamic shapes, other dtypes, tuple outputs and unlegalized graphs are rejected.

This is an independent, narrow deployment proof, not a general Relax VM replacement or a complete-model exporter. Only this graph is numerically qualified. The verifier records generated IR, commands, tool/library/source identities, ELF memory bounds, one resident constant copy, reused workspace and a painted-stack high-water observation. It requests 256 MiB in generic Spike; that is not a statement about deployed DRAM. No Gemmini plugin or accelerator execution is involved.

In the activated host environment, from the comparison checkout, supply the actual installed tool paths:

```sh
TVM_ROOT="$PWD/third_party/baselines/tvm-gemmini"
BUILD_ROOT="$PWD/out/build/baselines/tvm-gemmini"
RISCV_BIN=/path/to/riscv-tools/bin
SPIKE_SUPPORT=/path/to/spike-dependencies
python "$TVM_ROOT/apps/gemmini/verify_baremetal_cpu.py" --tvm-source "$TVM_ROOT" --tvm-build "$BUILD_ROOT/host" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --output-dir "$BUILD_ROOT/baremetal-cpu-check"
```

The output directory must be new. Spike's selected library directory must provide its compatible `libstdc++.so.6`; the verifier supplies this path and the selected `dtc` to the Spike process without sourcing global Chipyard setup. Review the recorded limits before extending this proof to additional graphs or a real platform startup/transport.

## Scheduled graph in Gemmini Spike

[verify_graph.py](verify_graph.py) builds a Relax graph containing a 17×33 by 33×19 integer matmul, bias and ReLU. TVM generates the Gemmini schedule and CPU operations; the static exporter supplies calls, constants and workspace. The guest checks six repeated invocations against an independent int64 oracle, primitive call counts, input/constant preservation, guards, retained outputs and recovery after rejected buffers. Missing-extension and wrong-oracle controls must fail, and both final ELFs must contain no FSM instructions.

```sh
python "$TVM_ROOT/apps/gemmini/verify_graph.py" --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$SIMULATOR_BUILD/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$SIMULATOR_BUILD/libgemmini.so" \
  --tile-i 2 --tile-j 2 --output-dir "$BUILD_ROOT/scheduled-graph-check"
```

Use matching adapter/simulator receipts and a fresh output directory. Both 1×1 and 2×2 schedules pass: A/B loads each fall from twelve to six, while computation and outputs are unchanged. The receipt retains semantic/tensorized IR, a replayable substitution trace, tool/source identities, final ELF bounds and a painted-stack high-water observation. The requested generic Spike map is 256 MiB; this does not establish deployed capacity. No full-model, timing or actual-platform coherence qualification is implied.

Add `--graph-mode optimized` to fold constants and fuse CPU operations around Gemmini. The baseline and optimized fixtures both pass exact functional checks; calls shrink from three to two and explicit workspace from 2,688 to 1,344 bytes. Replace both tile options with `--search-seed 7` to execute an untrained search proposal through the optimized graph. This mode binds the candidate to the actual compiler/adapter and generated device IR, then records its functional simulator gate after all controls pass. It supplies zero timing labels and selects no performance winner; trained diagnostic proposal application is covered separately by the search tests.

Add `--workload conv-residual --conv-kernel 3` to the same command for a convolution/residual block; kernels 1 and 7 are also supported. The fixed N1/C17/H7/W9/O17 fixture uses same padding, channel bias, a derived residual, floor division by 512 and clipping to [0,127] before int8 output. A second int32 output exposes the raw convolution for exact checking. The independent guest oracle traverses original OIHW kernel coordinates, and each invocation poisons workspace and checks both outputs' guards and retention. This explicit integer diagnostic does not select ResNet quantization or quality policy.

Kernels 1/3/7 pass optimized execution, and kernel 3 passes both graph modes plus an untrained search proposal. The 3×3 block shrinks from 12 calls/22,528 workspace bytes to 4 calls/13,952 bytes. Receipts retain tensor preflight, final ELF/stack accounting, primitive counts and no-FSM/intentional-failure gates. Pretrained ResNet quality, deployment capacity/coherence and performance remain unqualified; the complete random-model functional check is described below.

## Complete exported graphs in Gemmini Spike

[verify_exported_graph.py](verify_exported_graph.py) consumes a complete static export and independently prepared numerical fixtures. It reuses the adapter/runtime bindings and final-ELF instruction audit; it never loads a model or generates expected answers. Its NPZ requires exactly `input_N`/`output_N` arrays with a common positive sample dimension followed by each manifest tensor's exact shape and storage dtype. BF16 manifests require raw uint16 words; FP64 requires float64 without narrowing. Nonfinite FP32/FP64/BF16 payloads are refused. The provenance JSON binds `fixture_sha256`, ordered unique `sample_ids`, a nonempty `reference_contract` and `reference_sources` file identities (`path`/`sha256`). These declarations identify evidence; they do not authenticate the caller's numerical oracle.

```sh
python "$TVM_ROOT/apps/gemmini/verify_exported_graph.py" --graph-dir /absolute/path/to/export \
  --fixture /absolute/path/to/reference.npz --fixture-provenance /absolute/path/to/reference.json \
  --tvm-source "$TVM_ROOT" --tvm-build "$TVM_BUILD" \
  --adapter-receipt "$ADAPTER_BUILD/receipt.json" --simulator-receipt "$SIMULATOR_BUILD/receipt.json" \
  --riscv-gcc "$RISCV_BIN/riscv64-unknown-elf-gcc" --spike "$RISCV_BIN/spike" --dtc "$SPIKE_SUPPORT/bin/dtc" \
  --spike-library-dir "$SPIKE_SUPPORT/lib" --plugin "$SIMULATOR_BUILD/libgemmini.so" \
  --repeats 2 --timeout-seconds 7200 --output-dir "$BUILD_ROOT/exported-graph-new-run"
```

The verifier checks every sample on each repeat against frozen little-endian output bytes. It validates typed manifest extents, binds generated fixture blobs, checks their linked ELF symbols, poisons workspace/output padding, retains previous outputs, preserves inputs/constants and counts primitive calls from the selected TIR schedules. Missing-extension and deliberately wrong-oracle controls must fail. Focused admission tests and bounded mixed-output simulator checks pass, including BF16 nearest-even ties and FP64 increments below FP32 precision.

For a package from `export_graphs`, pass its root as `--graph-dir` and add `--session-route /absolute/path/to/route.json`:

```json
{"version": 1, "calls": [{"stage": "prefill", "inputs": [{"fixture": 0}]}, {"stage": "decode", "inputs": [{"fixture": 1}, {"output": [0, 0]}]}]}
```

Routes name contiguous external `input_N` fixtures or an earlier invocation/output with identical dtype and shape; they perform no conversion. Every packaged stage must execute. Flatten all invocation outputs in call order into the frozen `output_N` streams, including state and unused returns. Each sample/repeat starts fresh, retains every output, shares constants and reuses guarded workspace. Per-stage preservation/count checks supplement the final-image and failure controls. The 64-stage/64-call/256-tensor bounds permit prefill plus 32 growing-cache signatures. Merlin's verified semantic-session adapter can generate the routes and source fixtures; this driver checks their frozen bytes independently.

The default functional map is 256 MiB. `--guest-memory-bytes` selects an explicit page-aligned RV64 span used consistently for session/trace admission, the linker, ELF checks, Spike and receipt. Guarded retained outputs and frozen streams consume more memory than a production session. A requested simulator map does not establish deployed DRAM capacity or complete-model fit.

Graphs requiring CPU math may explicitly repeat `--cpu-runtime-library /absolute/path/to/library.a` for the selected toolchain's libm, libc and libgcc dependencies. No library is selected by default. The verifier binds ordinary archive bytes/ABI, enables section garbage collection, distinguishes extracted members from retained sections and audits the final RV64GC/lp64d image for unsupported ISA, TLS, constructors and heap/file callbacks. The caller must select libraries compatible with this single-hart startup; these checks do not prove arbitrary library initialization semantics. The reviewed static errno state is valid under the ELF loader. A bounded generated math/Gemmini identity smoke passes; an independent nontrivial Decimal fixture retains an exact-byte failure at sinf(1), one ULP from correctly rounded output. Neither proves full-model or general libm accuracy.

A complete optimized random-weight ResNet50 v1.5 graph passes two distinct 224×224 images, each repeated twice, with all controls. It uses 34,679,680 bytes of explicit tensors; the linked diagnostic image reserves 35,065,808 bytes and observes 720 stack bytes within a reserved 16 KiB. The full receipt precedes the final fixture-symbol admission refinement, which is covered by the bounded rerun. These random-model receipts establish no pretrained quality, RTL/platform coherence or deployed capacity. Simulator wall time supplies neither device timing nor learned-search labels.

The model order is canonical ResNet50, TinyLlama 1B, then SmolVLA. Prove small matmul and convolution cases using the selected target's arithmetic before full-model device execution. The proposed signed-int8/int32 contract requires matching hardware and generated headers; it does not apply to a floating-point configuration. Keep model inputs and quality thresholds fixed through performance tuning; functional simulator evidence and qualified timing evidence serve distinct roles. Final comparison timing comes from FireSim, with platform and capture details coordinated by the comparison team. Keep full-model inputs, comparisons and measurement orchestration in the parent comparison repository. Build products, environments, model weights, captures and outputs stay outside this source tree, using the parent's configured `out/build/baselines/tvm-gemmini/` and other `out/` roots.
