# Lowering and optimization ownership

This document describes the simplified `gemmini/direct-intrinsics` worktree.
The upstream reference remains TVM v0.19.0 `c4dc0c29ff81ddae688da24625a603da3b4a2c0e`.
Inherited frontend/TIR correctness fixes remain; this is not an unmodified
upstream binary. See [README.md](README.md) for status and the selected protocol.

## Active layers

| Layer | Stock TVM | Supplied integration |
| --- | --- | --- |
| Import and mathematical IR | Relax/TIR representation and frontend infrastructure | Inherited dtype, quantization, rounding and scheduling correctness fixes; explicit opt-in source arithmetic recipes. |
| Graph preparation | LegalizeOps, pattern annotation, FoldConstant, CanonicalizeBindings, FuseOps/FuseTIR, reshape rewrite, DCE; optional CSE | Pipeline selection and structural workload annotation. |
| Extraction | extract_tasks and call-count weights | Exact-family recognition and explicit placement/exclusion report. Recognition alone is not spec admission. |
| Search space | Schedule API, trace, sampling, mutations, union of generators | Gemmini contraction/secondary-family rules, capability limits and single-hart CPU settings. |
| Search/model | EvolutionarySearch, gradient task scheduler, XGBModel, PerStoreFeature | Bounded search settings and pre-postprocessing feature replay. |
| Tensorization | TensorIntrin registration/matching, RewriteTensorize | Mathematical descriptions, implementations and exact legality proofs. |
| Physical memory | Storage scopes and software-pipeline transformation | SRAM lifetime/address allocation, bounds/alias checks, host-DMA hazard waits and completion. |
| Machine code | LLVM RV64 | Primitive C command adapter; no hardware matmul/conv loop FSM. |
| Records | JSONDatabase and MetaScheduleApplyDatabase | Intrinsic registration on load, provenance checks and measured-baseline fallback policy. |
| Build/run | Builder/Runner/callback interfaces | Static ELF build, independent fixtures, FireSim transport and measurement evidence. |
| Complete model | Imported stage functions | Merlin numerical payloads, static session ABI, state/reset, graph buffers, loader and independent full-chain checks. |

## Removed custom optimization machinery

- `CanonicalizeGemminiBias` and bias-layout/algebraic folding: deleted. Preserve
  post-reduction bias; tensorize an explicit input-IR initialization when present.
- `SpecializeGemminiReadoutScales`: deleted. Frozen scale arrays retain their
  source representation; no replacement by a scalar merely to fit an intrinsic.
- `prepack_constants` / `gemmini_layout.py`: deleted. Source weights remain intact;
  required layout/packing work is explicit in the TensorIR schedule/runtime cost.
- `NormalizeGemminiReadout`: deleted. Compatible readout is recognized/proved
  during traced scheduling, without a separate mathematical preparation pass.
- Memory peepholes merging DMA and replacing zero-init with first-K overwrite:
  deleted. Required allocation and synchronization remain.

Preparation still returns its historical three-tuple for caller compatibility;
its two auxiliary dictionaries are empty. There is no alternate mathematical
reference or normalization restoration path. Requests for bias preload or
prepacking refuse explicitly.

## Target code that remains necessary

`gemmini_intrin.py` owns the primitive tile descriptions/implementations.
`gemmini_sketch.py` owns contraction schedules; `gemmini_meta_schedule.py` exposes
them to stock search and verifies memory legality. `gemmini_tensorir.py` applies
matching records/defaults and reports placement. `gemmini_memory.py` maps scoped
buffers to physical rows and orders accesses. `gemmini_bias.py` now contains only
explicit initialization matching/registration/scheduling. `gemmini_epilogue.py`
proves exact readout and schedules it; it contains no custom graph pass.

A TensorIntrin may issue several primitive commands. Keeping an affine or
sum-via-ones family requires its mathematical matcher, command implementation,
extra-memory accounting and exact software-spec proof. It does not require
restoring the removed graph transformations. Composite coverage is pending scope
review; do not advertise arbitrary FP32/wide-integer arithmetic or standalone
pooling based on similarity. The companion Merlin strict operation audit owns
that admission review.

Runtime configuration shadowing and compatible WS tile retention remain in
`matmul.c`. They are fixed runtime behavior, not independent MetaSchedule knobs.
Cross-function accelerator SRAM residency is not implemented. The static graph
exporter and package/session ABI are custom integration, not stock VM/AOT export.

## Legacy boundary and evidence

`gemmini_tuning.py`, `gemmini_schedule.py`, `gemmini_conv.py` and the explicit
legacy graph entrypoint retain consumers in older verification tools/tests.
They are not the current search algorithm. Migrate those consumers before
removal; never relabel historical receipts as results of the simplified source.

The complete predecessor operator inventory, primitive ABI and upstream-fix audit
are retained in [the source-pinned ownership record](https://github.com/grammy117/tvm/blob/42b3c1ef27abe878f930e48b335b24956beb0e56/apps/gemmini/OWNERSHIP.md).
The companion Merlin OPTIMIZATION_KNOBS.md records this cleanup's tests and
remaining cost-model/runner decisions. Neither a recognized pattern, a passing
emulator fixture nor a compiled archive proves full-model FireSim readiness.
