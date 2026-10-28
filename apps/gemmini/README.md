# Gemmini TensorIR integration

The active compiler is branch `gemmini/direct-intrinsics`, derived from TVM
v0.19.0 with inherited correctness fixes and the Gemmini integration.
It is a development snapshot, not a hardware-qualified release. The companion
Merlin `examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md` owns the campaign,
artifact retrieval and readiness status.

The selected model protocol is
[Merlin 8ec99308](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol).
Keep its numerical recipes, independent references and complete-chain checks.
The active campaign is ResNet50 plus 16 isolated kernel cases. Full TinyLlama
and SmolVLA are backlogged. Compiler smoke tests do not establish model quality.

## One active compiler path

```text
Relax mathematics -> stock legalization/fusion -> mathematical TensorIR
  -> stock MetaSchedule over Gemmini schedule rules
  -> TensorIntrin -> SRAM allocation and DMA synchronization
  -> LLVM RV64 objects -> static package/session -> measured FireSim guest
```

Removed: automatic bias relocation, constant weight prepacking, scale
specialization, custom readout normalization, zero-init elimination and DMA
command coalescing. Bias/scale epilogues retain source semantics. Explicit
accumulator initialization and compatible exact readout still tensorize.
Old graph plans requesting removed transforms refuse; re-extract tasks instead
of changing a saved plan's meaning.

Keep ordinary tiling/layout/cache schedules, exact intrinsic descriptions,
capacity/overflow/rounding checks and required memory synchronization. The user approved retaining only declared, proved composite TensorIntrins;
existing matchers do not establish software-spec admission. No model name or expected output chooses an optimization.

## Entry points

| Tool | Purpose |
| --- | --- |
| `tune_models.py --program PROGRAM --extract-only` | Extract actual prepared tasks and report exclusions. |
| `tune_models.py --program PROGRAM` | Stock MetaSchedule with the selected measurement adapter. |
| `compile_package.py` | Compile an archive, constants, ABI, placement and memory manifest; no implicit tuning. |
| `qualify_package.py` / `verify_tensorir.py` | Independent positive/wrong-oracle guests and functional results. |
| `tuning_library.py` | Validate measurement identities and merge structurally compatible records. |
| `tune_chunk.py` | Compare connected graph plans; its existing local runner is not the FireSim runner. |

[OWNERSHIP.md](OWNERSHIP.md) defines stock versus supplied code, including the
remaining placement, memory and host-inlining orchestration. The companion Merlin
[architecture and paper scope](https://github.com/ucb-bar/merlin/blob/comparisons/tvm-workload-tuning/examples/gemmini/comparisons/tvm/ARCHITECTURE.md)
is the top-down overview; its operation inventory separates implemented candidates
from software-spec and deployed-hardware qualification.
[TUNING.md](TUNING.md) defines schedule choices and cost-model behavior.
[FIRESIM.md](FIRESIM.md) defines build/handoff boundaries.

Use an isolated matching Python/native TVM environment, single-threaded BLAS and
bounded build workers. Verify `tvm.__file__` and the loaded `libtvm` identity.
Run relevant backend/application tests. Host emulation, Spike functional checks
and FireSim cycle observations are separate evidence tiers.

The legacy graph/template/bounded-search implementations are outside this path.
[LEGACY.md](LEGACY.md) lists their remaining consumers and the lazy compatibility
boundary; normal preparation does not import `gemmini_legacy.py`.
Their historical inventory and results remain at
[the predecessor source](https://github.com/grammy117/tvm/tree/42b3c1ef27abe878f930e48b335b24956beb0e56/apps/gemmini).

Full-domain INT32 add (including explicit modulo-32 and scalar forms) has no
active TensorIntrin in this revision. Its original fixtures keep their bytes
and host semantics. Bounded i8-source affine arithmetic remains a separate
contract. Affine source ranks stop at five; per-operand scaled sums require
rank two. These matcher restrictions do not replace selected-scale screens,
emitted-command audits or hardware qualification.

Unbounded standalone i32 readout is excluded from byte-affine admission;
contraction readout retains its separate producer/epilogue proof. Affine
metadata reports the conservative bound for every accumulator prefix and
weighted K=1 mesh partial. These bounds do not constitute hardware qualification.

The strict matcher admits contraction and pooled-readout result ranks 2–4;
standalone movement/affine/sum ranks stop at five. Rank-five 3D convolution
retains host mathematics. Byte sums require `128*K <= INT32_MAX`; accepting
`-2**31` solely because it fits the dtype would violate the spec's absolute
partial bound. These checks apply before ordinary schedule construction.
