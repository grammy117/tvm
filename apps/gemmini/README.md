# Gemmini TensorIR integration

This worktree simplifies TVM `42b3c1ef2` on branch `gemmini/direct-intrinsics`.
It is a development snapshot, not a hardware-qualified release. The companion
Merlin `examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md` owns the campaign,
artifact retrieval and readiness status.

The selected model protocol is
[Merlin 8ec99308](https://github.com/ucb-bar/merlin/tree/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol).
Keep its numerical recipes, independent references and complete-chain checks.
TinyLlama uses the approved 32-token prefill + 32 decode chain; its tuning probes
remain 16/1 rows. Compiler smoke tests do not establish full-model correctness.

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
capacity/overflow/rounding checks and required memory synchronization. Composite
intrinsic scope is under review; existing matchers do not establish software-spec
admission. No model name or expected output chooses an optimization.

## Entry points

| Tool | Purpose |
| --- | --- |
| `tune_models.py --program PROGRAM --extract-only` | Extract actual prepared tasks and report exclusions. |
| `tune_models.py --program PROGRAM` | Stock MetaSchedule with the selected measurement adapter. |
| `compile_package.py` | Compile an archive, constants, ABI, placement and memory manifest; no implicit tuning. |
| `qualify_package.py` / `verify_tensorir.py` | Independent positive/wrong-oracle guests and functional results. |
| `tuning_library.py` | Validate measurement identities and merge structurally compatible records. |
| `tune_chunk.py` | Compare connected graph plans; its existing local runner is not the FireSim runner. |

[OWNERSHIP.md](OWNERSHIP.md) defines stock versus supplied code.
[TUNING.md](TUNING.md) defines schedule choices and cost-model behavior.
[FIRESIM.md](FIRESIM.md) defines build/handoff boundaries.

Use an isolated matching Python/native TVM environment, single-threaded BLAS and
bounded build workers. Verify `tvm.__file__` and the loaded `libtvm` identity.
Run relevant backend/application tests. Host emulation, Spike functional checks
and FireSim cycle observations are separate evidence tiers.

The legacy graph/template/bounded-search implementations are outside this path.
Their historical inventory and results remain at
[the predecessor source](https://github.com/grammy117/tvm/tree/42b3c1ef27abe878f930e48b335b24956beb0e56/apps/gemmini).
