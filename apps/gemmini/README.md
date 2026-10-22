# Gemmini TensorIR compiler and EL4 handoff

This integration compiles semantic Relax programs through TensorIR into one-hart
RV64GC code issuing primitive Gemmini commands. The supported tuning path uses
stock TVM MetaSchedule and its JSONDatabase. Hardware matmul/conv loop engines
are forbidden. Primitive OS and WS schedules, checked software pipelining,
rectangular tails, SRAM reuse, bias preload and proved readout fusion are present.

Start with:

- [OWNERSHIP.md](OWNERSHIP.md): complete lowering responsibilities, custom passes,
  optimization ownership chart and audit of changes from Apache TVM v0.19.0.
- [TUNING.md](TUNING.md): search space, stock versus supplied components,
  measurement contracts, real-program intake and next tuning work.
- [FIRESIM.md](FIRESIM.md): package ABI, functional gates, hardware binding and
  the remaining full-model deployment work.
- [AGENTS.md](AGENTS.md): instructions for the agent continuing this work.
- Merlin's `examples/gemmini/comparisons/tvm/README.md`: actual model factories,
  source/reference gates, EL4 session contracts and queue workflow.

## Entry points

| Tool | Purpose | What it does not establish |
| --- | --- | --- |
| `tune_models.py --program … --extract-only` | Inventory all tasks of a real `.pt2` or semantic Relax JSON program | Compilation, model quality or device execution |
| `tune_models.py --program …` | Measure eligible operators and tune CPU/Gemmini placement | Complete session latency or correctness |
| `tune_models.py --model …` | Synthetic model-shaped operator study | Actual checkpoint execution |
| `compile_package.py` | Static archive, constants, ABI, placement and memory manifests; optional standard tuning database | Final platform boot or timing |
| `qualify_package.py` | Link archive with independent fixture and positive/wrong-oracle guests | FireSim deployment readiness |
| `qualify_tuning_package.py` | Check a representative operator linked from the delivered archive | All operators or full-model quality |
| `verify_tensorir.py run` | Functional Spike or selected Verilator execution | FireSim performance |
| `report_tuning.py`, `retry_capped.py`, `complete_baselines.py`, `merge_comparison_records.py` | Report or repair retained measurements using standard records | A new proposal/search algorithm |

`tune_tensorir.py` is a compatibility alias for `tune_models.py`. The retired
`--tuning-result` interface is unsupported; compilation takes
`--tuning-database DIRECTORY`. Do not use `gemmini_tuning.py`'s historical bounded
controller for new work. `prepare_gemmini_graph_legacy`, `gemmini_schedule.py`
and the old BYOC snapshot remain regression references, outside the default path.

## Qualification boundary

The completed operator study at original source `87f07883b65ee55dbd1b93f665aaab87cd8a1e50`
retains four static packages, representative positive/negative Spike and RTL
checks, and nine recorded task winners including two smoke tasks. Its source
bundle and raw measurement receipts remain authoritative for those results.
The curated branch groups the final implementation into reviewable commits;
changed commit IDs do not relabel historical hardware evidence.

Those packages use representative model-shaped programs. They are **not** full
ResNet50, TinyLlama or SmolVLA FireSim runs. In particular, a complete SmolVLA
16-layer diagnostic previously failed its floating gate, and matching a generated
header does not establish matching FireSim driver or whole-SoC behavior.
The handoff preserves those limitations and specifies their remaining gates.

## Build and verify

Use the LLVM/Python/toolchain versions recorded by the existing receipts or build
an isolated matching environment. Set `PYTHONPATH=$TVM/python` plus required
Python dependencies, `TVM_LIBRARY_PATH` to the matching build and `OMP_NUM_THREADS=1`.
Do not load a system TVM wheel alongside this fork. Verify `tvm.__file__` and
`tvm._ffi.base._LIB._name` before compiling. Initialize upstream submodules when
building a fresh checkout. The host-only CMake `USE_HOST_ONLY_AUTO_COPY_GUARD`
is an opt-in refusal implementation for builds missing automatic-copy lowering;
it must not be enabled alongside the normal implementation.

Run the focused tests under `tests/python/relax/test_backend_contrib_gemmini*`,
`test_apps_gemmini*`, the affected frontend tests and the arithmetic/TIR/search
regressions listed in the audit. Host tests and emulator tests do not replace
Spike, RTL or FireSim gates. Every performance claim must name the hardware,
ELF, inputs, oracle, compiler, adapter and measured interval.

The earlier narrative is retained in [README_HISTORICAL.md](README_HISTORICAL.md),
[EL4_STATUS_HISTORICAL.md](EL4_STATUS_HISTORICAL.md) and
[REVIEW_ACTIONS_HISTORICAL.md](REVIEW_ACTIONS_HISTORICAL.md).
[TENSORIR.md](TENSORIR.md) contains detailed development notes, including superseded
checkpoints; use the documents above as the current runbook.
