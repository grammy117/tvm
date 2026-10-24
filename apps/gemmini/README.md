# Gemmini TensorIR compiler and EL4 handoff

Current user profile: full **ResNet50, TinyLlama and SmolVLA** under the
[EL4 protocol](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml).
One selected `visible/v00` per model is sufficient for the functional benchmark.
ResNet is one image/one call; TinyLlama is **32-token prefill + 32 teacher-forced
decode calls**; SmolVLA is prefix + ten denoise calls + action decode. Repetitions
reset state. Calibration data prepares scales once; ResNet's 260 calibration
images are not 260 benchmark passes. The old numerical bundle is needed only for
exact historical reproduction; a fresh release needs its own shared bytes,
hashes and independent references. Full-model numerical/FireSim gates remain open.

This integration compiles semantic Relax programs through TensorIR into one-hart
RV64GC code issuing primitive Gemmini commands. The supported tuning path uses
stock TVM MetaSchedule and its JSONDatabase. Hardware matmul/conv loop engines
are forbidden. Primitive OS and WS schedules, checked software pipelining,
rectangular tails, SRAM reuse, bias preload and proved readout fusion are present.

The workload-tuning branch adds explicit FASED latency-pipe measurements,
fixed one-warmup/one-measured timing, and stock Relax graph-choice traces for
fusion level, constant packing, bias preload and common-subexpression elimination
(32 generic alternatives; the measured subset follows the experiment budget).
Per-operator MetaSchedule records remain
reusable through structural matching; connected-chunk timing selects graph
choices separately for each workload/phase. This does not add cross-function
SRAM residency. See [WORKLOAD_TUNING.md](WORKLOAD_TUNING.md) and the companion
Merlin `FASED_SWEEPS.md` for the parallel three-workload campaign and its limits.

Start with:

- [OWNERSHIP.md](OWNERSHIP.md): complete lowering responsibilities, custom passes,
  optimization ownership chart and audit of changes from Apache TVM v0.19.0.
- [WORKLOAD_TUNING.md](WORKLOAD_TUNING.md): per-workload operator/connected chunks,
  current knob coverage, memory audits and later isolated Verilator campaigns.
- [TUNING.md](TUNING.md): search space, stock versus supplied components,
  measurement contracts, real-program intake and next tuning work.
- [FIRESIM.md](FIRESIM.md): package ABI, functional gates, hardware binding and
  the remaining full-model deployment work.
- [AGENTS.md](AGENTS.md): instructions for the agent continuing this work.
- [Merlin comparison handoff](https://github.com/ucb-bar/merlin/blob/comparisons/tvm-el4-handoff/examples/gemmini/comparisons/tvm/README.md):
  per-workload preparation, shared numerical release, session contracts and
  direct FireSim integration.

## Entry points

| Tool | Purpose | What it does not establish |
| --- | --- | --- |
| `tune_models.py --program … --extract-only` | Inventory all tasks of a real `.pt2` or semantic Relax JSON program | Compilation, model quality or device execution |
| `tune_models.py --program …` | Measure eligible operators and tune CPU/Gemmini placement | Complete session latency or correctness |
| `tune_models.py --model …` | Synthetic model-shaped operator study | Actual checkpoint execution |
| `tune_chunk.py` | Measure compiler graph alternatives on a complete connected chunk with confirmed FASED memory | Cross-call SRAM residency or complete-model qualification |
| `tuning_library.py` | Validate fresh record evidence, merge compatible operator databases, report structural coverage | A matching operator record does not establish a connected-graph speedup |
| `compile_package.py` | Static archive, constants, ABI, placement and memory manifests; optional standard tuning database or shipped preset (`rtl-v1`, `fased-v1`) with an ordered fallback | Final platform boot or timing |
| `qualify_package.py` | Link archive with independent fixture and positive/wrong-oracle guests | FireSim deployment readiness |
| `qualify_tuning_package.py` | Check a representative operator linked from the delivered archive | All operators or full-model quality |
| `verify_tensorir.py run` | Functional Spike or selected Verilator execution | FireSim performance |
| `report_tuning.py`, `retry_capped.py`, `complete_baselines.py`, `merge_comparison_records.py` | Report or repair retained measurements using standard records | A new proposal/search algorithm |

`tune_tensorir.py` is a compatibility alias for `tune_models.py`. The retired
`--tuning-result` interface is unsupported; compilation takes
`--tuning-database DIRECTORY`. Do not use `gemmini_tuning.py`'s historical bounded
controller for new work. `prepare_gemmini_graph_legacy`, `gemmini_schedule.py`
and the old BYOC snapshot remain regression references, outside the default path.

For this handoff, pass **`--tuning-preset rtl-v1`** to static compilation.
The checked-in preset contains all nine verified synthetic-operator winners as
stock MetaSchedule records. It verifies capabilities/payloads, then lets stock
TVM apply structural matches. `placement.json` and `package.json` report actual
coverage; zero matches is allowed for this preset and means defaults remain
unmeasured. Explicit `--tuning-database` still rejects zero matches. Even with the
selected 32-token prefill, quantization/epilogue/IR differences can prevent the
32-row winner from matching. Inspect coverage; do not assign winners by model
name. See TUNING.md.

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

Current EL4 already implements native complete-chain images, compact comparison
of every output element, strict version-pinned upload/manager tools and a
lightweight actual-boot-file verification mode. TVM's package-session bridge has
different ABI/manifest/UART contracts and needs an explicit adapter to use these
paths. It provides aggregate warm/measured timing; the companion Merlin
session builder adds optional buffered stage timing via `--profile-stages`. The current audit passes 218 TVM backend/CLI tests. Companion model tests
also expose AVX512 environment requirements and a LeRobot/diffusers/TorchAO
dependency conflict; see the Merlin audit rather than assuming a fully validated
three-model environment. No full-model FireSim result is inferred.

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
