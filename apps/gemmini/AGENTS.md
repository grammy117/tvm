# Agent handoff: TVM Gemmini and EL4

Current campaign: ResNet50 plus the 16 cases in the companion Merlin
KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md. Full TinyLlama/SmolVLA
are backlogged; their full-model recipes below are retained future work.

Read README.md, OWNERSHIP.md, TUNING.md and FIRESIM.md before making changes.
The sole campaign protocol is Merlin 8ec99308 (superseding older protocol links
below). The current selection is full ResNet50, TinyLlama **32-token prefill plus 32
teacher-forced decode calls**, and SmolVLA's prefix/ten-denoise/action-decode
chain. Use the specified EL4 recipes, one selected `visible/v00` per model and
explicit warm/reset/measured repetitions. Prepare calibration once, then publish
shared numerical bytes and independent references. A previous capture is needed
only for exact historical identity. Direct FireSim transport is the current
workhorse path. This source is a compiler/tooling handoff, not evidence that
those full-model runs already passed.
The simplified compiler removes automatic bias/prepacking/scale/readout preparation.
Re-extract tasks and rebuild packages. Historical presets may seed structurally
matching traces, but do not promise coverage or measured winners for this source.

The follow-up in `WORKLOAD_TUNING.md` and Merlin `FASED_SWEEPS.md` uses explicit
latency-pipe Verilator receipts and fixed-warm timing. Keep its fresh databases
separate from historical magic-memory labels. `tune_chunk.py` enumerates stock
Relax graph traces; `tuning_library.py` validates actual record/trace/ELF evidence
before reuse. Preserve per-phase graph choices and partial coverage. Cross-function
SRAM residency remains unsupported; chunk passes do not resolve the historical
complete TinyLlama numerical failure or qualify a FireSim deployment.

## Source and evidence

- Apache upstream is v0.19.0 `c4dc0c29ff81ddae688da24625a603da3b4a2c0e`.
  Do not omit the inherited five ONNX fixes by starting at `fb78e0e3`.
- Historical qualified operator source is `87f07883b65ee55dbd1b93f665aaab87cd8a1e50`.
  Curated history preserves it in the original branch; source-equivalent history
  cleanup does not change a measurement's original attribution.
- Merlin integration is based on EL4 `0cc289de52a10bed256f32c90cfbb54a21b3c81f`.
  Read its root/local AGENT.md and comparison AGENTS.md. Do not alter frozen
  Phase 1 artifacts, campaign deadlines, hidden answers or admission contracts.
- Use separate worktrees. Original Merlin and `tvm-tensorir-source` have saved
  unrelated/uncommitted work. Do not reset, delete or sweep them into a commit.
- Generated artifacts belong under Merlin's configured `out/`, created with its
  artifact helpers. Use new roots; preserve old failures, logs and controls.

## Compiler invariants

- Preserve mathematical IR until the normal schedule/tensorize path. No model
  name, expected output or captured provenance ID may choose an optimization.
- Hardware loop FSM commands remain forbidden in the **whole final ELF**. Raw
  command software loops, primitive OS/WS, legal buffering and reuse are allowed.
- Keep capacity, signed-width, overflow, NaN/rounding, alias and DMA-completion
  proofs. Unsupported cases must retain CPU semantics or refuse explicitly.
- Use stock `extract_tasks`, `tune_tasks`, ReplayTrace (no learned model),
  JSONDatabase and MetaScheduleApplyDatabase. Do not revive the old
  custom result format/controller. New search decisions must live in traces.
- Tune and compile the same prepared IR and capability target. Inspect every
  function's placement; a partial database does not mean full coverage.
- Verify loaded Python and native TVM library paths. Rebuild the native library
  for any C++ change; do not test new Python against a stale ABI by accident.
- Source-specific ONNX arithmetic is explicitly versioned opt-in behavior in
  `_source_fidelity.py`; it is not default ONNX semantics or an accuracy waiver.

## Next-agent execution order

1. Reopen the handoff artifact's refs, file audit, tests and remaining gates.
2. Read the [selected EL4 contract](https://github.com/ucb-bar/merlin/blob/8ec99308cb1182b9b896a6075300312266fadf6f/examples/gemmini/protocol/experiment.yaml)
   and companion shared-assets guide. Model revisions, inputs, calibration,
   recurrence and host-math choices are already specified. Finish/publish their
   portable prepared payloads. Older factory recipes and synthetic cutouts do
   not replace selected EL4 capture programs.
3. Run real-program extraction; account for all unsupported CPU tasks and each
   session stage, cache update, denoising step and reset.
4. Compile and qualify real stages using independent fixtures. Resolve complete
   SmolVLA numerical failures before performance claims.
5. Use Merlin's explicit package-ABI/session bridge and existing runtime APIs.
   Select absolute weights/arena placement for large maps and preserve operator
   objects/constant bytes when recompiling the wrapper. Small near/high-address
   controls pass; verify actual model memory/stack and every final ELF instruction.
6. Adapt TVM's session manifest/UART to the current EL4 chain transport with real
   package/source/route/reference bindings. The native retained-object chain
   builder already exists, but its memref ABI is different from TVM's pointer
   array ABI. Do not rename schemas or invent native build receipts. Use the companion
   session builder's `--profile-stages` alongside aggregate warm/measured timing;
   record counter overhead and keep final uninstrumented comparisons.
7. Bind matching FireSim driver/image/loader identities and the full memory range.
   Follow FIRESIM.md: strict staging/manager installation, lightweight boot-file
   evidence or legacy full-array transport are distinct modes. The supplied
   strict publisher and lightweight mode require compact all-element comparison.
   Keep small positive/wrong-oracle controls on functional engines; paid FPGA
   runs measure the complete selected kernel case or full model, as declared. Report every output/state check and timing scope.
8. Tune only after correctness and deployment gates pass. Preserve default/CPU
   baselines, failed caps, pure-search minima, record provenance and replayability.

## Validation and reporting

Run relevant frontend/TIR/backend/CLI tests for changed behavior. Retain source
hashes and exact commands. Host/emulated checks, Spike functional checks, local
Verilator cycles and FireSim observations are different evidence tiers. A trained
XGB booster is not proof of post-training ranking; a zero exit status is not
complete output validation; header agreement is not whole-SoC equivalence.

Do not force-push or rewrite existing published history. Curated branches are
reviewable deliveries. No new PR is authorized. A future actual-model run needs
shared numerical payloads, the selected manager access and hardware receipts.
Report exact missing artifacts while continuing independent compilation/inventory work.
