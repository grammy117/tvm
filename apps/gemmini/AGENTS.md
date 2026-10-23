# Agent handoff: TVM Gemmini and EL4

Read README.md, OWNERSHIP.md, TUNING.md and FIRESIM.md before making changes.
The current user selection is full ResNet50 and full TinyLlama **16-token prefill
plus one decode**, using EL4 recipes with fresh calibration/reference hashes.
SmolVLA and exact reproduction of the old frozen run are outside this preliminary
handoff. The execution agent runs the selected queued FireSim device. This source
is a compiler/tooling handoff, not evidence that those runs already passed.
Select `--tuning-preset rtl-v1` for static compilation and inspect actual structural
coverage; the preset cannot promise measured winners for every new EL4 operation.

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
- Use stock `extract_tasks`, `tune_tasks`, EvolutionarySearch/ReplayTrace,
  XGBModel, JSONDatabase and MetaScheduleApplyDatabase. Do not revive the old
  custom result format/controller. New search decisions must live in traces.
- Tune and compile the same prepared IR and capability target. Inspect every
  function's placement; a partial database does not mean full coverage.
- Verify loaded Python and native TVM library paths. Rebuild the native library
  for any C++ change; do not test new Python against a stale ABI by accident.
- Source-specific ONNX arithmetic is explicitly versioned opt-in behavior in
  `_source_fidelity.py`; it is not default ONNX semantics or an accuracy waiver.

## Next-agent execution order

1. Reopen the handoff artifact's refs, file audit, tests and remaining gates.
2. Recover actual model/data/reference contracts from Merlin's factories and
   selected EL4 sources. Do not substitute synthetic operator shapes/cutouts.
3. Run real-program extraction; account for all unsupported CPU tasks and each
   session stage, cache update, denoising step and reset.
4. Compile and qualify real stages using independent fixtures. Resolve complete
   SmolVLA numerical failures before performance claims.
5. Use Merlin's explicit package-ABI/session bridge and existing runtime APIs.
   Select absolute weights/arena placement for large maps and preserve operator
   objects/constant bytes when recompiling the wrapper. Small near/high-address
   controls pass; verify actual model memory/stack and every final ELF instruction.
6. Recover matching FireSim driver/image/loader identities and probe the required
   memory range. Validate queued positive and wrong-oracle small controls.
7. Run full sessions through the existing checkpoint/receipt path. Report quality,
   complete output/state checks, peak memory and measured cycle boundaries.
8. Tune only after correctness and deployment gates pass. Preserve default/CPU
   baselines, failed caps, pure-search minima, record provenance and replayability.

## Validation and reporting

Run relevant frontend/TIR/backend/CLI tests for changed behavior. Retain source
hashes and exact commands. Host/emulated checks, Spike functional checks, local
Verilator cycles and FireSim observations are different evidence tiers. A trained
XGB booster is not proof of post-training ranking; a zero exit status is not
complete output validation; header agreement is not whole-SoC equivalence.

Do not force-push or rewrite existing published history. Curated branches are
local reviewable deliveries. No PR is authorized. A future actual-model run may
need operator-owned checkpoints, queue access and hardware receipts; report exact
missing artifacts while continuing independent compilation/inventory work.
