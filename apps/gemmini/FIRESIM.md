# FireSim build and handoff boundary

The companion Merlin `examples/gemmini/comparisons/tvm/FIRESIM_HANDOFF.md` owns
model inputs, portable chunk delivery, budgets, complete-chain construction and
remote qualification. The simplified compiler is published; use the companion Merlin handoff for
current source and campaign pins. Rebuild packages and re-extract workloads; do not relabel
old ELFs or graph plans with new source identities.

## Current campaign decisions

- Sole model protocol: Merlin 8ec99308, with approved TinyLlama prefill correction
  to 32; E2E is 32 prefill + 32 decode. Tune published 16/1 probes.
- Aggregate occupied tuning time: active ResNet 4h; backlogged TinyLlama/SmolVLA
  retain 1h each for a future reactivation, across chunks
  and graph choices. Stop new admissions at budget; finish active runs under their
  safeguards and report overrun. CPU preparation and final E2E are separate.
- Final E2E wall timeouts only: ResNet 1800s, TinyLlama/SmolVLA 25200s per run.
- Next campaign: ResNet plus the 16 kernel cases in the companion Merlin
  KERNEL_BENCHMARK_HANDOFF_IREE_TVM_VOYAGER_AGENTIC.md. Full TinyLlama/SmolVLA
  are backlogged. Use 25 total ReplayTrace proposals per kernel/chunk with no ML. The 16-case
  TVM kernel sweep ceiling is 24 occupied-slot-hours. Report unequal effort.
- Candidate median-based wall watchdog replaces incumbent-cycle cancellation.
  Use three real baseline timings, a 3x latest-nine median with a 3x baseline floor,
  and an explicit hard ceiling. Bootstrap/ceiling values come from the remote pilot.
- Remote tuning uses independent FireSim guest checks, not sequential Spike
  preflight. Local small functional controls remain a separate evidence tier.
- Remote worker owns manager/slots, deployed image, actual cancellation and clock/
  memory bindings. Do not assume the reported F2 and selected U250 are equivalent.

## Build interface

```sh
python apps/gemmini/compile_package.py --program PROGRAM \
  --output-dir NEW_PACKAGE --graph-fusion-level 2 \
  --adapter-receipt ADAPTER_RECEIPT --riscv-gcc RISCV_GCC
python apps/gemmini/compile_package.py --verify NEW_PACKAGE
```

Use source-bound tools and a fresh output directory. There is no automatic bias
preload or constant prepacking. Optional tuning uses `--tuning-database`; inspect
actual placement and misses. Presets/old costs are not new FireSim measurements.

The package contains operator objects, immutable constants, graph wrapper, ABI,
workspace/placement plans and source/toolchain evidence. Its entrypoint is
`model_run(inputs, outputs, workspace, workspace_bytes)`. Buffers need the declared
extent, alignment and nonoverlap. The package itself does not implement a complete
recurrent model session. Use Merlin's explicit package-session bridge; the native
MLIR memref ABI is different and cannot be substituted by renaming symbols.

Large constants can use the existing absolute-address wrapper convention while
retaining the operator objects and exact constant bytes. The remote linker/loader
must verify all load spans, stack, workspace, inputs, references and carried state.
No archive or small high-address probe establishes whole-model capacity.

## Evidence required after building

1. Verify source/adapter identities, static package manifest, unresolved symbols
   and whole-ELF prohibition of hardware loop FSM commands.
2. Check independent positive/wrong-oracle fixtures and every required output,
   guard and input immutability condition. Label diagnostic Spike separately
   from the spec-pinned oracle; its missing provenance is not waived.
3. Bind actual image/driver/loader/ISA, memory map and counter frequency remotely.
   Execute the qualified guest and confirm remote termination on watchdog expiry;
   killing a local transport process alone is insufficient.
4. Tune only under the selected policy, preserving real cycle labels and errors.
   Confirm replayed winners and connected chunks before baseline/tuned full chains.
5. Validate actual carried state, resets and the entire selected model chain using
   the pinned protocol. Operator/chunk tests do not establish model success.

Before launch, finish per-family proof admission, source-bound model artifacts
and remote wall-cancellation/correctness qualification. Historical transport/ABI details and
results are preserved in
[the source-pinned FireSim guide](https://github.com/grammy117/tvm/blob/42b3c1ef27abe878f930e48b335b24956beb0e56/apps/gemmini/FIRESIM.md).
