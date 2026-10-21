# Offline compiler and FireSim handoff

`compile_package.py` compiles a saved Torch export (`.pt2`) or semantic Relax JSON
into a portable RV64/lp64d static library. It runs ordinary Relax optimization,
constant packing/folding, exact integer legality checks and TensorIR scheduling;
unsupported arithmetic and materialized copies use the CPU. It does not launch
FireSim or allocate a board. Local Spike/RTL evidence is listed in
[EL4_STATUS.md](EL4_STATUS.md) and the follow-up report.

## Compile and link a reviewable result

Use the pinned TVM Python/compiler-library environment and RV64 GCC recorded in
the experiment receipts. Supply an audited adapter receipt built from this source.
Output directories must be fresh.

```bash
python apps/gemmini/compile_package.py \
  --program workload.pt2 --output-dir compiled-model \
  --adapter-receipt adapter/receipt.json --riscv-gcc /path/to/riscv64-unknown-elf-gcc
python apps/gemmini/compile_package.py --verify compiled-model
```

`--cpu-only` creates the ordinary CPU graph baseline, without accelerator layout
packing or bias scheduling. Repeat `--tuning-result search/result.json` to apply
measured per-function choices, including CPU winners. Each record must match the
exact mathematical workload, capability profile and a current legal trace; both
CPU/default baselines must have completed. Verilator choices are optimization
hints for FireSim and must be remeasured there. Compilation does not tune implicitly.

The package includes `libmodel.a`, `model.h`, `matmul.h`, the selected `libm.a` and
`libgcc.a`, generated LLVM/TIR/Relax and C, relative constant assembly, placement,
explicit tensor/workspace memory plans, source/toolchain/adapter identities and a
relative-path payload manifest. Constants are compiled into the archive before
handoff, so a final link does not depend on the original `.incbin` path. A
whole-archive partial link rejects unresolved runtime dependencies and audits
source-derived prohibited loop encodings. No guest TVM VM, workspace allocator,
heap or syscalls are introduced. Weak byte-memory helpers and a single-hart errno
cell are supplied; board code can override them. The fallback vendor diagnostic
suppresses text and traps on exit. The validation guest provides HTIF diagnostics.

The entry point is:

```c
int32_t model_run(const void *const *inputs, void *const *outputs,
                  uint8_t *workspace, uint64_t workspace_bytes);
```

Every buffer must have the manifest's full extent, 64-byte alignment and required
nonoverlap. Call inputs, outputs and explicit workspace remain caller-owned. Graph
constants are immutable. The generated admission code rejects missing, short,
misaligned or conflicting buffers. A production caller supplies startup, inputs,
outputs and the complete memory reservation. A static graph can represent explicit
cache/state inputs and outputs; automatic multi-program session orchestration is
not supplied by this CLI. The existing `export_graphs` API covers fixed stages.

## Build independent functional boot ELFs

```bash
python apps/gemmini/qualify_package.py \
  --package compiled-model --fixture arrays.npz \
  --fixture-provenance reference.json --output-dir checked-model \
  --riscv-gcc /path/to/riscv64-unknown-elf-gcc --rtol 1e-5 --atol 1e-6
python apps/gemmini/verify_tensorir.py run \
  --bundle checked-model --backend spike --name spike-valid \
  --simulator-receipt spike/receipt.json --seconds 120
python apps/gemmini/verify_tensorir.py run \
  --bundle checked-model --backend verilator --name rtl-valid \
  --hardware-receipt rtl/runtime-qualification.json --cycles 4000000 --seconds 900
```

Repeat each run with `--negative` and a fresh `--name`; successful negative control
means the guest rejected the deliberately wrong oracle with the expected code.
Never overwrite receipts. The fixture stores `input_0`, ... and `expected` (or
`expected_0`, ... for multiple outputs), with exact exported shapes and dtypes.
`reference.json` supplies `fixture_sha256`, a nonempty `reference_contract` and
`reference_sources: [{"path": "/absolute/reference", "sha256": "..."}]`.
Use independent frozen source/reference outputs, never outputs generated from the
scheduled candidate. Integer results compare exactly; floating tolerances are
explicit. These are compiler functional gates, not pretrained model quality gates.

The tool links the delivered archive itself, audits ELF load spans/entry/ABI and
no-FSM instructions, initializes FP state to RNE, checks inputs/output/workspace
canaries and constant immutability, and reports stack high-water use against the
16 KiB reserved stack. Both positive and wrong-oracle ELFs are retained. The
page-aligned guest map defaults to 256 MiB and is adjustable; this is a functional
memory reservation, not deployed DRAM capacity qualification. The linker enforces
all ELF sections plus stack within it. The explicit tensor budget alone does not
include code, loader/platform reservations or arbitrary operator stack use.

For timing, `--timing-repeats 3` adds one warmup and three fenced `model_run`
intervals, with output validation after every repeat. Startup/oracle/HTIF are
excluded from the interval. Spike counters are functional evidence only; completed
Verilator intervals use its selected magic-memory scope. Pin independent host
runner processes to distinct allowed cores when running concurrently; the verifier
uses the first core in its inherited affinity. Wall deadlines and target cycle
limits are separate bounds.

## Bind the actual deployment

[firesim_target.json](firesim_target.json) records the requested full U250 device:
`alveo_u250_firesim_gemmini_rocket_30mhz`, bitstream `0af6f26e...`, elaborated design
`7b628027...`, generated ABI header `3758ae96...`, one non-vector Rocket hart and
DTS-declared 16 GiB at `0x80000000`. The 64 MiB code reserve is a layout request.
The driver digest is deliberately unresolved: the previously available
`55d39776...` driver belongs to the different `a9a190b9...` bitstream. Do not pair
it with this build based on matching config names. Source-to-built lineage is
incomplete; the elaborated hardware digest is the content identity.

Before executing or interpreting FireSim performance, bind actual bitstream,
elaboration, matching driver, generated header, DTS, loader/HTIF behavior and full
reserved DRAM map to the final ELF. Check all load segments, stack high-water and
cache/state lifetimes at the intended full size; run independent inputs and
negative controls. The 30 MHz build label is not a cycle-to-time or DRAM model
calibration. Existing local RTL results cannot qualify these deployment facts.
No queue submission or hardware execution is performed by these tools.

The compiler software handoff is ready when its archive, linked boot ELF and local
proof receipts pass. A deployed-hardware or final whole-model performance claim
also requires the deployment bindings, realistic checkpoint/calibration/task-quality
coverage and repeated matching CPU/default/winner FireSim intervals. None is
inferred from the small synthetic recipe slices.
