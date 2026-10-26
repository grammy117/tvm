# Offline compiler and FireSim handoff

## Current remote handoff (2026-10-25)

Published source: `grammy117/tvm`, branch `gemmini/firesim-chunks`. A checkout
of this branch already includes the changes; the older delivered patch applies
only to its original base, not on top of the published branch.

The companion Merlin guide `examples/gemmini/comparisons/tvm/FIRESIM_CHUNK_TUNING.md`
supersedes the historical preset recommendations below. Its delivery
`out/artifacts/handoff/gemmini-tvm-firesim-refresh/v1/latest/` contains full
ResNet baseline/current-tuned/profiled guests and small trained-weight TinyLlama probes.
They pass functional Spike positive/wrong-oracle controls; FPGA execution is
deferred to the new machine. The ResNet explicit validated database matches
one classifier task out of 68, not the whole model. No new FireSim gain is claimed.

Use this isolated `tvm-firesim-chunks` tree at base `c50c4e2` plus the complete
delivered patch/overlay and the matching corrected native library. It retains
stock task extraction, gradient scheduling, EvolutionarySearch, XGB and standard
JSONDatabase. The scalar input-binding fix preserves rank-zero mathematical
constants during host propagation. The completed `tvm-dse-budgeted` evidence
tree remains unchanged.

Merlin's `tune_firesim_chunks.py` uses the existing builder and stock search
with a bounded command adapter to the actual remote manager. It batches 4–6
proposals, shares explicit slot locks across studies, and allocates no hardware.
The supplied bitstream already configures the expected DDR3 memory model;
deliver ELFs to the existing runner. Do not configure memory or infer F2 clock,
capacity or runtimes. Abort DSE candidates when active measured cycles reach
the current best, excluding boot/loading/initialization/warmup, and tighten
in-flight limits on improvements. There are no guessed per-job caps. Keep
FireSim and Verilator cost labels separate. Probe constants use a fresh W8A8
recipe; actual calibrated EL4 slices and full-shape confirmation precede E2E.

The remote order is full TinyLlama baseline plus ResNet baseline and current
tuned ResNet, then ML-guided TinyLlama chunk DSE, then tuned TinyLlama E2E.
Independent jobs may overlap on available slots; full TinyLlama ELFs are built
from the actual remote exports. EvolutionarySearch uses XGBoost ranking and
measurement feedback, not a static parameter sweep.

## Package interface and historical deployment notes

`compile_package.py` compiles a saved Torch export (`.pt2`) or semantic Relax JSON
into a portable RV64/lp64d static library. It runs ordinary Relax optimization,
constant packing/folding, exact integer legality checks and TensorIR scheduling;
unsupported arithmetic and materialized copies use the CPU. It does not launch
FireSim or allocate a board. The qualification boundary and current entrypoints are in
[README.md](README.md); lowering ownership is in [OWNERSHIP.md](OWNERSHIP.md).

The earlier FireSim compilation used the shipped FASED-measured records with
`compile_package.py --tuning-preset fased-v1 --fallback-tuning-preset rtl-v1`.
`fased-v1` holds 16 operator records measured under the FASED latency-pipe
memory model; `rtl-v1` supplies traces only where `fased-v1` has no structural
match. Add `--graph-cse` for TinyLlama stages: it shared repeated quantization
work and cut a connected K/V region's warm cycles by 25.95%. The companion Merlin
`FASED_SWEEPS.md` records the bounded three-workload results. These are chunk
measurements; read each package's `database-selection.json` and `placement.json`
for actual coverage on the exported stages. A matching latency-pipe policy alone
does not qualify a new bitstream or establish same-ELF FireSim agreement.

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
packing or bias scheduling. Use `--tuning-database "$RUN/database"` to apply a
standard MetaSchedule JSONDatabase, including CPU winners for supported tasks.
The compiler uses `MetaScheduleApplyDatabase`; it rejects an empty database,
a mismatched target, and a database matching no function. Partial coverage is
allowed, so inspect placement and account for every remaining default/fallback.
`--tuning-result` was removed. Verilator choices are hints for FireSim and must
be measured on the selected FireSim image. Compilation does not tune implicitly.

The earlier three-model EL4 handoff selected `--tuning-preset rtl-v1`. This portable
stock database supplies the nine existing operator winners and verifies payload
and capability identities. Unlike an explicit database, this preset permits zero
structural matches, leaving those functions on unmeasured defaults. Inspect
`tuned_functions`, `unmeasured_placement_is_provisional` and `placement.json`.
The selected TinyLlama prefill has 32 tokens, but matching row count alone does
not establish a structural match or a measurement for different EL4 arithmetic.

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

New graph manifests also declare `constants_binding`, with the default
`model_constants` link symbol and optional `TVM_MODEL_CONSTANTS_ADDRESS` macro.
A board adapter may recompile the delivered `graph/model.c` with that macro set
to an explicit aligned absolute address, retaining `graph/operators.o` and the
exact constant payload. This avoids medany relocations across multi-GB gaps.
The board adapter owns placement, loading, bounds/overlap checks and renewed
final-ELF auditing. The original archive remains the default near-symbol form.
Package hashes are streamed and the package instruction audit maps the object,
so verification need not copy an entire large weight blob into Python memory.

Merlin's `build_package_session.py` implements this option using its existing
`model_link.ld` weight region and a reserved NOLOAD arena. Its small two-stage
control passes Spike with 9 GiB weights/references and a 13 GiB arena, including
all-output readback, reset and wrong-oracle rejection. This establishes address
handling on that functional engine; full model sizes and FireSim loading still
need qualification. See the companion README for exact CLI and manifest formats.

## Build independent functional boot ELFs

```bash
python apps/gemmini/qualify_package.py \
  --package compiled-model --fixture arrays.npz \
  --fixture-provenance reference.json --output-dir checked-model \
  --riscv-gcc /path/to/riscv64-unknown-elf-gcc --rtol 1e-5 --atol 2e-6
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
No hardware submission or execution is performed by these compiler tools.

The compiler software handoff is ready when its archive, linked boot ELF and local
proof receipts pass. A deployed-hardware or final whole-model performance claim
also requires the deployment bindings, realistic checkpoint/calibration/task-quality
coverage and repeated matching CPU/default/winner FireSim intervals. None is
inferred from the small synthetic recipe slices.


## Actual EL4 workloads: ordered handoff

The companion Merlin delivery branch contains fetched EL4
`0cc289de52a10bed256f32c90cfbb54a21b3c81f`. The
[shared protocol](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/protocol/experiment.yaml)
owns model revisions, recipes, input shapes, calibration and recurrence. The
[comparison handoff](https://github.com/ucb-bar/merlin/blob/comparisons/tvm-el4-handoff/examples/gemmini/comparisons/tvm/README.md)
records their preparation/integration status. ResNet is one image/one call;
TinyLlama is 32-token prefill plus 32 teacher-forced decode calls; SmolVLA is
prefix + ten denoise calls + action decode. One selected `visible/v00` per
model suffices for the functional benchmark, with explicit repetitions and reset.
ResNet's 260 images are calibration data, used once to prepare scales.
Do not run the stopped
manual `el4_20261018/model_regression.py` campaign or transplant its candidate
admission receipts into the TVM comparison. Its saved compiler-object replacement
path is not a TVM whole-model executor.

1. Freeze actual model inputs and independent source outputs, source/checkpoint
   digests, exported-program signatures, precision and quantization policy.
   Inventory every actual stage with `tune_models.py --program … --extract-only`.
   The synthetic `--model` programs are useful smoke tests only.
2. Compile each real stage and verify its package manifest. Use a separate package
   directory per static signature and inspect both device placement and CPU work.
   A compiled subgraph is not the full session.
3. Execute full source-equivalent stages and recurrence on the selected functional
   engine, checking all outputs/state and resets, plus a wrong-oracle control.
   Preserve any refusal/failure. Do not relax gates to copy a faster schedule.
4. Integrate those package entrypoints with the existing Merlin session/board
   runtime. The `model_run` pointer-array ABI is different from a generated
   gemmini-mlir command-buffer entrypoint. Translate the ABI explicitly and bind
   source/object/symbol hashes; do not rename a symbol and assume compatibility.
   Merlin's comparison bridge now supplies these bindings and explicit routes;
   actual selected full-model qualification remains unfinished.
   The newer native `merlin.model_check.chain_image` builder already packages
   complete chains from retained Merlin program builds. Its memref/build ABI,
   `merlin.post_freeze.firesim.chain_image.v1` manifest and compact `CHAIN_*`
   transcript differ from the TVM bridge's `tvm_merlin_package_session_v1` and
   `TVM_SESSION_*`/OUT_BIN output. Add explicit source/package/reference bindings
   and compatible transport at this boundary; renaming a schema is insufficient.
5. Bind the actual bitstream **and matching driver**, startup/console/tohost ABI,
   clock/counter, ISA, one-hart policy, DRAM base/extent and memory reservations.
   Keep small positive/negative controls on functional engines. Current paid FPGA
   execution is reserved for complete models. A correct header alone does not
   qualify the deployed hardware or its memory model.
6. Check all ELF LOAD spans, .bss/.noinit, stack, inputs, outputs, shared constants,
   packed copies, caches, graph workspace, host staging and verification buffers.
   The 64-MiB request and 16-GiB declaration in `firesim_target.json` are not live
   loader/capacity evidence. Probe the needed high-DRAM ranges through the actual
   loader and recheck memory after linking the entire session. RV64 medany code
   references are bounded; large blobs need the existing explicit absolute-pointer
   loader convention, not ordinary relocations over multi-GB distances.
7. Use the existing post-freeze direct prepare/collect tools and the selected
   locked SSM path, one complete model at a time. Choose the mode below and fill
   actual assets/manager pins; the checked-in selection is a template. No old
   queue or new parallel ELF harness is required.
8. Preserve manager/UART logs, submission argv, SSM command ID, image/source hashes,
   all numerical results, negative controls and one defined cycle interval. Report
   both model-compute and declared E2E boundaries; include every required stage and
   repeated invocation. Never add operator minima and call the sum measured E2E.

The TVM package-session bridge already emits one warm and one measured full
invocation in one boot, with state reset and aggregate cycle windows. The companion
Merlin session builder adds `--profile-stages`: counters are buffered and printed after
validation, with instrumentation overhead recorded. Configurable counts are still
needed for any different repetition policy. The native chain's stage counters exclude descriptor preparation and
input transfers, so their sum is not whole-chain E2E latency. Retain both scopes.
Generic `qualify_package.py` HTIF guests/operator harnesses do not establish
complete-session execution by printing a pass marker.

## Select the current FireSim transport

The [EL4 execution guide](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/README.md)
defines three relevant direct paths:

| Mode | Tool/evidence contract | Current boundary |
| --- | --- | --- |
| Strict version-pinned manager provenance | Offline prepare selects `elf_upload` staging. Staging writes versioned ELF/request objects to S3 and prepares an unsent SSM plan. The manager observes the actual boot file before/after and records execution/cleanup. | Requires installation and pinned hashes of the supplied manager wrapper/SSM document. Its result publisher supports compact `all_element_comparison_v1`; prepare alone can admit a full-array request that this publisher cannot finish. |
| Lightweight `booted_elf_hash_v1` | The installed `fpga-run-elf ELF NAME` helper plus actual runtime-host `booted-elf.json`, matching simulator argv/`+prog0`, complete UART and original SSM/upload records. | Requires compact complete-chain comparison, but no strict-manager upgrade or manufactured result manifest. This establishes a separately labeled reproducibility observation. |
| Legacy direct full arrays | Old helper, chain manifest, manager result/output-file roster, uploaded/consumed identities and shared offline numerical checker. | TVM's full-array reader can supply decoded arrays, but does not itself create manager evidence. This mode does not provide strict boot-file custody; verify timing/cleanup separately. |

Sources: [strict manager integration](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/manager-provenance.md)
and [lightweight contract](https://github.com/ucb-bar/merlin/blob/0cc289de52a10bed256f32c90cfbb54a21b3c81f/examples/gemmini/validation/booted-elf-provenance.md).
Both newer paths need the TVM bridge adapter described above. Compact mode
compares every element of every declared output against admitted references
inside the guest, then sends comparison statistics; it does not replace
numerical checking with a digest. All modes leave `model_validation_complete`
false until independent source, placement, instruction and hardware gates pass.
The old helper's polling/exit/output limitations do not mean the newer strict
helpers are unimplemented, and local tests do not prove they are installed.
The historical queue/checkpoint path remains an explicitly separate deployment.

FireSim timing is an independent backend for the existing MetaSchedule runner
contract; it is not implemented by changing `--hardware-receipt` on the Verilator
runner. Keep its database and labels separate. First measure untuned CPU/default
and a known tuned candidate with the same full session and timing window. Only
then spend search budget on FireSim or infer that local RTL rankings transfer.
