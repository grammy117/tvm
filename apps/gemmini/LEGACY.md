# Historical compatibility code

The active compiler is mathematical TensorIR plus target schedule rules,
TensorIntrins, required memory lowering and the primitive runtime. See
[README.md](README.md), [OWNERSHIP.md](OWNERSHIP.md) and [TUNING.md](TUNING.md).

| Retained component | Why retained / boundary |
| --- | --- |
| `gemmini_legacy.py` | Old Relax linear/conv decomposition and early external/template substitution; lazy-loaded only for explicit historical API/route requests |
| `gemmini.py` historical symbol exports | Compatibility for old tests and `verify_graph.py`; normal TensorIR preparation does not load legacy implementation |
| `gemmini_conv.py` | Historical explicit graph im2col pass, called by old graph/template paths; not current TensorIR scheduling |
| `gemmini_tuning.py` | Old nine-template feature/XGB experiment, used by old diagnostics; not the stock ReplayTrace campaign |
| Evolutionary/XGB option in the common tuner | Explicit research/reproduction option; active FireSim entry point forces ReplayTrace |
| `verify_graph.py` and historical regression tests | Reproduction and compatibility, not current FireSim launch tools |
| `*_HISTORICAL.md`, old presets and receipts | Frozen provenance; use current handoff for launch settings |

Do not delete `gemmini_bias.py`, `gemmini_memory.py` or the whole
`gemmini_schedule.py`: explicit initialization, physical memory lowering and
shared intrinsic/schedule helpers are active dependencies. Automatic bias
relocation, constant prepacking and scale/readout normalization are already removed.
No new optimization is introduced by isolating the historical module.

Local branches with checked-out worktrees or uncommitted changes are not
cleanup candidates. Published historical refs and their evidence remain intact;
only duplicate unused local aliases can be removed after an explicit review.
