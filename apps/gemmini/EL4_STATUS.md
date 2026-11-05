# Qualification status

The current entrypoint is [README.md](README.md). See [OWNERSHIP.md](OWNERSHIP.md)
for the implementation and upstream audit, [TUNING.md](TUNING.md) for stock
MetaSchedule, and [FIRESIM.md](FIRESIM.md) for the deployment contract.

ResNet50 and 16 isolated kernels were measured on an AWS F2 FireSim image
(60 MHz, DDR3+LLC) on 2026-10-28; the companion Merlin `REPRODUCING_F2_RESULTS.md`
has the results. Full EL4 model execution (TinyLlama, SmolVLA) and the selected
U250 30 MHz target remain unqualified.

The previous narrative is preserved in [EL4_STATUS_HISTORICAL.md](EL4_STATUS_HISTORICAL.md),
and the review disposition in [REVIEW_ACTIONS_HISTORICAL.md](REVIEW_ACTIONS_HISTORICAL.md).
Their OS-only and missing-WS/pipeline claims describe an earlier revision.
