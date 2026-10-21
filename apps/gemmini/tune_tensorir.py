# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Cycle-measured CPU/Gemmini DSE with bounded native execution.

The initial command is a primitive calibration experiment, not a model-quality
benchmark. It uses the same MetaSchedule rule, scalar ABI, exporter and pinned
RTL as model slices. Functional Spike checks never contribute timing labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.backend.contrib.gemmini_dse import knob_inventory, search
from tvm.relax.backend.contrib.gemmini_target import GemminiCapabilities
from tvm.relax.backend.contrib.gemmini_tensorir import lower_tensorir_schedule

import verify_tensorir as verifier
from verify_baremetal_cpu import identity


def product_workload(m, k, n):
    """Untiled mathematical integer reference, with independent external buffers."""
    a = te.placeholder((m, k), "int8", "a")
    b = te.placeholder((k, n), "int8", "b")
    r = te.reduce_axis((0, k), "r")
    c = te.compute(
        (m, n),
        lambda i, j: te.sum(a[i, r].astype("int32") * b[r, j].astype("int32"), axis=r),
        name="product",
    )
    return te.create_prim_func([a, b, c])


def candidate_graph(func, candidate, capabilities):
    """Use the selected framework schedule without rescheduling it in export."""
    if candidate.placement == "gemmini":
        lowered = lower_tensorir_schedule(func, candidate.schedule, capabilities=capabilities)
        selected, metadata = lowered.scheduled_mod["main"], lowered.metadata
    else:
        selected, metadata = candidate.schedule.mod["main"], {}
    buffers = [func.buffer_map[param] for param in func.params]
    builder = relax.BlockBuilder()
    gv = builder.add_func(selected.with_attr("global_symbol", "kernel"), "kernel")
    inputs = [
        relax.Var(f"input_{i}", relax.TensorStructInfo(buffer.shape, buffer.dtype))
        for i, buffer in enumerate(buffers[:-1])
    ]
    output = buffers[-1]
    with builder.function("main", inputs):
        with builder.dataflow():
            result = builder.emit_output(
                relax.call_tir(
                    gv, inputs, out_sinfo=relax.TensorStructInfo(output.shape, output.dtype)
                )
            )
        builder.emit_func_output(result)
    report = [
        {
            "function": "kernel",
            "placement": candidate.placement,
            "origin": "MetaSchedule cycle experiment candidate",
            "metadata": metadata,
            "trace": candidate.schedule.trace.as_json(),
        }
    ]
    return builder.get(), report


class KernelMeasurer:
    """Build and verify the exact candidate ELF before admitting RTL cycles."""

    def __init__(self, args, func, arrays, expected):
        self.args, self.func, self.arrays, self.expected = args, func, arrays, expected
        self.capabilities = GemminiCapabilities()
        self.records = []
        body = {
            "hardware_receipt": identity(args.hardware_receipt),
            "adapter": identity(args.adapter_receipt),
            "capabilities": self.capabilities.fingerprint,
            "timer": "guest_cycle_csr_model_run_warmup1_v1",
            "repeats": args.repeats,
            "input_hashes": [hashlib.sha256(x.tobytes()).hexdigest() for x in arrays],
            "oracle_hash": hashlib.sha256(expected.tobytes()).hexdigest(),
        }
        self.profile = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        with (args.output_dir / "measurement-profile.json").open("x") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")

    def __call__(self, candidate):
        args, index = self.args, len(self.records)
        folder = args.output_dir / f"trial-{index:03d}-{candidate.id[:12]}"
        graph, report = candidate_graph(self.func, candidate, self.capabilities)
        native = candidate.placement == "gemmini"
        case = (
            "kernel",
            graph,
            self.arrays,
            self.expected,
            "gemmini" if native else "host",
            {},
            int(native),
        )
        build_start = time.monotonic()
        verifier.prepare(
            SimpleNamespace(
                output_dir=folder,
                adapter_receipt=args.adapter_receipt,
                riscv_gcc=args.riscv_gcc,
                cases=["kernel"],
                timing_repeats=args.repeats,
                rtol=0.0,
                atol=0.0,
            ),
            cases=[case],
            prepared_reports={"kernel": report},
            additional_bindings=[Path(__file__), args.output_dir / "measurement-profile.json"]
            + ([args.workload_json] if args.workload_json is not None else []),
        )
        build_seconds = time.monotonic() - build_start
        for backend, negative, name in (
            ("spike", False, "spike-valid"),
            ("spike", True, "spike-negative"),
            ("verilator", False, "rtl-valid"),
        ):
            verifier.run(
                SimpleNamespace(
                    bundle=folder,
                    backend=backend,
                    negative=negative,
                    name=name,
                    simulator_receipt=args.spike_receipt,
                    hardware_receipt=args.hardware_receipt,
                    seconds=args.seconds,
                    cycles=args.cycles,
                )
            )
        receipt_path = folder / "rtl-valid-result.json"
        receipt = json.loads(receipt_path.read_text())
        timing = receipt["timing"]
        if not timing or not timing["usable_for_dse"] or timing["metric"] != "target_cycles":
            raise ValueError("Candidate has no qualified RTL-model cycle interval")
        record = {
            "metric": "target_cycles",
            "engine": "verilator",
            "capability_fingerprint": self.capabilities.fingerprint,
            "measurement_profile": self.profile,
            "correctness": "passed",
            "timed_out": False,
            "samples": timing["samples_by_stage"][0],
            "receipt": identity(receipt_path),
            "build_seconds": build_seconds,
            "scope": timing["scope"],
        }
        self.records.append(record)
        print(
            json.dumps(
                {
                    "trial": index,
                    "placement": candidate.placement,
                    "median_cycles": statistics.median(record["samples"]),
                }
            ),
            flush=True,
        )
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--hardware-receipt", type=Path, required=True)
    parser.add_argument("--spike-receipt", type=Path, required=True)
    parser.add_argument("--riscv-gcc", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, default=(16, 16, 16), metavar=("M", "K", "N"))
    parser.add_argument(
        "--workload-json",
        type=Path,
        help="An exact pre-scheduling TVM IRModule with one mathematical main product",
    )
    parser.add_argument("--strategy", choices=("sweep", "ml"), default="sweep")
    parser.add_argument("--max-trials", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=300000)
    parser.add_argument("--time-budget-seconds", type=float, default=900)
    args = parser.parse_args()
    func = None
    if args.workload_json is not None:
        args.workload_json = args.workload_json.resolve(strict=True)
        mod = tvm.ir.load_json(args.workload_json.read_text())
        func = mod["main"]
        buffers = [func.buffer_map[param] for param in func.params]
        if (
            len(buffers) != 3
            or any(len(buffer.shape) != 2 for buffer in buffers)
            or [buffer.dtype for buffer in buffers] != ["int8", "int8", "int32"]
        ):
            raise ValueError(
                "This measurement adapter requires a two-input signed-byte matrix product"
            )
        m, k = map(int, buffers[0].shape)
        kb, n = map(int, buffers[1].shape)
        if k != kb or tuple(map(int, buffers[2].shape)) != (m, n):
            raise ValueError("Captured product dimensions disagree")
        args.shape = (m, k, n)
    if (
        any(value <= 0 for value in args.shape)
        or args.shape[1] > GemminiCapabilities().max_reduction
    ):
        raise ValueError("Primitive experiment requires positive, prefix-bounded dimensions")
    if max(
        args.shape[0] * args.shape[1], args.shape[1] * args.shape[2], args.shape[0] * args.shape[2]
    ) > (1 << 20):
        raise ValueError("Primitive experiment exceeds the bounded input/output element budget")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_receipt", "hardware_receipt", "spike_receipt", "riscv_gcc"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    rng = np.random.default_rng(args.seed)
    m, k, n = args.shape
    arrays = [
        rng.integers(-128, 128, (m, k), dtype="int8"),
        rng.integers(-128, 128, (k, n), dtype="int8"),
    ]
    expected = (arrays[0].astype("int64") @ arrays[1].astype("int64")).astype("int32")
    func = product_workload(m, k, n) if func is None else func
    measure = KernelMeasurer(args, func, arrays, expected)
    # Short RTL experiments bound padding/cache-group expansion. The backend's
    # full group choices remain available through its public rule options.
    options = {"row_tiles": (1, 2), "col_tiles": (1, 2)}
    result = search(
        func,
        measure,
        strategy=args.strategy,
        max_trials=args.max_trials,
        warmup=args.warmup,
        seed=args.seed,
        time_budget_seconds=args.time_budget_seconds,
        rule_options=options,
    )
    body = result.as_json()
    body.update(
        {
            "shape_mkn": args.shape,
            "scope": "primitive_calibration_not_model_quality",
            "knobs": knob_inventory(),
            "rule_options": options,
            "script": identity(__file__),
            "workload_source": identity(args.workload_json) if args.workload_json else None,
        }
    )
    with (args.output_dir / "search-result.json").open("x") as handle:
        json.dump(body, handle, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "result": str(args.output_dir / "search-result.json"),
                "winner": result.winner.placement,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
