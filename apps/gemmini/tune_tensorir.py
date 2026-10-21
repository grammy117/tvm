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
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.backend.contrib.gemmini_dse import TrialFailure, knob_inventory, search
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
        from tvm.relax.backend.contrib.gemmini_tensorir import (
            contraction_block,
            _normalizable_contractions,
        )
        from tvm.relax.backend.contrib.gemmini_elementwise import elementwise_blocks
        from tvm.relax.backend.contrib.gemmini_reduction import reduction_blocks, reduction_metadata

        metadata = None
        if not (contraction_block(func) or _normalizable_contractions(func)):
            reductions, pointwise = reduction_blocks(func), elementwise_blocks(func)
            metadata = (
                reduction_metadata(reductions[0]) if reductions else {"operation": pointwise[0][1]}
            )
        lowered = lower_tensorir_schedule(
            func, candidate.schedule, capabilities=capabilities, workload_metadata=metadata
        )
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
        self.attempts = 0
        self.deadline = time.monotonic() + args.time_budget_seconds
        body = {
            "hardware_receipt": identity(args.hardware_receipt),
            "adapter": identity(args.adapter_receipt),
            "capabilities": self.capabilities.fingerprint,
            "timer": "guest_cycle_csr_model_run_warmup1_v1",
            "repeats": args.repeats,
            "input_hashes": [hashlib.sha256(x.tobytes()).hexdigest() for x in arrays],
            "buffer_abis": [{"dtype": x.dtype.name, "shape": list(x.shape)} for x in [*arrays, expected]],
            "oracle_hash": hashlib.sha256(expected.tobytes()).hexdigest(),
            "toolchain": identity(args.riscv_gcc),
            "comparison": {"rtol": getattr(args, "rtol", 0.0), "atol": getattr(args, "atol", 0.0)},
        }
        body["math_libraries"] = []
        if getattr(args, "libm", False):
            for name in ("libm.a", "libgcc.a"):
                path = subprocess.check_output(
                    [str(args.riscv_gcc), "-march=rv64gc", "-mabi=lp64d", "-print-file-name=" + name],
                    text=True, timeout=args.build_seconds,
                ).strip()
                body["math_libraries"].append(identity(Path(path)))
        from tvm._ffi.base import _LIB

        source = Path(tvm.__file__).resolve().parent
        body["compiler"] = [identity(_LIB._name), identity(__file__)] + [
            identity(path)
            for folder in (
                source / "relax/backend/contrib",
                source / "tir/transform",
                Path(__file__).resolve().parent,
            )
            for path in sorted(folder.glob("*.py"))
        ]
        self.profile = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        np.savez(
            args.output_dir / "measurement-arrays.npz",
            **{f"input_{i}": x for i, x in enumerate(arrays)},
            expected=expected,
        )
        with (args.output_dir / "measurement-profile.json").open("x") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")

    def build_candidate(self, candidate):
        try:
            return self._build_candidate(candidate)
        except TrialFailure:
            raise
        except Exception as error:
            import traceback

            path = self.args.output_dir / f"build-failure-{self.attempts:03d}.json"
            with path.open("x") as handle:
                json.dump(
                    {
                        "candidate_id": candidate.id,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    handle,
                    indent=2,
                )
            raise TrialFailure(
                "Candidate compilation refused: " + str(error), evidence=identity(path)
            ) from error

    def _build_candidate(self, candidate):
        if time.monotonic() >= self.deadline:
            raise TrialFailure("Campaign wall budget exhausted before build")
        args, index = self.args, self.attempts
        self.attempts += 1
        folder = args.output_dir / f"trial-{index:03d}-{candidate.id[:12]}"
        graph, report = candidate_graph(self.func, candidate, self.capabilities)
        native = candidate.placement == "gemmini"
        graph_path = folder.with_name(folder.name + "-graph.json")
        graph_path.write_text(tvm.ir.save_json(graph))
        request_path = folder.with_name(folder.name + "-build-request.json")
        request = {
            "output_dir": str(folder),
            "adapter_receipt": str(args.adapter_receipt),
            "riscv_gcc": str(args.riscv_gcc),
            "repeats": args.repeats,
            "graph": str(graph_path),
            "arrays": str(args.output_dir / "measurement-arrays.npz"),
            "report": report,
            "native": native,
            "libm": getattr(args, "libm", False),
            "rtol": getattr(args, "rtol", 0.0),
            "atol": getattr(args, "atol", 0.0),
            "bindings": [
                str(Path(__file__).resolve()),
                str(args.output_dir / "measurement-profile.json"),
            ]
            + ([str(args.workload_json)] if args.workload_json is not None else []),
        }
        request_path.write_text(json.dumps(request, indent=2) + "\n")
        build_start = time.monotonic()
        log_path = folder.with_name(folder.name + "-build.log")
        with log_path.open("xb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--build-request",
                    str(request_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(
                    timeout=max(0.001, min(args.build_seconds, self.deadline - time.monotonic()))
                )
            except subprocess.TimeoutExpired as error:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                raise TrialFailure(
                    "Candidate build exceeded its wall deadline", evidence=identity(log_path)
                ) from error
        if code:
            raise TrialFailure("Candidate build failed", evidence=identity(log_path))
        build_seconds = time.monotonic() - build_start
        return {
            "folder": str(folder),
            "build_seconds": build_seconds,
            "index": index,
            "placement": candidate.placement,
        }

    def run_candidate(self, artifact):
        args = self.args
        folder, index = Path(artifact["folder"]), artifact["index"]
        for backend, negative, name in (
            ("spike", False, "spike-valid"),
            ("spike", True, "spike-negative"),
            ("verilator", False, "rtl-valid"),
        ):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TrialFailure("Campaign wall budget exhausted before " + name)
            try:
                verifier.run(
                    SimpleNamespace(
                        bundle=folder,
                        backend=backend,
                        negative=negative,
                        name=name,
                        simulator_receipt=args.spike_receipt,
                        hardware_receipt=args.hardware_receipt,
                        seconds=min(args.seconds, remaining),
                        cycles=args.cycles,
                    )
                )
            except SystemExit as error:
                raise TrialFailure(
                    "Candidate " + name + " failed or timed out",
                    evidence=identity(folder / (name + "-result.json")),
                ) from error
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
            "build_seconds": artifact["build_seconds"],
            "scope": timing["scope"],
        }
        self.records.append(record)
        print(
            json.dumps(
                {
                    "trial": index,
                    "placement": artifact["placement"],
                    "median_cycles": statistics.median(record["samples"]),
                }
            ),
            flush=True,
        )
        return record

    def __call__(self, candidate):
        return self.run_candidate(self.build_candidate(candidate))


def build_request(path):
    """Run TVM code generation in a killable process, retaining build evidence."""
    request = json.loads(Path(path).read_text())
    graph = tvm.ir.load_json(Path(request["graph"]).read_text())
    with np.load(request["arrays"]) as data:
        arrays = [data[f"input_{i}"].copy() for i in range(len(data.files) - 1)]
        expected = data["expected"].copy()
    native = request["native"]
    case = ("kernel", graph, arrays, expected, "gemmini" if native else "host", {}, int(native))
    verifier.prepare(
        SimpleNamespace(
            output_dir=Path(request["output_dir"]),
            adapter_receipt=Path(request["adapter_receipt"]),
            riscv_gcc=Path(request["riscv_gcc"]),
            cases=["kernel"],
            timing_repeats=request["repeats"],
            rtol=request.get("rtol", 0.0),
            atol=request.get("atol", 0.0),
            libm=request.get("libm", False),
        ),
        cases=[case],
        prepared_reports={"kernel": request["report"]},
        additional_bindings=[
            Path(path),
            Path(request["graph"]),
            Path(request["arrays"]),
            *map(Path, request["bindings"]),
        ],
    )


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--build-request":
        build_request(sys.argv[2])
        return
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
    parser.add_argument(
        "--inputs-npz",
        type=Path,
        help="Actual function input_N/expected tensors from extract_tensorir.py",
    )
    parser.add_argument(
        "--libm", action="store_true", help="Explicitly link the pinned GCC math archives"
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        choices=("sweep", "ml", "metaschedule"),
        default="sweep",
        help="ml/metaschedule use TVM tune_tasks, EvolutionarySearch and XGBModel",
    )
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument(
        "--search-seed", type=int, default=0, help="Independent of the input-data seed"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=int, help="Explicit per-simulator wall deadline override")
    parser.add_argument("--build-seconds", type=float, default=120)
    parser.add_argument(
        "--minimum-cycles-per-second",
        type=float,
        default=1000,
        help="Conservative simulation throughput for deriving wall deadlines",
    )
    parser.add_argument("--cycles", type=int, default=300000)
    parser.add_argument("--time-budget-seconds", type=float, default=3600)
    parser.add_argument(
        "--resume", type=Path, help="Previous search directory with durable trials.jsonl"
    )
    args = parser.parse_args()
    if (
        any(
            not math.isfinite(x) or x <= 0
            for x in (args.build_seconds, args.minimum_cycles_per_second)
        )
        or args.cycles <= 0
    ):
        raise ValueError("Build and simulation budgets must be finite and positive")
    if args.seconds is None:
        args.seconds = max(120, math.ceil(args.cycles / args.minimum_cycles_per_second) + 60)
    if args.seconds <= 0:
        raise ValueError("Wall deadline must be positive")
    func = None
    if args.inputs_npz is not None and args.workload_json is None:
        raise ValueError("Function input arrays require an exact workload JSON")
    if args.workload_json is not None:
        args.workload_json = args.workload_json.resolve(strict=True)
        mod = tvm.ir.load_json(args.workload_json.read_text())
        func = mod["main"]
        buffers = [func.buffer_map[param] for param in func.params]
        if args.inputs_npz is not None:
            args.inputs_npz = args.inputs_npz.resolve(strict=True)
            with np.load(args.inputs_npz) as data:
                if set(data.files) != {
                    "expected",
                    *[f"input_{i}" for i in range(len(buffers) - 1)],
                }:
                    raise ValueError("Function arrays do not match the mathematical buffer ABI")
                arrays = [data[f"input_{i}"].copy() for i in range(len(buffers) - 1)]
                expected = data["expected"].copy()
            for array, buffer in zip([*arrays, expected], buffers):
                if (
                    array.dtype.name != buffer.dtype
                    or tuple(array.shape) != tuple(map(int, buffer.shape))
                    or array.size > (1 << 20)
                ):
                    raise ValueError("Function arrays exceed the bounded shape/dtype ABI")
            args.shape = None
        elif (
            len(buffers) != 3
            or any(len(buffer.shape) != 2 for buffer in buffers)
            or [buffer.dtype for buffer in buffers] != ["int8", "int8", "int32"]
        ):
            raise ValueError(
                "This measurement adapter requires a two-input signed-byte matrix product"
            )
        if args.inputs_npz is None:
            m, k = map(int, buffers[0].shape)
            kb, n = map(int, buffers[1].shape)
            if k != kb or tuple(map(int, buffers[2].shape)) != (m, n):
                raise ValueError("Captured product dimensions disagree")
            args.shape = (m, k, n)
    if args.shape is not None and (
        any(value <= 0 for value in args.shape)
        or args.shape[1] > GemminiCapabilities().max_reduction
    ):
        raise ValueError("Primitive experiment requires positive, prefix-bounded dimensions")
    if args.shape is not None and max(
        args.shape[0] * args.shape[1], args.shape[1] * args.shape[2], args.shape[0] * args.shape[2]
    ) > (1 << 20):
        raise ValueError("Primitive experiment exceeds the bounded input/output element budget")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("adapter_receipt", "hardware_receipt", "spike_receipt", "riscv_gcc"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    if args.inputs_npz is None:
        rng = np.random.default_rng(args.seed)
        m, k, n = args.shape
        arrays = [
            rng.integers(-128, 128, (m, k), dtype="int8"),
            rng.integers(-128, 128, (k, n), dtype="int8"),
        ]
        expected = (arrays[0].astype("int64") @ arrays[1].astype("int64")).astype("int32")
        func = product_workload(m, k, n) if func is None else func
    measure = KernelMeasurer(args, func, arrays, expected)
    resume_records = []
    workload_json = tvm.ir.save_json(tvm.IRModule({"main": func}))
    (args.output_dir / "workload.json").write_text(workload_json)
    if args.resume is not None:
        previous = args.resume.resolve(strict=True)
        if not tvm.ir.structural_equal(
            tvm.ir.load_json((previous / "workload.json").read_text()),
            tvm.ir.load_json(workload_json),
        ):
            raise ValueError("Resume workload differs")
        if json.loads((previous / "measurement-profile.json").read_text()) != json.loads(
            (args.output_dir / "measurement-profile.json").read_text()
        ):
            raise ValueError("Resume measurement profile differs")
        resume_records = [
            json.loads(line) for line in (previous / "trials.jsonl").read_text().splitlines()
        ]

    journal = (args.output_dir / "trials.jsonl").open("x")

    def save_trial(row):
        journal.write(json.dumps(row, sort_keys=True) + "\n")
        journal.flush()
        os.fsync(journal.fileno())

    for row in resume_records:
        save_trial(row)
    # Short RTL experiments bound padding/cache-group expansion. The backend's
    # full group choices remain available through its public rule options.
    options = {"row_tiles": (1, 2), "col_tiles": (1, 2)}
    if args.strategy in ("ml", "metaschedule"):
        from tvm.relax.backend.contrib.gemmini_search import tune

        result = tune(
            func,
            measure.build_candidate,
            measure.run_candidate,
            work_dir=args.output_dir / "metaschedule",
            max_trials=args.max_trials,
            seed=args.search_seed,
            warmup=args.warmup,
            rule_options=options,
            on_record=save_trial,
            resume_records=resume_records,
        )
    else:
        result = search(
            func,
            measure,
            strategy=args.strategy,
            max_trials=args.max_trials,
            warmup=args.warmup,
            seed=args.search_seed,
            time_budget_seconds=args.time_budget_seconds,
            rule_options=options,
            resume_records=resume_records,
            on_record=save_trial,
        )
    journal.close()
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
