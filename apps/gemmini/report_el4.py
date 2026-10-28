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
"""Regenerate per-op and benchmark tables from explicit EL4 capture/run receipts."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


# These are semantic limits, separate from the generated execution observations.
LIMITS = {
    "aten._int_mm.default": (
        "INT8 x INT8 -> INT32",
        "Array product; CPU staging and result conversion remain explicit",
        "Static positive K <= 131071, exact INT32 prefixes; PE chunks <= 16; admitted layout and SRAM. Larger K keeps modulo-32 CPU math.",
    ),
    "aten.conv2d.default": (
        "FP32 carriers of signed INT8 codes",
        "Array after integer-carrier proof; runtime windows on CPU; frozen weight packing folds at compile time",
        "Scale-one/zero-point-zero INT8 carriers: K <= 1024 or frozen-weight L1 bound proves exact FP32 prefixes. Proven integral bias may preload; eligible uniform-scale RNE/INT8 ReLU epilogues fuse by a full-INT32 proof. General floating conv stays CPU; no LOOP_CONV or fused pool.",
    ),
    "aten.linear.default": (
        "FP32 carriers of signed INT8 codes",
        "Array after carrier proof; integral bias and exact scalar readout may fuse",
        "K <= 1024 or frozen-weight L1 prefix proof; integral bias needs its own FP32 exactness bound. Fractional/unproved bias and per-channel/dynamic readout scales remain explicit CPU math; no implicit calibration.",
    ),
    "aten.matmul.default": (
        "FP32 in retained source sites",
        "CPU when the canonical recipe retains a floating contraction",
        "The public recipe leaves activation reductions <= 16 floating. These reduced slices do not qualify an all-integer-contraction deployment gate.",
    ),
    "quantized_decomposed.quantize_per_tensor.default": (
        "FP32 -> INT8",
        "CPU multiply/RNE/add/clamp/cast unless the exact scalar contraction readout proof admits fusion",
        "Finite positive static scale, bounded integer zero point. Exact source rounding; not searched. Tensor-valued qparams and runtime per-channel quantization remain unsupported.",
    ),
    "quantized_decomposed.dequantize_per_tensor.default": (
        "INT8 -> FP32",
        "CPU affine dequantization; scale-one/zero-zero becomes exact cast",
        "FP32 result only in this intake. Other scales stay explicit CPU arithmetic; they are not discarded to gain offload.",
    ),
    "aten.round.default": (
        "FP32 in these graphs",
        "CPU nearbyint; bounded guarded byte quantization uses clamp plus hardware RNE conversion",
        "RNE floating environment; scalar/tensor rounding stays distinct from hardware integer-shift RNE and portable half-up.",
    ),
    "aten.relu.default": (
        "FP32 carrier/epilogue",
        "Observed source ReLU; placement must be read from the function-level dispatch evidence",
        "FP32 ReLU alone has no native arithmetic implementation. The two eligible CNN requant sites now fuse into scalar scaled ReLU readout after a full-INT32-domain threshold proof. Arbitrary floating ReLU, per-lane scales, nonzero zero points and unequal rounding chains retain CPU math.",
    ),
    "aten.add.Tensor": (
        "FP32 in recipe graphs",
        "CPU bias, residual or plane accumulation",
        "Only proved bounded byte-source affine/sum intrinsics remain candidates. Full-domain INT32 addition and FP32 residual addition use host arithmetic.",
    ),
    "aten.sub.Tensor": (
        "FP32 residual-plane arithmetic",
        "CPU",
        "No floating subtract unit. Full-domain INT32 constant offsets are also outside the bounded-exact device contract.",
    ),
    "aten.mul.Tensor": (
        "FP32 scaling, RoPE and gate arithmetic",
        "CPU",
        "Input/readout scalers do not implement general FP32 elementwise multiply. No vector of lane scales in one store command.",
    ),
    "aten.div.Tensor": (
        "FP32",
        "CPU",
        "Host dequantization and softmax denominator division; not a Gemmini arithmetic primitive.",
    ),
    "aten.neg.default": (
        "FP32 RoPE",
        "CPU",
        "INT8 load scaling saturates and cannot stand in for generic wrapping negation; FP32 negation is host arithmetic.",
    ),
    "aten.clamp.default": (
        "FP32 before cast",
        "CPU",
        "Only a proved integer-derived scalar FP32 RNE/clamp/cast chain can fuse into narrow readout; general clamp remains CPU.",
    ),
    "aten.exp.default": (
        "FP32",
        "CPU via pinned libm expf",
        "No floating exponential hardware. Single-hart errno ABI is supplied; no libc syscall or allocator stubs.",
    ),
    "aten.softmax.int": (
        "FP32",
        "CPU, or removed after canonical plane rewrite",
        "Stable row max/exp/sum/divide stay host work. Softmax planes introduce multiple integer products, not a native softmax unit.",
    ),
    "aten.amax.default": (
        "FP32",
        "CPU reduction",
        "No registered arbitrary FP32 row-max intrinsic; store max-pool is a different operation.",
    ),
    "aten.sum.dim_IntList": (
        "FP32 softmax denominator",
        "CPU reduction",
        "INT8 bounded sum via ones-vector contraction exists; it cannot replace this FP32 sum without changing the recipe.",
    ),
    "aten.mean.dim": (
        "FP32 window/global mean",
        "CPU reduction/divide",
        "Max-pool readout is not average pooling. No implicit INT8 mean recipe is introduced.",
    ),
    "aten.layer_norm.default": (
        "FP32",
        "CPU reduction, rsqrt and affine math",
        "No floating normalization unit. FP32 payload DMA is transport only.",
    ),
    "aten.rms_norm.default": (
        "FP32",
        "CPU reduction, rsqrt and affine math",
        "Importer now handles FP32, including default epsilon. Reduced precision RMSNorm remains explicitly unqualified.",
    ),
    "aten.silu.default": (
        "FP32",
        "CPU",
        "Float island required by the recipe; sigmoid/exponential and multiply are not native array operations.",
    ),
    "aten.embedding.default": (
        "INT64 indices, FP32 stored values",
        "CPU gather",
        "No runtime indexed embedding primitive. Raw FP32 DMA alone does not implement an arbitrary index lookup.",
    ),
    "aten.eq.Scalar": (
        "FP32 -> bool",
        "CPU predicate",
        "No native comparison/predicate intrinsic.",
    ),
    "aten.where.ScalarSelf": (
        "bool predicate and FP32 values",
        "CPU select",
        "Runtime data predicates are outside static region DMA proofs.",
    ),
    "aten._assert_tensor_metadata.default": (
        "Captured size/dtype/device/strided layout",
        "Compile-time checked; no arithmetic",
        "Explicit physical stride assertions are refused. Only admitted static tensor metadata may fold away.",
    ),
    "<built-in function getitem>": (
        "Tuple selection",
        "Compile-time graph selection",
        "Not an accelerator instruction.",
    ),
}
MOVEMENT = (
    "cat",
    "chunk",
    "flatten",
    "reshape",
    "select",
    "slice",
    "squeeze",
    "stack",
    "transpose",
    "unsqueeze",
)
for name in (
    "cat.default",
    "chunk.default",
    "flatten.using_ints",
    "reshape.default",
    "select.int",
    "slice.Tensor",
    "squeeze.dim",
    "stack.default",
    "transpose.int",
    "unsqueeze.default",
):
    LIMITS["aten." + name] = (
        "INT8, INT32 or FP32 payloads in these graphs",
        "Metadata when folded; materialized copies default to CPU; native DMA is explicit or measured",
        "Static positive spans and bounded nonoverlap proof. Native copies require opt-in or a measured record; rectangular failure keeps CPU fallback. Scalar DMA is an explicit diagnostic choice, never an implicit fallback.",
    )
for name in ("aten._to_copy.default", "aten.to.dtype"):
    LIMITS[name] = (
        "INT8/INT32/FP32 conversions",
        "CPU cast; exact INT8 -> INT32 widening may use DMA",
        "INT32 -> FP32 conversion and FP32 -> INT8 rounding are CPU. Bit-preserving FP32 DMA is not conversion or FP32 arithmetic.",
    )


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, bindings):
    bindings.append(identity(path))
    return json.loads(Path(path).read_text())


def cells(values):
    return (
        "| "
        + " | ".join(str(value).replace("|", "/").replace("\n", " ") for value in values)
        + " |"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, nargs="+", required=True)
    parser.add_argument("--bundles", type=Path, nargs="+", required=True)
    parser.add_argument("--searches", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bindings, benchmarks, appearances, runs, searches = [], [], defaultdict(list), [], []
    for folder in args.captures:
        body = load(folder / "result.json", bindings)
        for op, count in body.get("aten_ops", {}).items():
            if op not in LIMITS:
                raise ValueError("An observed operation lacks an explicit contract: " + op)
            appearances[op].append({"workload": body["workload"], "sites": count})
        calls, externs = (
            body.get("call_tir_sites_by_function", {}),
            body.get("externs_by_function", {}),
        )
        array = {
            name: count
            for name, count in calls.items()
            if any("tvm_gemmini_compute" in key for key in externs.get(name, {}))
        }
        dma = {
            name: count
            for name, count in calls.items()
            if name not in array and any("tvm_gemmini_dma_" in key for key in externs.get(name, {}))
        }
        benchmarks.append(
            {
                "capture": str(folder.resolve()),
                "workload": body["workload"],
                "status": body["status"],
                "stage": body["stage"],
                "input_shapes": body.get("input_shapes"),
                "input_dtypes": body.get("input_dtypes"),
                "recipe": body.get("recipe"),
                "calibration_count": body.get("calibration_count"),
                "array_call_sites": sum(array.values()),
                "movement_call_sites": sum(dma.values()),
                "total_call_sites": sum(calls.values()),
                "function_placement_and_refusals": body.get("scheduling"),
                "quantization": body.get("quantization"),
                "partial_tuning": body.get("partial_tuning"),
                "error": body.get("error"),
            }
        )
    for folder in args.bundles:
        bundle = load(folder / "bundle.json", bindings)
        for path in sorted(folder.glob("*-result.json")):
            body = load(path, bindings)
            for key in ("ELF", "log"):
                binding = body[key]
                if identity(binding["path"])["sha256"] != binding["sha256"]:
                    raise ValueError("Executed artifact changed: " + binding["path"])
            runs.append(
                {
                    "bundle": str(folder.resolve()),
                    "cases": bundle["cases"],
                    **{
                        key: body[key]
                        for key in (
                            "status",
                            "backend",
                            "negative",
                            "timed_out",
                            "wall_seconds",
                            "cycle_limit",
                            "timing",
                        )
                    },
                    "receipt": str(path.resolve()),
                    "comparison": bundle.get("comparison"),
                }
            )
    for path in args.searches:
        body = load(path, bindings)
        searches.append(
            {
                "receipt": str(path.resolve()),
                "summary": body["summary"],
                "winner": body["winner"],
                "records": body["records"],
            }
        )
    ops = [
        {
            "op": op,
            "contract_source": "maintained semantic explanation in report_el4.LIMITS; not inferred dispatch",
            "dtype": LIMITS[op][0],
            "execution": LIMITS[op][1],
            "limits": LIMITS[op][2],
            "observed_in": rows,
        }
        for op, rows in sorted(appearances.items())
    ]
    report = {
        "scope": "reduced_synthetic_recipe_diagnostics; not full model or FireSim qualification",
        "operator_table_scope": "Operator spellings/site counts are observed. Dtype/execution/limit prose is maintained contract documentation; exact dispatch appears in function_placement_and_refusals, not a proven source-op-to-function mapping.",
        "benchmarks": benchmarks,
        "ops": ops,
        "runs": runs,
        "searches": searches,
        "bindings": bindings + [identity(__file__)],
    }
    with args.output.with_suffix(".json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    lines = [
        "# EL4 recipe diagnostics",
        "",
        report["scope"],
        "",
        "Array and movement counts are static call_tir sites, not dynamic ISA command counts. A function using DMA may still execute CPU arithmetic.",
        "",
        cells(
            [
                "Workload",
                "Input shapes",
                "Recipe / calibration",
                "Array sites",
                "Movement sites",
                "All sites",
                "Status",
            ]
        ),
        cells(["---"] * 7),
    ]
    for row in benchmarks:
        lines.append(
            cells(
                [
                    row["workload"] + (" (score tuned)" if row["partial_tuning"] else ""),
                    row["input_shapes"],
                    f"{row['recipe']} / {row['calibration_count']}",
                    row["array_call_sites"],
                    row["movement_call_sites"],
                    row["total_call_sites"],
                    row["status"],
                ]
            )
        )
    lines += [
        "",
        "## Operator contracts",
        "",
        "Operator spellings and occurrence counts come from captures. The descriptions below are maintained semantic contracts, not receipt-derived per-op dispatch. Fusion can merge source ops; consult each benchmark's function dispatch evidence for measured placement.",
        "",
        cells(["Observed operation", "Contract dtype", "Execution contract", "Specific limits"]),
        cells(["---"] * 4),
    ]
    lines += [cells([row[key] for key in ("op", "dtype", "execution", "limits")]) for row in ops]
    lines += [
        "",
        "## Native checks",
        "",
        "Spike counters never provide timing labels. Wall time is experiment cost. RTL cycle limit is a timeout ceiling, not elapsed cycles.",
        "",
        cells(
            [
                "Bundle",
                "Engine",
                "Wrong-oracle control",
                "Verdict",
                "Host wall s",
                "Cycle ceiling",
                "Measured intervals",
            ]
        ),
        cells(["---"] * 7),
    ]
    for row in runs:
        timing = row["timing"]
        lines.append(
            cells(
                [
                    Path(row["bundle"]).name,
                    row["backend"],
                    row["negative"],
                    row["status"],
                    f"{row['wall_seconds']:.3f}",
                    row["cycle_limit"],
                    (timing["metric"] + ": " + str(timing["samples_by_stage"]))
                    if timing
                    else "none",
                ]
            )
        )
    lines += [
        "",
        "## Measured search",
        "",
        "These are bounded four-trial experiments, not exhaustive sweeps or validated ML generalization.",
        "",
        cells(["Search", "Candidate", "Placement", "Parameters", "Cycle samples", "Median cycles"]),
        cells(["---"] * 6),
    ]
    for experiment in searches:
        for row in experiment["records"]:
            lines.append(
                cells(
                    [
                        Path(experiment["receipt"]).parent.name,
                        row["candidate_id"][:12],
                        row["placement"],
                        row["parameters"],
                        row["measurement"]["samples"],
                        row["median_cycles"],
                    ]
                )
            )
    with args.output.with_suffix(".md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
