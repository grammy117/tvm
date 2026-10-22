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
"""Extract mathematical functions and their actual CPU/Gemmini comparison inputs.

The CPU VM evaluates ordinary mathematical TIR and returns the first call site's
arguments/result for each shared function. These are numerical references and
measurement inputs, never host-time performance labels. Measure functions with
tune_tensorir.py and compare complete graphs separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini_tensorir import prepare_gemmini_tensorir_workloads

from verify_baremetal_cpu import identity


def extract(program, inputs_path, output_dir, *, max_functions=32, max_capture_bytes=16 << 20):
    output_dir.mkdir(parents=True, exist_ok=False)
    if program.suffix == ".pt2":
        import torch
        import torch.ao.quantization.fx._decomposed  # Registers serialized Q/DQ operators.
        from tvm.relax.frontend.torch import from_exported_program

        mod = from_exported_program(torch.export.load(program), unwrap_unit_return_tuple=True)
    else:
        mod = tvm.ir.load_json(program.read_text())
    mod, _, _ = prepare_gemmini_tensorir_workloads(mod)
    mod = relax.transform.ToNonDataflow()(mod)
    main = mod["main"]
    sites, seen, values = [], set(), []
    for block in main.body.blocks:
        for binding in block.bindings:
            call = binding.value
            if not isinstance(call, relax.Call) or call.op != tvm.ir.Op.get("relax.call_tir"):
                continue
            gv = call.args[0]
            if gv in seen:
                continue
            if not isinstance(binding.var.struct_info, relax.TensorStructInfo):
                raise ValueError("Function extraction currently requires single-tensor results")
            seen.add(gv)
            arguments = list(call.args[1])
            sites.append((gv, len(values), len(arguments)))
            values.extend(arguments + [binding.var])
    if not sites or len(sites) > max_functions:
        raise ValueError("Function extraction exceeds its explicit function budget")
    capture_bytes = 0
    for value in values:
        info = value.struct_info
        if not isinstance(info, relax.TensorStructInfo) or not isinstance(
            info.shape, relax.ShapeExpr
        ):
            raise ValueError("Function comparison needs static tensor arguments")
        size = 1
        for dim in info.shape:
            if not isinstance(dim, tir.IntImm) or int(dim) <= 0:
                raise ValueError("Function comparison needs positive static dimensions")
            size *= int(dim)
        capture_bytes += size * ((tvm.DataType(info.dtype).bits + 7) // 8)
    if capture_bytes > max_capture_bytes:
        raise ValueError("Function argument capture exceeds its explicit byte budget")
    values.append(main.body.body)
    proxy = relax.Function(
        main.params,
        relax.SeqExpr(main.body.blocks, relax.Tuple(values)),
        relax.TupleStructInfo([value.struct_info for value in values]),
    )
    diagnostic = tvm.IRModule(dict(mod.functions))
    diagnostic.update_func(diagnostic.get_global_var("main"), proxy)
    diagnostic = relax.transform.Normalize()(diagnostic)
    with np.load(inputs_path) as data:
        arrays = [data[f"input_{i}"].copy() for i in range(len(main.params))]
        expected = data["expected"].copy() if "expected" in data else None
    vm = relax.VirtualMachine(relax.build(diagnostic, "llvm"), tvm.cpu())
    captured = vm["main"](*[tvm.nd.array(value) for value in arrays])
    if expected is not None:
        np.testing.assert_allclose(captured[-1].numpy(), expected, rtol=1e-5, atol=1e-6)
    records = []
    for index, (gv, offset, count) in enumerate(sites):
        folder = output_dir / f"function-{index:03d}"
        folder.mkdir()
        mathematical = tvm.IRModule({"main": mod[gv]})
        (folder / "workload.json").write_text(tvm.ir.save_json(mathematical))
        arrays = {f"input_{i}": captured[offset + i].numpy() for i in range(count)}
        arrays["expected"] = captured[offset + count].numpy()
        np.savez(folder / "arrays.npz", **arrays)
        records.append(
            {
                "function": gv.name_hint,
                "directory": str(folder.resolve()),
                "scope": "first actual call site; full-graph cache and lifetime costs need separate comparison",
                "workload": identity(folder / "workload.json"),
                "arrays": identity(folder / "arrays.npz"),
            }
        )
    result = {
        "status": "passed",
        "functions": records,
        "source": identity(program),
        "inputs": identity(inputs_path),
        "whole_graph_reference": "passed" if expected is not None else "not_supplied",
        "script": identity(__file__),
    }
    (output_dir / "extraction.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-functions", type=int, default=32)
    parser.add_argument("--max-capture-bytes", type=int, default=16 << 20)
    args = parser.parse_args()
    result = extract(
        args.program.resolve(strict=True),
        args.inputs.resolve(strict=True),
        args.output_dir.resolve(),
        max_functions=args.max_functions,
        max_capture_bytes=args.max_capture_bytes,
    )
    print(json.dumps({"status": result["status"], "functions": len(result["functions"])}))


if __name__ == "__main__":
    main()
