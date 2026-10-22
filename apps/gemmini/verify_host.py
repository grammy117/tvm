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
"""Verify the configured TVM checkout with a tiny Relax matmul on the host CPU."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import tvm
from tvm import relax
from tvm._ffi.base import _LIB
from tvm.script import tir as T


@T.prim_func
def _annotated_copy(left: T.Buffer((4,), "float32"), right: T.Buffer((4,), "float32")):
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    for i in T.serial(4):
        with T.block("copy"):
            vi = T.axis.spatial(4, i)
            T.block_attr({"auto_copy": 1})
            right[vi] = left[vi]


def verify_auto_copy_guard():
    """Require explicit rejection, including after the normal driver's early passes."""
    marker = tvm.get_global_func("tir.transform.HostOnlyAutoCopyGuardEnabled", allow_missing=True)
    if marker is None:
        return {"enabled": False}
    if not marker():
        raise RuntimeError("Unexpected disabled host-only guard marker")

    annotated = tvm.IRModule({"main": _annotated_copy})
    schedule = tvm.tir.Schedule(annotated)
    schedule.unannotate(schedule.get_block("copy"), "auto_copy")
    clean = schedule.mod
    tvm.ir.assert_structural_equal(tvm.tir.transform.LowerAutoCopy()(clean), clean)

    loop_schedule = tvm.tir.Schedule(clean)
    loop_schedule.annotate(loop_schedule.get_loops(loop_schedule.get_block("copy"))[0], "auto_copy", 1)
    func = clean["main"]
    attr_stmt = tvm.tir.AttrStmt(tvm.tir.IntImm("int32", 0), "auto_copy", 1, func.body)
    cases = {
        "block": annotated,
        "loop": loop_schedule.mod,
        "function": tvm.IRModule({"main": func.with_attr("auto_copy", 1)}),
        "attribute_statement": tvm.IRModule({"main": func.with_body(attr_stmt)}),
    }
    checks = []
    for name, mod in cases.items():
        for route in ("direct_pass", "driver_build"):
            try:
                if route == "direct_pass":
                    tvm.tir.transform.LowerAutoCopy()(mod)
                else:
                    tvm.build(mod, target="llvm")
            except tvm.TVMError as err:
                if "HostOnlyAutoCopyGuard" not in str(err):
                    raise
            else:
                raise AssertionError(f"Auto-copy guard did not reject {name} via {route}")
            checks.append(f"{name}:{route}")
    return {"enabled": True, "unannotated_ir_preserved": True, "rejection_checks": checks}


def verify():
    """Compile, execute, and return provenance plus numerical verification."""
    source_root = Path(__file__).resolve().parents[2]
    python_location = Path(tvm.__file__).resolve()
    if not python_location.is_relative_to(source_root / "python"):
        raise RuntimeError(f"Expected this checkout's Python package, got {python_location}")
    if not tvm.runtime.enabled("llvm"):
        raise RuntimeError("The loaded TVM library does not have LLVM enabled")

    left = relax.Var("left", relax.TensorStructInfo((2, 3), "float32"))
    right = relax.Var("right", relax.TensorStructInfo((3, 4), "float32"))
    builder = relax.BlockBuilder()
    with builder.function("main", [left, right]):
        with builder.dataflow():
            result = builder.emit_output(relax.op.matmul(left, right))
        builder.emit_func_output(result)
    executable = relax.build(builder.get(), target="llvm")
    device = tvm.cpu()
    vm = relax.VirtualMachine(executable, device)
    left_np = (np.arange(6, dtype="float32").reshape(2, 3) - 2.5) / 3
    right_np = (np.arange(12, dtype="float32").reshape(3, 4) - 5.5) / 7
    actual = vm["main"](tvm.nd.array(left_np, device), tvm.nd.array(right_np, device)).numpy()
    expected = left_np @ right_np
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    guard = verify_auto_copy_guard()
    info = tvm.support.libinfo()
    fields = ("GIT_COMMIT_HASH", "LLVM_VERSION", "USE_LLVM", "TVM_CXX_COMPILER_PATH", "USE_CUDA", "USE_ROCM", "USE_OPENCL", "USE_VULKAN", "USE_METAL", "USE_LIBBACKTRACE", "HIDE_PRIVATE_SYMBOLS")
    return {
        "status": "passed",
        "scope": "Host LLVM code generation and CPU execution of one synthetic Relax matmul",
        "tvm_python_location": str(python_location),
        "tvm_version": tvm.__version__,
        "loaded_libtvm_path": str(Path(_LIB._name).resolve()),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "llvm_enabled": True,
        "host_only_auto_copy_guard": guard,
        "build_info": {key: str(info[key]) for key in fields if key in info},
        "matmul": {
            "left_shape": list(left_np.shape), "right_shape": list(right_np.shape),
            "dtype": "float32", "target": "llvm", "device": "cpu",
            "max_absolute_error": float(np.max(np.abs(actual - expected))),
            "rtol": 1e-5, "atol": 1e-6,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Explicit path for an optional JSON record")
    args = parser.parse_args()
    record = json.dumps(verify(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(record, encoding="utf-8")
    print(record, end="")


if __name__ == "__main__":
    main()
