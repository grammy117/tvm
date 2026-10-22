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
"""Package a captured EL4 slice for the ordinary bounded native verifier."""

import argparse
import json
from pathlib import Path

import numpy as np
import tvm

import verify_tensorir as verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--riscv-gcc", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=0)
    args = parser.parse_args()
    source = args.capture.resolve(strict=True)
    result = json.loads((source / "result.json").read_text())
    if result["status"] != "passed":
        raise ValueError("The capture must first pass import, scheduling and CPU numerical checks")
    for binding in result["bindings"]:
        if verifier.identity(binding["path"])["sha256"] != binding["sha256"]:
            raise ValueError("Capture input changed: " + binding["path"])
    graph = tvm.ir.load_json((source / "scheduled.json").read_text())
    with np.load(source / "arrays.npz") as data:
        arrays = [data[f"input_{index}"].copy() for index in range(len(result["input_shapes"]))]
        expected = data["expected"].copy()
    report = result["scheduling"]
    native = sum(row["placement"] == "gemmini" for row in report)
    case = ("slice", graph, arrays, expected, "gemmini" if native else "host", {}, native)
    args.cases, args.libm, args.rtol, args.atol = ["slice"], True, 1e-5, 1e-6
    verifier.prepare(
        args,
        cases=[case],
        prepared_reports={"slice": report},
        additional_bindings=[
            Path(__file__),
            source / "result.json",
            source / "arrays.npz",
            source / "scheduled.json",
        ],
    )


if __name__ == "__main__":
    main()
