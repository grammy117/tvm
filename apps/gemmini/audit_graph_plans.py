#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the
# specific language governing permissions and limitations
# under the License.
"""Audit effective graph choices on the exact input used for compilation."""

import argparse
import json
from pathlib import Path

import tvm

from graph_plans import GraphPlan, audit_plans
from memory_model import identity
from program_io import import_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--plans", nargs="+", help="Canonical graph choices; default audits all 32")
    args = parser.parse_args()
    # Refuse to replace a previous audit, including an incomplete one.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = identity(args.program)
    context = {"program": source, "tvm_python": identity(tvm.__file__),
               "tvm_library": identity(tvm._ffi.base._LIB._name),
               "audit_source": identity(__file__),
               "graph_plan_source": identity(Path(__file__).with_name("graph_plans.py"))}
    (args.output_dir / "inputs.json").write_text(json.dumps(context, indent=2) + "\n")
    alternatives = None if args.plans is None else [GraphPlan.from_name(p) for p in args.plans]
    mod = import_program(args.program)
    result = audit_plans(mod, alternatives)
    if identity(args.program) != source:
        raise ValueError("Program changed during audit")
    result.update(context)
    (args.output_dir / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in
                     ("configurations", "distinct_prepared_graphs", "option_effects")}))


if __name__ == "__main__":
    main()
