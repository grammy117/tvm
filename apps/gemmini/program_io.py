# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared semantic program intake for static compilation and tuning."""

from pathlib import Path

import tvm


def import_program(path):
    path = Path(path)
    if path.suffix == ".pt2":
        import torch
        import torch.ao.quantization.fx._decomposed  # registers serialized Q/DQ operators
        from tvm.relax.frontend.torch import from_exported_program

        return from_exported_program(torch.export.load(path), unwrap_unit_return_tuple=True)
    mod = tvm.ir.load_json(path.read_text())
    if not isinstance(mod, tvm.IRModule) or "main" not in mod:
        raise ValueError("Program must be a semantic Relax module with main")
    return mod
