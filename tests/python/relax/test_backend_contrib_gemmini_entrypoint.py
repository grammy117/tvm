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
"""Default preparation must not load the historical graph-replacement backend."""
import subprocess
import sys


def test_tensorir_preparation_loads_legacy_only_when_requested():
    subprocess.run(
        [sys.executable, "-c", '''
import sys
from tvm import relax
from tvm.relax.backend.contrib import gemmini
x = relax.Var("x", relax.TensorStructInfo((16, 16), "int8"))
y = relax.Var("y", relax.TensorStructInfo((16, 16), "int8"))
bb = relax.BlockBuilder()
with bb.function("main", [x, y]):
    bb.emit_func_output(relax.op.matmul(x, y, out_dtype="int32"))
gemmini.prepare_gemmini_graph(bb.get())
assert "tvm.relax.backend.contrib.gemmini_legacy" not in sys.modules
from tvm.relax.backend.contrib.gemmini import LowerGemminiMatmul
assert LowerGemminiMatmul is not None
assert "tvm.relax.backend.contrib.gemmini_legacy" in sys.modules
'''], check=True, timeout=30,
    )
