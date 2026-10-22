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
"""Explicit compiler capabilities for the pinned integer Gemmini configuration.

The target remains LLVM/RV64: Rocket issues the accelerator commands. A target
key selects Gemmini scheduling, and the model fingerprint binds the capability
profile into ordinary TVM tuning database identities. Registration is not device
discovery; deployment must bind the selected headers, RTL and runtime separately.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

from tvm.target import Target


@dataclass(frozen=True)
class GemminiCapabilities:
    """Source-backed geometry plus the implemented primitive-only ABI subset."""

    dim: int = 16
    spad_rows: int = 16384
    acc_rows: int = 1024
    pe_bits: int = 20
    input_bits: int = 8
    accumulator_bits: int = 32
    dataflow: str = "OS"
    hardware_loops: bool = False
    accumulator_load_scale: bool = False
    full_width_readout: bool = True
    readout_relu: bool = True
    params_sha256: str = "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
    config_sha256: str = "178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b"

    def __post_init__(self):
        if (
            self.dim,
            self.pe_bits,
            self.input_bits,
            self.accumulator_bits,
            self.dataflow,
            self.hardware_loops,
            self.accumulator_load_scale,
            self.full_width_readout,
            self.readout_relu,
        ) != (16, 20, 8, 32, "OS", False, False, True, True):
            raise ValueError("Unsupported Gemmini primitive ABI configuration")
        if not 0 < self.spad_rows <= 16384 or not 0 < self.acc_rows <= 1024:
            raise ValueError("Gemmini SRAM limits exceed the bound hardware profile")

    @property
    def max_reduction(self):
        return ((1 << (self.accumulator_bits - 1)) - 1) // (1 << (2 * (self.input_bits - 1)))

    @property
    def max_chunk(self):
        return min(self.dim, ((1 << (self.pe_bits - 1)) - 1) // (1 << (2 * (self.input_bits - 1))))

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def target(self):
        """Return a supported LLVM target with explicit accelerator dispatch keys."""
        return Target(
            {
                "kind": "llvm",
                "keys": ["gemmini", "cpu"],
                "model": "gemmini-" + self.fingerprint,
                "mtriple": "riscv64-unknown-elf",
                "mcpu": "generic-rv64",
                "mabi": "lp64d",
                "mattr": ["+m", "+a", "+f", "+d", "+c"],
                "num-cores": 1,
            }
        )
