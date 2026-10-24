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
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Static graph/memory evidence, distinct from measured DMA or cache traffic."""
from collections import Counter


def audit_graph(graph, placement, capabilities):
    by_symbol = {row["function"]: row for row in placement}
    tensors = {row["name"]: row for group in ("inputs", "outputs", "intermediates", "constants", "views")
               for row in graph.get(group, [])}
    producers, edges, calls = {}, [], []
    counts = Counter()
    for index, call in enumerate(graph["calls"]):
        row = by_symbol.get(call["symbol"], {})
        destination = row.get("placement", "cpu")
        counts[destination] += 1
        metadata = row.get("metadata", {})
        spad, acc = metadata.get("spad_rows"), metadata.get("acc_rows")
        calls.append({"index": index, "symbol": call["symbol"], "placement": destination,
                      "origin": row.get("origin", "CPU-only build or no placement record"),
                      "scratchpad_bytes": None if spad is None else spad * capabilities.dim * capabilities.input_bits // 8,
                      "accumulator_bytes": None if acc is None else acc * capabilities.dim * capabilities.accumulator_bits // 8,
                      "waited_host_buffers": metadata.get("waited_host_buffers"),
                      "extra_completion_fences": metadata.get("completion_fences")})
        for name in dict.fromkeys(call["inputs"]):
            if name in producers:
                edges.append({"producer": producers[name], "consumer": index, "tensor": name,
                              "logical_bytes": tensors[name]["bytes"], "storage": "host"})
        for name in call["outputs"]:
            producers[name] = index
    return {"schema": "gemmini_package_memory_audit_v1", "calls_by_placement": dict(counts),
            "unique_functions": len({c["symbol"] for c in calls}), "calls": calls,
            "host_materialized_edges": edges,
            "logical_edge_bytes": sum(e["logical_bytes"] for e in edges),
            "edge_scope": "direct named producer-consumer edges only; aliases/views not expanded; logical bytes, not DRAM/DMA counters",
            "memory_preflight": graph["memory_preflight"],
            "cross_function_sram_residency": False,
            "capacity_bytes": {"scratchpad": capabilities.spad_rows * capabilities.dim * capabilities.input_bits // 8,
                               "accumulator": capabilities.acc_rows * capabilities.dim * capabilities.accumulator_bits // 8},
            "timing": "not measured; do not sum historical operator minima as full-graph latency"}
