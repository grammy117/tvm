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
"""Experiment inputs, never workload-name dispatch in the compiler.

Single operators retain real M/N/K/spatial dimensions. Connected surrogates
exercise intermediate materialization and CPU glue; they are not EL4 numerical
captures. ``slice_program`` instead cuts exact bindings from a supplied prepared
program, retaining every live-out and exposing every external dependency.
"""

import numpy as np
import tvm
from tvm import relax

import model_workloads

# Calls per selected full inference, not repetitions inside the small graph.
# These are architectural counts; quantization can split an actual IR site.
CASES = {
    "tinyllama-decode-kv": dict(model="tinyllama-decode", operator=1, stage_calls=32, sites=44),
    "tinyllama-decode-qo": dict(model="tinyllama-decode", operator=0, stage_calls=32, sites=44),
    "tinyllama-prefill-kv": dict(model="tinyllama-prefill", operator=1, stage_calls=1, sites=44),
    "smolvla-action-in": dict(model="smolvla-expert", operator=5, stage_calls=10, sites=1),
    "smolvla-action-out": dict(model="smolvla-expert", operator=8, stage_calls=10, sites=1),
    "resnet50-layer4-reduce": dict(model="resnet50", operator=21, stage_calls=1, sites=2),
    "resnet50-layer4-spatial": dict(model="resnet50", operator=22, stage_calls=1, sites=2),
    "tinyllama-decode-mlp": dict(chain="mlp", rows=1, stage_calls=32, sites=22),
    "tinyllama-prefill-mlp": dict(chain="mlp", rows=32, stage_calls=1, sites=22),
    "smolvla-action-time": dict(chain="action_time", rows=50, stage_calls=10, sites=1),
    "resnet50-layer4-pair": dict(chain="conv_pair", stage_calls=1, sites=2),
}


def bindings(mod):
    function = mod["main"]
    if not isinstance(function, relax.Function) or not isinstance(function.body, relax.SeqExpr):
        raise ValueError("Require a normalized semantic Relax main")
    result = [binding for block in function.body.blocks for binding in block.bindings]
    if any(not isinstance(binding, relax.VarBinding) for binding in result):
        raise ValueError("Only static VarBinding chunks are supported")
    return result


def inventory(mod):
    return [{"index": i, "name": b.var.name_hint, "value": str(b.value),
             "struct_info": str(b.var.struct_info)} for i, b in enumerate(bindings(mod))]


@relax.expr_functor.mutator
class _Substitute(relax.PyExprMutator):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def visit_var_(self, var):
        return self.mapping.get(var, var)

    def visit_dataflow_var_(self, var):
        return self.mapping.get(var, var)


def slice_program(mod, start, stop):
    """Extract [start, stop), preserving expressions, constants and boundary types.

    Only pure semantic calls are admitted. No reductions/dimensions are cut.
    External constant bindings are cloned; other live-ins become explicit inputs.
    A downstream user of any selected value makes it an output, even if that
    value also has an internal consumer. This preserves residual/cache edges.
    """
    rows = bindings(mod)
    if not 0 <= start < stop <= len(rows):
        raise ValueError("Require a nonempty in-range binding interval")
    selected = rows[start:stop]
    produced = {b.var for b in selected}
    definitions = {b.var: b.value for b in rows}
    external, live_out = [], set()
    for b in selected:
        def check(node):
            if isinstance(node, (relax.ExternFunc, tvm.ir.GlobalVar)):
                raise ValueError("Chunks require pure semantic operators; opaque/global calls refused")
            if isinstance(node, relax.Call):
                if not isinstance(node.op, tvm.ir.Op) or not bool(node.op.get_attr("FPurity")):
                    raise ValueError("Chunks require pure semantic operators; opaque/global calls refused")
        relax.analysis.post_order_visit(b.value, check)
        for var in relax.analysis.free_vars(b.value):
            if var not in produced and var not in external:
                external.append(var)
    for value in [b.value for b in rows[stop:]] + [mod["main"].body.body]:
        live_out.update(v for v in relax.analysis.free_vars(value) if v in produced)
    outputs = [b.var for b in selected if b.var in live_out]
    if not outputs:
        raise ValueError("Selected bindings have no live output")
    params, mapping, input_names = [], {}, []
    for var in external:
        if isinstance(definitions.get(var), relax.Constant):
            mapping[var] = definitions[var]
        else:
            fresh = relax.Var(var.name_hint, var.struct_info)
            params.append(fresh)
            mapping[var] = fresh
            input_names.append(var.name_hint)
    bb = relax.BlockBuilder()
    with bb.function("main", params):
        with bb.dataflow():
            for b in selected:
                mapping[b.var] = bb.emit(_Substitute(mapping).visit_expr(b.value), name_hint=b.var.name_hint)
            out = bb.emit_output(relax.Tuple([mapping[v] for v in outputs]))
        bb.emit_func_output(out)
    result = bb.get()
    # Preserve numerical policy annotations; only the runtime signature changes.
    function = result["main"]
    for key, value in (mod["main"].attrs or {}).items():
        function = function.with_attr(key, len(params) if key == "num_input" else value)
    result.update_func(result.get_global_var("main"), function)
    for key, value in (mod.attrs or {}).items():
        result = result.with_attr(key, value)
    if not relax.analysis.well_formed(result):
        raise ValueError("Extracted chunk is not well formed")
    return result, {"binding_interval": [start, stop], "live_inputs": input_names,
                    "live_outputs": [v.name_hint for v in outputs],
                    "shape_policy": "unchanged; no reduction, channel or spatial cropping"}


def single_operator(model, operator):
    mod, description = model_workloads.build(model, operators=[operator])
    rows = bindings(mod)
    # The representative factory returns a tuple of independent repeated sites.
    # Retain exactly the first site's dependency prefix, not all layer copies.
    returned = rows[-1].value
    if not isinstance(returned, relax.Tuple):
        raise ValueError("Representative factory no longer returns its operator tuple")
    output = returned.fields[0]
    end = next(i + 1 for i, b in enumerate(rows) if b.var.same_as(output))
    chunk, boundary = slice_program(mod, 0, end)
    return chunk, dict(boundary, operators=description, executed_sites=1)


class Chain:
    """Legacy W8A8/ResNet surrogate semantics, explicit CPU work between GEMMs."""

    def __init__(self, seed=0):
        self.bb = relax.BlockBuilder()
        self.rng = np.random.default_rng(seed)
        self.params = []

    def emit(self, value):
        return self.bb.emit(value)

    def input(self, name, shape, dtype="float32"):
        value = relax.Var(name, relax.TensorStructInfo(shape, dtype))
        self.params.append(value)
        return value

    def quantize(self, x):
        scale = self.emit(relax.op.max(self.emit(relax.op.abs(x)), axis=-1, keepdims=True))
        scale = self.emit(relax.op.maximum(scale, relax.const(1e-6, "float32")))
        scale = self.emit(relax.op.divide(scale, relax.const(127., "float32")))
        codes = self.emit(relax.op.divide(x, scale))
        codes = self.emit(relax.op.round(codes))
        codes = self.emit(relax.op.clip(codes, -127, 127))
        return self.emit(relax.op.astype(codes, "int8")), scale

    def projection(self, x, k, n):
        codes, scale = self.quantize(x)
        w = relax.const(self.rng.integers(-127, 128, (k, n), dtype="int8"))
        ws = relax.const(self.rng.uniform(.001, .01, (n,)).astype("float32"))
        product = self.emit(relax.op.matmul(codes, w, out_dtype="int32"))
        value = self.emit(relax.op.astype(product, "float32"))
        return self.emit(relax.op.multiply(self.emit(relax.op.multiply(value, scale)), ws))

    def silu(self, x):
        return self.emit(relax.op.multiply(x, self.emit(relax.op.sigmoid(x))))

    def conv(self, x, cin, cout, kernel):
        w = relax.const(self.rng.integers(-127, 128, (cout, kernel, kernel, cin), dtype="int8"))
        bias = relax.const(self.rng.integers(-100, 101, (1, 1, 1, cout), dtype="int32"))
        divisor = relax.const(np.full((1, 1, 1, cout), 1024, dtype="int64"), "int64")
        half = relax.const(np.full((1, 1, 1, cout), 512, dtype="int64"), "int64")
        x = self.emit(relax.op.nn.conv2d(x, w, padding=kernel // 2, data_layout="NHWC",
                                        kernel_layout="OHWI", out_dtype="int32"))
        x = self.emit(relax.op.astype(self.emit(relax.op.add(x, bias)), "int64"))
        magnitude = self.emit(relax.op.abs(x))
        rounded = self.emit(relax.op.floor_divide(self.emit(relax.op.add(magnitude, half)), divisor))
        signed = self.emit(relax.op.where(self.emit(relax.op.less(x, relax.const(0, "int64"))),
                                         self.emit(relax.op.negative(rounded)), rounded))
        return self.emit(relax.op.astype(self.emit(relax.op.clip(signed, 0, 127)), "int8"))


def connected(kind, rows=1):
    c = Chain()
    with c.bb.function("main"):
        with c.bb.dataflow():
            if kind == "mlp":
                x = c.input("normalized_hidden", (rows, 2048))
                residual = c.input("residual", (rows, 2048))
                gate = c.silu(c.projection(x, 2048, 5632))
                up = c.projection(x, 2048, 5632)
                down = c.projection(c.emit(relax.op.multiply(gate, up)), 5632, 2048)
                result = c.emit(relax.op.add(residual, down))
            elif kind == "action_time":
                x = c.input("actions", (50, 32))
                t = c.input("time_embedding", (50, 720))
                x = c.projection(x, 32, 720)
                x = c.emit(relax.op.concat([x, t], axis=-1))
                x = c.silu(c.projection(x, 1440, 720))
                result = c.projection(x, 720, 720)
            elif kind == "conv_pair":
                x = c.input("activation", (1, 7, 7, 2048), "int8")
                result = c.conv(c.conv(x, 2048, 512, 1), 512, 512, 3)
            else:
                raise ValueError("Unknown connected surrogate")
            out = c.bb.emit_output(result)
        c.bb.emit_func_output(out, params=c.params)
    return c.bb.get()


def build_case(name):
    case = CASES[name]
    if "operator" in case:
        mod, detail = single_operator(case["model"], case["operator"])
    else:
        mod = connected(case["chain"], case.get("rows", 1))
        detail = {"connected": True, "executed_blocks": 1}
    return mod, dict(case, **detail, case=name, source_kind="synthetic_shape_surrogate",
                     numerical_identity="legacy pilot recipe; not the selected EL4 capture",
                     full_model_qualified=False,
                     session_occurrences=case["stage_calls"] * case["sites"])
