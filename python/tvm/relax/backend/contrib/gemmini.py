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

"""Gemmini graph preparation and preserved legacy external-call lowering.

The default graph entry point retains ordinary mathematical legalization and
fusion before TensorIR scheduling. Legacy Relax replacement passes run before
LegalizeOps and remain available explicitly. Link apps/gemmini/matmul.h's
primitive implementation when exporting accelerator code. Unsupported calls
retain CPU lowering.
"""

from math import prod

import tvm
from tvm import relax, tir
from tvm._ffi.runtime_ctypes import DataTypeCode
from tvm.relax.expr_functor import PyExprMutator, mutator
from tvm.relax.pipeline import default_build_pipeline, register_pipeline

_SYMBOL = "tvm_gemmini_matmul_i8_i32"
_MAX_K = ((1 << 31) - 1) // (128 * 128)
_MAX_BYTES = (1 << 63) - 1


def _shape(info, dtype, ndim=2):
    if not isinstance(info, relax.TensorStructInfo) or info.dtype != dtype or info.ndim != ndim:
        return None
    if not isinstance(info.shape, relax.ShapeExpr):
        return None
    if info.vdevice is not None and info.vdevice.target.kind.name not in ("llvm", "c"):
        return None
    if any(not isinstance(dim, tir.IntImm) or dim.value <= 0 for dim in info.shape):
        return None
    return tuple(int(dim.value) for dim in info.shape)


@mutator
class _LinearLowerer(PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        if not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.matmul":
            return call
        rank = call.args[0].struct_info.ndim
        if rank < 1 or rank == 2:
            return call
        left = _shape(call.args[0].struct_info, "int8", rank)
        right = _shape(call.args[1].struct_info, "int8")
        output = _shape(call.struct_info, "int32", rank)
        if left is None or right is None or output is None:
            return call
        k, n = right
        m = prod(left[:-1])
        if left[-1] != k or output != left[:-1] + (n,) or k > _MAX_K or n * 4 > (1 << 32) - 1:
            return call
        if max(m * k, k * n, m * n * 4) > _MAX_BYTES:
            return call
        flattened = self.builder_.emit(relax.op.reshape(call.args[0], (m, k)))
        product = self.builder_.emit(relax.op.matmul(flattened, call.args[1], out_dtype="int32"))
        return self.builder_.normalize(relax.op.reshape(product, output))

    def visit_function_(self, func):
        if func.attrs and ("Codegen" in func.attrs or "Composite" in func.attrs):
            return func
        return super().visit_function_(func)


@tvm.transform.module_pass(opt_level=0, name="DecomposeGemminiLinear")
class DecomposeGemminiLinear:
    """Flatten static INT8 vectors/batches multiplying a shared 2D weight.

    Reshapes preserve the INT32 contraction and original output shape. Quantization,
    bias and dequantization remain explicit graph operations. Batched weights,
    dynamic/empty shapes, non-CPU tensors and unsupported arithmetic stay unchanged.
    """

    def transform_module(self, mod, _ctx):
        lowerer = _LinearLowerer(mod)
        for gv, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        return lowerer.builder_.get()


def _eligible(call):
    if not isinstance(call.op, tvm.ir.Op) or call.op.name != "relax.matmul":
        return None
    left, right = (_shape(arg.struct_info, "int8") for arg in call.args)
    output = _shape(call.struct_info, "int32")
    if left is None or right is None or output is None:
        return None
    m, k = left
    inner, n = right
    if inner != k or output != (m, n) or k > _MAX_K:
        return None
    if max(m * k, k * n, m * n * 4) > _MAX_BYTES:
        return None
    return m, n, k


def _wrapper(m, n, k):
    a = tir.decl_buffer((m, k), "int8", name="a")
    b = tir.decl_buffer((k, n), "int8", name="b")
    c = tir.decl_buffer((m, n), "int32", name="c")
    dims = [tir.IntImm("int64", value) for value in (m, n, k, k, n, n)]
    result = tir.call_extern("int32", _SYMBOL, a.data, b.data, c.data, *dims)
    status = tir.Var("status", "int32")
    error = tir.SeqStmt([tir.Evaluate(tir.call_extern("void", "TVMAPISetLastError", tir.StringImm("Gemmini external matmul returned an error"))), tir.Evaluate(tir.tvm_throw_last_error())])
    body = tir.LetStmt(status, result, tir.IfThenElse(status != 0, error, None))
    return tir.PrimFunc([a.data, b.data, c.data], body, buffer_map={a.data: a, b.data: b, c.data: c}).with_attr("tir.noalias", True)


@mutator
class _MatmulLowerer(PyExprMutator):
    def __init__(self, mod, wrapper=_wrapper, name="gemmini_matmul", eligible=_eligible):
        super().__init__(mod)
        self.wrappers = {}
        self.wrapper = wrapper
        self.name = name
        self.eligible = eligible

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        shape = self.eligible(call)
        if shape is None:
            return call
        if shape not in self.wrappers:
            self.wrappers[shape] = self.builder_.add_func(self.wrapper(*shape), self.name)
        return self.builder_.normalize(relax.call_tir(self.wrappers[shape], tuple(call.args), call.struct_info))

    def visit_function_(self, func):
        if func.attrs and ("Codegen" in func.attrs or "Composite" in func.attrs):
            return func
        return super().visit_function_(func)


@tvm.transform.module_pass(opt_level=0, name="LowerGemminiMatmul")
class LowerGemminiMatmul:
    """Lower positive static 2D signed-int8 matmul with signed-int32 output.

    The external function must synchronously write the complete output, preserve
    inputs, and honor the C ABI. No bias, scaling, activation or batching is fused.
    The reduction bound guarantees exact int32 sums for every int8 input value.
    DLTensor arguments must be compact; generated TIR wrappers check that ABI.
    This opt-in pass does not choose hardware headers, runtime or instruction policy.
    """

    def transform_module(self, mod, _ctx):
        lowerer = _MatmulLowerer(mod)
        for gv, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        return lowerer.builder_.get()


@tvm.transform.module_pass(opt_level=0, name="LowerGemminiScheduledMatmul")
class LowerGemminiScheduledMatmul:
    """Opt-in static 2D matmul scheduling with genuine TensorIntrin substitution.

    Run before LegalizeOps; unsupported operations keep their ordinary lowering.
    This first integration converts matmul to call_tir before later FuseTIR, so
    it does not claim fused-contraction scheduling or whole-graph paper fidelity.
    gemmini_schedule exposes semantic construction and tensorization separately
    for future graph/TIR integration before final intrinsic substitution.
    Link the primitive C ABI with its pinned no-FSM hardware/header contract.
    """

    def __init__(self, tile_i=1, tile_j=1, tile_choices=None):
        if any(type(x) is not int or x not in (1, 2, 4) for x in (tile_i, tile_j)):
            raise ValueError("Gemmini macro tiles must be 1, 2 or 4")
        self.tile_i = tile_i
        self.tile_j = tile_j
        if tile_choices is not None:
            if not isinstance(tile_choices, dict):
                raise ValueError("Tile choices must be a dictionary")
            for shape, tiles in tile_choices.items():
                if not isinstance(shape, tuple) or len(shape) != 3 or any(type(value) is not int or value <= 0 for value in shape):
                    raise ValueError("Tile choices require positive (M,N,K) shapes")
                if not isinstance(tiles, tuple) or len(tiles) != 2 or any(type(value) is not int or value not in (1, 2, 4) for value in tiles):
                    raise ValueError("Tile choices require legal (I,J) macro tiles")
        self.tile_choices = dict(tile_choices or {})

    def transform_module(self, mod, _ctx):
        from .gemmini_schedule import make_gemmini_matmul  # pylint: disable=import-outside-toplevel

        def wrapper(m, n, k):
            tiles = self.tile_choices.get((m, n, k), (self.tile_i, self.tile_j))
            return make_gemmini_matmul(m, n, k, *tiles).scheduled_mod["main"].without_attr("global_symbol").with_attr("op_pattern", 8)

        def eligible(call):
            shape = _eligible(call)
            if shape is not None and max(shape[2], shape[1] * 4) > (1 << 32) - 1:
                return None
            return shape

        lowerer = _MatmulLowerer(mod, wrapper, "gemmini_scheduled_matmul", eligible)
        for gv, func in list(mod.functions_items()):
            if isinstance(func, relax.Function):
                lowerer.builder_.update_func(gv, lowerer.visit_expr(func))
        return lowerer.builder_.get()


def prepare_gemmini_graph_legacy(mod, tile_i=1, tile_j=1, optimize=True, tile_choices=None):
    """Prepare a semantic Relax graph while keeping device calls opaque.

    Eligible integer convolutions first become explicit CPU packing/restoration
    and semantic matmul. The optimized path folds constants and canonicalizes bindings before Gemmini
    substitution, then legalizes and fuses surrounding CPU operations. Internal
    pointwise producers are inlined by a TIR Schedule when it proves legality;
    returned buffers remain explicit. Fully
    constant matmuls can fold on the CPU before any device IR exists. The signed
    int8/int32 contraction and separate bias, clipping, casts and shifts retain
    their original graph order; no affine quantization correction is invented.

    Gemmini's admission/primitive body has op_pattern=8 (opaque), so FuseOps does
    not group it with CPU operations and FuseTIR does not rewrite its envelope.
    This is fusion around a boundary, not fusion inside the accelerator schedule.
    Both paths inline legal operator-local pointwise buffers so padding does not
    require a baremetal heap. optimize=False otherwise supplies the scheduled/
    LegalizeOps baseline without constant folding or graph fusion. The optimized
    path requires semantic input, rather than an already
    device-lowered module: generic FoldConstant may CPU-evaluate call_tir.
    Candidate search can pass its bound tile_i/tile_j without altering this order.
    """
    lower = LowerGemminiScheduledMatmul(tile_i, tile_j, tile_choices)
    if type(optimize) is not bool:
        raise ValueError("optimize must be a boolean")
    if optimize:
        device_calls = []
        for func in mod.functions.values():
            if isinstance(func, tir.PrimFunc):
                def find_device(node):
                    if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern") and isinstance(node.args[0], tir.StringImm) and node.args[0].value.startswith("tvm_gemmini_"):
                        device_calls.append(node)

                tir.stmt_functor.post_order_visit(func.body, find_device)
        if device_calls:
            raise ValueError("Optimized Gemmini preparation requires a semantic graph before device lowering")
    from .gemmini_conv import DecomposeGemminiConv2D  # pylint: disable=import-outside-toplevel

    mod = DecomposeGemminiLinear()(DecomposeGemminiConv2D()(mod))
    if optimize:
        mod = tvm.transform.Sequential([relax.transform.FoldConstant(), relax.transform.CanonicalizeBindings()])(mod)
    mod = relax.transform.LegalizeOps()(lower(mod))
    if optimize:
        mod = tvm.transform.Sequential([relax.transform.AnnotateTIROpPattern(), relax.transform.FuseOps(fuse_opt_level=2), relax.transform.FuseTIR()])(mod)
    return _inline_cpu_producers(mod)


def prepare_gemmini_graph(mod, tile_i=1, tile_j=1, optimize=True, tile_choices=None,
                          *, route="tensorir", capabilities=None, order="ij", schedule_options=None,
                          report=None, bias_preload=True, offload_copies=False, prepack_weights=True,
                          movement_strategy="rectangular", database=None, fusion_level=-1,
                          eliminate_common_subexpr=False):
    """Prepare Gemmini through mathematical TensorIR; retain explicit legacy mode.

    The default route legalizes/fuses ordinary computations before structural
    scheduling. Legacy template tile parameters are accepted only with the
    explicitly selected regression route, never silently ignored.
    """
    if route == "legacy":
        if eliminate_common_subexpr:
            raise ValueError("Graph CSE choices require route='tensorir'")
        if fusion_level != -1:
            raise ValueError("Graph fusion choices require route='tensorir'")
        if offload_copies:
            raise ValueError("Copy offload choices require route='tensorir'")
        if bias_preload is not True:
            raise ValueError("bias_preload choices require route='tensorir'")
        if schedule_options:
            raise ValueError("TensorIR schedule options require route='tensorir'")
        return prepare_gemmini_graph_legacy(mod, tile_i, tile_j, optimize, tile_choices)
    if route != "tensorir":
        raise ValueError("Gemmini route must be tensorir or legacy")
    if (tile_i, tile_j) != (1, 1) or tile_choices:
        raise ValueError("Legacy macro tile parameters require route='legacy'; use TensorIR schedule rules")
    if type(optimize) is not bool:
        raise ValueError("optimize must be a boolean")
    from .gemmini_tensorir import prepare_gemmini_tensorir_graph  # pylint: disable=import-outside-toplevel

    return prepare_gemmini_tensorir_graph(mod, optimize=optimize, capabilities=capabilities,
                                         order=order, schedule_options=schedule_options, report=report,
                                         bias_preload=bias_preload, offload_copies=offload_copies,
                                         prepack_weights=prepack_weights, movement_strategy=movement_strategy,
                                         database=database, fusion_level=fusion_level,
                                         eliminate_common_subexpr=eliminate_common_subexpr)


@register_pipeline("gemmini")
def gemmini_pipeline(**options):
    """Select the backend through ``relax.get_pipeline('gemmini', ...)``.

    Import this contrib module once to register it. Options are the same as
    prepare_gemmini_graph, including explicit legacy regression selection.
    """
    @tvm.transform.module_pass(opt_level=0, name="GemminiPipeline")
    def pipeline(mod, _ctx):
        return prepare_gemmini_graph(mod, **options)
    return pipeline


@register_pipeline("gemmini_build")
def gemmini_build_pipeline(**options):
    """Compose Gemmini preparation with TVM's stock Relax VM lowering."""
    return tvm.transform.Sequential([gemmini_pipeline(**options),default_build_pipeline()])


def _inline_cpu_producers(mod):
    """Remove legal operator-local pointwise temporaries, including padding.

    A raw baremetal exporter has no implicit TVM workspace allocator. Inlining
    removes these intermediates through ordinary scheduling, not an allocation
    override. Remaining unsupported allocations require downstream runtime
    support or rejection. Accelerator envelopes are never scheduled here.
    """
    for gv, func in list(mod.functions_items()):
        if not isinstance(func, tir.PrimFunc) or not isinstance(func.body, tir.BlockRealize):
            continue
        if func.attrs and int(func.attrs.get("op_pattern", -1)) == 8:
            continue
        internal = {buffer.data for buffer in func.body.block.alloc_buffers}
        parameters = {buffer.data for buffer in func.buffer_map.values()}
        candidates = []

        def find_producer(node):
            if isinstance(node, tir.Block) and node.init is None and isinstance(node.body, tir.BufferStore) and len(node.writes) == 1:
                buffer = node.writes[0].buffer
                # Inlining would erase the logical low-precision storage rounding.
                if tvm.DataType(buffer.dtype).type_code in (DataTypeCode.BFLOAT, DataTypeCode.E4M3Float, DataTypeCode.E5M2Float):
                    return
                data = buffer.data
                if data in internal and data not in parameters and all(axis.iter_type == tir.IterVar.DataPar for axis in node.iter_vars):
                    candidates.append(node.name_hint)

        tir.stmt_functor.post_order_visit(func.body, find_producer)
        if not candidates:
            continue
        schedule = tir.Schedule(tvm.IRModule({gv: func}), debug_mask=1)
        for name in candidates:
            try:
                schedule.compute_inline(schedule.get_block(name, func_name=gv.name_hint))
            except tir.schedule.ScheduleError:
                # Reduction/non-complete/otherwise illegal producers stay in
                # TIR. A downstream exporter must account for their memory.
                continue
        mod.update_func(gv, schedule.mod[gv])
    return mod
