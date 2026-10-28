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

"""Current mathematical TensorIR entry point and shared host cleanup.

Historical Relax substitution is isolated in gemmini_legacy and imported only
for explicit legacy requests. The default path has no graph-replacement passes.
"""
import tvm
from tvm import relax, tir
from tvm._ffi.runtime_ctypes import DataTypeCode
from tvm.relax.pipeline import default_build_pipeline, register_pipeline


def prepare_gemmini_graph(mod, tile_i=1, tile_j=1, optimize=True, tile_choices=None,
                          *, route="tensorir", capabilities=None, order="ij", schedule_options=None,
                          report=None, bias_preload=False, offload_copies=False, prepack_weights=False,
                          movement_strategy="rectangular", database=None, fusion_level=-1,
                          eliminate_common_subexpr=False):
    """Prepare Gemmini through mathematical TensorIR; retain explicit legacy mode.

    The default route legalizes/fuses ordinary computations before structural
    scheduling. Legacy template tile parameters are accepted only with the
    explicitly selected regression route, never silently ignored.
    """
    if route == "legacy":
        from .gemmini_legacy import prepare_gemmini_graph_legacy

        if eliminate_common_subexpr:
            raise ValueError("Graph CSE choices require route='tensorir'")
        if fusion_level != -1:
            raise ValueError("Graph fusion choices require route='tensorir'")
        if offload_copies:
            raise ValueError("Copy offload choices require route='tensorir'")
        if bias_preload is not False:
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


def __getattr__(name):
    """Preserve explicit historical imports without loading them in active runs."""
    if name in {
        "DecomposeGemminiLinear", "LowerGemminiMatmul", "LowerGemminiScheduledMatmul",
        "prepare_gemmini_graph_legacy", "_shape", "_eligible", "_wrapper",
        "_LinearLowerer", "_MatmulLowerer", "_SYMBOL", "_MAX_K", "_MAX_BYTES",
    }:
        from . import gemmini_legacy

        return getattr(gemmini_legacy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
