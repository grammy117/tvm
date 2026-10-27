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
"""Structural scheduling of mathematical TensorIR for Gemmini.

Unlike gemmini_schedule (the retained legacy route), this module starts with
ordinary legalized tensor functions and records transformations of those functions.
"""

import math
import json
from dataclasses import dataclass

import tvm
from tvm import tir

from .gemmini_intrin import ACC, SPAD, register_intrinsics


def _contraction_update(node):
    """Read an ordinary or explicitly modulo-32 product update structurally."""
    value = node.body.value
    wrapped = False
    if isinstance(value, tir.Cast) and value.dtype == "int32":
        add = value.value
        if (
            not isinstance(add, tir.Add)
            or add.dtype != "int64"
            or not all(isinstance(x, tir.Cast) and x.dtype == "int64" for x in (add.a, add.b))
        ):
            return None
        current, product, wrapped = add.a.value, add.b.value, True
    elif isinstance(value, tir.Add):
        current, product = value.a, value.b
    else:
        return None
    if not isinstance(current, tir.BufferLoad) or not isinstance(product, tir.Mul):
        return None
    return current, product, wrapped


def _supported_contraction_init(node):
    value = node.init.value
    return (isinstance(value, tir.IntImm) and int(value) == 0) or (
        isinstance(value, tir.BufferLoad) and value.dtype == "int32"
    )


def _normalizable_contractions(func):
    """Identify exact contractions whose access maps can form matrix caches.

    Spatial axes used by both operands are batches; axes used by only one
    operand form the two matrix dimensions. All reduction axes must occur in
    both operands. TVM's index-map and reindex primitives prove and perform the
    layout transformation, including convolution's sliding-window accesses.
    """
    matches = []

    def variables(expressions):
        result = set()
        for expr in expressions:
            tir.stmt_functor.post_order_visit(
                expr, lambda node: result.add(node) if isinstance(node, tir.Var) else None
            )
        return result

    def visit(node):
        if not isinstance(node, tir.Block) or node.init is None or len(node.writes) != 1:
            return
        if not isinstance(node.body, tir.BufferStore) or not isinstance(node.init, tir.BufferStore):
            return
        output = node.body.buffer
        update = _contraction_update(node)
        if output.dtype != "int32" or not _supported_contraction_init(node) or update is None:
            return
        current, product, _ = update
        equal = tvm.ir.structural_equal
        if (
            not current.buffer.same_as(output)
            or not node.init.buffer.same_as(output)
            or not equal(current.indices, node.body.indices)
            or not equal(node.init.indices, node.body.indices)
        ):
            return
        loads = []
        for expr in (product.a, product.b):
            if (
                not isinstance(expr, tir.Cast)
                or expr.dtype != "int32"
                or not isinstance(expr.value, tir.BufferLoad)
                or expr.value.dtype != "int8"
            ):
                return
            loads.append(expr.value)
        axes = list(node.iter_vars)
        if any(
            axis.iter_type not in (0, 2)
            or not isinstance(axis.dom.min, tir.IntImm)
            or int(axis.dom.min) != 0
            or not isinstance(axis.dom.extent, tir.IntImm)
            or int(axis.dom.extent) <= 0
            for axis in axes
        ):
            return
        spatial = [index for index, axis in enumerate(axes) if axis.iter_type == 0]
        reduction = [index for index, axis in enumerate(axes) if axis.iter_type == 2]
        if not reduction or len(node.body.indices) != len(spatial):
            return
        # Reindexing the output must preserve every original spatial element.
        if set(node.body.indices) != {axes[index].var for index in spatial}:
            return
        left, right = [variables(load.indices) for load in loads]
        if any(axes[index].var not in left or axes[index].var not in right for index in reduction):
            return
        batch = [index for index in spatial if axes[index].var in left & right]
        rows = [index for index in spatial if axes[index].var in left - right]
        cols = [index for index in spatial if axes[index].var in right - left]
        if not rows or not cols or len(batch + rows + cols) != len(spatial):
            return
        shape = tuple(
            math.prod(int(axes[index].dom.extent) for index in group)
            for group in (rows, cols, reduction)
        )
        if shape[2] > 131071:
            return
        matches.append((node, shape, (batch, rows, cols, reduction)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


def contraction_block(func):
    """Find an exact signed INT8 x INT8 -> INT32 matrix reduction by structure."""
    matches = []

    def visit(node):
        if not isinstance(node, tir.Block) or node.init is None:
            return
        if len(node.iter_vars) < 3 or len(node.writes) != 1:
            return
        if [axis.iter_type for axis in node.iter_vars] != [0] * (len(node.iter_vars) - 1) + [2]:
            return
        if not isinstance(node.body, tir.BufferStore) or not isinstance(node.init, tir.BufferStore):
            return
        output = node.body.buffer
        if output.dtype != "int32" or len(output.shape) != len(node.iter_vars) - 1:
            return
        if not _supported_contraction_init(node):
            return
        update = _contraction_update(node)
        if update is None:
            return
        current, product, _ = update
        if not current.buffer.same_as(output):
            return
        if (
            not node.init.buffer.same_as(output)
            or not tvm.ir.structural_equal(current.indices, node.body.indices)
            or not tvm.ir.structural_equal(node.init.indices, node.body.indices)
        ):
            return
        operands = []
        for expr in (product.a, product.b):
            if (
                not isinstance(expr, tir.Cast)
                or expr.dtype != "int32"
                or not isinstance(expr.value, tir.BufferLoad)
            ):
                return
            load = expr.value
            if load.buffer.dtype != "int8" or len(load.buffer.shape) < 2:
                return
            operands.append(load)
        batch = [axis.var for axis in node.iter_vars[:-3]]
        i, j, k = [axis.var for axis in node.iter_vars[-3:]]
        expected = ((i, k), (k, j))
        equal = tvm.ir.structural_equal
        if not all(equal(x, y) for x, y in zip(node.body.indices, (*batch, i, j))):
            return
        if not all(
            all(equal(x, y) for x, y in zip(load.indices[-2:], indices))
            for load, indices in zip(operands, expected)
        ):
            return
        for load in operands:
            if len(load.indices) > len(batch) + 2:
                return
            for index, axis in zip(reversed(load.indices[:-2]), reversed(batch)):
                if not equal(index, axis) and not (
                    isinstance(index, tir.IntImm) and int(index) == 0
                ):
                    return
        if not all(isinstance(axis.dom.extent, tir.IntImm) for axis in node.iter_vars):
            return
        m, n, depth = [int(axis.dom.extent) for axis in node.iter_vars[-3:]]
        if min(m, n, depth) <= 0 or depth > 131071:
            return
        matches.append((node, (m, n, depth)))

    tir.stmt_functor.post_order_visit(func.body, visit)
    return matches


@dataclass(frozen=True)
class GemminiTensorIRSchedule:
    """The original workload, full transformation trace and deployable result."""

    workload: tvm.IRModule
    mathematical: tvm.IRModule
    scheduled_mod: tvm.IRModule
    trace: tir.schedule.Trace
    metadata: dict


def finalize_gemmini_function(scheduled, *, capabilities=None):
    """Lower memory and add the admission envelope to an already scheduled function."""
    return lower_tensorir_schedule(
        None, tir.Schedule(scheduled), capabilities=capabilities
    ).scheduled_mod["main"]


def make_tensorir_schedule(
    func,
    *,
    order="ij",
    reduction_chunk=None,
    row_tiles=1,
    col_tiles=1,
    capabilities=None,
):
    """Apply the default contraction sketch and lower target scopes with admission checks."""
    from .gemmini_sketch import default_schedule  # pylint: disable=import-outside-toplevel

    sch = default_schedule(
        func,
        capabilities=capabilities,
        order=order,
        chunk=reduction_chunk,
        row_tiles=row_tiles,
        col_tiles=col_tiles,
    )
    return lower_tensorir_schedule(func, sch, order=order, capabilities=capabilities)


def lower_tensorir_schedule(
    func, sch, *, mathematical=None, order="database", capabilities=None, workload_metadata=None
):
    """Finalize a TVM-selected schedule without replacing its scheduling choices.

    ``func`` is the mathematical workload when available. A function already
    scheduled elsewhere (for example replayed from a tuning database) passes
    ``func=None``; buffer ownership then comes from the scheduled program.
    """
    from .gemmini_memory import lower_memory  # pylint: disable=import-outside-toplevel
    from .gemmini_target import GemminiCapabilities  # pylint: disable=import-outside-toplevel

    capabilities = capabilities or GemminiCapabilities()
    scheduled_only = func is None
    if scheduled_only:
        func = sch.mod["main"]
        if workload_metadata is None:
            workload_metadata = {}
    if any(param not in func.buffer_map for param in func.params):
        raise ValueError("Gemmini runtime ABI requires buffer parameters")
    mathematical = mathematical if mathematical is not None else tvm.IRModule({"main": func})
    mod, resources = lower_memory(
        sch.mod, spad_rows=capabilities.spad_rows, acc_rows=capabilities.acc_rows
    )
    device = mod["main"]
    buffers = [device.buffer_map[param] for param in device.params]
    if any(
        not all(isinstance(dim, tir.IntImm) and int(dim) > 0 for dim in buffer.shape)
        or buffer.strides
        for buffer in buffers
    ):
        raise ValueError("TensorIR admission requires positive static compact external buffers")
    sizes = [
        math.prod(int(dim) for dim in buffer.shape)
        * max(1, (tvm.DataType(buffer.dtype).bits + 7) // 8)
        for buffer in buffers
    ]
    if max(sizes) > (1 << 63) - 1:
        raise ValueError("Tensor buffer byte span exceeds the signed ABI")
    if any(
        (int(buffer.shape[-1]) if len(buffer.shape) else 1)
        * max(1, (tvm.DataType(buffer.dtype).bits + 7) // 8)
        > (1 << 32) - 1
        for buffer in buffers
    ):
        raise ValueError("External DMA row stride exceeds the Gemmini ABI")
    written = set()

    def record_write(node):
        if isinstance(node, tir.BufferStore):
            written.add(node.buffer.data)
        elif (
            isinstance(node, tir.Call)
            and node.op.same_as(tvm.ir.Op.get("tir.tvm_access_ptr"))
            and isinstance(node.args[4], tir.IntImm)
            and int(node.args[4]) & 2
        ):
            written.add(node.args[1])

    # Without the mathematical function, ownership is read from the lowered
    # program, where intrinsic views have become parameter access pointers.
    tir.stmt_functor.post_order_visit((device if scheduled_only else func).body, record_write)
    # MetaSchedule may renew every Var while cloning a candidate. Ownership
    # comes from the original mathematical ABI's parameter positions, not
    # pointer identity shared accidentally with one scheduling path.
    original_buffers = [func.buffer_map[param] for param in func.params]
    if len(original_buffers) != len(buffers) or any(
        source.dtype != selected.dtype or not tvm.ir.structural_equal(source.shape, selected.shape)
        for source, selected in zip(original_buffers, buffers)
    ):
        raise ValueError("Scheduled function changed the mathematical tensor ABI")
    roles = [buffer.data in written for buffer in (buffers if scheduled_only else original_buffers)]
    inputs = [(buffer, size) for buffer, size, output in zip(buffers, sizes, roles) if not output]
    outputs = [(buffer, size) for buffer, size, output in zip(buffers, sizes, roles) if output]
    if not outputs:
        raise ValueError("TensorIR admission could not establish input/output ownership")
    failure = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_extern(
                    "void", "TVMAPISetLastError", tir.StringImm("Gemmini tensor buffers rejected")
                )
            ),
            tir.Evaluate(tir.tvm_throw_last_error()),
        ]
    )
    if not isinstance(device.body, tir.BlockRealize):
        raise ValueError("Gemmini lowering requires a TensorIR root block")
    realization = device.body
    root = realization.block
    body = tir.SeqStmt(
        [
            tir.Evaluate(
                tir.call_extern("void", "tvm_gemmini_begin", *[tir.IntImm("int64", 1)] * 3)
            ),
            root.body,
            tir.Evaluate(tir.call_extern("void", "tvm_gemmini_end")),
        ]
    )
    for output, output_bytes in outputs:
        checks = [
            tir.call_extern(
                "int32",
                "tvm_gemmini_validate_tensor_buffers",
                source.data,
                tir.IntImm("int64", source_bytes),
                source.data,
                tir.IntImm("int64", source_bytes),
                output.data,
                tir.IntImm("int64", output_bytes),
            )
            for source, source_bytes in inputs
        ]
        if not inputs:
            checks.append(
                tir.call_extern(
                    "int32",
                    "tvm_gemmini_validate_tensor_output",
                    output.data,
                    tir.IntImm("int64", output_bytes),
                )
            )
        for admission in checks:
            status = tir.Var("gemmini_admission", "int32")
            body = tir.LetStmt(status, admission, tir.IfThenElse(status != 0, failure, body))
    # Keep the ordinary TensorIR root for stock Relax analyses, including
    # RewriteDataflowReshape on single-input/single-output functions. Admission
    # and ownership still precede every device command within that root.
    root = tir.Block(
        root.iter_vars,
        root.reads,
        root.writes,
        root.name_hint,
        body,
        root.init,
        root.alloc_buffers,
        root.match_buffers,
        root.annotations,
    )
    device = device.with_body(
        tir.BlockRealize(realization.iter_values, realization.predicate, root)
    )
    if workload_metadata is None:
        matches = contraction_block(func) or _normalizable_contractions(func)
        shape = matches[0][1]
        workload_metadata = dict(zip(("m", "n", "k"), shape))
        if isinstance(matches[0][0].init.value, tir.BufferLoad):
            workload_metadata.update(operation="matmul_bias", bias_preloaded=True)
    cpu_reductions = []
    tir.stmt_functor.post_order_visit(
        sch.mod["main"].body,
        lambda node: cpu_reductions.append(node)
        if isinstance(node, tir.Block) and node.init is not None
        else None,
    )
    if cpu_reductions:
        workload_metadata = dict(workload_metadata, cpu_reduction_blocks=len(cpu_reductions))
    device = device.with_attr("gemmini.tensorir", 1)
    device = device.with_attr("gemmini.capability_id", capabilities.fingerprint)
    device = device.with_attr("op_pattern", 8)
    for key in ("m", "n", "k"):
        if key in workload_metadata:
            device = device.with_attr("gemmini." + key, workload_metadata[key])
    metadata = dict(
        resources,
        **workload_metadata,
        order=order,
        instruction_policy="primitive_no_fsm",
        performance_measured=False,
        capability_id=capabilities.fingerprint,
    )
    root = sch.mod["main"].body.block
    if root.annotations.get("gemmini.readout_fused", False):
        metadata["readout_fused"] = True
        metadata["readout_proof"] = json.loads(str(func.attrs["gemmini.readout_proof"]))
    metadata["schedule_parameters"] = {
        str(key)[8:]: int(value) if isinstance(value, tir.IntImm) else str(value)
        for key, value in root.annotations.items()
        if str(key).startswith("gemmini.")
    }
    return GemminiTensorIRSchedule(
        tvm.IRModule({"main": func}),
        mathematical,
        tvm.IRModule({"main": device}),
        sch.trace,
        metadata,
    )


def _uses_gemmini_memory(func):
    found = []
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: found.extend(
            buffer for buffer in node.alloc_buffers if buffer.scope().startswith("local.gemmini_")
        )
        if isinstance(node, tir.Block)
        else None,
    )
    return bool(found)


@tvm.transform.module_pass(opt_level=0, name="ScheduleGemminiTensorIR")
class ScheduleGemminiTensorIR:
    """Finalize database schedules and give untuned functions the default placement.

    Run after ``relax.transform.MetaScheduleApplyDatabase``. A function it marked
    ``tir.is_scheduled`` keeps its tuned schedule: Gemmini placements only get
    physical memory lowering and the admission envelope, CPU placements are left
    unchanged. Untuned supported functions get the default Gemmini sketch;
    anything unsupported stays on the CPU. A target-specific schedule failure
    retains the original function and records its reason.
    """

    def __init__(
        self,
        *,
        capabilities=None,
        order="ij",
        schedule_options=None,
        report=None,
        references=None,
        normalizations=None,
        offload_copies=False,
        movement_strategy="rectangular",
    ):
        from .gemmini_target import GemminiCapabilities  # pylint: disable=import-outside-toplevel

        self.capabilities = capabilities or GemminiCapabilities()
        self.order = order
        self.schedule_options = dict(schedule_options or {})
        if set(self.schedule_options) - {"reduction_chunk", "row_tiles", "col_tiles"}:
            raise ValueError("Unknown Gemmini TensorIR schedule option")
        self.report = report if report is not None else []
        self.references = references if references is not None else {}
        self.normalizations = normalizations if normalizations is not None else {}
        if type(offload_copies) is not bool:
            raise ValueError("offload_copies must be a boolean")
        self.offload_copies = offload_copies
        if movement_strategy not in ("rectangular", "scalar"):
            raise ValueError(
                "Movement strategy must be rectangular or an explicit scalar experiment"
            )
        self.movement_strategy = movement_strategy

    def transform_module(self, mod, _ctx):
        from .gemmini_elementwise import elementwise_blocks, make_elementwise_schedule
        from .gemmini_reduction import reduction_blocks, reduction_metadata, make_reduction_schedule
        from .gemmini_pool import pool_blocks, make_pool_schedule

        for gv, func in list(mod.functions_items()):
            if not isinstance(func, tir.PrimFunc) or (
                func.attrs and "gemmini.tensorir" in func.attrs
            ):
                continue
            if func.attrs and bool(func.attrs.get("tir.is_scheduled", False)):
                if not _uses_gemmini_memory(func):
                    self.report.append(
                        {
                            "function": gv.name_hint,
                            "placement": "cpu",
                            "origin": "TVM MetaSchedule database",
                            "reason": "Tuned CPU schedule selected by measurement",
                        }
                    )
                    continue
                result = lower_tensorir_schedule(
                    None, tir.Schedule(func), capabilities=self.capabilities
                )
                mod.update_func(gv, result.scheduled_mod["main"])
                self.report.append(
                    {
                        "function": gv.name_hint,
                        "placement": "gemmini",
                        "metadata": result.metadata,
                        "origin": "TVM MetaSchedule database",
                    }
                )
                continue
            matches = contraction_block(func) or _normalizable_contractions(func)
            reductions = reduction_blocks(func) if not matches else []
            pools = pool_blocks(func) if not matches and not reductions else []
            pointwise = elementwise_blocks(func) if not matches and not reductions and not pools else []
            if not matches and not pointwise and not reductions and not pools:
                self.report.append(
                    {
                        "function": gv.name_hint,
                        "placement": "cpu",
                        "reason": "No supported exact integer contraction, bounded sum/window, affine/readout, copy, widening or addition",
                    }
                )
                continue
            try:
                if (
                    pointwise
                    and not self.offload_copies
                    and any(
                        not kind.startswith("add_i32") and kind not in ("relu_i8", "affine_i8")
                        for _, kind in pointwise
                    )
                ):
                    self.report.append(
                        {
                            "function": gv.name_hint,
                            "placement": "cpu",
                            "reason": "Copy offload requires an explicit opt-in or a measured tuning record",
                        }
                    )
                    continue
                if pools:
                    result = make_pool_schedule(func, capabilities=self.capabilities)
                    origin = "TensorIR values-only pooled readout rule"
                elif reductions:
                    result = make_reduction_schedule(func, capabilities=self.capabilities)
                    origin = "deterministic TensorIR reduction rule"
                elif pointwise:
                    result = make_elementwise_schedule(
                        func,
                        capabilities=self.capabilities,
                        scalar=self.movement_strategy == "scalar",
                    )
                    origin = (
                        "explicit scalar DMA experiment"
                        if self.movement_strategy == "scalar"
                        else "deterministic TensorIR DMA rule"
                    )
                else:
                    result = make_tensorir_schedule(
                        func,
                        capabilities=self.capabilities,
                        order=self.order,
                        **self.schedule_options,
                    )
                    origin = "default Gemmini contraction sketch"
            except (ValueError, tvm.TVMError) as error:
                if gv.name_hint in self.references:
                    mod.update_func(gv, self.references[gv.name_hint])
                self.report.append(
                    {"function": gv.name_hint, "placement": "cpu", "reason": str(error)}
                )
                continue
            mod.update_func(gv, result.scheduled_mod["main"])
            record = {
                "function": gv.name_hint,
                "placement": "gemmini",
                "metadata": result.metadata,
                "trace": result.trace.as_json(),
                "origin": origin,
            }
            if gv.name_hint in self.normalizations:
                record["mathematical_normalization"] = self.normalizations[gv.name_hint]
            self.report.append(record)
        return mod


def prepare_gemmini_tensorir_workloads(
    mod, *, optimize=True, bias_preload=True, prepack_weights=True, fusion_level=-1,
    eliminate_common_subexpr=False
):
    """Expose the same mathematical workloads used by compile and DSE paths."""
    from tvm import relax  # pylint: disable=import-outside-toplevel

    if type(optimize) is not bool:
        raise ValueError("optimize must be a boolean")
    if type(fusion_level) is not int or fusion_level not in (-1, 0, 1, 2, 3):
        raise ValueError("fusion_level must be -1 (PassContext) or an integer from 0 to 3")
    if not optimize and fusion_level != -1:
        raise ValueError("Explicit fusion_level requires optimize=True")
    if type(eliminate_common_subexpr) is not bool:
        raise ValueError("eliminate_common_subexpr must be a boolean")
    if eliminate_common_subexpr:
        if not optimize:
            raise ValueError("Graph CSE requires optimize=True")
        mod = relax.transform.EliminateCommonSubexpr()(mod)
    passes = [relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern()]
    if optimize:
        passes.extend(
            [
                relax.transform.FoldConstant(),
                relax.transform.CanonicalizeBindings(),
                relax.transform.FuseOps(fuse_opt_level=fusion_level),
                relax.transform.FuseTIR(),
                relax.transform.RewriteDataflowReshape(),
                relax.transform.DeadCodeElimination(),
            ]
        )
    mod = tvm.transform.Sequential(passes)(mod)
    from .gemmini_epilogue import SpecializeGemminiReadoutScales

    mod = SpecializeGemminiReadoutScales()(mod)
    if type(prepack_weights) is not bool:
        raise ValueError("prepack_weights must be a boolean")
    if prepack_weights:
        from .gemmini_layout import prepack_constants

        mod = prepack_constants(mod)
    from .gemmini_bias import CanonicalizeGemminiBias

    if type(bias_preload) is not bool:
        raise ValueError("bias_preload must be a boolean")
    references, normalizations = {}, {}
    if bias_preload:
        mod = CanonicalizeGemminiBias(references=references, report=normalizations)(mod)
    from .gemmini_epilogue import NormalizeGemminiReadout
    from .gemmini_search import AnnotateGemminiWorkloads

    mod = NormalizeGemminiReadout()(mod)
    # Workload identity for tuning records; extraction and application both
    # see functions prepared by this same pipeline.
    mod = AnnotateGemminiWorkloads()(mod)
    return mod, references, normalizations


def prepare_gemmini_tensorir_graph(
    mod,
    *,
    optimize=True,
    capabilities=None,
    order="ij",
    schedule_options=None,
    report=None,
    bias_preload=True,
    offload_copies=False,
    prepack_weights=True,
    movement_strategy="rectangular",
    database=None,
    fusion_level=-1,
    eliminate_common_subexpr=False,
):
    """Use normal mathematical legalization/fusion before Gemmini scheduling.

    Tuned schedules come from ``database`` (or the current MetaSchedule
    database) through the stock ``MetaScheduleApplyDatabase`` pass.
    """
    from .gemmini import _inline_cpu_producers  # pylint: disable=import-outside-toplevel
    from .gemmini_target import GemminiCapabilities  # pylint: disable=import-outside-toplevel

    mod, references, normalizations = prepare_gemmini_tensorir_workloads(
        mod, optimize=optimize, bias_preload=bias_preload, prepack_weights=prepack_weights,
        fusion_level=fusion_level, eliminate_common_subexpr=eliminate_common_subexpr,
    )
    database = database if database is not None else tvm.meta_schedule.Database.current()
    if database is not None:
        target = (capabilities or GemminiCapabilities()).target()
        with target, database:
            mod = tvm.relax.transform.MetaScheduleApplyDatabase()(mod)
    mod = ScheduleGemminiTensorIR(
        capabilities=capabilities,
        order=order,
        schedule_options=schedule_options,
        report=report,
        references=references,
        normalizations=normalizations,
        offload_copies=offload_copies,
        movement_strategy=movement_strategy,
    )(mod)
    return _inline_cpu_producers(mod)
