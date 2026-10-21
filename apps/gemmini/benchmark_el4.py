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
"""Capture reduced EL4 diagnostic workloads with the pinned public recipes.

Source is supplied explicitly, with file digests checked before import. No model
checkpoint, frozen capture, reference backend or prewritten device kernel is used.
The attention slices exercise the public static-head rewrite directly; learned
contractions use the public PT2E recipe. These are synthetic numerical/compiler
diagnostics, not accuracy results for the named pretrained models.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch
import tvm
from tvm import relax, tir
from tvm.relax.backend.contrib.gemmini import prepare_gemmini_graph
from tvm.relax.frontend.torch import from_exported_program


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_source(root, receipts):
    """Only import files from the explicitly supplied, content-bound snapshot."""
    pinned, files = set(), {}
    for receipt in receipts:
        body = json.loads(receipt.read_text())
        pinned.add(body.get("revision", body.get("source_revision")))
        for row in body["files"]:
            relative = row.get("path", row.get("source_path"))
            path = root / relative
            if digest(path) != row["sha256"]:
                raise ValueError("Public source digest changed: " + str(path))
            files[relative] = row["sha256"]
    if len(pinned) != 1 or None in pinned:
        raise ValueError("Public source receipts must identify one exact revision")
    sys.path[:0] = [str(root / "src"), str(root / "src/merlin/targetgen")]
    os.environ["MERLIN_SCHEMAS_DIR"] = str(root / "merlin/schemas")
    return {"revision": next(iter(pinned)), "files": files}


def load_public(root, workload):
    path = root / "examples/workloads" / workload / "loader.py"
    spec = importlib.util.spec_from_file_location("el4_public_" + workload, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionHead(torch.nn.Module):
    """One head; separate cache inputs for decode; host RoPE and stable softmax."""

    def __init__(self, rows, *, decode=False, rotary=False):
        super().__init__()
        self.decode, self.rotary = decode, rotary
        positions = torch.arange(17, dtype=torch.float32)
        frequencies = 1 / (10000 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32))
        angles = torch.outer(positions, frequencies).repeat(1, 2)
        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())
        self.register_buffer("query_cos", angles.cos()[-rows:])
        self.register_buffer("query_sin", angles.sin()[-rows:])
        self.register_buffer(
            "mask",
            torch.where(
                torch.arange(17)[None, :] > torch.arange(17 - rows, 17)[:, None], float("-inf"), 0.0
            ),
        )

    def rotate(self, value, cosine, sine):
        return value * cosine + torch.cat((-value[..., 16:], value[..., :16]), -1) * sine

    def forward(self, query, key, value, key_cache=None, value_cache=None):
        if self.rotary:
            query = self.rotate(query, self.query_cos, self.query_sin)
            key = self.rotate(
                key,
                self.query_cos if self.decode else self.cos,
                self.query_sin if self.decode else self.sin,
            )
        if self.decode:
            key = torch.cat((key_cache, key), 2)
            value = torch.cat((value_cache, value), 2)
        scores = (query @ key.transpose(-2, -1)) / (32**0.5)
        if self.rotary and not self.decode:
            scores = scores + self.mask
        return torch.softmax(scores, -1) @ value


def workload(root, name, seed):
    """Name chooses an experiment geometry, never a production compiler rule."""
    generator = torch.Generator().manual_seed(seed)

    def rand(shape):
        return torch.randn(shape, generator=generator) * 0.25

    if name in ("tiny_prefill_head", "tiny_decode_head", "smol_prefix_head"):
        decode = name == "tiny_decode_head"
        rows = 1 if decode else 3 if name == "tiny_prefill_head" else 2
        model = AttentionHead(rows, decode=decode, rotary=name.startswith("tiny"))
        inputs = (
            rand((1, 1, rows, 32)),
            rand((1, 1, 1 if decode else 17, 32)),
            rand((1, 1, 1 if decode else 17, 32)),
        )
        if decode:
            inputs += (rand((1, 1, 16, 32)), rand((1, 1, 16, 32)))
        return model, inputs, "smolvla" if name.startswith("smol") else "tiny_llama", True
    if name == "residual_bottleneck":
        source = load_public(root, "residual_cnn")
        model = source.ResidualCNN(channels=16, blocks=1, bottleneck=True, batchnorm=True)
        with torch.no_grad():
            source._seed_batchnorm(model)
        return model, (rand((1, 3, 4, 4)),), "resnet50", False
    if name == "decoder_block":
        source = load_public(root, "causal_decoder")
        return (
            source.CausalDecoder(seq=3, hidden=32, heads=1, ffn=32, rope=True),
            (torch.tensor([[1, 7, 9]], dtype=torch.int64),),
            "tiny_llama",
            False,
        )
    if name == "policy_block":
        source = load_public(root, "multimodal_policy")
        model, inputs = source.get_model_and_inputs()
        return model, inputs, "smolvla", False
    if name == "coverage_mlp":
        source = load_public(root, "coverage_mlp")
        model, inputs = source.get_model_and_inputs()
        return model, inputs, "resnet50", False
    raise ValueError("Unknown reduced workload: " + name)


def seeded_weights(model):
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    index = torch.arange(parameter.numel(), dtype=torch.int64).reshape(
                        parameter.shape
                    )
                    parameter.copy_(
                        (((index * 37 + ordinal * 101 + field * 53) % 251) - 125).to(
                            parameter.dtype
                        )
                        / 512
                    )


def capture(root, name, *, seed=1907, calibration_count=8):
    model, inputs, recipe_name, attention = workload(root, name, seed)
    model.eval()
    seeded_weights(model)
    recipe_path = root / "examples/gemmini/protocol/recipes" / (recipe_name + ".json")
    recipe = json.loads(recipe_path.read_text())
    calibration = [workload(root, name, seed + i + 1)[1] for i in range(calibration_count)]
    if attention:
        import _static_attention as sa
        import _dynamic_contractions as dc

        quantized = torch.export.export(model, inputs).module()
        sites = [
            node
            for node in quantized.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.matmul.default
        ]
        rows = sa.rewrite_sites(
            quantized,
            sites,
            recipe["dynamic"],
            calibration,
            low=-128,
            high=127,
            eps=2**-12,
            owner=lambda node: None,
        )
        dc.verify_sites(quantized, rows, calibration + [inputs])
        gate = recipe["dynamic"]["site_agreement"]
        for row in rows:
            observed = row.get("verification", {})
            if (
                not observed.get("finite")
                or observed["cosine_min"] < gate["cosine_min"]
                or observed["rel_rmse_max"] > gate["rel_rmse_max"]
            ):
                raise ValueError(
                    "Reduced attention site fails the unchanged recipe agreement gate: "
                    + str(row.get("node"))
                )
        stats = {
            "dynamic_sites": rows,
            "site_agreement_gate": gate,
            "site_agreement": "passed",
            "calibration_count": calibration_count,
            "scope": "public_static_head_algorithm; held_out_seed_separate_from_calibration",
        }
    else:
        import _recipe_quantizer as quantizer

        quantized = quantizer.apply_recipe(
            model,
            recipe,
            example_inputs=inputs,
            calibration_inputs=calibration,
            calibration_samples=calibration_count,
        )
        stats = quantized._recipe_quantization_stats
    with torch.no_grad():
        expected = quantized(*inputs)
        floating = model(*inputs)
    if isinstance(expected, (tuple, list)):
        if len(expected) != 1:
            raise ValueError("This static benchmark runner currently requires one tensor output")
        expected = expected[0]
    ep = torch.export.export(quantized, inputs)
    metadata = {
        "workload": name,
        "recipe": recipe_name,
        "recipe_file_sha256": digest(recipe_path),
        "recipe_sha256": recipe.get("recipe_sha256"),
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [str(value.dtype) for value in inputs],
        "calibration_count": calibration_count,
        "seed": seed,
        "weight_initialization": "benchmark_el4.seeded_weights: ((index*37 + module_ordinal*101 + field*53)%251 - 125)/512; overrides learned public defaults",
        "scope": "reduced_synthetic_compiler_diagnostic; no_pretrained_accuracy_claim",
        "quantization": stats,
        "aten_ops": dict(
            Counter(str(node.target) for node in ep.graph.nodes if node.op == "call_function")
        ),
        "float_reference_max_abs_error": float((expected - floating).abs().max()),
    }
    return ep, inputs, expected, metadata


def function_calls(func):
    calls = Counter()
    tir.stmt_functor.post_order_visit(
        func.body,
        lambda node: calls.update([str(node.args[0])])
        if isinstance(node, tir.Call) and node.op == tvm.ir.Op.get("tir.call_extern")
        else None,
    )
    return dict(calls)


def benchmark(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "stage": "capture", "source": args.source_identity}
    try:
        ep, inputs, expected, metadata = capture(args.source_root, args.case)
        result.update(metadata)
        torch.export.save(ep, args.output_dir / "workload.pt2")
        np.savez(
            args.output_dir / "arrays.npz",
            expected=expected.detach().numpy(),
            **{f"input_{i}": value.detach().numpy() for i, value in enumerate(inputs)},
        )
        result["stage"] = "import"
        mod = from_exported_program(ep, keep_params_as_input=False, unwrap_unit_return_tuple=True)
        (args.output_dir / "imported.relax.py").write_text(mod.script())
        result["stage"] = "schedule"
        report = []
        prepared = prepare_gemmini_graph(mod, report=report)
        (args.output_dir / "scheduled.relax.py").write_text(prepared.script())
        result["scheduling"] = report
        result["externs_by_function"] = {
            gv.name_hint: function_calls(func)
            for gv, func in prepared.functions.items()
            if isinstance(func, tir.PrimFunc)
        }
        calls = Counter()
        relax.analysis.post_order_visit(
            prepared["main"],
            lambda node: calls.update([node.args[0].name_hint])
            if isinstance(node, relax.Call) and node.op == tvm.ir.Op.get("relax.call_tir")
            else None,
        )
        result["call_tir_sites_by_function"] = dict(calls)
        (args.output_dir / "scheduled.json").write_text(tvm.ir.save_json(prepared))
        result["stage"] = "host_cpu_reference"
        # The CPU route checks importer mathematics. It does not emulate RTL or
        # contribute latency labels. Native correctness is checked separately.
        cpu = relax.transform.LegalizeOps()(mod)
        vm = relax.VirtualMachine(relax.build(cpu, "llvm"), tvm.cpu())
        got = vm["main"](*[tvm.nd.array(value.detach().numpy()) for value in inputs])
        np.testing.assert_allclose(got.numpy(), expected.detach().numpy(), rtol=1e-5, atol=1e-6)
        result.update(
            status="passed",
            stage="captured_imported_scheduled_cpu_checked",
            host_reference="passed",
            native_runtime="not_run",
        )
    except Exception as error:  # Retain the precise boundary for every attempted workload.
        result.update(status="blocked", error_type=type(error).__name__, error=str(error))
        (args.output_dir / "failure.txt").write_text(traceback.format_exc())
    result["bindings"] = [
        {"path": str(path.resolve()), "sha256": digest(path)}
        for path in (Path(__file__), *args.source_receipts)
    ]
    compiler = Path(tvm.__file__).resolve().parent
    for path in sorted((compiler / "relax/backend/contrib").glob("gemmini*.py")):
        result["bindings"].append({"path": str(path), "sha256": digest(path)})
    for name in (
        "exported_program_translator.py",
        "base_fx_graph_translator.py",
        "_quantization.py",
    ):
        path = compiler / "relax/frontend/torch" / name
        result["bindings"].append({"path": str(path), "sha256": digest(path)})
    result["environment"] = {
        "torch": torch.__version__,
        "tvm": tvm.__version__,
        "numpy": np.__version__,
    }
    for name in (
        "_recipe_quantizer",
        "_static_attention",
        "_dynamic_contractions",
        "m2m.capture.pt2e_padding",
    ):
        module = sys.modules.get(name)
        if module is not None:
            path = Path(module.__file__).resolve()
            result["bindings"].append({"path": str(path), "sha256": digest(path)})
    with (args.output_dir / "result.json").open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: result.get(key) for key in ("status", "stage", "error")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "tiny_prefill_head",
            "tiny_decode_head",
            "smol_prefix_head",
            "residual_bottleneck",
            "decoder_block",
            "policy_block",
            "coverage_mlp",
        ),
    )
    args = parser.parse_args()
    args.source_root = args.source_root.resolve(strict=True)
    args.source_identity = public_source(args.source_root, args.source_receipts)
    benchmark(args)


if __name__ == "__main__":
    main()
