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
"""Real program intake must preserve coverage and refuse campaign replacement."""

import json
from pathlib import Path
import sys

import pytest
import tvm
from tvm import relax

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from program_io import import_program
from tune_models import main


def program(dtype="int8", with_exp=False):
    bb = relax.BlockBuilder()
    a = relax.Var("a", relax.TensorStructInfo((2, 16), dtype))
    b = relax.Var("b", relax.TensorStructInfo((16, 7), dtype))
    x = relax.Var("x", relax.TensorStructInfo((3,), "float32"))
    with bb.function("main", [a, b, x] if with_exp else [a, b]):
        with bb.dataflow():
            product = bb.emit(relax.op.matmul(a, b, out_dtype="int32" if dtype == "int8" else dtype))
            value = relax.Tuple([product, bb.emit(relax.op.exp(x))]) if with_exp else product
            out = bb.emit_output(value)
        bb.emit_func_output(out)
    return bb.get()


def test_real_program_inventory_includes_excluded_cpu_work_without_simulation(tmp_path, monkeypatch):
    path = tmp_path / "real.json"
    original = program(with_exp=True)
    path.write_text(tvm.ir.save_json(original))
    assert tvm.ir.structural_equal(import_program(path), original)
    import measure
    monkeypatch.setattr(measure, "Toolchain", lambda *_: pytest.fail("extraction touched toolchain"))
    out = tmp_path / "inventory"
    main(["--program", str(path), "--work-dir", str(out), "--extract-only"])
    report = json.loads((out / "extraction.json").read_text())
    assert report["scope"] == "real_program"
    assert report["eligible_tasks"] >= 1
    assert report["total_tasks"] > report["eligible_tasks"]
    assert any(row.get("reason") for row in report["extracted_tasks"])
    assert not (out / "candidates").exists()
    assert not (out / "hardware-provenance.json").exists()
    before = (out / "program.json").read_bytes()
    main(["--program", str(path), "--work-dir", str(out), "--extract-only"])
    path.write_text(tvm.ir.save_json(program(dtype="float32")))
    with pytest.raises(ValueError, match="different program"):
        main(["--program", str(path), "--work-dir", str(out), "--extract-only"])
    assert (out / "program.json").read_bytes() == before


def test_program_input_rejects_synthetic_operator_selection(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--program", str(tmp_path / "absent.json"), "--operators", "0",
              "--work-dir", str(tmp_path / "out"), "--extract-only"])
    assert error.value.code == 2


def test_semantic_json_requires_main(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(tvm.ir.save_json(tvm.IRModule()))
    with pytest.raises(ValueError, match="semantic Relax module with main"):
        import_program(path)
