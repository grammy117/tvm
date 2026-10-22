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
"""Portable payload integrity and final ELF deployment memory admission."""

import hashlib
import json
import mmap
from pathlib import Path
import shutil
import struct
import sys

import pytest
import tvm

sys.path.insert(0, str(Path(tvm.__file__).resolve().parents[2] / "apps/gemmini"))
from compile_package import verify_package
from qualify_package import elf_memory
from verify_matmul import audit_instructions


def test_package_remains_verifiable_after_move_and_detects_changed_payload(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    payload = original / "libmodel.a"
    payload.write_bytes(b"independent static archive bytes")
    manifest = {
        "schema": "gemmini_static_package_v1",
        "files": [
            {
                "path": payload.name,
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }
    (original / "package.json").write_text(json.dumps(manifest))
    moved = tmp_path / "relocated"
    shutil.move(original, moved)
    assert verify_package(moved) == manifest
    (moved / payload.name).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="payload changed"):
        verify_package(moved)


@pytest.mark.parametrize("name", ["../external", "link"])
def test_package_cannot_reference_external_payloads(tmp_path, name):
    outside = tmp_path / "external"
    outside.write_bytes(b"external")
    folder = tmp_path / "package"
    folder.mkdir()
    (folder / "link").symlink_to(outside)
    (folder / "package.json").write_text(
        json.dumps(
            {
                "schema": "gemmini_static_package_v1",
                "files": [
                    {"path": name, "bytes": 8, "sha256": hashlib.sha256(b"external").hexdigest()}
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="escapes"):
        verify_package(folder)


def elf(*, entry=0x80000000, extent=4096, flags=4):
    header = struct.pack(
        "<16sHHIQQQIHHHHHH",
        b"\x7fELF\x02\x01" + bytes(10),
        2,
        243,
        1,
        entry,
        64,
        0,
        flags,
        64,
        56,
        1,
        64,
        0,
        0,
    )
    return header + struct.pack("<IIQQQQQQ", 1, 5, 0, 0x80000000, 0x80000000, 120, extent, 4096)


def test_elf_entry_abi_and_complete_load_storage_must_fit_declared_dram():
    assert elf_memory(elf(), 4096)["load_segments"][0]["memory_bytes"] == 4096
    with pytest.raises(ValueError, match="load segment"):
        elf_memory(elf(extent=4097), 4096)
    with pytest.raises(ValueError, match="entry"):
        elf_memory(elf(entry=0x80001000), 4096)
    with pytest.raises(ValueError, match="double-float ABI"):
        elf_memory(elf(flags=0), 4096)


def test_mapped_elf_audits_metadata_and_instructions_beyond_four_gib(tmp_path):
    # Sparse file: exercise 64-bit file offsets without allocating a multi-GB
    # Python byte string or writing a large weight payload to the test disk.
    path = tmp_path / "large.elf"
    code_offset = (1 << 32) + 4096
    section_offset = code_offset + 4096
    names = b"\0.text\0.shstrtab\0"
    header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01" + bytes(10),
                         2, 243, 1, 0x80000000, 64, section_offset, 4, 64, 56, 1, 64, 3, 2)
    segment = struct.pack("<IIQQQQQQ", 1, 5, code_offset, 0x80000000, 0x80000000, 4, 4, 4096)
    with path.open("wb") as stream:
        stream.write(header + segment)
        stream.seek(code_offset)
        stream.write(b"\x13\x00\x00\x00" + names)  # actual RV64 NOP
        stream.seek(section_offset)
        stream.write(bytes(64))
        stream.write(struct.pack("<IIQQQQIIQQ", 1, 1, 6, 0x80000000, code_offset, 4, 0, 0, 4, 0))
        stream.write(struct.pack("<IIQQQQIIQQ", 7, 3, 0, 0, code_offset + 4, len(names), 0, 0, 1, 0))
    with path.open("r+b") as stream, mmap.mmap(stream.fileno(), 0) as data:
        assert elf_memory(data, 4096)["load_segments"][0]["file_bytes"] == 4
        assert audit_instructions(data, 0x0b, {2})["instructions"] == 1
        data[code_offset:code_offset + 4] = struct.pack("<I", (2 << 25) | 0x0b)
        assert len(audit_instructions(data, 0x0b, {2})["prohibited_hits"]) == 1
