#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging

from .sourcemap_utils import load_ir_contents


logger = logging.getLogger("IRAnalysis")


def process_amd_bufferop(ir_content: str, io_keys: list[str]) -> dict[str, int]:
    def make_key(prefix: str) -> str:
        return f"{prefix}_count"

    io_keys = [(make_key(prefix), prefix) for prefix in io_keys]
    output: dict[str, int] = {}
    for dict_key, _ in io_keys:
        output[dict_key] = 0
    if ir_content:
        for line in ir_content.split("\n"):
            for dict_key, code_key in io_keys:
                if code_key in line:
                    output[dict_key] += 1
    return output


def process_amd_ttgir_bufferops(
    key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, int]:
    ir_content = load_ir_contents(key, file_content, file_path)
    # TODO: Add atomics
    io_keys = ["tt.load", "tt.store", "amdgpu.buffer_load", "amdgpu.buffer_store"]
    return process_amd_bufferop(ir_content, io_keys)


def process_amd_gcn_bufferops(
    key: str,
    file_content: dict[str, str],
    file_path: dict[str, str],
) -> dict[str, int]:
    ir_content = load_ir_contents(key, file_content, file_path)
    # TODO: Add atomics
    io_keys = ["global_load_", "global_store_", "buffer_load_", "buffer_store_"]
    return process_amd_bufferop(ir_content, io_keys)


def _generate_ir_analysis(entry: str):
    payload = entry.setdefault("payload", {})
    file_content = payload.get("file_content", {})
    file_path = payload.get("file_path", {})

    # Find the IR file keys
    ttgir_key = next((k for k in file_content if k.endswith(".ttgir")), None)
    amdgcn_key = next((k for k in file_content if k.endswith(".amdgcn")), None)
    # Skip if no IR files found
    if not (ttgir_key or amdgcn_key):
        logger.debug("No AMD IR found")
        return {}
    ir_analysis = {}
    if amdgcn_key:
        ttgir_bufferops_info = process_amd_ttgir_bufferops(
            ttgir_key, file_content, file_path
        )
        gcn_bufferops_info = process_amd_gcn_bufferops(
            amdgcn_key, file_content, file_path
        )
        # NDJSON format requires a newline at the end of each line
        if ttgir_bufferops_info:
            ir_analysis["amd_ttgir_bufferops_count"] = ttgir_bufferops_info
        if gcn_bufferops_info:
            ir_analysis["amd_gcn_bufferops_count"] = gcn_bufferops_info
    return {"ir_analysis": ir_analysis}
