# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
End-to-end tests for tritonparse reproducer functionality.

Test Plan:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=0 python -m unittest tests.gpu.test_reproducer_e2e -v
```
"""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import triton  # @manual=//triton:triton
import tritonparse.structured_logging
from tests.test_utils import GPUTestBase
from tritonparse.reproducer.orchestrator import reproduce
from tritonparse.shared_vars import TEST_KEEP_OUTPUT
from tritonparse.tools.prettify_ndjson import load_ndjson


class TestReproducerE2E(GPUTestBase):
    """End-to-end tests for reproducer functionality."""

    # Kernel source code used by all tests
    KERNEL_SRC = (
        "import triton\n"
        "import triton.language as tl\n"
        "import torch\n"
        "\n"
        "@triton.jit\n"
        "def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(axis=0)\n"
        "    block_start = pid * BLOCK_SIZE\n"
        "    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n_elements\n"
        "    x = tl.load(x_ptr + offsets, mask=mask)\n"
        "    y = tl.load(y_ptr + offsets, mask=mask)\n"
        "    tl.store(out_ptr + offsets, x + y, mask=mask)\n"
    )

    def _setup_temp_dirs(self):
        """Create temporary directory structure for tests."""
        temp_dir = tempfile.mkdtemp()
        logs_dir = os.path.join(temp_dir, "logs")
        out_dir = os.path.join(temp_dir, "repro_output")
        kernel_dir = os.path.join(temp_dir, "kernels")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(kernel_dir, exist_ok=True)
        return temp_dir, logs_dir, out_dir, kernel_dir

    def _write_kernel_file(self, kernel_dir):
        """Write the test kernel to a file."""
        kernel_file = os.path.join(kernel_dir, "simple_kernel.py")
        with open(kernel_file, "w", encoding="utf-8") as f:
            f.write(self.KERNEL_SRC)
        return kernel_file

    def _run_kernel_with_logging(self, logs_dir, kernel_dir):
        """Run the kernel with structured logging enabled."""
        tritonparse.structured_logging.init(
            logs_dir, enable_trace_launch=True, enable_more_tensor_information=True
        )
        try:
            if kernel_dir not in sys.path:
                sys.path.insert(0, kernel_dir)

            mod = importlib.import_module("simple_kernel")
            mod = importlib.reload(mod)
            device = torch.device("cuda:0")
            torch.manual_seed(0)
            n = 256
            x = torch.randn((n,), device=device, dtype=torch.float32)
            y = torch.randn((n,), device=device, dtype=torch.float32)
            out = torch.empty_like(x)
            BLOCK_SIZE = 64
            grid = (triton.cdiv(n, BLOCK_SIZE),)
            mod.add_kernel[grid](x, y, out, n, BLOCK_SIZE)
            torch.cuda.synchronize()
        finally:
            tritonparse.structured_logging.clear_logging_config()

    def _find_launch_event(self, logs_dir):
        """Find the NDJSON file and return (ndjson_path, line_index)."""
        ndjson_files = [
            os.path.join(logs_dir, f)
            for f in os.listdir(logs_dir)
            if f.endswith(".ndjson")
        ]
        assert ndjson_files, f"No ndjson found in {logs_dir}"
        ndjson_path = max(ndjson_files, key=os.path.getmtime)

        events = load_ndjson(Path(ndjson_path))
        launch_indices = [
            i for i, ev in enumerate(events) if ev.get("event_type") == "launch"
        ]
        assert launch_indices, "No launch event found in ndjson"
        return ndjson_path, launch_indices[0]

    def _cleanup(self, temp_dir):
        """Clean up temporary directory."""
        if TEST_KEEP_OUTPUT:
            print(f"✓ Preserving temporary directory (TEST_KEEP_OUTPUT=1): {temp_dir}")
        else:
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary directory")

    def _generate_kernel_logs(self):
        """
        Set up temp dirs, write kernel, run it, and find launch event.

        Returns:
            Tuple of (ndjson_path, line_index, out_dir, temp_dir)
        """
        temp_dir, logs_dir, out_dir, kernel_dir = self._setup_temp_dirs()
        self._write_kernel_file(kernel_dir)
        self._run_kernel_with_logging(logs_dir, kernel_dir)
        ndjson_path, line_index = self._find_launch_event(logs_dir)
        return ndjson_path, line_index, out_dir, temp_dir

    def test_reproducer_end_to_end(self):
        """End-to-end test for reproducer: generate logs, build script, run it."""
        ndjson_path, line_index, out_dir, temp_dir = self._generate_kernel_logs()

        try:
            # Build reproducer
            reproduce(
                input_path=ndjson_path,
                line_index=line_index,
                out_dir=out_dir,
                template="example",
            )

            # Locate generated script and context under out_dir/add_kernel/
            kernel_out_dir = os.path.join(out_dir, "add_kernel")
            assert os.path.isdir(kernel_out_dir), (
                f"Kernel output dir not found: {kernel_out_dir}"
            )
            gen_scripts = [f for f in os.listdir(kernel_out_dir) if f.endswith(".py")]
            gen_jsons = [f for f in os.listdir(kernel_out_dir) if f.endswith(".json")]
            assert gen_scripts, f"No generated script in {kernel_out_dir}"
            assert gen_jsons, f"No generated context json in {kernel_out_dir}"
            script_path = os.path.join(kernel_out_dir, sorted(gen_scripts)[-1])

            # Execute generated script and assert success output
            proc = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Kernel execution finished.", proc.stdout)
        finally:
            self._cleanup(temp_dir)

    def test_reproducer_embed_context(self):
        """Test reproducer with embed_context=True generates standalone script."""
        ndjson_path, line_index, out_dir, temp_dir = self._generate_kernel_logs()

        try:
            # Build reproducer with embed_context=True
            result = reproduce(
                input_path=ndjson_path,
                line_index=line_index,
                out_dir=out_dir,
                template="example",
                embed_context=True,
            )

            # Verify repro_context is None when embedding
            self.assertIsNone(result["repro_context"])

            # Locate generated script under out_dir/add_kernel/
            kernel_out_dir = os.path.join(out_dir, "add_kernel")
            assert os.path.isdir(kernel_out_dir), (
                f"Kernel output dir not found: {kernel_out_dir}"
            )
            gen_scripts = [f for f in os.listdir(kernel_out_dir) if f.endswith(".py")]
            # Verify NO JSON file was created
            gen_jsons = [f for f in os.listdir(kernel_out_dir) if f.endswith(".json")]
            self.assertEqual(
                len(gen_jsons), 0, f"Expected no JSON files, found: {gen_jsons}"
            )
            assert gen_scripts, f"No generated script in {kernel_out_dir}"
            script_path = os.path.join(kernel_out_dir, sorted(gen_scripts)[-1])

            # Verify script contains embedded JSON
            with open(script_path, "r", encoding="utf-8") as f:
                script_content = f.read()
            self.assertIn('CONTEXT_JSON = r"""', script_content)
            self.assertIn("json.loads(CONTEXT_JSON)", script_content)
            # Should NOT contain file-based loading
            self.assertNotIn("create_args_from_json_file", script_content)

            # Execute generated script and assert success output
            proc = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Kernel execution finished.", proc.stdout)
        finally:
            self._cleanup(temp_dir)


if __name__ == "__main__":
    unittest.main()
