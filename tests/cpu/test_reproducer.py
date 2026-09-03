# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for reproducer functionality (CPU-only, no kernel execution)."""

import argparse
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

import tritonparse.reproducer.orchestrator
from tests.test_utils import (
    cleanup_temp_dir,
    get_test_ndjson_file,
    setup_temp_reproduce_dir,
)
from tritonparse.reproducer.cli import _add_reproducer_args
from tritonparse.reproducer.ingestion.ndjson import get_kernel_info
from tritonparse.reproducer.placeholder_replacer import DefaultPlaceholderReplacer
from tritonparse.reproducer.utils import _parse_kernel_signature, dedent_kernel_source


class TestReproducer(unittest.TestCase):
    """Tests for reproducer generation."""

    def test_reproduce_mutual_exclusivity(self):
        """Test that --line and --kernel/--launch-id are mutually exclusive."""
        parser = argparse.ArgumentParser()
        _add_reproducer_args(parser)

        # Test: both --line and --kernel provided should raise error
        # Create a mock parser with error method
        mock_parser = argparse.ArgumentParser()
        _add_reproducer_args(mock_parser)
        args = mock_parser.parse_args(
            ["test.ndjson", "--line", "5", "--kernel", "matmul_kernel"]
        )

        # The mutual exclusivity check happens in cli.py main()
        # We test that args are parsed correctly, and the check will happen there
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 5)

        # Test: only --kernel should work (line defaults to 0, which is allowed)
        args = parser.parse_args(["test.ndjson", "--kernel", "matmul_kernel"])
        self.assertEqual(args.kernel, "matmul_kernel")
        self.assertEqual(args.line, 0)  # default value, allowed with --kernel

        # Test: only --line should work
        args = parser.parse_args(["test.ndjson", "--line", "5"])
        self.assertEqual(args.line, 5)
        self.assertIsNone(args.kernel)

    def test_reproduce_kernel_launch_id(self):
        """End-to-end test: reproduce using --kernel and --launch-id."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # Test reproducing fused_op_kernel launch_id=0
            result = tritonparse.reproducer.orchestrator.reproduce(
                input_path=str(gz_file),
                line_index=0,  # Placeholder, will be recalculated from kernel_name
                out_dir=out_dir,
                template="example",
                kernel_name="fused_op_kernel",
                launch_id=0,
            )

            # Verify output structure
            self.assertIn("kernel", result)
            self.assertIn("repro_script", result)
            self.assertIn("repro_context", result)
            self.assertTrue(os.path.exists(result["repro_script"]))
            self.assertTrue(os.path.exists(result["repro_context"]))

            # Verify the script contains kernel name
            script_content = Path(result["repro_script"]).read_text()
            self.assertIn("fused_op_kernel", script_content)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_kernel_not_found(self):
        """Test that proper error is raised when kernel not found."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="nonexistent_kernel",
                    launch_id=0,
                )

            error_msg = str(cm.exception)
            self.assertIn("not found", error_msg)
            self.assertIn("nonexistent_kernel", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)

    def test_reproduce_launch_id_out_of_range(self):
        """Test that proper error is raised when launch_id is out of range."""
        gz_file = get_test_ndjson_file()
        temp_dir, out_dir = setup_temp_reproduce_dir()

        try:
            # fused_op_kernel has only 4 launches (0-3), test with launch_id=10
            with self.assertRaises(ValueError) as cm:
                tritonparse.reproducer.orchestrator.reproduce(
                    input_path=str(gz_file),
                    line_index=0,  # Placeholder, will be recalculated from kernel_name
                    out_dir=out_dir,
                    template="example",
                    kernel_name="fused_op_kernel",
                    launch_id=10,
                )

            error_msg = str(cm.exception)
            self.assertIn("has only 4 launches", error_msg)
            self.assertIn("--launch-id 10", error_msg)
            self.assertIn("Valid range: 0 to 3", error_msg)

        finally:
            cleanup_temp_dir(temp_dir)


_INDENTED_KERNEL = '''\
    @triton.jit
    def fused_op_kernel(
        a_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """A kernel defined inside a function, so its source is indented."""
        pid = tl.program_id(axis=0)
        tl.store(a_ptr + pid, pid, mask=pid < n_elements)
'''


def _compilation_event(code):
    return {
        "event_type": "compilation",
        "stack": [],
        "payload": {
            "metadata": {"name": "fused_op_kernel"},
            "python_source": {"file_path": "/tritonparse/bench.py", "code": code},
        },
    }


class IndentedKernelSourceTest(unittest.TestCase):
    """A kernel defined inside a function must survive ingestion.

    Kernels nested in a function or class are captured with their original
    indentation. Slicing the captured source at the character offset of
    '@triton.jit' dedents only that first line and leaves the body indented,
    which is not valid Python -- it breaks signature parsing and, worse, gets
    embedded verbatim into the generated reproducer script.
    """

    def test_dedent_kernel_source_removes_common_indent(self):
        dedented = dedent_kernel_source(_INDENTED_KERNEL)
        self.assertTrue(dedented.startswith("@triton.jit\n"))
        self.assertIn("\ndef fused_op_kernel(\n", dedented)

    def test_dedent_kernel_source_is_a_noop_when_unindented(self):
        source = "@triton.jit\ndef k():\n    pass\n"
        self.assertEqual(dedent_kernel_source(source), source)

    def test_dedent_kernel_source_tolerates_whitespace_only_lines(self):
        """A line of trailing spaces would otherwise make the common prefix ''."""
        source = "    @triton.jit\n    \n    def k():\n        pass\n"
        self.assertTrue(dedent_kernel_source(source).startswith("@triton.jit\n"))

    def test_ingested_source_is_parseable(self):
        info = get_kernel_info(_compilation_event(_INDENTED_KERNEL))
        # The whole point: what ingestion hands downstream must be valid Python.
        ast.parse(info.source_code)
        self.assertTrue(info.source_code.startswith("@triton.jit"))
        self.assertTrue(info.is_nested)

    def test_ingested_source_keeps_body_indentation(self):
        """Dedent must be uniform, not a per-line lstrip."""
        info = get_kernel_info(_compilation_event(_INDENTED_KERNEL))
        lines = info.source_code.splitlines()
        self.assertEqual(lines[1], "def fused_op_kernel(")
        self.assertEqual(lines[2], "    a_ptr,")

    def test_signature_parses_from_indented_source(self):
        info = get_kernel_info(_compilation_event(_INDENTED_KERNEL))
        pos_args, kw_args = _parse_kernel_signature(info.source_code)
        self.assertEqual(pos_args + kw_args, ["a_ptr", "n_elements", "BLOCK_SIZE"])

    def test_preamble_before_the_decorator_is_dropped(self):
        """Slicing still starts at the decorator, just at its line boundary."""
        code = "    x = 1\n" + _INDENTED_KERNEL
        info = get_kernel_info(_compilation_event(code))
        self.assertTrue(info.source_code.startswith("@triton.jit"))
        self.assertNotIn("x = 1", info.source_code)

    def test_module_level_kernel_is_unchanged(self):
        code = "@triton.jit\ndef k(a_ptr):\n    pass\n"
        info = get_kernel_info(_compilation_event(code))
        self.assertEqual(info.source_code, code)
        self.assertFalse(info.is_nested)

    def test_nested_kernel_with_jit_alias_is_dedented(self):
        code = _INDENTED_KERNEL.replace("@triton.jit", "@jit")
        info = get_kernel_info(_compilation_event(code))
        ast.parse(info.source_code)
        self.assertTrue(info.source_code.startswith("@jit"))
        self.assertTrue(info.is_nested)


class NestedKernelReproducerTest(unittest.TestCase):
    def _write_trace(self, root: Path, preserve_autotune: bool = False) -> Path:
        root.joinpath("pyproject.toml").write_text("[project]\nname = 'fixture'\n")
        decorators = ""
        if preserve_autotune:
            decorators = (
                "    @triton.autotune(\n"
                "        configs=[triton.Config({'BLOCK_SIZE': 1})],\n"
                "        key=[],\n"
                "    )\n"
            )
        source_path = root / "nested_kernel_source.py"
        source_path.write_text(
            "import triton\n"
            "import triton.language as tl\n\n"
            "def make_kernel():\n"
            f"{decorators}"
            f"{_INDENTED_KERNEL}"
            "    return fused_op_kernel\n",
            encoding="utf-8",
        )

        compilation = _compilation_event(_INDENTED_KERNEL)
        compilation["payload"]["metadata"]["hash"] = "nested-hash"
        compilation["payload"]["python_source"]["file_path"] = str(source_path)
        launch = {
            "event_type": "launch",
            "name": "fused_op_kernel",
            "grid": [1, 1, 1],
            "extracted_args": {
                "a_ptr": {
                    "type": "tensor",
                    "shape": [1],
                    "dtype": "torch.float32",
                    "device": "cuda",
                    "stride": [1],
                    "is_contiguous": True,
                    "numel": 1,
                },
                "n_elements": {"type": "int", "value": 1},
                "BLOCK_SIZE": {"type": "int", "value": 1},
            },
            "compilation_metadata": {
                "hash": "nested-hash",
                "num_warps": 4,
                "num_stages": 1,
                "backend_name": "cuda",
            },
        }
        trace_path = root / "trace.ndjson"
        trace_path.write_text(
            f"{json.dumps(compilation)}\n{json.dumps(launch)}\n", encoding="utf-8"
        )
        return trace_path

    def _reproduce(self, root: Path, preserve_autotune: bool = False) -> str:
        trace_path = self._write_trace(root, preserve_autotune)
        replacer = (
            DefaultPlaceholderReplacer(preserve_autotune=True)
            if preserve_autotune
            else None
        )
        result = tritonparse.reproducer.orchestrator.reproduce(
            input_path=str(trace_path),
            line_index=1,
            out_dir=str(root / "out"),
            template="example",
            replacer=replacer,
            skip_logger=True,
        )
        return Path(result["repro_script"]).read_text(encoding="utf-8")

    def test_default_mode_embeds_nested_kernel_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = self._reproduce(Path(tmp))
        ast.parse(script)
        self.assertEqual(script.count("def fused_op_kernel("), 1)
        self.assertNotIn("import fused_op_kernel", script)
        self.assertIn("imported_kernel_function = fused_op_kernel", script)

    def test_preserved_autotune_source_is_dedented_and_filters_config_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = self._reproduce(Path(tmp), preserve_autotune=True)
        ast.parse(script)
        self.assertEqual(script.count("def fused_op_kernel("), 1)
        self.assertIn("@triton.autotune", script)
        self.assertNotIn('BLOCK_SIZE=args_dict["BLOCK_SIZE"]', script)


if __name__ == "__main__":
    unittest.main()
