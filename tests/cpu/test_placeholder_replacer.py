# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for placeholder_replacer module constants and helper functions."""

import unittest
from unittest.mock import MagicMock

from tritonparse.reproducer.ingestion.ndjson import ContextBundle, KernelInfo
from tritonparse.reproducer.placeholder_replacer import (
    _BASE_IMPORT_LINES,
    _detect_extra_imports,
    _EXTRA_IMPORT_PATTERNS,
    _SKIP_BARE_MODULES,
    _SKIP_IMPORTS,
    DefaultPlaceholderReplacer,
)


class TestDetectExtraImports(unittest.TestCase):
    """Tests for _detect_extra_imports()."""

    def test_detects_tlx_dot_usage(self):
        source = "result = tlx.async_task(fn, args)"
        imports = _detect_extra_imports(source)
        self.assertEqual(imports, ["import triton.language.extra.tlx as tlx"])

    def test_detects_tlx_space_usage(self):
        source = "import tlx "
        imports = _detect_extra_imports(source)
        self.assertEqual(imports, ["import triton.language.extra.tlx as tlx"])

    def test_no_match_returns_empty(self):
        source = "result = tl.load(ptr)"
        imports = _detect_extra_imports(source)
        self.assertEqual(imports, [])

    def test_deduplicates_imports(self):
        source = "tlx.foo()\ntlx.bar()\ntlx "
        imports = _detect_extra_imports(source)
        self.assertEqual(imports, ["import triton.language.extra.tlx as tlx"])

    def test_empty_source(self):
        imports = _detect_extra_imports("")
        self.assertEqual(imports, [])


class TestModuleConstants(unittest.TestCase):
    """Tests for module-level constants used in import filtering."""

    def test_skip_imports_contains_core_imports(self):
        self.assertIn("import triton", _SKIP_IMPORTS)
        self.assertIn("import torch", _SKIP_IMPORTS)
        self.assertIn("import numpy", _SKIP_IMPORTS)
        self.assertIn("import numpy as np", _SKIP_IMPORTS)

    def test_skip_bare_modules_contains_submodule_refs(self):
        for mod in ("triton", "language", "tl", "tlx", "torch", "numpy", "np"):
            self.assertIn(mod, _SKIP_BARE_MODULES)

    def test_base_import_lines_has_required_imports(self):
        joined = "\n".join(_BASE_IMPORT_LINES)
        self.assertIn("import torch", joined)
        self.assertIn("import triton", joined)
        self.assertIn("import triton.language as tl", joined)

    def test_base_import_lines_consistent_with_skip_imports(self):
        """Simple 'import X' lines in _BASE_IMPORT_LINES should be in _SKIP_IMPORTS
        to prevent duplicates from the analyzer."""
        for line in _BASE_IMPORT_LINES:
            # Only check plain 'import X' or 'import X as Y' (no dots = top-level module)
            if line.startswith("import ") and "." not in line.split()[1]:
                self.assertIn(
                    line,
                    _SKIP_IMPORTS,
                    f"'{line}' in _BASE_IMPORT_LINES but missing from _SKIP_IMPORTS",
                )

    def test_extra_import_patterns_are_tuples(self):
        for item in _EXTRA_IMPORT_PATTERNS:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)


def _make_context_bundle(
    global_scratch_size=None,
    function_name="test_kernel",
    source_code="@triton.jit\ndef test_kernel(): pass",
    file_path="/tmp/test.py",
):
    """Helper to build a minimal ContextBundle for testing."""
    kernel_info = KernelInfo(
        file_path=file_path,
        function_name=function_name,
        source_code=source_code,
        call_stack=[],
    )
    compile_block = {
        "num_warps": 4,
        "num_stages": 2,
        "global_scratch_size": global_scratch_size,
    }
    launch_block = {"grid": [1, 1, 1], "kwargs": {}}
    return ContextBundle(
        kernel_info=kernel_info,
        compile=compile_block,
        launch=launch_block,
        args={},
        tensor_args={},
        raw_launch_event={},
        raw_comp_event={},
    )


class TestAllocatorInjection(unittest.TestCase):
    """Tests for allocator injection in _replace_launch_kernel_body."""

    def _run_replace(self, global_scratch_size):
        replacer = DefaultPlaceholderReplacer()
        ctx = _make_context_bundle(global_scratch_size=global_scratch_size)
        template = "# {{LAUNCH_KERNEL_BODY_PLACEHOLDER}}"
        result = replacer._replace_launch_kernel_body(
            template,
            ctx,
            embed_context=False,
            temp_json_path=MagicMock(name="ctx.json"),
        )
        return result

    def test_allocator_injected_when_scratch_size_positive(self):
        result = self._run_replace("1024")
        self.assertIn("triton.set_allocator(_alloc_fn)", result)
        self.assertIn("global_scratch_size=1024", result)
        self.assertIn("default allocator", result.lower())

    def test_no_allocator_when_scratch_size_zero(self):
        result = self._run_replace("0")
        self.assertNotIn("set_allocator", result)

    def test_no_allocator_when_scratch_size_none(self):
        result = self._run_replace(None)
        self.assertNotIn("set_allocator", result)

    def test_no_allocator_when_scratch_size_missing(self):
        replacer = DefaultPlaceholderReplacer()
        ctx = _make_context_bundle()
        ctx.compile = {"num_warps": 4}
        template = "# {{LAUNCH_KERNEL_BODY_PLACEHOLDER}}"
        result = replacer._replace_launch_kernel_body(
            template,
            ctx,
            embed_context=False,
            temp_json_path=MagicMock(name="ctx.json"),
        )
        self.assertNotIn("set_allocator", result)

    def test_allocator_comment_mentions_customization(self):
        result = self._run_replace("512")
        self.assertIn("replace this with the allocator", result)


class TestBuildContextBundleScratchSize(unittest.TestCase):
    """Tests that build_context_bundle captures global_scratch_size."""

    def _make_events(self, global_scratch_size=None):
        comp_meta = {
            "hash": "abc123",
            "num_warps": 4,
            "num_stages": 2,
        }
        if global_scratch_size is not None:
            comp_meta["global_scratch_size"] = global_scratch_size

        launch_event = {
            "event_type": "launch",
            "grid": [1, 1, 1],
            "extracted_args": {},
            "compilation_metadata": comp_meta,
        }
        comp_event = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"hash": "abc123", "name": "my_kernel"},
                "python_source": {
                    "file_path": "/tmp/kernel.py",
                    "code": "@triton.jit\ndef my_kernel(): pass",
                },
            },
            "stack": [],
        }
        return [launch_event, comp_event]

    def test_scratch_size_captured(self):
        from tritonparse.reproducer.ingestion.ndjson import build_context_bundle

        events = self._make_events(global_scratch_size=2048)
        bundle = build_context_bundle(events, line_index=0)
        self.assertEqual(bundle.compile["global_scratch_size"], 2048)

    def test_scratch_size_none_when_absent(self):
        from tritonparse.reproducer.ingestion.ndjson import build_context_bundle

        events = self._make_events(global_scratch_size=None)
        bundle = build_context_bundle(events, line_index=0)
        self.assertIsNone(bundle.compile["global_scratch_size"])


if __name__ == "__main__":
    unittest.main()
