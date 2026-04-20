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
from tritonparse.reproducer.types import KernelImportMode


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


class TestBuildContextBundleCudaGraphCapture(unittest.TestCase):
    """Tests that build_context_bundle raises RuntimeError for CUDA graph capture launches.

    When a kernel is launched during CUDA graph capture, argument extraction
    is skipped (see D86722827) and extracted_args contains only a _note string
    instead of per-argument dicts. build_context_bundle should detect this and
    raise a clear RuntimeError.
    """

    def _make_events_with_note(self, note_message):
        """Create launch + compilation events where extracted_args has a _note sentinel."""
        launch_event = {
            "event_type": "launch",
            "grid": [1, 1, 1],
            "extracted_args": {
                "_note": note_message,
            },
            "compilation_metadata": {
                "hash": "abc123",
                "num_warps": 4,
                "num_stages": 2,
            },
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

    def test_note_sentinel_raises_runtime_error(self):
        """Test that _note in extracted_args raises RuntimeError."""
        from tritonparse.reproducer.ingestion.ndjson import build_context_bundle

        note = "Argument extraction skipped during CUDA graph capture"
        events = self._make_events_with_note(note)

        with self.assertRaises(RuntimeError) as cm:
            build_context_bundle(events, line_index=0)

        error_msg = str(cm.exception)
        self.assertIn("Cannot generate reproducer", error_msg)
        self.assertIn("my_kernel", error_msg)
        self.assertIn(note, error_msg)

    def test_no_note_sentinel_succeeds(self):
        """Test that normal extracted_args without _note works fine."""
        from tritonparse.reproducer.ingestion.ndjson import build_context_bundle

        launch_event = {
            "event_type": "launch",
            "grid": [1, 1, 1],
            "extracted_args": {
                "x": {"type": "tensor", "shape": [4, 4], "dtype": "float32"},
                "BLOCK_SIZE": {"type": "int", "value": 128},
            },
            "compilation_metadata": {
                "hash": "abc123",
                "num_warps": 4,
                "num_stages": 2,
            },
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
        events = [launch_event, comp_event]

        # Should not raise
        bundle = build_context_bundle(events, line_index=0)
        self.assertEqual(bundle.kernel_info.function_name, "my_kernel")


class InterleavedConstexprInvocationTest(unittest.TestCase):
    """Tests for _replace_kernel_invocation with interleaved constexpr params.

    Regression test for a bug where the OVERRIDE_TTIR path dropped constexpr
    names from the positional arg list while passing the remaining
    non-constexpr args positionally. When constexprs are interleaved with
    regular params (e.g. ``..., N_CTX, is_predict: constexpr, Q_SHAPE_0, ...``),
    the resulting call shifted later positional args into the constexpr's
    parameter slot, colliding with the autotune-supplied kwarg for that
    constexpr ("got multiple values for argument 'is_predict'"). The fix
    passes all non-constexpr args as keyword args in the override path.
    """

    _INTERLEAVED_SOURCE = (
        "@triton.jit\n"
        "def my_kernel(\n"
        "    Q,\n"
        "    K,\n"
        "    N_CTX,\n"
        "    is_predict: tl.constexpr,\n"
        "    Q_SHAPE_0,\n"
        "    FUSED_QKV: tl.constexpr,\n"
        "    B,\n"
        "    BLOCK_M: tl.constexpr,\n"
        "):\n"
        "    pass\n"
    )

    def _make_bundle(self):
        kernel_info = KernelInfo(
            file_path="/tmp/kernel.py",
            function_name="my_kernel",
            source_code=self._INTERLEAVED_SOURCE,
            call_stack=[],
        )
        args = {
            "Q": {"value": None},
            "K": {"value": None},
            "N_CTX": {"value": 200},
            "is_predict": {"value": False},
            "Q_SHAPE_0": {"value": 204800},
            "FUSED_QKV": {"value": False},
            "B": {"value": None},
            "BLOCK_M": {"value": 256},
        }
        return ContextBundle(
            kernel_info=kernel_info,
            compile={"num_warps": 8, "num_stages": 4},
            launch={"grid": [1, 1, 1], "kwargs": {}},
            args=args,
            tensor_args={},
            raw_launch_event={},
            raw_comp_event={},
        )

    def _render_override_invocation(self):
        replacer = DefaultPlaceholderReplacer()
        template = "    # {{KERNEL_INVOCATION_PLACEHOLDER}}"
        return replacer._replace_kernel_invocation(
            template,
            self._make_bundle(),
            kernel_import=KernelImportMode.OVERRIDE_TTIR,
        )

    def test_override_snippet_passes_nonconstexpr_as_kwargs(self):
        """Non-constexpr args in the override branch must be kwargs."""
        result = self._render_override_invocation()
        override_branch = result.split("else:")[0]
        # Non-constexpr args must appear as keyword args, not positional.
        self.assertIn('Q=args_dict["Q"]', override_branch)
        self.assertIn('K=args_dict["K"]', override_branch)
        self.assertIn('N_CTX=args_dict["N_CTX"]', override_branch)
        self.assertIn('Q_SHAPE_0=args_dict["Q_SHAPE_0"]', override_branch)
        self.assertIn('B=args_dict["B"]', override_branch)

    def test_override_snippet_omits_constexprs(self):
        """Constexpr args must not appear in the override branch.

        They are supplied by the autotune Config.kwargs instead.
        """
        result = self._render_override_invocation()
        override_branch = result.split("else:")[0]
        self.assertNotIn("is_predict", override_branch)
        self.assertNotIn("FUSED_QKV", override_branch)
        self.assertNotIn("BLOCK_M", override_branch)

    def test_override_snippet_has_no_positional_args(self):
        """No positional ``args_dict["X"]`` entries should appear in the
        override branch — the fix routes everything through kwargs to avoid
        position shifts when a constexpr is interleaved in the signature.
        """
        result = self._render_override_invocation()
        override_branch = result.split("else:")[0]
        call_line = next(
            (
                line
                for line in override_branch.splitlines()
                if "imported_kernel_function" in line
            ),
            "",
        )
        self.assertIn("imported_kernel_function", call_line)
        open_paren = call_line.index("(", call_line.index("imported_kernel_function"))
        arg_text = call_line[open_paren + 1 :]
        # Every occurrence of args_dict["X"] should be immediately preceded
        # by 'X=' (keyword form), never appear as a bare positional.
        for segment in arg_text.split('args_dict["')[1:]:
            name = segment.split('"]', 1)[0]
            self.assertIn(f'{name}=args_dict["{name}"]', call_line)

    def test_fallback_branch_still_passes_all_args(self):
        """The fallback branch (no IR override) keeps passing every arg,
        including constexprs, so the kernel launches correctly when the
        captured_irs/ directory is missing.
        """
        result = self._render_override_invocation()
        _, fallback_branch = result.split("else:", 1)
        self.assertIn('args_dict["is_predict"]', fallback_branch)
        self.assertIn('args_dict["FUSED_QKV"]', fallback_branch)
        self.assertIn('args_dict["BLOCK_M"]', fallback_branch)


if __name__ == "__main__":
    unittest.main()
