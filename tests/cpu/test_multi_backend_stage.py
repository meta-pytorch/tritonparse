import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from tritonparse.backend import (
    AmdTritonAdapter,
    AnalyzerContext,
    COMPILE_LEVEL,
    get_backend_registry,
    LAUNCH_LEVEL,
    NvidiaTritonAdapter,
    PipelineAdapterRegistry,
)
from tritonparse.parse.ir_analysis import _generate_ir_analysis
from tritonparse.parse.trace_processor import (
    _resolve_source_mappable_stage_keys,
    generate_source_mappings,
    parse_single_trace_content,
)
from tritonparse.shared_vars import (
    get_enabled_analyses,
    get_enabled_derived_artifacts,
    set_runtime_sass_dump_override,
)


class TestMultiBackendStage(unittest.TestCase):
    def test_pre_opt_stage_has_a_semantic_name(self):
        for adapter in (NvidiaTritonAdapter(), AmdTritonAdapter()):
            stage = adapter.get_stage_by_extension(".source")
            self.assertIsNotNone(stage)
            self.assertEqual(stage.name, "pre_opt_ir")
            self.assertEqual(stage.display_order, 0)
            self.assertTrue(stage.supports_source_mapping)

    def test_source_artifact_is_captured_and_labeled_as_ttir_pre_opt(self):
        from tritonparse import structured_logging

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "kernel.source"
            source_ir = '%0 = arith.muli %arg0, %c4_i32 : i32 loc("k.py":7:8)'
            source_path.write_text(source_ir, encoding="utf-8")
            payload = {
                "metadata": {"backend_name": "cuda"},
                "file_path": {},
                "file_content": {},
            }
            metadata_group = {"kernel.source": str(source_path)}
            src = SimpleNamespace(
                ext="ttir",
                language=SimpleNamespace(name="TRITON"),
            )

            with patch.object(structured_logging, "TRITONPARSE_DUMP_SOURCE_IR", True):
                structured_logging.extract_file_content(payload, metadata_group, "cuda")
                structured_logging._extract_compilation_input_metadata(
                    payload, src, metadata_group
                )

            self.assertEqual(payload["file_content"]["kernel.source"], source_ir)
            self.assertEqual(
                payload["pre_opt_ir"],
                {
                    "file": "kernel.source",
                    "stage": "ttir_pre_opt",
                    "ir_kind": "ttir",
                    "producer": "SimpleNamespace.make_ir",
                    "language": "triton",
                    "before_stage": "ttir",
                    "optimized": False,
                },
            )

    def test_source_artifact_capture_can_be_disabled(self):
        from tritonparse import structured_logging

        with tempfile.TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "kernel.source"
            source_path.write_text("module {}", encoding="utf-8")
            payload = {
                "metadata": {"backend_name": "cuda"},
                "file_path": {},
                "file_content": {},
            }
            metadata_group = {"kernel.source": str(source_path)}
            src = SimpleNamespace(
                ext="ttir",
                language=SimpleNamespace(name="TRITON"),
            )

            with patch.object(structured_logging, "TRITONPARSE_DUMP_SOURCE_IR", False):
                structured_logging.extract_file_content(payload, metadata_group, "cuda")
                structured_logging._extract_compilation_input_metadata(
                    payload, src, metadata_group
                )

            self.assertNotIn("kernel.source", payload["file_content"])
            self.assertNotIn("pre_opt_ir", payload)

    def test_ir_source_is_labeled_as_compilation_input(self):
        from tritonparse import structured_logging

        payload = {"metadata": {}, "file_path": {}, "file_content": {}}
        src = SimpleNamespace(
            ext="ttgir",
            language=SimpleNamespace(name="TRITON"),
            path="input.ttgir",
        )

        structured_logging._extract_compilation_input_metadata(
            payload, src, {"input.ttgir": "/tmp/input.ttgir"}
        )

        self.assertEqual(
            payload["input_ir"],
            {
                "file": "input.ttgir",
                "stage": "ttgir",
                "ir_kind": "ttgir",
                "producer": "SimpleNamespace.make_ir",
                "language": "triton",
            },
        )

    def test_processed_source_artifact_uses_logical_stage_name(self):
        source_ir = "\n".join(
            [
                '#loc1 = loc("kernel.py":7:8)',
                "%0 = arith.muli %arg0, %c4_i32 : i32 loc(#loc1)",
            ]
        )
        entry = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"backend_name": "cuda"},
                "file_content": {
                    "kernel.source": source_ir,
                    "kernel.ttir": source_ir,
                },
                "file_path": {},
                "pre_opt_ir": {
                    "file": "kernel.source",
                    "stage": "ttir_pre_opt",
                    "ir_kind": "ttir",
                    "producer": "ASTSource.make_ir",
                    "language": "triton",
                    "before_stage": "ttir",
                    "optimized": False,
                },
            },
        }

        stage_keys = _resolve_source_mappable_stage_keys(entry)
        self.assertEqual(stage_keys["ttir_pre_opt"], "kernel.source")
        processed = json.loads(parse_single_trace_content(json.dumps(entry)))
        payload = processed["payload"]
        self.assertIn("ttir_pre_opt", payload["source_mappings"])
        source_stage = next(
            stage for stage in payload["ir_stages"] if stage["extension"] == ".source"
        )
        self.assertEqual(source_stage["name"], "ttir_pre_opt")
        self.assertEqual(source_stage["display_name"], "TTIR (pre-optimization)")

    def test_old_trace_without_source_does_not_advertise_pre_opt_stage(self):
        entry = {
            "event_type": "compilation",
            "payload": {
                "metadata": {"backend_name": "cuda"},
                "file_content": {"kernel.ttir": "module {}"},
                "file_path": {},
            },
        }
        processed = json.loads(parse_single_trace_content(json.dumps(entry)))
        extensions = {stage["extension"] for stage in processed["payload"]["ir_stages"]}
        self.assertNotIn(".source", extensions)

    def test_legacy_fallback_resolution_order(self):
        # No metadata provided -> use hardcoded fallback order
        file_content = {"a.ptx": "...", "b.ttir": "..."}
        event = {"payload": {"file_content": file_content, "metadata": {}}}

        stage_keys = _resolve_source_mappable_stage_keys(event)

        # Fallback dict orders ttir before ptx, so keys should reflect that order
        self.assertEqual(list(stage_keys.keys()), ["ttir", "ptx"])
        # Check values mapped correctly
        self.assertEqual(stage_keys["ttir"], "b.ttir")
        self.assertEqual(stage_keys["ptx"], "a.ptx")

    def test_no_artifacts_returns_empty(self):
        event = {"payload": {"file_content": {}, "metadata": {}}}
        stage_keys = _resolve_source_mappable_stage_keys(event)
        self.assertEqual(stage_keys, {})

    def test_legacy_fallback_honors_derived_artifacts_env(self):
        from tritonparse.structured_logging import extract_file_content

        original_derived_artifacts_env = os.environ.get("TRITONPARSE_DERIVED_ARTIFACTS")
        original_dump_sass_env = os.environ.get("TRITONPARSE_DUMP_SASS")

        try:
            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "sass"
            os.environ.pop("TRITONPARSE_DUMP_SASS", None)
            set_runtime_sass_dump_override(None)

            cubin_path = "/tmp/tritonparse-test/kernel.cubin"
            payload = {
                "metadata": {},
                "file_path": {},
                "file_content": {},
            }
            metadata_group = {
                "kernel.cubin": cubin_path,
            }

            with patch("tritonparse.tools.disasm.extract", return_value="sass output"):
                extract_file_content(
                    payload,
                    metadata_group,
                    payload["metadata"].get("backend_name", ""),
                )
            self.assertEqual(payload["file_content"]["kernel.sass"], "sass output")
        finally:
            if original_derived_artifacts_env is None:
                os.environ.pop("TRITONPARSE_DERIVED_ARTIFACTS", None)
            else:
                os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = (
                    original_derived_artifacts_env
                )

            if original_dump_sass_env is None:
                os.environ.pop("TRITONPARSE_DUMP_SASS", None)
            else:
                os.environ["TRITONPARSE_DUMP_SASS"] = original_dump_sass_env

            set_runtime_sass_dump_override(None)

    def test_register_and_resolve(self):
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        # resolve by exact name
        cuda_adapter = registry.resolve(adapter_name="cuda_triton")
        self.assertEqual(cuda_adapter.adapter_name, "cuda_triton")

        # resolve should be case-insensitive for lookup
        hip_adapter = registry.resolve(adapter_name="HIP_TRITON")
        self.assertEqual(hip_adapter.adapter_name, "hip_triton")

        # unknown adapter should raise
        with self.assertRaises(ValueError):
            registry.resolve(adapter_name="unknown_adapter")

    def test_resolve_from_trace(self):
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        metadata = {"backend_name": "cuda"}
        adapter = registry.resolve_from_trace(metadata=metadata)
        self.assertEqual(adapter.adapter_name, "cuda_triton")

        # missing or invalid backend_name should raise
        with self.assertRaises(ValueError):
            registry.resolve_from_trace(metadata={})

    def test_parser_registry_and_layered_registration(self):
        """Test per-adapter parser isolation: each adapter has common + its own parsers."""
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        nvidia = registry.resolve(adapter_name="cuda_triton")
        amd = registry.resolve(adapter_name="hip_triton")

        # Common parsers present in both
        for adapter in (nvidia, amd):
            self.assertIsNotNone(adapter.get_parser("generic_loc"))
            self.assertIsNotNone(adapter.get_parser("none"))

        # NVIDIA-specific parsers only in NVIDIA adapter
        self.assertIsNotNone(nvidia.get_parser("ptx_loc"))
        self.assertIsNotNone(nvidia.get_parser("sass_loc"))
        with self.assertRaises(ValueError):
            nvidia.get_parser("amdgcn_loc")

        # AMD-specific parser only in AMD adapter
        self.assertIsNotNone(amd.get_parser("amdgcn_loc"))
        with self.assertRaises(ValueError):
            amd.get_parser("ptx_loc")
        with self.assertRaises(ValueError):
            amd.get_parser("sass_loc")

    def test_adapter_get_parser_method(self):
        """Test adapter.get_parser() with isolated per-adapter registries."""
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        nvidia = registry.resolve(adapter_name="cuda_triton")
        amd = registry.resolve(adapter_name="hip_triton")

        # Common parser works on both
        generic_parser = nvidia.get_parser("generic_loc")
        self.assertIsNotNone(generic_parser)
        generic_parser_amd = amd.get_parser("generic_loc")
        self.assertIsNotNone(generic_parser_amd)

        # Backend-specific parsers only work on their own adapter
        ptx_parser = nvidia.get_parser("ptx_loc")
        self.assertIsNotNone(ptx_parser)
        amdgcn_parser = amd.get_parser("amdgcn_loc")
        self.assertIsNotNone(amdgcn_parser)

        # Cross-backend access should fail (isolation)
        with self.assertRaises(ValueError):
            nvidia.get_parser("amdgcn_loc")
        with self.assertRaises(ValueError):
            amd.get_parser("ptx_loc")

        # Unknown parser
        with self.assertRaises(ValueError):
            nvidia.get_parser("unknown_parser")

    def test_adapter_driven_parser_selection_and_fallback(self):
        """Test adapter-driven parser selection with backward compatibility fallback."""
        # Use module-level registry so generate_source_mappings sees the same adapter
        nvidia = get_backend_registry().resolve(adapter_name="cuda_triton")

        # Test content (TTIR with #loc directives)
        ttir_content = """
            #loc = loc("test.py":10:5)
            #loc1 = loc("test.py":20:10)
            %0 = arith.constant 42 loc(#loc1)
            """

        sentinel_mapping = {
            "sentinel": {"file": "adapter_selected.py", "line": 999, "ttir_line": 1}
        }

        def sentinel_generic_loc_parser(*args, **kwargs):
            return sentinel_mapping

        original_generic_parser = nvidia.get_parser("generic_loc")
        self.assertIsNotNone(original_generic_parser)

        # Test with metadata (adapter-driven parser selection)
        metadata_cuda = {"backend_name": "cuda"}
        try:
            nvidia.register_backend_parser("generic_loc", sentinel_generic_loc_parser)

            result_with_metadata = generate_source_mappings(
                ttir_content, "ttir", None, metadata_cuda
            )
            self.assertEqual(
                result_with_metadata,
                sentinel_mapping,
                "Expected metadata-driven source mapping to use the adapter-selected parser",
            )

            # Empty metadata simulates legacy traces after payload.setdefault("metadata", {}).
            result_empty_metadata = generate_source_mappings(
                ttir_content, "ttir", None, {}
            )
            self.assertIsInstance(result_empty_metadata, dict)
            self.assertIn("4", result_empty_metadata)
            self.assertEqual(result_empty_metadata["4"]["file"], "test.py")
            self.assertEqual(result_empty_metadata["4"]["line"], 20)
            self.assertIn("ttir_line", result_empty_metadata["4"])

            result_fallback = generate_source_mappings(ttir_content, "ttir", None, None)
            self.assertIsInstance(result_fallback, dict)
            self.assertIn("4", result_fallback)
            self.assertEqual(result_fallback["4"]["file"], "test.py")
            self.assertEqual(result_fallback["4"]["line"], 20)
            self.assertIn("ttir_line", result_fallback["4"])
        finally:
            nvidia.register_backend_parser("generic_loc", original_generic_parser)

    def test_adapter_parser_execution_errors_are_not_silently_swallowed(self):
        """Adapter-selected parser execution failures should propagate instead of falling back."""
        # Use module-level registry so generate_source_mappings sees the same adapter
        nvidia = get_backend_registry().resolve(adapter_name="cuda_triton")

        ttir_content = """
            #loc = loc("test.py":10:5)
            #loc1 = loc("test.py":20:10)
            %0 = arith.constant 42 loc(#loc1)
            """

        metadata_cuda = {"backend_name": "cuda"}

        def failing_generic_loc_parser(*args, **kwargs):
            raise RuntimeError("parser execution failed")

        original_generic_parser = nvidia.get_parser("generic_loc")
        self.assertIsNotNone(original_generic_parser)

        try:
            nvidia.register_backend_parser("generic_loc", failing_generic_loc_parser)

            with self.assertRaisesRegex(RuntimeError, "parser execution failed"):
                generate_source_mappings(ttir_content, "ttir", None, metadata_cuda)
        finally:
            nvidia.register_backend_parser("generic_loc", original_generic_parser)


class TestAnalysisAdapterDriven(unittest.TestCase):
    """Comprehensive tests for adapter-driven IR analysis (new traces with metadata)."""

    def setUp(self):
        """Set up test fixtures."""
        # Use the module-level registry (same one the dispatcher uses)
        self.registry = get_backend_registry()

        # Save original environment variable
        self.original_env = os.environ.get("TRITONPARSE_ANALYSIS")

    def tearDown(self):
        """Clean up after tests."""
        # Restore original environment variable
        if self.original_env is None:
            os.environ.pop("TRITONPARSE_ANALYSIS", None)
        else:
            os.environ["TRITONPARSE_ANALYSIS"] = self.original_env

    def test_analysis_adapter_driven_path(self):
        """Adapter-driven path: returns dict with artifacts, empty dict without."""
        ctx = AnalyzerContext()
        # With artifacts (both backends)
        for backend in ("hip", "cuda"):
            trace = {
                "payload": {
                    "metadata": {"backend_name": backend},
                    "file_content": {
                        "kernel.ttir": "ttir content",
                        "kernel.ttgir": "ttgir content",
                    },
                    "file_path": {},
                    "source_mappings": {},
                }
            }
            self.assertIsInstance(
                _generate_ir_analysis(trace, ctx),
                dict,
                f"{backend} backend should return dict",
            )

        # Without artifacts
        empty_trace = {
            "payload": {
                "metadata": {"backend_name": "hip"},
                "file_content": {},
                "file_path": {},
                "source_mappings": {},
            }
        }
        self.assertEqual(_generate_ir_analysis(empty_trace, ctx), {})

    def test_analysis_legacy_path(self):
        """Fallback to legacy when adapter resolution fails."""
        ctx = AnalyzerContext()
        # With artifacts
        fallback_trace = {
            "payload": {
                "metadata": {"backend_name": "unknown"},
                "file_content": {
                    "kernel.ttgir": "ttgir content",
                    "kernel.amdgcn": "amdgcn content",
                },
                "file_path": {},
                "source_mappings": {},
            }
        }
        result = _generate_ir_analysis(fallback_trace, ctx)
        self.assertIsInstance(result, dict)
        self.assertIn("io_counts", result)

        # Without artifacts
        self.assertEqual(
            _generate_ir_analysis(
                {
                    "payload": {
                        "metadata": {"backend_name": "unknown"},
                        "file_content": {},
                        "file_path": {},
                        "source_mappings": {},
                    }
                },
                ctx,
            ),
            {},
        )

    def test_analysis_env_var_disables_all(self):
        """TRITONPARSE_ANALYSIS='none' or '' disables all analyses."""
        ctx = AnalyzerContext()
        trace = {
            "payload": {
                "metadata": {"backend_name": "hip"},
                "file_content": {
                    "kernel.ttgir": "ttgir content",
                    "kernel.amdgcn": "amdgcn content",
                },
                "file_path": {},
                "source_mappings": {},
            }
        }

        for val in ("none", ""):
            os.environ["TRITONPARSE_ANALYSIS"] = val
            self.assertEqual(
                _generate_ir_analysis(trace, ctx), {}, f"env='{val}' should disable all"
            )

    def test_analysis_registry_common_and_backend_analyzers(self):
        """Common and backend-specific analyzers are registered in per-adapter registries."""
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        amd = self.registry.resolve(adapter_name="hip_triton")

        # Common analyzers present in both
        for adapter in (nvidia, amd):
            analyzers = adapter.list_analyzer_keys()
            self.assertIn("loop_schedules", analyzers)
            self.assertIn("procedure_checks", analyzers)

        # Only AMD adapter has amd_buffer_ops
        self.assertIn("amd_buffer_ops", amd.list_analyzer_keys())
        self.assertNotIn("amd_buffer_ops", nvidia.list_analyzer_keys())

    def test_analysis_adapter_passes_differ_by_backend(self):
        """NVIDIA only has common passes; AMD additionally has amd_buffer_ops."""
        nvidia_passes = set(
            self.registry.resolve(adapter_name="cuda_triton").list_analyzer_keys()
        )
        amd_passes = set(
            self.registry.resolve(adapter_name="hip_triton").list_analyzer_keys()
        )

        self.assertIn("loop_schedules", nvidia_passes)
        self.assertNotIn("amd_buffer_ops", nvidia_passes)

        self.assertIn("loop_schedules", amd_passes)
        self.assertIn("amd_buffer_ops", amd_passes)

    def test_analysis_adapter_run_analysis_pass(self):
        """run_analysis_pass: empty artifacts → None; invalid analyzer → ValueError."""
        amd_adapter = self.registry.resolve(adapter_name="hip_triton")
        test_entry = {
            "payload": {
                "metadata": {"backend_name": "hip"},
                "file_content": {},
                "file_path": {},
                "source_mappings": {},
            }
        }
        ctx = AnalyzerContext()

        # Empty artifacts → analyzer skips due to missing required_stages
        self.assertIsNone(
            amd_adapter.run_analysis_pass("amd_buffer_ops", test_entry, ctx)
        )

        with self.assertRaises(ValueError):
            amd_adapter.run_analysis_pass("nonexistent_analyzer", test_entry, ctx)

    def test_get_enabled_analyses_helper_function(self):
        """Test get_enabled_analyses() with default, ALL/none keywords, and comma list."""
        # Default (no env var) → enable all
        if "TRITONPARSE_ANALYSIS" in os.environ:
            del os.environ["TRITONPARSE_ANALYSIS"]
        self.assertIsNone(get_enabled_analyses())

        # "ALL" → enable all (also covers case insensitivity)
        os.environ["TRITONPARSE_ANALYSIS"] = "ALL"
        self.assertIsNone(get_enabled_analyses())

        # "none" → disable all
        os.environ["TRITONPARSE_ANALYSIS"] = "none"
        self.assertEqual(get_enabled_analyses(), set())

        # Comma-separated with spaces → trimmed set
        os.environ["TRITONPARSE_ANALYSIS"] = " amd_buffer_ops , loop_schedules "
        self.assertEqual(get_enabled_analyses(), {"amd_buffer_ops", "loop_schedules"})

    def test_get_enabled_derived_artifacts_env_parsing(self):
        """Derived artifact env parsing should cover defaults, keywords, lists, and compat."""
        original_derived_artifacts_env = os.environ.get("TRITONPARSE_DERIVED_ARTIFACTS")
        original_dump_sass_env = os.environ.get("TRITONPARSE_DUMP_SASS")

        try:
            set_runtime_sass_dump_override(None)

            os.environ.pop("TRITONPARSE_DERIVED_ARTIFACTS", None)
            os.environ.pop("TRITONPARSE_DUMP_SASS", None)
            self.assertEqual(get_enabled_derived_artifacts(), set())

            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "all"
            self.assertIsNone(get_enabled_derived_artifacts())

            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "none"
            self.assertEqual(get_enabled_derived_artifacts(), set())

            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = " example , sass "
            self.assertEqual(get_enabled_derived_artifacts(), {"example", "sass"})

            os.environ.pop("TRITONPARSE_DERIVED_ARTIFACTS", None)
            os.environ["TRITONPARSE_DUMP_SASS"] = "1"
            self.assertEqual(get_enabled_derived_artifacts(), {"sass"})

            # Unknown names are passed through (validated at adapter level)
            os.environ.pop("TRITONPARSE_DUMP_SASS", None)
            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "example,unknown"
            self.assertEqual(get_enabled_derived_artifacts(), {"example", "unknown"})
        finally:
            if original_derived_artifacts_env is None:
                os.environ.pop("TRITONPARSE_DERIVED_ARTIFACTS", None)
            else:
                os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = (
                    original_derived_artifacts_env
                )

            if original_dump_sass_env is None:
                os.environ.pop("TRITONPARSE_DUMP_SASS", None)
            else:
                os.environ["TRITONPARSE_DUMP_SASS"] = original_dump_sass_env

            set_runtime_sass_dump_override(None)

    def test_register_backend_analyzer_isolation(self):
        """register_backend_analyzer: registered analyzer isolated to target adapter."""
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        amd = self.registry.resolve(adapter_name="hip_triton")

        sentinel_result = {"custom_analysis": {"key": "value"}}

        def custom_analyzer(entry, ctx):
            return sentinel_result

        # Save original loop_schedules on NVIDIA
        original_info = nvidia._analysis_registry.get_analyzer_info("loop_schedules")
        self.assertIsNotNone(original_info)
        assert original_info is not None
        original_analyzer = original_info.func
        original_stages = original_info.required_stages

        try:
            # Overwrite loop_schedules on NVIDIA with custom analyzer
            nvidia.register_backend_analyzer(
                "loop_schedules", custom_analyzer, required_stages=("ttir",)
            )

            # Custom analyzer works on NVIDIA
            self.assertEqual(
                nvidia.run_analysis_pass(
                    "loop_schedules",
                    {
                        "payload": {
                            "file_content": {},
                            "file_path": {},
                            "source_mappings": {},
                        }
                    },
                    AnalyzerContext(),
                ),
                sentinel_result,
            )

            # AMD's loop_schedules is unaffected (isolation)
            amd_info = amd._analysis_registry.get_analyzer_info("loop_schedules")
            self.assertIsNotNone(amd_info)
            assert amd_info is not None
            amd_stages = amd_info.required_stages
            self.assertEqual(amd_stages, original_stages)
        finally:
            nvidia.register_backend_analyzer(
                "loop_schedules", original_analyzer, original_stages
            )

    def test_roofline_registered_as_launch_level_analyzer(self):
        """roofline is a common launch-level analyzer on every adapter."""
        for adapter_name in ("cuda_triton", "hip_triton", "cann_triton"):
            adapter = self.registry.resolve(adapter_name=adapter_name)
            self.assertIn("roofline", adapter.list_analyzer_keys())
            info = adapter._analysis_registry.get_analyzer_info("roofline")
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.level, LAUNCH_LEVEL)

        # The common compile-level analyzers default to COMPILE_LEVEL.
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        loop_info = nvidia._analysis_registry.get_analyzer_info("loop_schedules")
        self.assertIsNotNone(loop_info)
        assert loop_info is not None
        self.assertEqual(loop_info.level, COMPILE_LEVEL)

    def test_list_executable_analyzers_filters_by_level(self):
        """list_executable_analyzers returns only analyzers of the requested level."""
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        file_content = {"kernel.ttir": "t", "kernel.ttgir": "t"}

        compile_execs = nvidia.list_executable_analyzers(file_content)
        self.assertIn("loop_schedules", compile_execs)
        self.assertNotIn("roofline", compile_execs)

        launch_execs = nvidia.list_executable_analyzers(
            file_content, level=LAUNCH_LEVEL
        )
        self.assertIn("roofline", launch_execs)
        self.assertNotIn("loop_schedules", launch_execs)

    def test_register_backend_launch_analyzer_isolation(self):
        """register_backend_launch_analyzer: registered analyzer isolated to target adapter."""
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        amd = self.registry.resolve(adapter_name="hip_triton")

        sentinel_result = {"custom_launch": {"key": "value"}}

        def custom_launch_analyzer(entry, ctx):
            return sentinel_result

        try:
            nvidia.register_backend_launch_analyzer(
                "custom_launch", custom_launch_analyzer, required_stages=("ttir",)
            )

            info = nvidia._analysis_registry.get_analyzer_info("custom_launch")
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info.level, LAUNCH_LEVEL)

            # Visible only at the launch level on NVIDIA.
            file_content = {"kernel.ttir": "t"}
            self.assertIn(
                "custom_launch",
                nvidia.list_executable_analyzers(file_content, level=LAUNCH_LEVEL),
            )
            self.assertNotIn(
                "custom_launch", nvidia.list_executable_analyzers(file_content)
            )

            # AMD adapter is unaffected (isolation).
            self.assertNotIn("custom_launch", amd.list_analyzer_keys())
        finally:
            nvidia._analysis_registry._analyzer_infos.pop("custom_launch", None)

    def test_register_backend_derived_artifact_isolation(self):
        """register_backend_derived_artifact: registered artifact isolated to target adapter."""
        nvidia = self.registry.resolve(adapter_name="cuda_triton")
        amd = self.registry.resolve(adapter_name="hip_triton")

        def sentinel_derive(path):
            return "derived content"

        # Save original sass derived artifact on NVIDIA
        original_artifacts = [
            info
            for info in nvidia.list_applicable_derived_artifacts()
            if info.target_stage_name == "sass"
        ]
        original_sass = original_artifacts[0] if original_artifacts else None

        try:
            # Overwrite sass on NVIDIA with custom derived artifact
            nvidia.register_backend_derived_artifact(
                source_stage_name="cubin",
                target_stage_name="sass",
                tool_name="test_tool",
                derive_func=sentinel_derive,
            )

            # Custom artifact visible on NVIDIA
            nvidia_artifacts = nvidia.list_applicable_derived_artifacts()
            target_names = [info.target_stage_name for info in nvidia_artifacts]
            self.assertIn("sass", target_names)

            # AMD is unaffected (isolation) — AMD has no sass
            amd_artifacts = amd.list_applicable_derived_artifacts()
            amd_target_names = [info.target_stage_name for info in amd_artifacts]
            self.assertNotIn("sass", amd_target_names)
        finally:
            if original_sass:
                nvidia.register_backend_derived_artifact(
                    source_stage_name=original_sass.source_stage_name,
                    target_stage_name=original_sass.target_stage_name,
                    tool_name=original_sass.tool_name,
                    derive_func=original_sass.derive_func,
                )


class TestDeviceStringHelpers(unittest.TestCase):
    """Tests for device-string normalization helpers."""

    def test_public_helper_normalizes_cuda(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("cuda"), "cuda:0")
        self.assertEqual(normalize_accelerator_device_string("cuda:2"), "cuda:0")

    def test_public_helper_normalizes_hip(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("hip"), "hip:0")
        self.assertEqual(normalize_accelerator_device_string("hip:3"), "hip:0")

    def test_public_helper_keeps_cpu(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("cpu"), "cpu")

    def test_public_helper_normalizes_indexed_device(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("hip:3"), "hip:0")

    def test_public_helper_maps_empty_to_cpu(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string(""), "cpu")


class TestAmdBufferOpsStatus(unittest.TestCase):
    """Derived AMD buffer-ops enablement status in ir_analysis."""

    _TTGIR = """\
tt.func @kernel(%arg0: !tt.ptr<f32>) {
  %0 = tt.load %arg0 : !tt.ptr<f32>
  %1 = amdg.buffer_load %arg0 : !tt.ptr<f32>
  %2 = amdg.buffer_load_to_local %arg0 : !tt.ptr<f32>
  %3 = tt.atomic_rmw add, %arg0, %0 : !tt.ptr<f32>
  %4 = amdg.buffer_atomic_rmw add, %arg0, %0 : !tt.ptr<f32>
  tt.store %arg0, %1 : !tt.ptr<f32>
}"""

    _GCN_ALL_BUFFER = """\
buffer_load_dwordx4 v[1:4], off, s[4:7], 0
buffer_store_dwordx2 v[1:2], v3, s[4:5]
"""

    _GCN_PARTIAL = """\
buffer_load_dwordx4 v[1:4], off, s[4:7], 0
global_store_dwordx2 v[1:2], v3, s[4:5]
"""

    _GCN_NONE = """\
global_load_dwordx4 v[1:4], off, s[4:7], 0
global_store_dwordx2 v[1:2], v3, s[4:5]
"""

    _GCN_NO_MEMOPS = """\
s_waitcnt vmcnt(0)
v_add_u32 v1, v2, v3
"""

    _GCN_ALL_BUFFER_ATOMICS = """\
buffer_atomic_add v1, off, s[4:7], v2
buffer_atomic_cmpswap_x2 v[1:2], off, s[4:7], v[3:4]
"""

    _GCN_PARTIAL_ATOMIC = """\
buffer_load_dwordx4 v[1:4], off, s[4:7], 0
global_atomic_add v1, off, s[4:7], v2
"""

    _GCN_ONLY_GLOBAL_ATOMICS = """\
global_atomic_add v1, off, s[4:7], v2
global_atomic_cmpswap v1, off, s[4:7], v[2:3]
flat_atomic_add v1, off, s[4:7], v2
"""

    def _entry(self, gcn: str):
        return {
            "payload": {
                "metadata": {"backend_name": "hip"},
                "file_content": {
                    "kernel.ttgir": self._TTGIR,
                    "kernel.amdgcn": gcn,
                },
                "file_path": {},
                "source_mappings": {},
            }
        }

    def _run_pass(self, gcn: str):
        adapter = get_backend_registry().resolve(adapter_name="hip_triton")
        return adapter.run_analysis_pass(
            "amd_buffer_ops", self._entry(gcn), AnalyzerContext()
        )

    def test_all_buffer_enabled(self):
        result = self._run_pass(self._GCN_ALL_BUFFER)
        status = result["amd_buffer_ops"]
        self.assertTrue(status["enabled"])
        self.assertEqual(status["status"], "all_buffer")
        self.assertEqual(status["buffer_load_count"], 1)
        self.assertEqual(status["buffer_store_count"], 1)
        self.assertEqual(status["global_load_count"], 0)
        self.assertEqual(status["global_store_count"], 0)

    def test_partial_not_enabled(self):
        result = self._run_pass(self._GCN_PARTIAL)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "partial")

    def test_no_buffer_ops(self):
        result = self._run_pass(self._GCN_NONE)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "none")

    def test_no_memops_unknown(self):
        result = self._run_pass(self._GCN_NO_MEMOPS)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "unknown")

    def test_ttgir_prefix_and_atomics(self):
        """Real amdg. prefix counted; load_to_local not double-counted."""
        result = self._run_pass(self._GCN_ALL_BUFFER)
        ttgir = result["io_counts"]["amd_ttgir_bufferops_count"]
        self.assertEqual(ttgir["tt.load_count"], 1)
        self.assertEqual(ttgir["tt.store_count"], 1)
        self.assertEqual(ttgir["tt.atomic_rmw_count"], 1)
        self.assertEqual(ttgir["tt.atomic_cas_count"], 0)
        self.assertEqual(ttgir["amdg.buffer_load_count"], 1)
        self.assertEqual(ttgir["amdg.buffer_load_to_local_count"], 1)
        self.assertEqual(ttgir["amdg.buffer_store_count"], 0)
        self.assertEqual(ttgir["amdg.buffer_atomic_rmw_count"], 1)
        self.assertEqual(ttgir["amdg.buffer_atomic_cas_count"], 0)

    def test_legacy_amdgpu_prefix_counted(self):
        """Historical amdgpu. spelling still matches."""
        from tritonparse.parse.ir_analysis import process_amd_ttgir_bufferops

        counts = process_amd_ttgir_bufferops(
            "k.ttgir",
            {"k.ttgir": "%x = amdgpu.buffer_load %p : !tt.ptr<f32>"},
            {},
        )
        self.assertEqual(counts["amdg.buffer_load_count"], 1)

    def test_all_buffer_atomics_enabled(self):
        result = self._run_pass(self._GCN_ALL_BUFFER_ATOMICS)
        status = result["amd_buffer_ops"]
        self.assertTrue(status["enabled"])
        self.assertEqual(status["status"], "all_buffer")
        self.assertEqual(status["buffer_atomic_count"], 2)

    def test_partial_with_global_atomics(self):
        result = self._run_pass(self._GCN_PARTIAL_ATOMIC)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["global_atomic_count"], 1)

    def test_only_global_atomics(self):
        result = self._run_pass(self._GCN_ONLY_GLOBAL_ATOMICS)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "none")
        self.assertEqual(status["global_atomic_count"], 2)
        self.assertEqual(status["flat_atomic_count"], 1)

    def test_status_keeps_io_counts(self):
        """The new field is additive: existing io_counts are unchanged."""
        result = self._run_pass(self._GCN_ALL_BUFFER)
        self.assertIn("io_counts", result)
        self.assertIn("amd_buffer_ops", result)
        self.assertEqual(
            result["io_counts"]["amd_gcn_bufferops_count"]["buffer_load_count"], 1
        )

    def test_legacy_path_emits_status(self):
        """Legacy fallback (unknown backend) emits amd_buffer_ops too."""
        old = os.environ.pop("TRITONPARSE_ANALYSIS", None)
        try:
            entry = self._entry(self._GCN_ALL_BUFFER)
            entry["payload"]["metadata"] = {"backend_name": "unknown"}
            result = _generate_ir_analysis(entry, AnalyzerContext())
        finally:
            if old is not None:
                os.environ["TRITONPARSE_ANALYSIS"] = old
        self.assertIn("amd_buffer_ops", result)
        self.assertTrue(result["amd_buffer_ops"]["enabled"])

    _GCN_REALISTIC_BLOCK = """\
s_waitcnt vmcnt(0) lgkmcnt(0)
s_buffer_load_dwordx4 s[0:3], s[4:7], 0
buffer_load_dword v1, off, s[4:7], 0
buffer_load_dwordx2 v[2:3], off, s[4:7], 0
buffer_store_dwordx4 v[1:4], off, s[4:7], 0
global_load_dwordx4 v[5:8], off, s[8:11], 0
; spill: buffer_load_dword v9, off, s[12:15], 0
v_add_u32 v1, v2, v3
"""

    def test_realistic_gcn_block_counts(self):
        """Realistic disassembly text: scalar s_ prefix and comment lines
        count (line-based substring matching), one global op flips the
        verdict to partial."""
        result = self._run_pass(self._GCN_REALISTIC_BLOCK)
        status = result["amd_buffer_ops"]
        self.assertFalse(status["enabled"])
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["buffer_load_count"], 4)
        self.assertEqual(status["buffer_store_count"], 1)
        self.assertEqual(status["global_load_count"], 1)
        self.assertEqual(status["global_store_count"], 0)
        self.assertTrue(status["reason"])


class TestStageParserContract(unittest.TestCase):
    """Guards for the bug class that left LLIR silently unmapped.

    LLIR was declared `supports_source_mapping=True` with
    `parser_id="generic_loc"` -- an MLIR parser -- on both the NVIDIA and AMD
    adapters. It matched nothing in a .llir file, so the stage advertised a
    capability it never delivered and nothing failed. These two tests make that
    shape detectable: the first at declaration time, the second at runtime.
    """

    # A mappable stage's parser must suit the IR's own syntax. `syntax_id`
    # already records the format, so a mismatch is visible in the descriptor
    # without running anything.
    #
    # Deliberately EXHAUSTIVE: a mappable stage whose syntax_id is absent here
    # fails. Leaving it permissive would reproduce the original bug's failure
    # mode -- a new backend could add a stage with an unrecognised syntax and
    # the wrong parser, and this test would stay green. Adding a backend
    # therefore requires one deliberate line below.
    ALLOWED_PARSERS = {
        "mlir": {"generic_loc"},
        "llvm": {"llvm_dbg"},
        "ptx": {"ptx_loc"},
        # both are legitimate: NVIDIA SASS and AMD GCN are different assemblies
        "asm": {"sass_loc", "amdgcn_loc"},
    }

    def _adapters(self):
        from tritonparse import backend as backend_mod

        seen = []
        for name, obj in vars(backend_mod).items():
            if (
                isinstance(obj, type)
                and name.endswith("Adapter")
                and obj is not backend_mod.CompilationPipelineAdapter
            ):
                try:
                    seen.append((name, obj()))
                except Exception:  # adapters needing a live backend
                    continue
        self.assertTrue(seen, "no adapters discovered")
        return seen

    def test_parser_id_matches_syntax_id_for_mappable_stages(self):
        """Catches `generic_loc` on a non-MLIR stage, which is the original bug."""
        bad = []
        for adapter_name, adapter in self._adapters():
            for stage in adapter.list_ir_stages():
                if not stage.supports_source_mapping:
                    continue
                allowed = self.ALLOWED_PARSERS.get(stage.syntax_id)
                if allowed is None:
                    bad.append(
                        f"{adapter_name}.{stage.name}: syntax_id="
                        f"{stage.syntax_id!r} is not in ALLOWED_PARSERS. Add it "
                        f"with the parser_id(s) valid for that syntax, so a new "
                        f"backend cannot ship a mappable stage with a parser "
                        f"that does not understand its format."
                    )
                elif stage.parser_id not in allowed:
                    bad.append(
                        f"{adapter_name}.{stage.name}: parser_id="
                        f"{stage.parser_id!r} but syntax_id={stage.syntax_id!r} "
                        f"expects one of {sorted(allowed)}"
                    )
        self.assertEqual(bad, [], "parser/syntax mismatch on a mappable stage")

    def test_mappable_stages_actually_produce_mappings(self):
        """A registered parser that silently matches nothing must fail loudly.

        Runs every mappable stage's own parser over the checked-in example
        trace, which inlines `file_content` for each stage, and requires a
        non-empty result. Catches a parser that is wired up correctly but does
        not understand the format it was given.
        """
        import json
        from importlib.resources import files

        log = files("tests").joinpath(
            "example_output",
            "logs",
            "dedicated_log_triton_trace_findhao_.ndjson",
        )
        self.assertTrue(log.is_file(), f"Required example trace not available: {log}")

        adapter = NvidiaTritonAdapter()
        by_extension = {s.extension: s for s in adapter.list_ir_stages()}
        checked = set()
        with log.open() as fh:
            for line in fh:
                event = json.loads(line)
                if event.get("event_type") != "compilation":
                    continue
                for fname, content in event["payload"]["file_content"].items():
                    suffix = f".{fname.rsplit('.', 1)[-1]}"
                    stage = by_extension.get(suffix)
                    if stage is None or not stage.supports_source_mapping:
                        continue
                    parser = adapter.get_parser(stage.parser_id)
                    self.assertIsNotNone(
                        parser, f"{stage.name}: parser {stage.parser_id} missing"
                    )
                    result = parser(content, None, stage.name)
                    self.assertTrue(
                        result,
                        f"{stage.name} declares supports_source_mapping=True but "
                        f"parser {stage.parser_id!r} produced no mappings",
                    )
                    checked.add(stage.name)
        self.assertIn("llir", checked, "llir not exercised by this fixture")
        self.assertGreaterEqual(len(checked), 3, f"only checked {checked}")


class TestCompilerIdentity(unittest.TestCase):
    def _load_package(self, package_dir: Path, name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            name,
            package_dir / "__init__.py",
            submodule_search_locations=[str(package_dir)],
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        return module

    def _write_package(self, root: Path, name: str = "fake_triton_pkg") -> Path:
        package_dir = root / "triton"
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").write_text(
            '__version__ = "3.8.0"\n', encoding="utf-8"
        )
        return package_dir

    def test_reads_packaged_identity_from_real_package(self):
        from tritonparse import compiler_identity

        with tempfile.TemporaryDirectory() as tmp:
            package_dir = self._write_package(Path(tmp))
            (package_dir / "src_hash.txt").write_text("source-hash\n", encoding="utf-8")
            (package_dir / "fb_version_info.json").write_text(
                json.dumps(
                    {
                        "internal_version_label": "beta",
                        "oss_version": "20261008",
                        "upstream_hash": "upstream-hash",
                        "ignored": "not part of the contract",
                    }
                ),
                encoding="utf-8",
            )
            triton_module = self._load_package(package_dir, "fake_triton_pkg")
            identity = compiler_identity._build_compiler_identity(triton_module)

        self.assertEqual(
            identity,
            {
                "triton_version": "3.8.0",
                "src_hash": "source-hash",
                "fb_version_info": {
                    "internal_version_label": "beta",
                    "oss_version": "20261008",
                    "upstream_hash": "upstream-hash",
                },
            },
        )

    def test_unreadable_packaged_files_drop_only_their_fields(self):
        from tritonparse import compiler_identity

        with tempfile.TemporaryDirectory() as tmp:
            package_dir = self._write_package(Path(tmp), "fake_triton_bad_pkg")
            (package_dir / "src_hash.txt").write_bytes(b"\xff\xfe not utf-8")
            (package_dir / "fb_version_info.json").write_text(
                "{not json", encoding="utf-8"
            )
            triton_module = self._load_package(package_dir, "fake_triton_bad_pkg")
            identity = compiler_identity._build_compiler_identity(triton_module)

        self.assertEqual(identity, {"triton_version": "3.8.0"})

    def test_missing_version_is_unknown(self):
        from tritonparse import compiler_identity

        triton_module = ModuleType("triton")
        triton_module.__file__ = "/nonexistent/triton/__init__.py"

        self.assertEqual(
            compiler_identity._build_compiler_identity(triton_module),
            {"triton_version": "unknown"},
        )

    def test_identity_is_computed_once_and_returned_by_value(self):
        from tritonparse import compiler_identity

        compiler_identity._get_compiler_identity.cache_clear()
        triton_module = ModuleType("triton")
        expected = {"triton_version": "test", "fb_version_info": {"x": "y"}}
        try:
            with (
                patch.dict(sys.modules, {"triton": triton_module}),
                patch.object(
                    compiler_identity,
                    "_build_compiler_identity",
                    side_effect=lambda module: copy.deepcopy(expected),
                ) as build_identity,
            ):
                first = compiler_identity.get_compiler_identity()
                first["fb_version_info"]["x"] = "changed"
                second = compiler_identity.get_compiler_identity()

            build_identity.assert_called_once_with(triton_module)
            self.assertEqual(second, expected)
        finally:
            compiler_identity._get_compiler_identity.cache_clear()

    def test_failed_identity_is_cached_as_unknown(self):
        from tritonparse import compiler_identity

        compiler_identity._get_compiler_identity.cache_clear()
        triton_module = ModuleType("triton")
        try:
            with (
                patch.dict(sys.modules, {"triton": triton_module}),
                patch.object(
                    compiler_identity,
                    "_build_compiler_identity",
                    side_effect=RuntimeError("probe failed"),
                ) as build_identity,
            ):
                with self.assertLogs(compiler_identity.log, level="WARNING"):
                    first = compiler_identity.get_compiler_identity()
                second = compiler_identity.get_compiler_identity()

            build_identity.assert_called_once_with(triton_module)
            self.assertEqual(first, {"triton_version": "unknown"})
            self.assertEqual(second, {"triton_version": "unknown"})
        finally:
            compiler_identity._get_compiler_identity.cache_clear()


if __name__ == "__main__":
    unittest.main()
