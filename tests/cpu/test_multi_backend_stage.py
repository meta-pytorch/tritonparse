import os
import unittest
from unittest.mock import patch

from tritonparse.backend import (
    AmdTritonAdapter,
    AnalysisRegistry,
    get_backend_registry,
    NvidiaTritonAdapter,
    ParserRegistry,
    PipelineAdapterRegistry,
)
from tritonparse.parse.ir_analysis import _generate_ir_analysis
from tritonparse.parse.trace_processor import (
    _resolve_source_mappable_stage_keys,
    generate_source_mappings,
)
from tritonparse.shared_vars import (
    get_enabled_analyses,
    get_enabled_derived_artifacts,
    set_runtime_sass_dump_override,
)


class TestMultiBackendStage(unittest.TestCase):
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
        """Test ParserRegistry and layered parser registration (common + backend-specific)."""
        # Step 1: Verify common parsers are pre-registered
        common_parsers = {"generic_loc", "none"}
        for parser_id in common_parsers:
            parser = ParserRegistry.get_parser(parser_id)
            self.assertIsNotNone(
                parser, f"Common parser '{parser_id}' should be pre-registered"
            )

        # Step 2: Create adapters (triggers backend-specific parser registration)
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        # Step 3: Verify all parsers are listed (check registry list functionality)
        all_parsers = ParserRegistry.list_parsers()
        self.assertIn("generic_loc", all_parsers)
        self.assertIn("none", all_parsers)
        self.assertIn("ptx_loc", all_parsers)
        self.assertIn("sass_loc", all_parsers)
        self.assertIn("amdgcn_loc", all_parsers)

    def test_adapter_get_parser_method(self):
        """Test adapter.get_parser() method with both common and backend-specific parsers."""
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        # Get NVIDIA adapter
        nvidia_adapter = registry.resolve(adapter_name="cuda_triton")

        # Test getting common parser (generic_loc)
        generic_parser = nvidia_adapter.get_parser("generic_loc")
        self.assertIsNotNone(generic_parser)

        # Test getting NVIDIA-specific parser
        ptx_parser = nvidia_adapter.get_parser("ptx_loc")
        self.assertIsNotNone(ptx_parser)

        # Test getting AMD parser (also works because parsers are globally registered)
        amdgcn_parser = nvidia_adapter.get_parser("amdgcn_loc")
        self.assertIsNotNone(amdgcn_parser)

        # Get AMD adapter and test NVIDIA parser (also works for symmetrical access)
        amd_adapter = registry.resolve(adapter_name="hip_triton")
        ptx_parser_amd = amd_adapter.get_parser("ptx_loc")
        self.assertIsNotNone(ptx_parser_amd)

        # Test getting unknown parser should raise ValueError
        with self.assertRaises(ValueError):
            nvidia_adapter.get_parser("unknown_parser")

    def test_adapter_driven_parser_selection_and_fallback(self):
        """Test adapter-driven parser selection with backward compatibility fallback."""
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

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

        original_generic_parser = ParserRegistry.get_parser("generic_loc")
        self.assertIsNotNone(original_generic_parser)

        # Test with metadata (adapter-driven parser selection)
        metadata_cuda = {"backend_name": "cuda"}
        try:
            ParserRegistry.register("generic_loc", sentinel_generic_loc_parser)

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
            ParserRegistry.register("generic_loc", original_generic_parser)

    def test_adapter_parser_execution_errors_are_not_silently_swallowed(self):
        """Adapter-selected parser execution failures should propagate instead of falling back."""
        registry = PipelineAdapterRegistry()
        registry.register(NvidiaTritonAdapter)
        registry.register(AmdTritonAdapter)

        ttir_content = """
            #loc = loc("test.py":10:5)
            #loc1 = loc("test.py":20:10)
            %0 = arith.constant 42 loc(#loc1)
            """

        metadata_cuda = {"backend_name": "cuda"}

        def failing_generic_loc_parser(*args, **kwargs):
            raise RuntimeError("parser execution failed")

        original_generic_parser = ParserRegistry.get_parser("generic_loc")
        self.assertIsNotNone(original_generic_parser)

        try:
            ParserRegistry.register("generic_loc", failing_generic_loc_parser)

            with self.assertRaisesRegex(RuntimeError, "parser execution failed"):
                generate_source_mappings(ttir_content, "ttir", None, metadata_cuda)
        finally:
            ParserRegistry.register("generic_loc", original_generic_parser)


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
                _generate_ir_analysis(trace),
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
        self.assertEqual(_generate_ir_analysis(empty_trace), {})

    def test_analysis_legacy_path(self):
        """Fallback to legacy when adapter resolution fails."""
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
        result = _generate_ir_analysis(fallback_trace)
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
                }
            ),
            {},
        )

    def test_analysis_env_var_disables_all(self):
        """TRITONPARSE_ANALYSIS='none' or '' disables all analyses."""
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
                _generate_ir_analysis(trace), {}, f"env='{val}' should disable all"
            )

    def test_analysis_registry_common_and_backend_analyzers(self):
        """Common and backend-specific analyzers are all registered after adapter init."""
        for analyzer_id in ("loop_schedules", "procedure_checks", "amd_buffer_ops"):
            self.assertIsNotNone(
                AnalysisRegistry.get_analyzer(analyzer_id),
                f"Analyzer '{analyzer_id}' should be registered",
            )

    def test_analysis_adapter_passes_differ_by_backend(self):
        """NVIDIA only has common passes; AMD additionally has amd_buffer_ops."""
        nvidia_passes = set(
            self.registry.resolve(adapter_name="cuda_triton").get_analysis_passes()
        )
        amd_passes = set(
            self.registry.resolve(adapter_name="hip_triton").get_analysis_passes()
        )

        self.assertIn("loop_schedules", nvidia_passes)
        self.assertNotIn("amd_buffer_ops", nvidia_passes)

        self.assertIn("loop_schedules", amd_passes)
        self.assertIn("amd_buffer_ops", amd_passes)

    def test_analysis_adapter_required_stages_consistency(self):
        """Each analysis pass's required_stages should exist in the adapter."""
        amd_adapter = self.registry.resolve(adapter_name="hip_triton")
        for pass_name in amd_adapter.get_analysis_passes():
            info = AnalysisRegistry.get_analyzer_info(pass_name)
            self.assertIsNotNone(info, f"Analyzer '{pass_name}' should be registered")
            for stage_name in info.required_stages:
                self.assertIsNotNone(
                    amd_adapter.get_stage_by_name(stage_name),
                    f"Required stage '{stage_name}' should exist in AMD adapter",
                )

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

        # Empty artifacts → analyzer skips due to missing required_stages
        self.assertIsNone(
            amd_adapter.run_analysis_pass("amd_buffer_ops", test_entry, None)
        )

        with self.assertRaises(ValueError):
            amd_adapter.run_analysis_pass("nonexistent_analyzer", test_entry, None)

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
        """Derived artifact env parsing should cover defaults, keywords, lists, compat, and unknown filtering."""
        original_derived_artifacts_env = os.environ.get("TRITONPARSE_DERIVED_ARTIFACTS")
        original_dump_sass_env = os.environ.get("TRITONPARSE_DUMP_SASS")

        try:
            set_runtime_sass_dump_override(None)

            with patch(
                "tritonparse.backend.DerivedArtifactRegistry.list_target_stage_names",
                return_value=["sass", "example"],
            ):
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

                os.environ.pop("TRITONPARSE_DUMP_SASS", None)
                os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "example,unknown"
                with self.assertLogs("tritonparse", level="WARNING") as logs:
                    self.assertEqual(get_enabled_derived_artifacts(), {"example"})
                self.assertIn("unknown target stage names", "\n".join(logs.output))
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


class TestDeviceStringHelpers(unittest.TestCase):
    """Tests for device-string normalization helpers."""

    def test_adapter_returns_canonical_cuda_device(self):
        adapter = NvidiaTritonAdapter()
        self.assertEqual(adapter.get_canonical_device_string(), "cuda:0")

    def test_adapter_returns_canonical_hip_device(self):
        adapter = AmdTritonAdapter()
        self.assertEqual(adapter.get_canonical_device_string(), "cuda:0")

    def test_public_helper_keeps_cpu(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("cpu"), "cpu")

    def test_public_helper_normalizes_indexed_device(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string("hip:3"), "hip:0")

    def test_public_helper_maps_empty_to_cpu(self):
        from tritonparse.backend import normalize_accelerator_device_string

        self.assertEqual(normalize_accelerator_device_string(""), "cpu")


if __name__ == "__main__":
    unittest.main()
