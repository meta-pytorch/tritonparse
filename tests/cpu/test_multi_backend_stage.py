import unittest

from tritonparse.backend import (
    AmdTritonAdapter,
    deserialize_stage_descriptors_from_event,
    NvidiaTritonAdapter,
    PipelineAdapterRegistry,
)
from tritonparse.parse.ir_parser import ParserRegistry
from tritonparse.parse.trace_processor import (
    _resolve_source_mappable_stage_keys,
    generate_source_mappings,
)


def make_event(file_content: dict, stage_descriptors: list | None = None):
    payload = {"file_content": file_content}
    if stage_descriptors is not None:
        payload["metadata"] = {"stage_descriptors": stage_descriptors}
    else:
        payload["metadata"] = {}
    return {"payload": payload}


class TestMultiBackendStage(unittest.TestCase):
    def test_metadata_resolution_and_ordering(self):
        # Two stages provided out-of-order; display_order should control final order
        stages = [
            {
                "name": "ptx",
                "extension": ".ptx",
                "display_name": "PTX",
                "display_order": 20,
                "is_text": True,
                "supports_source_mapping": True,
                "parser_id": "p",
                "syntax_id": "s",
            },
            {
                "name": "ttir",
                "extension": ".ttir",
                "display_name": "TTIR",
                "display_order": 10,
                "is_text": True,
                "supports_source_mapping": True,
                "parser_id": "p",
                "syntax_id": "s",
            },
        ]

        file_content = {"kernel.ptx": "...", "kernel.ttir": "..."}
        event = make_event(file_content, stages)

        stage_keys = _resolve_source_mappable_stage_keys(event)

        # Expect keys ordered by display_order (ttir then ptx)
        self.assertEqual(list(stage_keys.keys()), ["ttir", "ptx"])
        self.assertEqual(stage_keys["ttir"], "kernel.ttir")
        self.assertEqual(stage_keys["ptx"], "kernel.ptx")

    def test_legacy_fallback_resolution_order(self):
        # No metadata provided -> use hardcoded fallback order
        file_content = {"a.ptx": "...", "b.ttir": "..."}
        event = make_event(file_content, stage_descriptors=None)

        stage_keys = _resolve_source_mappable_stage_keys(event)

        # Fallback dict orders ttir before ptx, so keys should reflect that order
        self.assertEqual(list(stage_keys.keys()), ["ttir", "ptx"])
        # Check values mapped correctly
        self.assertEqual(stage_keys["ttir"], "b.ttir")
        self.assertEqual(stage_keys["ptx"], "a.ptx")

    def test_no_artifacts_returns_empty(self):
        event = make_event({}, stage_descriptors=None)
        stage_keys = _resolve_source_mappable_stage_keys(event)
        self.assertEqual(stage_keys, {})

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

    def test_deserialize_and_ordering(self):
        event = {
            "payload": {
                "metadata": {
                    "stage_descriptors": [
                        {
                            "name": "b_stage",
                            "extension": ".bst",
                            "display_name": "B",
                            "display_order": 20,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "p",
                            "syntax_id": "s",
                        },
                        {
                            "name": "a_stage",
                            "extension": ".ast",
                            "display_name": "A",
                            "display_order": 10,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "p",
                            "syntax_id": "s",
                        },
                    ]
                }
            }
        }

        stages = deserialize_stage_descriptors_from_event(event)
        self.assertIsInstance(stages, list)
        self.assertEqual(len(stages), 2)
        # Should be ordered by display_order (10 then 20)
        self.assertEqual(stages[0].name, "a_stage")
        self.assertEqual(stages[1].name, "b_stage")

    def test_missing_required_field_raises(self):
        # missing 'extension' should raise ValueError
        event = {
            "payload": {
                "metadata": {
                    "stage_descriptors": [
                        {
                            "name": "bad_stage",
                            # "extension": ".bad",  # omitted
                            "display_name": "Bad",
                            "display_order": 1,
                            "is_text": True,
                            "supports_source_mapping": True,
                            "parser_id": "p",
                            "syntax_id": "s",
                        }
                    ]
                }
            }
        }

        with self.assertRaises(ValueError):
            deserialize_stage_descriptors_from_event(event)

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

        # Test with metadata (adapter-driven parser selection)
        metadata_cuda = {"backend_name": "cuda"}
        result_with_metadata = generate_source_mappings(
            ttir_content, "ttir", None, metadata_cuda
        )
        self.assertIsInstance(result_with_metadata, dict)
        self.assertIn("4", result_with_metadata)
        self.assertEqual(result_with_metadata["4"]["file"], "test.py")
        self.assertEqual(result_with_metadata["4"]["line"], 20)
        self.assertIn("ttir_line", result_with_metadata["4"])

        # Test without metadata (fallback to hardcoded parser selection)
        result_fallback = generate_source_mappings(ttir_content, "ttir", None, None)
        self.assertIsInstance(result_fallback, dict)
        self.assertIn("4", result_fallback)
        self.assertEqual(result_fallback["4"]["file"], "test.py")
        self.assertEqual(result_fallback["4"]["line"], 20)
        self.assertIn("ttir_line", result_fallback["4"])

        # Verify both paths produce similar results
        self.assertEqual(
            result_with_metadata["4"]["file"],
            result_fallback["4"]["file"],
            "Adapter-driven and fallback should produce same file",
        )
        self.assertEqual(
            result_with_metadata["4"]["line"],
            result_fallback["4"]["line"],
            "Adapter-driven and fallback should produce same line",
        )


if __name__ == "__main__":
    unittest.main()
