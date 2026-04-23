import unittest

from tritonparse.backend import (
    AmdTritonAdapter,
    deserialize_stage_descriptors_from_event,
    NvidiaTritonAdapter,
    PipelineAdapterRegistry,
)
from tritonparse.parse.trace_processor import _resolve_source_mappable_stage_keys


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
        self.assertIn("ttir", stage_keys)
        self.assertIn("ptx", stage_keys)
        self.assertEqual(list(stage_keys.keys()), [k for k in list(stage_keys.keys())])
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


if __name__ == "__main__":
    unittest.main()
