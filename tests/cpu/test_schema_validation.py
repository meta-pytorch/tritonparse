#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for TritonParse trace JSON schema validation."""

import json
import unittest

from tests.test_utils import (
    get_inductor_ndjson_file,
    get_raw_trace_file,
    get_test_ndjson_file,
)
from tritonparse.tools.compression import open_compressed_file
from tritonparse.validation.json_validator import validate_record, validate_trace_file
from tritonparse.validation.schema_loader import (
    get_all_schemas,
    get_schema,
    get_supported_event_types,
)


class SchemaLoaderTest(unittest.TestCase):
    """Tests for schema loading functionality."""

    def test_get_supported_event_types(self):
        event_types = get_supported_event_types()
        self.assertIn("compilation", event_types)
        self.assertIn("launch", event_types)
        self.assertIn("launch_diff", event_types)
        self.assertIn("ir_analysis", event_types)
        self.assertIn("roofline", event_types)
        self.assertIn("autotune", event_types)
        self.assertIn("autotune_analysis", event_types)
        self.assertIn("autotune_summary", event_types)
        self.assertEqual(len(event_types), 8)

    def test_get_schema_compilation(self):
        schema = get_schema("compilation")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["$schema"], "http://json-schema.org/draft-07/schema#")
        self.assertEqual(schema["properties"]["event_type"]["enum"], ["compilation"])
        self.assertIn("payload", schema["properties"])

    def test_get_schema_launch(self):
        schema = get_schema("launch")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["properties"]["event_type"]["enum"], ["launch"])
        self.assertIn("grid", schema["properties"])
        self.assertIn("extracted_args", schema["properties"])

    def test_get_schema_launch_diff(self):
        schema = get_schema("launch_diff")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["properties"]["event_type"]["enum"], ["launch_diff"])
        self.assertIn("total_launches", schema["properties"])
        self.assertIn("diffs", schema["properties"])
        # Verify DiffEntry is wired up via additionalProperties $ref
        diffs_schema = schema["properties"]["diffs"]
        self.assertIn("additionalProperties", diffs_schema)
        self.assertIn("$ref", diffs_schema["additionalProperties"])

    def test_get_schema_ir_analysis(self):
        schema = get_schema("ir_analysis")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["properties"]["event_type"]["enum"], ["ir_analysis"])
        self.assertIn("ir_analysis", schema["properties"])
        # Verify inner structure definitions exist
        self.assertIn("definitions", schema)
        self.assertIn("BlockPingpongResult", schema["definitions"])
        self.assertIn("IOCounts", schema["definitions"])
        self.assertIn("LoopSchedule", schema["definitions"])

    def test_get_schema_autotune(self):
        schema = get_schema("autotune")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["properties"]["event_type"]["enum"], ["autotune"])
        self.assertIn("configs_timings", schema["properties"])
        self.assertIn("best_config", schema["properties"])

    def test_get_schema_autotune_analysis(self):
        schema = get_schema("autotune_analysis")
        self.assertIsNotNone(schema)
        self.assertEqual(
            schema["properties"]["event_type"]["enum"], ["autotune_analysis"]
        )
        self.assertIn("session_id", schema["properties"])
        self.assertIn("winner_compilation_hash", schema["properties"])
        # Verify inner structure definitions exist
        self.assertIn("definitions", schema)
        for name in (
            "CompilationAnalysis",
            "LaunchAnalysis",
            "AutotuneArgsSummary",
            "AutotuneResult",
        ):
            self.assertIn(name, schema["definitions"])

    def test_get_schema_autotune_summary(self):
        schema = get_schema("autotune_summary")
        self.assertIsNotNone(schema)
        self.assertEqual(
            schema["properties"]["event_type"]["enum"], ["autotune_summary"]
        )
        # Counts are wired up as a typed additionalProperties map
        counts_schema = schema["properties"]["winner_run_counts"]
        self.assertEqual(counts_schema["additionalProperties"]["type"], "integer")

    def test_get_schema_unknown_returns_none(self):
        schema = get_schema("nonexistent_event_type")
        self.assertIsNone(schema)

    def test_get_all_schemas(self):
        schemas = get_all_schemas()
        self.assertEqual(len(schemas), 8)
        for event_type in get_supported_event_types():
            self.assertIn(event_type, schemas)

    def test_schemas_are_valid_json_schema(self):
        """Verify each schema has the required JSON Schema structure."""
        for event_type in get_supported_event_types():
            schema = get_schema(event_type)
            with self.subTest(event_type=event_type):
                self.assertEqual(
                    schema["$schema"], "http://json-schema.org/draft-07/schema#"
                )
                self.assertEqual(schema["type"], "object")
                self.assertIn("required", schema)
                self.assertIn("properties", schema)
                self.assertIn("event_type", schema["properties"])
                # Event type discriminator should be a single-value enum
                self.assertEqual(
                    schema["properties"]["event_type"]["enum"], [event_type]
                )


class ValidateRecordTest(unittest.TestCase):
    """Tests for individual record validation."""

    def test_valid_compilation_record(self):
        record = {
            "event_type": "compilation",
            "pid": 12345,
            "timestamp": "2025-01-01T00:00:00Z",
            "stack": [
                {"line": 10, "name": "main", "filename": "test.py", "loc": "foo()"}
            ],
            "payload": {
                "metadata": {"hash": "abc123", "name": "my_kernel"},
                "file_content": {"my_kernel.ttir": "module ..."},
                "file_path": {"my_kernel.ttir": "/tmp/my_kernel.ttir"},
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_valid_launch_record(self):
        record = {
            "event_type": "launch",
            "pid": 12345,
            "timestamp": "2025-01-01T00:00:00Z",
            "name": "my_kernel",
            "grid": [1, 1, 1],
            "stream": 0,
            "function": 148350704,
            "stack": [
                {"line": 20, "name": "run", "filename": "test.py", "loc": "kernel()"}
            ],
            "compilation_metadata": {"hash": "abc123", "name": "my_kernel"},
            "extracted_args": {
                "x": {"type": "tensor", "shape": [1024], "dtype": "torch.float32"}
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_valid_launch_record_function_null(self):
        """function field accepts null as well as integer."""
        record = {
            "event_type": "launch",
            "pid": 1,
            "timestamp": "t",
            "name": "k",
            "grid": [1],
            "stream": 0,
            "function": None,
            "stack": [],
            "compilation_metadata": {"hash": "a", "name": "k"},
            "extracted_args": {},
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_valid_launch_record_function_summary_string(self):
        """function field accepts the summary string emitted for huge handles."""
        record = {
            "event_type": "launch",
            "pid": 1,
            "timestamp": "t",
            "name": "k",
            "grid": [1],
            "stream": 0,
            "function": "<bytes: 4168024 bytes omitted>",
            "stack": [],
            "compilation_metadata": {"hash": "a", "name": "k"},
            "extracted_args": {},
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_launch_record_function_wrong_type(self):
        """function field rejects composite types like list and object."""
        for bad_value in ([1, 2, 3], {"handle": 1}):
            with self.subTest(function=bad_value):
                record = {
                    "event_type": "launch",
                    "pid": 1,
                    "timestamp": "t",
                    "name": "k",
                    "grid": [1],
                    "stream": 0,
                    "function": bad_value,
                    "stack": [],
                    "compilation_metadata": {"hash": "a", "name": "k"},
                    "extracted_args": {},
                }
                is_valid, errors = validate_record(record)
                self.assertFalse(is_valid)
                self.assertTrue(any("function" in e and "type" in e for e in errors))

    def test_valid_launch_diff_record(self):
        record = {
            "event_type": "launch_diff",
            "hash": "abc123",
            "name": "my_kernel",
            "total_launches": 10,
            "launch_index_map": [{"start": 0, "end": 10}],
            "sames": {"stream": 0},
            "diffs": {
                "grid": {
                    "diff_type": "distribution",
                    "values": [
                        {
                            "value": [1],
                            "count": 5,
                            "launches": [{"start": 0, "end": 5}],
                        }
                    ],
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_valid_ir_analysis_record(self):
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "blockpingpong": {
                    "category": "none",
                    "detected": False,
                    "num_warps": None,
                    "num_pp_clusters": None,
                    "cond_barrier_count": 0,
                    "setprio_count": 0,
                    "dot_count": 0,
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_missing_event_type(self):
        record = {"pid": 12345}
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertIn("missing 'event_type' field", errors[0])

    def test_wrong_event_type_value(self):
        record = {
            "event_type": "wrong_type",
            "pid": 12345,
            "timestamp": "2025-01-01T00:00:00Z",
            "payload": {"metadata": {"hash": "abc", "name": "k"}},
        }
        # Unknown event types should pass (no schema to validate against)
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid)

    def test_compilation_missing_required_fields(self):
        record = {"event_type": "compilation"}
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        error_text = " ".join(errors)
        self.assertIn("pid", error_text)
        self.assertIn("timestamp", error_text)
        self.assertIn("payload", error_text)

    def test_launch_missing_required_fields(self):
        record = {"event_type": "launch", "pid": 1, "timestamp": "t"}
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        error_text = " ".join(errors)
        self.assertIn("name", error_text)
        self.assertIn("grid", error_text)

    def test_compilation_wrong_pid_type(self):
        record = {
            "event_type": "compilation",
            "pid": "not_an_int",
            "timestamp": "2025-01-01T00:00:00Z",
            "payload": {"metadata": {"hash": "abc", "name": "k"}},
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("pid" in e and "type" in e for e in errors))

    def test_stack_frame_validation(self):
        record = {
            "event_type": "compilation",
            "pid": 1,
            "timestamp": "t",
            "stack": [{"line": 1, "name": "f", "filename": "x.py", "bad_field": 1}],
            "payload": {"metadata": {"hash": "a", "name": "k"}},
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("bad_field" in e for e in errors))

    def test_compilation_additional_top_level_fields_allowed(self):
        """Top-level additionalProperties is true to support processing fields."""
        record = {
            "event_type": "compilation",
            "pid": 1,
            "timestamp": "t",
            "payload": {"metadata": {"hash": "a", "name": "k"}},
            "some_future_field": "value",
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_metadata_additional_fields_allowed(self):
        """Metadata additionalProperties is true since fields are defined by Triton."""
        record = {
            "event_type": "compilation",
            "pid": 1,
            "timestamp": "t",
            "payload": {
                "metadata": {
                    "hash": "a",
                    "name": "k",
                    "some_triton_field": 42,
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_fake_compilation_record(self):
        record = {
            "event_type": "compilation",
            "pid": 1,
            "timestamp": "t",
            "payload": {"metadata": {"hash": "a", "name": "k"}},
            "is_fake": True,
            "fake_reason": "No compilation event found; inferred from launch event",
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_launch_diff_distribution_values(self):
        record = {
            "event_type": "launch_diff",
            "hash": "abc",
            "name": "k",
            "total_launches": 2,
            "diffs": {
                "grid": {
                    "diff_type": "distribution",
                    "values": [
                        {
                            "value": [1],
                            "count": 1,
                            "launches": [{"start": 0, "end": 1}],
                        },
                        {
                            "value": [4],
                            "count": 1,
                            "launches": [{"start": 1, "end": 2}],
                        },
                    ],
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_unresolved_ref_returns_error(self):
        """An unresolved $ref should produce a validation error, not silently pass."""
        from tritonparse.validation.json_validator import _validate_record

        schema = {
            "type": "object",
            "properties": {
                "field": {"$ref": "#/definitions/DoesNotExist"},
            },
        }
        record = {"field": "any_value"}
        errors = _validate_record(record, schema)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("unresolved" in e for e in errors))

    def test_numeric_maximum_constraint(self):
        """maximum constraint rejects values above the limit."""
        from tritonparse.validation.json_validator import _validate_record

        schema = {"type": "integer", "minimum": 0, "maximum": 100}
        self.assertEqual(_validate_record(50, schema), [])
        self.assertEqual(_validate_record(100, schema), [])
        errors = _validate_record(101, schema)
        self.assertTrue(any("maximum" in e for e in errors))

    def test_numeric_exclusive_minimum_constraint(self):
        """exclusiveMinimum rejects values equal to or below the limit."""
        from tritonparse.validation.json_validator import _validate_record

        schema = {"type": "integer", "exclusiveMinimum": 0}
        self.assertEqual(_validate_record(1, schema), [])
        errors = _validate_record(0, schema)
        self.assertTrue(any("exclusiveMinimum" in e for e in errors))

    def test_numeric_exclusive_maximum_constraint(self):
        """exclusiveMaximum rejects values equal to or above the limit."""
        from tritonparse.validation.json_validator import _validate_record

        schema = {"type": "integer", "exclusiveMaximum": 10}
        self.assertEqual(_validate_record(9, schema), [])
        errors = _validate_record(10, schema)
        self.assertTrue(any("exclusiveMaximum" in e for e in errors))

    def test_launch_diff_summary_diff_type(self):
        """DiffEntry with diff_type 'summary' validates correctly."""
        record = {
            "event_type": "launch_diff",
            "hash": "abc",
            "name": "k",
            "total_launches": 10,
            "diffs": {
                "function": {
                    "diff_type": "summary",
                    "summary_text": "Varies across 2 unique values",
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_launch_diff_argument_diff_type(self):
        """extracted_args in diffs maps arg names to DiffEntry with argument_diff."""
        record = {
            "event_type": "launch_diff",
            "hash": "abc",
            "name": "k",
            "total_launches": 441,
            "diffs": {
                "extracted_args": {
                    "b_ptr": {
                        "diff_type": "argument_diff",
                        "sames": {"type": "tensor", "dtype": "torch.float16"},
                        "diffs": {
                            "shape": {
                                "diff_type": "distribution",
                                "values": [
                                    {
                                        "value": [16, 16],
                                        "count": 66,
                                        "launches": [{"start": 0, "end": 65}],
                                    }
                                ],
                            }
                        },
                    }
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_launch_diff_missing_diff_type(self):
        """DiffEntry missing required diff_type field should fail."""
        record = {
            "event_type": "launch_diff",
            "hash": "abc",
            "name": "k",
            "total_launches": 2,
            "diffs": {
                "grid": {
                    "values": [
                        {
                            "value": [1],
                            "count": 1,
                            "launches": [{"start": 0, "end": 0}],
                        }
                    ]
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("diff_type" in e for e in errors))

    def test_ir_analysis_blockpingpong_detected(self):
        """BlockPingpong with detected=True and full fields."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "blockpingpong": {
                    "category": "pingpong_medium",
                    "detected": True,
                    "num_warps": 8,
                    "num_pp_clusters": 2,
                    "cond_barrier_count": 3,
                    "setprio_count": 4,
                    "dot_count": 2,
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_ir_analysis_blockpingpong_bad_category(self):
        """BlockPingpong with invalid category should fail."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "blockpingpong": {
                    "category": "invalid_category",
                    "detected": False,
                    "num_warps": None,
                    "num_pp_clusters": None,
                    "cond_barrier_count": 0,
                    "setprio_count": 0,
                    "dot_count": 0,
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("enum" in e for e in errors))

    def test_ir_analysis_blockpingpong_unexpected_field(self):
        """BlockPingpong with unexpected field should fail (additionalProperties false)."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "blockpingpong": {
                    "category": "none",
                    "detected": False,
                    "num_warps": None,
                    "num_pp_clusters": None,
                    "cond_barrier_count": 0,
                    "setprio_count": 0,
                    "dot_count": 0,
                    "extra_field": True,
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("extra_field" in e for e in errors))

    def test_ir_analysis_loop_schedules(self):
        """Loop schedules with prologue/loop_body/epilogue arrays."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "loop_schedules": [
                    {
                        "prologue": ["x = tl.load(ptr)"],
                        "loop_body": ["acc += tl.dot(a, b)"],
                        "epilogue": ["tl.store(out_ptr, acc)"],
                    }
                ]
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_ir_analysis_io_counts(self):
        """IO counts for AMD buffer operations."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {
                "io_counts": {
                    "amd_ttgir_bufferops_count": {
                        "tt.load_count": 4,
                        "tt.store_count": 1,
                    },
                    "amd_gcn_bufferops_count": {
                        "buffer_load_count": 8,
                        "buffer_store_count": 2,
                    },
                }
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_ir_analysis_unknown_analysis_type_allowed(self):
        """Unknown analysis types pass (additionalProperties true on ir_analysis)."""
        record = {
            "event_type": "ir_analysis",
            "hash": "abc123",
            "ir_analysis": {"future_analysis": {"some_data": True}},
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_compilation_pt_info_attempt(self):
        """pt_info carries 'attempt' (not 'attempt_id'); see structured_logging."""
        record = {
            "event_type": "compilation",
            "pid": 12345,
            "timestamp": "2025-01-01T00:00:00Z",
            "payload": {
                "metadata": {"hash": "abc123", "name": "triton_poi_fused_add_0"},
                "pt_info": {"frame_id": 0, "frame_compile_id": 1, "attempt": 0},
            },
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_compilation_pt_info_attempt_wrong_type(self):
        record = {
            "event_type": "compilation",
            "pid": 12345,
            "timestamp": "2025-01-01T00:00:00Z",
            "payload": {
                "metadata": {"hash": "abc123"},
                "pt_info": {"frame_id": 0, "attempt": "zero"},
            },
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("attempt" in e for e in errors), errors)


def _autotune_analysis_record(**overrides):
    """A minimal but complete autotune_analysis record, as the parser emits it."""
    record = {
        "event_type": "autotune_analysis",
        "session_id": "6852348bc10a0d9e",
        "session_stack": [
            {"line": 222, "name": "matmul", "filename": "bench.py", "loc": "mm(a, b)"}
        ],
        "name": "matmul_kernel",
        "selected_hash": "9d4505033ff1cc25",
        "winner_compilation_hash": "c024454524b8d34e",
        "possible_groups": [["c024454524b8d34e", "479daaee1705cbd5"]],
        "compilation_analysis": {
            "configs": [
                {
                    "compilation_config_params": {"num_warps": 1, "num_stages": 1},
                    "compilation_hash": "c024454524b8d34e",
                }
            ],
            "compilation_hashes": ["c024454524b8d34e"],
            "common_info": {
                "stack": [{"line": 10, "name": "main", "filename": "bench.py"}],
                "python_source": {"code": "@triton.jit\ndef matmul_kernel(): ..."},
            },
        },
        "launch_analysis": {
            "launch_group_hashes": ["9d4505033ff1cc25"],
            "launch_params_diff": {"sames": {"grid": [1]}, "diffs": {}},
        },
        "cache_usage": False,
        "launch_ranges": {"benchmark": "1-689", "winner": "690"},
        "launch_occurrence_ids": {"benchmark": [1, 2], "winner": [690]},
        "occurrence_id": 2540,
    }
    record.update(overrides)
    return record


def _autotune_record(**overrides):
    """A raw autotune record, as the AutotuneListener callback writes it."""
    record = {
        "event_type": "autotune",
        "pid": 732176,
        "kernel_name": "matmul_kernel",
        "cache_key": "ad39063f45fa6c849f13ea8b14e942f3",
        "best_config": "BLOCK_SIZE_M: 16, num_warps: 1, num_stages: 1",
        "configs_timings": {
            "BLOCK_SIZE_M: 16, num_warps: 1, num_stages: 1": [0.29, 0.27, 0.34],
            "BLOCK_SIZE_M: 32, num_warps: 1, num_stages: 1": [0.36, 0.31, 0.45],
        },
        "duration": 2.62,
        "cache_hit": False,
        "autotune_key": "(16, 16, 16, 'torch.float16')",
        "stack": [{"line": 10, "name": "matmul", "filename": "bench.py"}],
        "timestamp": "2026-01-01T00:00:00Z",
    }
    record.update(overrides)
    return record


class AutotuneSchemaTest(unittest.TestCase):
    """Tests for the autotune / autotune_analysis / autotune_summary schemas."""

    def test_valid_autotune_record(self):
        is_valid, errors = validate_record(_autotune_record())
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_cache_hit_record(self):
        """A cache hit reports no timings and near-zero duration."""
        record = _autotune_record(cache_hit=True, configs_timings={}, duration=0.0)
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_missing_required_field(self):
        record = _autotune_record()
        del record["best_config"]
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("best_config" in e for e in errors), errors)

    def test_autotune_best_config_must_be_stringified(self):
        """The writer stringifies the Config; a raw object is a writer bug."""
        record = _autotune_record(best_config={"BLOCK_SIZE_M": 16})
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("best_config" in e for e in errors), errors)

    def test_autotune_negative_duration_rejected(self):
        record = _autotune_record(duration=-1.0)
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("minimum" in e for e in errors), errors)

    def test_autotune_boolean_duration_rejected(self):
        record = _autotune_record(duration=True)
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("duration" in e and "type" in e for e in errors), errors)

    def test_valid_autotune_analysis_record(self):
        is_valid, errors = validate_record(_autotune_analysis_record())
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_analysis_cached_session(self):
        """A cached session leaves the derived fields null but still validates."""
        record = _autotune_analysis_record(
            name=None,
            compilation_analysis=None,
            launch_analysis=None,
            cache_usage=True,
            possible_groups=[],
            launch_ranges={"benchmark": "", "winner": "1"},
            launch_occurrence_ids={"benchmark": [], "winner": [1]},
        )
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_analysis_missing_required_field(self):
        record = _autotune_analysis_record()
        del record["cache_usage"]
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("cache_usage" in e for e in errors), errors)

    def test_autotune_analysis_nullable_object_still_validated(self):
        """A non-null nullable object must still have its structure checked.

        compilation_analysis is typed ["object", "null"]. A validator that keys
        structural checks off `type == "object"` skips the whole subtree for
        such fields, so this asserts the nested required-field check fires.
        """
        record = _autotune_analysis_record()
        del record["compilation_analysis"]["common_info"]
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(
            any("compilation_analysis.common_info" in e for e in errors), errors
        )

    def test_autotune_analysis_nested_unexpected_field(self):
        record = _autotune_analysis_record()
        record["compilation_analysis"]["bogus"] = 1
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("compilation_analysis.bogus" in e for e in errors), errors)

    def test_autotune_analysis_launch_occurrence_ids_item_type(self):
        record = _autotune_analysis_record()
        record["launch_occurrence_ids"]["benchmark"] = ["not-an-int"]
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(
            any("launch_occurrence_ids.benchmark[0]" in e for e in errors), errors
        )

    def test_autotune_analysis_with_result_and_args_summary(self):
        record = _autotune_analysis_record(
            autotune_result={
                "best_config": "BLOCK_SIZE_M: 16, num_warps: 1",
                "configs_timings": {"BLOCK_SIZE_M: 16, num_warps: 1": [0.1, 0.2, 0.3]},
                "benchmark_duration": 0.42,
                "cache_hit": False,
            },
            autotune_args_summary={
                "summary_version": 1,
                "unchanged_args": {"M": {"type": "int", "value": 16}},
                "per_config_args": {"c024454524b8d34e": {"M": {"values": []}}},
                "arg_order": ["a", "b", "M"],
                "autotune_configs": {"sames": {"num_warps": 1}, "varies": {}},
            },
        )
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_analysis_result_all_null(self):
        """Every autotune_result member is a .get() off the raw event, so all
        of them can be null when the autotuner event omitted the key."""
        record = _autotune_analysis_record(
            autotune_result={
                "best_config": None,
                "configs_timings": None,
                "benchmark_duration": None,
                "cache_hit": None,
            }
        )
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_analysis_args_summary_version_below_minimum(self):
        record = _autotune_analysis_record(
            autotune_args_summary={
                "summary_version": 0,
                "unchanged_args": {},
                "per_config_args": {},
                "arg_order": [],
                "autotune_configs": {"sames": {}, "varies": {}},
            }
        )
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("summary_version" in e for e in errors), errors)

    def test_valid_autotune_summary_record(self):
        record = {
            "event_type": "autotune_summary",
            "winner_run_counts": {"c024454524b8d34e": 1, "479daaee1705cbd5": 2},
            "occurrence_id": 2545,
        }
        is_valid, errors = validate_record(record)
        self.assertTrue(is_valid, f"Unexpected errors: {errors}")

    def test_autotune_summary_missing_counts(self):
        record = {"event_type": "autotune_summary"}
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("winner_run_counts" in e for e in errors), errors)

    def test_autotune_summary_count_wrong_type(self):
        record = {
            "event_type": "autotune_summary",
            "winner_run_counts": {"c024454524b8d34e": "three"},
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(
            any("winner_run_counts.c024454524b8d34e" in e for e in errors), errors
        )

    def test_autotune_summary_count_below_minimum(self):
        record = {
            "event_type": "autotune_summary",
            "winner_run_counts": {"c024454524b8d34e": 0},
        }
        is_valid, errors = validate_record(record)
        self.assertFalse(is_valid)
        self.assertTrue(any("minimum" in e for e in errors), errors)


class ValidateTraceFileTest(unittest.TestCase):
    """Integration tests: validate actual TritonParse trace files against schemas.

    These tests run the validator against real trace files from the example_output
    directory to ensure schemas match the actual format produced by tritonparse.
    Files are located via test_utils helpers (which resolve paths correctly in
    both Buck and local environments).
    """

    def test_validate_raw_trace_dedicated_log(self):
        """Validate raw trace: dedicated_log_triton_trace_findhao_.ndjson.

        A raw NDJSON trace straight off the writer: 4 compilation, 20 launch
        and 2 autotune records (the launch tail is capped by the generator).
        Validates that every record conforms to its schema.
        """
        raw_file = get_raw_trace_file()
        result = validate_trace_file(str(raw_file))
        self.assertTrue(result["valid"], f"Validation errors: {result['errors']}")
        self.assertGreater(result["record_count"], 0)
        self.assertIn("compilation", result["event_type_counts"])
        self.assertIn("launch", result["event_type_counts"])

    def test_inductor_trace_has_full_launches(self):
        """The inductor fixture must contain launches with metadata.

        Full launches (carrying ``compilation_metadata``) prove the
        inductor JIT post-compile hook simulation ran, i.e. init() reached
        torch's gate. Regression test for inductor traces that contained
        compilations but zero launches.
        """
        full_inductor_launches = 0
        with open_compressed_file(get_inductor_ndjson_file()) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("event_type") != "launch":
                    continue
                if isinstance(record.get("compilation_metadata"), dict) and str(
                    record.get("name", "")
                ).startswith("triton_"):
                    full_inductor_launches += 1
        self.assertGreater(
            full_inductor_launches,
            0,
            "No full inductor launches in the inductor fixture",
        )

    def test_validate_parsed_trace_complex_mapped(self):
        """Validate parsed trace: dedicated_log_triton_trace_findhao__mapped.ndjson.gz.

        A processed, gzip-compressed trace covering every event type the parser
        emits. Validates that each record conforms to its schema.
        """
        gz_file = get_test_ndjson_file()
        result = validate_trace_file(str(gz_file))
        self.assertTrue(result["valid"], f"Validation errors: {result['errors']}")
        self.assertGreater(result["record_count"], 0)
        self.assertIn("compilation", result["event_type_counts"])
        self.assertIn("launch", result["event_type_counts"])
        self.assertIn("launch_diff", result["event_type_counts"])

    def test_parsed_trace_event_counts(self):
        """Verify expected event counts in the complex parsed trace.

        Two autotuned matmul configs plus two fused_op ACTIVATION
        specializations give 4 compilations, and one derived event of each
        per-kernel kind alongside them.
        """
        gz_file = get_test_ndjson_file()
        result = validate_trace_file(str(gz_file))
        counts = result["event_type_counts"]
        self.assertEqual(counts.get("compilation", 0), 4)
        self.assertEqual(counts.get("launch_diff", 0), 4)
        self.assertGreater(counts.get("launch", 0), 100)

    def test_parsed_trace_covers_the_derived_event_types(self):
        """The fixture must keep exercising every event type the parser emits.

        roofline, ir_analysis and the autotune events postdate the original
        fixture, so for a long time their schemas were never checked against
        real data. Asserting their presence here keeps a future regeneration
        from quietly dropping back to a compilation/launch-only trace.
        """
        counts = validate_trace_file(str(get_test_ndjson_file()))["event_type_counts"]
        for event_type in (
            "ir_analysis",
            "roofline",
            "autotune_analysis",
            "autotune_summary",
        ):
            with self.subTest(event_type=event_type):
                self.assertGreater(counts.get(event_type, 0), 0)

    def test_validate_parsed_trace_inductor(self):
        """Validate the inductor trace: dedicated_log_triton_trace_inductor__mapped.

        The hand-written example has no `pt_info`, so before this fixture
        existed the PtInfo half of the compilation schema was never checked
        against real data.
        """
        result = validate_trace_file(str(get_inductor_ndjson_file()))
        self.assertTrue(result["valid"], f"Validation errors: {result['errors']}")
        self.assertGreater(result["event_type_counts"].get("compilation", 0), 0)

    def test_inductor_trace_carries_pt_info(self):
        with open_compressed_file(get_inductor_ndjson_file()) as f:
            compilations = [
                record
                for record in (json.loads(line) for line in f if line.strip())
                if record.get("event_type") == "compilation"
            ]
        self.assertGreater(len(compilations), 0)
        for comp in compilations:
            name = comp["payload"]["metadata"]["name"]
            with self.subTest(kernel=name):
                pt_info = comp["payload"].get("pt_info")
                if pt_info is None:
                    self.fail(f"{name} has no pt_info")
                # 'attempt', not 'attempt_id' -- see the PtInfo schema.
                self.assertIn("attempt", pt_info)
                self.assertIsInstance(pt_info.get("frame_id"), int)

    def test_every_record_in_inductor_trace(self):
        with open_compressed_file(get_inductor_ndjson_file()) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                is_valid, errors = validate_record(record)
                self.assertTrue(
                    is_valid,
                    f"Record at line {line_num} "
                    f"(event_type={record.get('event_type')}) failed: {errors}",
                )

    def test_raw_trace_covers_the_writer_event_types(self):
        """Same guard for the raw fixture, which the writer produces directly."""
        counts = validate_trace_file(str(get_raw_trace_file()))["event_type_counts"]
        for event_type in ("compilation", "launch", "autotune"):
            with self.subTest(event_type=event_type):
                self.assertGreater(counts.get(event_type, 0), 0)

    def test_validate_nonexistent_file(self):
        result = validate_trace_file("/nonexistent/path/trace.ndjson")
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["errors"]), 0)

    def test_every_record_in_raw_trace(self):
        """Validate each record individually in dedicated_log_triton_trace_findhao_.ndjson.

        Iterates every line and reports the exact failing line/event_type on error.
        """
        raw_file = get_raw_trace_file()
        with open_compressed_file(raw_file) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                is_valid, errors = validate_record(record)
                self.assertTrue(
                    is_valid,
                    f"Record at line {line_num} (event_type={record.get('event_type')}) "
                    f"failed validation: {errors}",
                )

    def test_every_record_in_parsed_trace(self):
        """Validate each record in dedicated_log_triton_trace_findhao__mapped.ndjson.gz.

        Iterates every record and reports the exact failing line/event_type on
        error, rather than only the first failure the file-level check finds.
        """
        gz_file = get_test_ndjson_file()
        with open_compressed_file(gz_file) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                is_valid, errors = validate_record(record)
                self.assertTrue(
                    is_valid,
                    f"Record at line {line_num} (event_type={record.get('event_type')}) "
                    f"failed validation: {errors}",
                )
