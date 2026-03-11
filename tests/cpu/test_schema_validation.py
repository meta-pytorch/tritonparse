#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for TritonParse trace JSON schema validation."""

import json
import unittest

from tests.test_utils import get_raw_trace_file, get_test_ndjson_file
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
        self.assertEqual(len(event_types), 4)

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

    def test_get_schema_unknown_returns_none(self):
        schema = get_schema("nonexistent_event_type")
        self.assertIsNone(schema)

    def test_get_all_schemas(self):
        schemas = get_all_schemas()
        self.assertEqual(len(schemas), 4)
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

    def test_launch_record_function_wrong_type(self):
        """function field rejects non-integer/non-null types like string."""
        record = {
            "event_type": "launch",
            "pid": 1,
            "timestamp": "t",
            "name": "k",
            "grid": [1],
            "stream": 0,
            "function": "not_an_int",
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


class ValidateTraceFileTest(unittest.TestCase):
    """Integration tests: validate actual TritonParse trace files against schemas.

    These tests run the validator against real trace files from the example_output
    directory to ensure schemas match the actual format produced by tritonparse.
    Files are located via test_utils helpers (which resolve paths correctly in
    both Buck and local environments).
    """

    def test_validate_raw_trace_dedicated_log(self):
        """Validate raw trace: dedicated_log_triton_trace_findhao_.ndjson.

        This is a raw NDJSON trace file with 2 compilation events captured
        from a live Triton kernel run. Validates that every record conforms
        to the compilation schema.
        """
        raw_file = get_raw_trace_file()
        result = validate_trace_file(str(raw_file))
        self.assertTrue(result["valid"], f"Validation errors: {result['errors']}")
        self.assertGreater(result["record_count"], 0)
        self.assertIn("compilation", result["event_type_counts"])

    def test_validate_parsed_trace_complex_mapped(self):
        """Validate parsed trace: dedicated_log_triton_trace_findhao__mapped.ndjson.gz.

        This is a processed NDJSON trace (gzip-compressed) containing 5 compilation
        events, 1557 launch events, and 5 launch_diff events. Validates that every
        record across all event types conforms to the appropriate schema.
        """
        gz_file = get_test_ndjson_file()
        result = validate_trace_file(str(gz_file))
        self.assertTrue(result["valid"], f"Validation errors: {result['errors']}")
        self.assertGreater(result["record_count"], 0)
        self.assertIn("compilation", result["event_type_counts"])
        self.assertIn("launch", result["event_type_counts"])
        self.assertIn("launch_diff", result["event_type_counts"])

    def test_parsed_trace_event_counts(self):
        """Verify expected event counts in the complex parsed trace."""
        gz_file = get_test_ndjson_file()
        result = validate_trace_file(str(gz_file))
        counts = result["event_type_counts"]
        # The complex trace has 5 compilations, many launches, and 5 launch_diffs
        self.assertEqual(counts.get("compilation", 0), 5)
        self.assertEqual(counts.get("launch_diff", 0), 5)
        self.assertGreater(counts.get("launch", 0), 100)

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

        Iterates all 1567 records (5 compilation + 1557 launch + 5 launch_diff)
        and reports the exact failing line/event_type on error.
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
