#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Validates TritonParse NDJSON trace files against JSON schemas.

Supports both raw trace files (.ndjson) and processed output files
(.ndjson.gz, .ndjson.zst). Validates each record against the schema
for its event_type.

Known limitations of this lightweight validator:
- Only ``#/definitions/<Name>`` style ``$ref`` pointers are supported;
  other JSON Pointer paths (e.g., ``#/properties/...``) are not resolved.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from tritonparse._json_compat import JSONDecodeError, loads
from tritonparse.tools.compression import open_compressed_file

from .schema_loader import get_schema, get_supported_event_types

log = logging.getLogger(__name__)

# Mapping from JSON Schema type names to Python types, created once at import.
_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _fmt_path(path: str, suffix: str = "") -> str:
    """Format a validation error path, avoiding a bare leading dot."""
    if suffix:
        return f"{path}.{suffix}" if path else suffix
    return path or "."


def _resolve_ref(schema: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a ``$ref`` pointer within a schema.

    Only ``#/definitions/<Name>`` pointers are supported.  Other JSON
    Pointer styles are not resolved and will return ``None``.

    Returns the resolved schema dict, or ``None`` if the reference cannot
    be resolved.
    """
    if not ref.startswith("#/definitions/"):
        return None
    name = ref[len("#/definitions/") :]
    resolved = schema.get("definitions", {}).get(name)
    return resolved


def _validate_type(value: Any, type_spec: Any) -> bool:
    """Check if a value matches a JSON Schema type specifier."""
    if isinstance(type_spec, list):
        return any(_validate_type(value, t) for t in type_spec)
    expected = _TYPE_MAP.get(type_spec)
    if expected is None:
        return True
    # In JSON, booleans are not integers
    if type_spec == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _validate_record(
    record: Dict[str, Any],
    schema: Dict[str, Any],
    root_schema: Optional[Dict[str, Any]] = None,
    path: str = "",
) -> List[str]:
    """Validate a single record against a schema, returning a list of errors.

    This is a lightweight validator that checks:
    - required fields are present
    - field types match
    - enum constraints
    - minimum / maximum / exclusiveMinimum / exclusiveMaximum constraints
    - additionalProperties: false
    - array item types
    - $ref resolution (``#/definitions/`` only)

    It does NOT implement the full JSON Schema spec.  See module docstring
    for known limitations.
    """
    if root_schema is None:
        root_schema = schema
    errors = []

    # Handle $ref
    if "$ref" in schema:
        resolved = _resolve_ref(root_schema, schema["$ref"])
        if resolved is not None:
            return _validate_record(record, resolved, root_schema, path)
        ref_target = schema["$ref"]
        log.warning("Unresolved $ref %r at %s", ref_target, _fmt_path(path))
        errors.append(f"{_fmt_path(path)}: unresolved schema $ref '{ref_target}'")
        return errors

    schema_type = schema.get("type")
    if schema_type and not _validate_type(record, schema_type):
        errors.append(
            f"{_fmt_path(path)}: expected type '{schema_type}', "
            f"got {type(record).__name__}"
        )
        return errors

    if schema_type == "object" and isinstance(record, dict):
        # Check required fields
        for field in schema.get("required", []):
            if field not in record:
                errors.append(f"{_fmt_path(path, field)}: required field missing")

        properties = schema.get("properties", {})

        # Check additionalProperties
        if schema.get("additionalProperties") is False:
            for key in record:
                if key not in properties:
                    errors.append(
                        f"{_fmt_path(path, key)}: unexpected field "
                        f"(additionalProperties is false)"
                    )

        # Validate each known property
        for key, prop_schema in properties.items():
            if key in record:
                sub_errors = _validate_record(
                    record[key], prop_schema, root_schema, _fmt_path(path, key)
                )
                errors.extend(sub_errors)

        # Validate additionalProperties schema (when it's an object, not bool)
        add_props = schema.get("additionalProperties")
        if isinstance(add_props, dict):
            for key in record:
                if key not in properties:
                    sub_errors = _validate_record(
                        record[key],
                        add_props,
                        root_schema,
                        _fmt_path(path, key),
                    )
                    errors.extend(sub_errors)

    elif schema_type == "array" and isinstance(record, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(record):
                sub_errors = _validate_record(
                    item, items_schema, root_schema, f"{path}[{i}]"
                )
                errors.extend(sub_errors)

    # Check enum constraint
    if "enum" in schema and record not in schema["enum"]:
        errors.append(
            f"{_fmt_path(path)}: value {record!r} not in enum {schema['enum']}"
        )

    # Check numeric constraints
    if isinstance(record, (int, float)) and not isinstance(record, bool):
        if "minimum" in schema and record < schema["minimum"]:
            errors.append(
                f"{_fmt_path(path)}: value {record} is less than "
                f"minimum {schema['minimum']}"
            )
        if "maximum" in schema and record > schema["maximum"]:
            errors.append(
                f"{_fmt_path(path)}: value {record} is greater than "
                f"maximum {schema['maximum']}"
            )
        if "exclusiveMinimum" in schema and record <= schema["exclusiveMinimum"]:
            errors.append(
                f"{_fmt_path(path)}: value {record} is not greater than "
                f"exclusiveMinimum {schema['exclusiveMinimum']}"
            )
        if "exclusiveMaximum" in schema and record >= schema["exclusiveMaximum"]:
            errors.append(
                f"{_fmt_path(path)}: value {record} is not less than "
                f"exclusiveMaximum {schema['exclusiveMaximum']}"
            )

    return errors


def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a single trace record against the appropriate schema.

    Args:
        record: A parsed JSON object from a trace file.

    Returns:
        A tuple of (is_valid, list_of_errors).
    """
    event_type = record.get("event_type")
    if event_type is None:
        return False, ["missing 'event_type' field"]

    schema = get_schema(event_type)
    if schema is None:
        # Unknown event types are allowed (future-proofing)
        return True, []

    errors = _validate_record(record, schema)
    return len(errors) == 0, errors


def validate_trace_file(filepath: str, max_errors: int = 50) -> Dict[str, Any]:
    """Validate an NDJSON trace file against TritonParse schemas.

    Args:
        filepath: Path to an .ndjson, .ndjson.gz, or .ndjson.zst trace file.
        max_errors: Maximum number of errors to collect before stopping.

    Returns:
        A dict with keys:
        - valid (bool): True if all records pass validation.
        - record_count (int): Total number of records processed.
        - event_type_counts (dict): Count of each event_type seen.
        - errors (list): List of error dicts with 'line', 'event_type', 'errors'.
        - supported_event_types (list): Event types that have schemas.
    """
    result: Dict[str, Any] = {
        "valid": True,
        "record_count": 0,
        "event_type_counts": {},
        "errors": [],
        "supported_event_types": get_supported_event_types(),
    }

    try:
        with open_compressed_file(filepath) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON
                try:
                    record = loads(line)
                except JSONDecodeError as e:
                    result["valid"] = False
                    result["errors"].append(
                        {
                            "line": line_num,
                            "event_type": None,
                            "errors": [f"JSON parse error: {e}"],
                        }
                    )
                    if len(result["errors"]) >= max_errors:
                        break
                    continue

                result["record_count"] += 1
                event_type = record.get("event_type", "unknown")
                result["event_type_counts"][event_type] = (
                    result["event_type_counts"].get(event_type, 0) + 1
                )

                # Validate against schema
                is_valid, errors = validate_record(record)
                if not is_valid:
                    result["valid"] = False
                    result["errors"].append(
                        {
                            "line": line_num,
                            "event_type": event_type,
                            "errors": errors[:3],  # Cap per-record errors
                        }
                    )
                    if len(result["errors"]) >= max_errors:
                        break

    except OSError as e:
        result["valid"] = False
        result["errors"].append(
            {"line": -1, "event_type": None, "errors": [f"File error: {e}"]}
        )

    return result
