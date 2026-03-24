#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Loads JSON schema files for TritonParse trace event validation.

Works in both normal filesystem and PAR (Python Archive) environments
by using importlib.resources for schema file access.
"""

from importlib.resources import files as pkg_files
from typing import Any, Dict, List, Optional

import orjson

_SCHEMAS_PACKAGE = "tritonparse.validation.schemas"

# Schema file names for each event type.
_SCHEMA_FILES = {
    "compilation": "compilation.schema.json",
    "launch": "launch.schema.json",
    "launch_diff": "launch_diff.schema.json",
    "ir_analysis": "ir_analysis.schema.json",
}

# Cache of loaded schemas.
_loaded_schemas: Dict[str, Any] = {}


def _load_schema_file(filename: str) -> Any:
    """Load a JSON schema file from the schemas package."""
    ref = pkg_files(_SCHEMAS_PACKAGE).joinpath(filename)
    return orjson.loads(ref.read_bytes())


def get_schema(event_type: str) -> Optional[Any]:
    """Get the JSON schema for a given event type.

    Args:
        event_type: The event type string (e.g., 'compilation', 'launch').

    Returns:
        The parsed JSON schema dict, or None if no schema exists for the type.
    """
    if event_type not in _SCHEMA_FILES:
        return None

    if event_type not in _loaded_schemas:
        _loaded_schemas[event_type] = _load_schema_file(_SCHEMA_FILES[event_type])

    return _loaded_schemas[event_type]


def get_all_schemas() -> Dict[str, Any]:
    """Load and return all available schemas keyed by event type."""
    for event_type in _SCHEMA_FILES:
        get_schema(event_type)
    return dict(_loaded_schemas)


def get_supported_event_types() -> List[str]:
    """Return the list of event types that have schemas."""
    return list(_SCHEMA_FILES.keys())
