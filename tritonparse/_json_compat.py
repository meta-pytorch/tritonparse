#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
JSON compatibility layer for tritonparse.

Provides a unified interface that uses orjson when available (for performance)
and falls back to stdlib json (for environments where orjson is unavailable,
e.g. CPython 3.14 free-threading builds).

``loads()`` accepts ``str | bytes | bytearray | memoryview`` inputs.
``dumps()`` returns ``str``.
"""

try:
    import orjson as _orjson

    _HAS_ORJSON = True
except ImportError:
    import json as _json

    _HAS_ORJSON = False


def _coerce_keys(obj):
    """Recursively convert non-string dict keys to strings.

    stdlib ``json.dumps`` raises ``TypeError`` on non-string keys, whereas
    orjson's ``OPT_NON_STR_KEYS`` converts them automatically. This helper
    replicates that behavior for the fallback path.
    """
    if isinstance(obj, dict):
        return {str(k): _coerce_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_keys(v) for v in obj]
    return obj


if _HAS_ORJSON:
    JSONDecodeError = _orjson.JSONDecodeError

    def loads(data):
        """Deserialize JSON string/bytes to a Python object."""
        return _orjson.loads(data)

    def dumps(obj, *, indent=False, sort_keys=False):
        """Serialize a Python object to a JSON ``str``.

        Args:
            obj: The object to serialize.
            indent: If True, pretty-print with 2-space indent.
            sort_keys: If True, sort dictionary keys.
        """
        option = _orjson.OPT_NON_STR_KEYS
        if indent:
            option |= _orjson.OPT_INDENT_2
        if sort_keys:
            option |= _orjson.OPT_SORT_KEYS
        return _orjson.dumps(obj, option=option).decode()

else:
    from json import JSONDecodeError  # noqa: F401

    def loads(data):
        """Deserialize JSON string/bytes to a Python object."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = (
                bytes(data).decode() if isinstance(data, memoryview) else data.decode()
            )
        return _json.loads(data)

    def dumps(obj, *, indent=False, sort_keys=False):
        """Serialize a Python object to a JSON ``str``.

        Args:
            obj: The object to serialize.
            indent: If True, pretty-print with 2-space indent.
            sort_keys: If True, sort dictionary keys.
        """
        obj = _coerce_keys(obj)
        kwargs = {"ensure_ascii": False}
        if indent:
            kwargs["indent"] = 2
        else:
            kwargs["separators"] = (",", ":")
        if sort_keys:
            kwargs["sort_keys"] = True
        return _json.dumps(obj, **kwargs)
