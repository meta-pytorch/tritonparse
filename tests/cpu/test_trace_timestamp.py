# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for the trace timestamp written by `TritonJsonFormatter`.

`logging.Formatter.formatTime` delegates to `time.strftime`, which does not
support `%f` (microseconds) and emits it as a literal string. These tests pin
that formatted trace records carry a real `YYYY-MM-DDTHH:MM:SS.ffffffZ`
timestamp.
"""

import re
import time
import unittest

from tritonparse._json_compat import loads
from tritonparse.structured_logging import create_triton_log_record, TritonJsonFormatter

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def _format_timestamp(created: float) -> str:
    record = create_triton_log_record(metadata={"event_type": "launch"})
    record.created = created
    return loads(TritonJsonFormatter().format(record))["timestamp"]


class TraceTimestampTest(unittest.TestCase):
    def test_no_literal_percent_f(self) -> None:
        timestamp = _format_timestamp(1759197943.5)
        self.assertNotIn("%f", timestamp)
        self.assertRegex(timestamp, _TIMESTAMP_RE)

    def test_microseconds_match_record_created(self) -> None:
        created = 1759197943.5
        expected = (
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created)) + ".500000Z"
        )
        self.assertEqual(_format_timestamp(created), expected)
