#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Output utilities for the diff module.

This module provides functions for serializing diff results to JSON/ndjson
and formatting results for CLI display.

Output format: Consolidated ndjson with unique compilation events first,
followed by diff events. Use ConsolidatedDiffWriter for streaming writes.
"""

from tritonparse.diff.output.event_writer import (
    append_diff_to_file,
    ConsolidatedDiffWriter,
    create_diff_event,
    write_consolidated_output,
)
from tritonparse.diff.output.summary_formatter import format_summary

__all__ = [
    "append_diff_to_file",
    "ConsolidatedDiffWriter",
    "create_diff_event",
    "format_summary",
    "write_consolidated_output",
]
