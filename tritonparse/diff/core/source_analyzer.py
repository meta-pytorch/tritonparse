#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Source analyzer for the diff module.

This module provides Level 2 comparison of Python source code
between two compilation events.

Per the design document Section 3.2:
- Normalizes sources for comparison using black/isort formatting when available
- Falls back to simple normalization (stripping comments and empty lines)
- Computes similarity using difflib.SequenceMatcher
- Generates unified diff hunks using ORIGINAL source to preserve line numbers

IMPORTANT: Formatting is used ONLY for comparison. Diff hunks and line-level
analysis use original source to maintain consistency with source_mapping.
"""

import difflib
import logging
from typing import Any

from tritonparse.diff.core.diff_types import PythonSourceDiff

logger = logging.getLogger(__name__)


class SourceAnalyzer:
    """Analyzer for Level 2 Python source comparison.

    Compares Python source code between two compilation events.
    Uses black/isort formatting for normalization when available,
    falling back to comment/whitespace stripping.

    IMPORTANT: Formatting is used only for equality/similarity comparison.
    Diff hunks use original source to preserve line number consistency
    with source_mapping.

    Attributes:
        comp_a: First compilation event.
        comp_b: Second compilation event.
    """

    def __init__(
        self,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
    ):
        """Initialize the source analyzer.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
        """
        self.comp_a = comp_a
        self.comp_b = comp_b

    def analyze(self) -> PythonSourceDiff:
        """Analyze Python source differences.

        Uses normalized sources (via black/isort formatting) for equality
        and similarity comparison, but generates diff hunks using ORIGINAL
        source to preserve line number consistency with source_mapping.

        Returns:
            PythonSourceDiff containing status, similarity, and hunks.
        """
        source_a = self._get_python_source(self.comp_a)
        source_b = self._get_python_source(self.comp_b)

        # Normalize sources for comparison (uses formatter when available)
        norm_a = normalize_python_source(source_a)
        norm_b = normalize_python_source(source_b)

        if norm_a == norm_b:
            return PythonSourceDiff(status="identical", similarity=1.0, hunks=[])

        # Compute similarity using normalized sources
        similarity = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

        # Generate diff hunks using ORIGINAL source to preserve line numbers
        diff_lines = list(
            difflib.unified_diff(
                source_a.splitlines(),
                source_b.splitlines(),
                fromfile="compilation_a",
                tofile="compilation_b",
                lineterm="",
            )
        )
        hunks = [{"line": line} for line in diff_lines]

        return PythonSourceDiff(status="different", similarity=similarity, hunks=hunks)

    def _get_python_source(self, comp: dict[str, Any]) -> str:
        """Extract Python source from a compilation event.

        Tries multiple possible locations:
        1. payload.python_source.content
        2. payload.python

        Args:
            comp: Compilation event dictionary.

        Returns:
            Python source code string, or empty string if not found.
        """
        payload = comp.get("payload", {})

        # Try python_source.content first
        python_source = payload.get("python_source", {})
        if isinstance(python_source, dict) and "content" in python_source:
            return python_source["content"]

        # Try python field
        python_field = payload.get("python", "")
        if python_field:
            return python_field

        return ""

    def get_python_line_code(self, comp: dict[str, Any], line_number: int) -> str:
        """Get the Python source code at a specific line number.

        Args:
            comp: Compilation event dictionary.
            line_number: Line number in the original source.

        Returns:
            The code at that line, or empty string if not found.
        """
        source = self._get_python_source(comp)
        if not source:
            return ""

        lines = source.splitlines()
        python_source = comp.get("payload", {}).get("python_source", {})
        start_line = python_source.get("start_line", 1)

        # Adjust for start_line offset
        idx = line_number - start_line
        if 0 <= idx < len(lines):
            return lines[idx]

        return ""


def normalize_python_source(source: str, use_formatter: bool = True) -> str:
    """Normalize Python source for comparison.

    Uses black/isort formatting for deep normalization when available,
    falling back to simple comment/whitespace stripping.

    IMPORTANT: The normalized result is used ONLY for equality/similarity
    comparison. Diff hunks and line-level analysis use ORIGINAL source
    to preserve line number consistency with source_mapping.

    Args:
        source: Python source code string.
        use_formatter: If True, attempt to use black/isort for deep
                      normalization. Falls back to simple normalization
                      if formatting fails.

    Returns:
        Normalized source string for comparison purposes.
    """
    if not source:
        return ""

    if use_formatter:
        try:
            # Lazy import to avoid heavy torch dependency from reproducer.utils
            from tritonparse.reproducer.utils import format_python_code

            # Deep normalization: black + isort via format_python_code
            formatted = format_python_code(source)
            if formatted:
                return formatted
        except ImportError:
            logger.debug("format_python_code not available, using simple normalization")
        except Exception as e:
            logger.debug(f"Formatter failed, falling back to simple normalization: {e}")

    # Simple normalization: strip comments and empty lines
    lines = source.splitlines()
    code_lines = [
        line for line in lines if line.strip() and not line.strip().startswith("#")
    ]
    return "\n".join(code_lines)


def is_python_source_identical(source_a: str, source_b: str) -> bool:
    """Check if two Python sources are identical (ignoring comments).

    Convenience function per design doc Section 3.2.

    Args:
        source_a: First Python source.
        source_b: Second Python source.

    Returns:
        True if normalized sources are identical.
    """
    return normalize_python_source(source_a) == normalize_python_source(source_b)


def analyze_python_source(
    comp_a: dict[str, Any],
    comp_b: dict[str, Any],
) -> PythonSourceDiff:
    """Convenience function to analyze Python source differences.

    Args:
        comp_a: First compilation event.
        comp_b: Second compilation event.

    Returns:
        PythonSourceDiff containing status, similarity, and hunks.
    """
    analyzer = SourceAnalyzer(comp_a, comp_b)
    return analyzer.analyze()
