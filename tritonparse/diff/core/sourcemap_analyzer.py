#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Source mapping analyzer for the diff module.

This module provides Level 4 comparison: organizing IR differences by Python
source line. This is the core functionality that lets users see "the same line
of Python code compiled to different IR lines".

Per the design document Section 4:
- Uses source_mappings from parsed compilation events
- Extracts IR line numbers corresponding to each Python line
- Computes expansion (line count difference) per IR type per Python line
"""

from typing import Any

from tritonparse.diff.core.diff_types import PythonLineDiff


# IR types to analyze for source mapping comparison
IR_TYPES = ["ttir", "ttgir", "llir", "ptx", "amdgcn"]


class SourcemapAnalyzer:
    """Analyzer for Level 4 source mapping-based comparison.

    Organizes IR differences by Python source line, enabling users to
    see how the same Python code compiles to different IR in each compilation.

    Attributes:
        comp_a: First compilation event (with source_mappings).
        comp_b: Second compilation event (with source_mappings).
        ir_types: List of IR types to analyze.
    """

    def __init__(
        self,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
        ir_types: list[str] | None = None,
    ):
        """Initialize the sourcemap analyzer.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
            ir_types: Optional list of IR types to analyze.
                      Defaults to IR_TYPES if not provided.
        """
        self.comp_a = comp_a
        self.comp_b = comp_b
        self.ir_types = ir_types or IR_TYPES

    def analyze(self) -> dict[int, PythonLineDiff]:
        """Analyze IR differences organized by Python source line.

        For each Python line that has source mappings in either compilation,
        extracts the corresponding IR lines and computes the expansion
        (difference in IR line count).

        Returns:
            Dictionary mapping Python line number to PythonLineDiff.
        """
        mappings_a = self.comp_a.get("payload", {}).get("source_mappings", {})
        mappings_b = self.comp_b.get("payload", {}).get("source_mappings", {})

        python_a = mappings_a.get("python", {})
        python_b = mappings_b.get("python", {})

        # Get all Python line numbers from both compilations
        all_lines: set[int] = set()
        all_lines.update(int(k) for k in python_a.keys() if k.isdigit())
        all_lines.update(int(k) for k in python_b.keys() if k.isdigit())

        result: dict[int, PythonLineDiff] = {}

        for py_line in sorted(all_lines):
            a_mapping: dict[str, list[int]] = {}
            b_mapping: dict[str, list[int]] = {}
            expansion: dict[str, int] = {}

            for ir_type in self.ir_types:
                lines_a = get_ir_lines_for_python_line(mappings_a, py_line, ir_type)
                lines_b = get_ir_lines_for_python_line(mappings_b, py_line, ir_type)

                if lines_a or lines_b:
                    a_mapping[ir_type] = lines_a
                    b_mapping[ir_type] = lines_b
                    expansion[ir_type] = len(lines_b) - len(lines_a)

            # Get Python code for this line
            python_code = get_python_line_code(self.comp_a, py_line)

            result[py_line] = PythonLineDiff(
                python_line=py_line,
                python_code=python_code,
                a=a_mapping,
                b=b_mapping,
                expansion=expansion,
            )

        return result

    def get_stats(self, by_python_line: dict[int, PythonLineDiff]) -> dict[str, int]:
        """Compute statistics from the by_python_line analysis.

        Args:
            by_python_line: Result from analyze().

        Returns:
            Dictionary with statistics:
            - python_lines_compared: Total lines analyzed
            - lines_with_ir_diff: Lines where IR expansion != 0
            - total_ir_line_expansion: Sum of all expansions
        """
        python_lines_compared = len(by_python_line)
        lines_with_ir_diff = sum(
            1
            for diff in by_python_line.values()
            if any(v != 0 for v in diff.expansion.values())
        )
        total_ir_line_expansion = sum(
            sum(diff.expansion.values()) for diff in by_python_line.values()
        )

        return {
            "python_lines_compared": python_lines_compared,
            "lines_with_ir_diff": lines_with_ir_diff,
            "total_ir_line_expansion": total_ir_line_expansion,
        }


def get_ir_lines_for_python_line(
    source_mappings: dict[str, Any], python_line: int, ir_type: str
) -> list[int]:
    """Get IR line numbers corresponding to a Python source line.

    Per design doc Section 4.1, uses the existing source_mappings structure:
    {
        "python": {
            "20": {"ttir_lines": [5,6], "ttgir_lines": [7,8,9,10]},
            ...
        }
    }

    Args:
        source_mappings: Source mappings from a compilation event.
        python_line: Python source line number.
        ir_type: IR type (ttir, ttgir, llir, ptx, amdgcn).

    Returns:
        List of IR line numbers, or empty list if none found.
    """
    python_mappings = source_mappings.get("python", {})
    line_mapping = python_mappings.get(str(python_line), {})
    ir_lines = line_mapping.get(f"{ir_type}_lines", [])

    # Ensure all line numbers are integers
    return [int(ln) if isinstance(ln, str) else ln for ln in ir_lines]


def get_python_line_code(comp: dict[str, Any], line_number: int) -> str:
    """Get the Python source code at a specific line number.

    Handles both python_source (with start_line offset) and
    plain python field formats.

    Args:
        comp: Compilation event dictionary.
        line_number: Line number in the original source file.

    Returns:
        The source code at that line, or empty string if not found.
    """
    payload = comp.get("payload", {})

    # Try python_source.content first (has start_line offset)
    python_source = payload.get("python_source", {})
    if isinstance(python_source, dict) and "content" in python_source:
        source = python_source["content"]
        start_line = python_source.get("start_line", 1)
    else:
        # Fall back to python field
        source = payload.get("python", "")
        start_line = 1

    if not source:
        return ""

    lines = source.splitlines()
    idx = line_number - start_line
    if 0 <= idx < len(lines):
        return lines[idx]

    return ""


def analyze_by_python_line(
    comp_a: dict[str, Any],
    comp_b: dict[str, Any],
    ir_types: list[str] | None = None,
) -> dict[int, PythonLineDiff]:
    """Convenience function to analyze IR differences by Python line.

    Args:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        ir_types: Optional list of IR types to analyze.

    Returns:
        Dictionary mapping Python line number to PythonLineDiff.
    """
    analyzer = SourcemapAnalyzer(comp_a, comp_b, ir_types)
    return analyzer.analyze()
