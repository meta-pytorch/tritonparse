# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
ConsolidatedResult: Data structures for multi-file call graph analysis results.

This module defines the output structure for the MultiFileCallGraphAnalyzer,
containing all extracted functions, imports, edges, and statistics.
"""

from dataclasses import dataclass

from tritonparse.reproducer.ast_analyzer import Edge
from tritonparse.reproducer.import_info import ImportInfo


@dataclass
class AnalysisStats:
    """Statistics about the multi-file analysis."""

    total_files_analyzed: int
    total_functions_found: int
    total_imports: int
    external_imports: int
    internal_imports: int


@dataclass
class ConsolidatedResult:
    """Consolidated analysis results across all files."""

    # All dependent functions with their source code
    functions: dict[str, str]  # qualified_name -> source_code

    # Mapping of function to file
    function_locations: dict[str, str]  # qualified_name -> file_path

    # Short names for standalone file generation (just the function name)
    function_short_names: dict[str, str]  # qualified_name -> short_name

    # All required imports, deduplicated and organized
    imports: list[ImportInfo]

    # Call graph edges across all files
    edges: list[Edge]

    # Files analyzed
    analyzed_files: set[str]

    # Statistics
    stats: AnalysisStats
