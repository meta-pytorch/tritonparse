#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Parse module for tritonparse.

This module contains all the functionality for parsing triton trace logs,
extracting source mappings, and processing IR files.
"""

# Public API exports
from .common import (
    copy_local_to_tmpdir,
    gzip_single_file,
    parse_logs,
    print_parsed_files_summary,
    Rank,
    RankConfig,
    save_logs,
)
from .ir_parser import (
    extract_code_locations,
    extract_loc_definitions,
    extract_ptx_amdgcn_mappings,
)
from .mapper import (
    create_bidirectional_mapping,
    create_ir_mapping,
    create_python_mapping,
)
from .source_type import Source, SourceType
from .trace_processor import (
    generate_source_mappings,
    parse_single_file,
    parse_single_trace_content,
)
from .utils import _add_parse_args, oss_run, unified_parse

__all__ = [
    # Common utilities
    "copy_local_to_tmpdir",
    "gzip_single_file",
    "parse_logs",
    "print_parsed_files_summary",
    "Rank",
    "RankConfig",
    "save_logs",
    # Source type
    "Source",
    "SourceType",
    # Trace processor
    "generate_source_mappings",
    "parse_single_file",
    "parse_single_trace_content",
    # Utils
    "_add_parse_args",
    "oss_run",
    "unified_parse",
    # IR parsing
    "extract_code_locations",
    "extract_loc_definitions",
    "extract_ptx_amdgcn_mappings",
    # Mapper
    "create_bidirectional_mapping",
    "create_ir_mapping",
    "create_python_mapping",
]
