#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
AI module for tritonparse.

This module provides LLM client abstractions and utilities for AI-powered
analysis features like diff analysis and build error fixing.
"""

from tritonparse.ai.client import (
    ClaudeCodeClient,
    LLMClient,
    Message,
    MockClient,
    Response,
    ToolCall,
)
from tritonparse.ai.parsers import extract_code_block, extract_diff_patch, extract_json
from tritonparse.ai.utils import format_messages, truncate_context

__all__ = [
    # Data structures
    "Message",
    "Response",
    "ToolCall",
    # Client classes
    "LLMClient",
    "MockClient",
    "ClaudeCodeClient",
    # Parsers
    "extract_json",
    "extract_code_block",
    "extract_diff_patch",
    # Utils
    "format_messages",
    "truncate_context",
]
