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

__all__ = [
    # Data structures
    "Message",
    "Response",
    "ToolCall",
    # Client classes
    "LLMClient",
    "MockClient",
    "ClaudeCodeClient",
]
