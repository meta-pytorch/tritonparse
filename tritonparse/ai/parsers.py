#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Output parsers for LLM responses.

These parsers extract structured content from LLM text responses.
They are primarily used as backup when LLM doesn't use tools to write files directly.

All parsers return None on invalid input (fail-safe design).
"""

import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[dict]:
    """Extract a JSON object from text.

    Supports:
    - Direct JSON text: {"key": "value"}
    - Markdown code blocks: ```json\\n{...}\\n```
    - Code blocks without language: ```\\n{...}\\n```

    Args:
        text: Text potentially containing JSON

    Returns:
        Parsed dict if valid JSON found, None otherwise
    """
    if not text or not text.strip():
        return None

    # Try direct JSON parsing
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    return None


def extract_code_block(text: str, language: Optional[str] = None) -> Optional[str]:
    """Extract a code block from markdown text.

    BACKUP FUNCTION: This is a fallback parser. The preferred approach is to
    instruct the AI to write code/patches directly to files using tools,
    rather than extracting from text responses.

    Args:
        text: Markdown text containing code blocks
        language: Optional language identifier (e.g., "python", "diff").
                 If None, matches any code block.

    Returns:
        Code block content if found, None otherwise
    """
    if not text:
        return None

    if language:
        pattern = rf"```{re.escape(language)}\s*\n(.*?)\n?```"
    else:
        pattern = r"```(?:\w+)?\s*\n(.*?)\n?```"

    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None


_DIFF_LINE_PREFIXES = (
    "diff --git",
    "index ",
    "--- ",
    "+++ ",
    "@@ ",
    " ",
    "+",
    "-",
    "\\ ",
)


def _is_diff_line(line: str) -> bool:
    """Check if a line looks like part of a unified diff."""
    return line.startswith(_DIFF_LINE_PREFIXES)


def extract_diff_patch(text: str) -> Optional[str]:
    """Extract git diff/patch content from text.

    BACKUP FUNCTION: This is a fallback parser. The preferred approach is to
    instruct the AI to write patches directly to files using tools (e.g.,
    write to a .patch file), rather than extracting from text responses.

    Supports:
    - Markdown code blocks: ```diff\\n...\\n```
    - Raw unified diff format (starts with "diff --git" or "--- ")

    For raw diff extraction, trailing non-diff lines (e.g., LLM explanations
    after the patch) are automatically trimmed.

    Args:
        text: Text potentially containing a diff/patch

    Returns:
        Extracted patch content if found, None otherwise
    """
    if not text:
        return None

    # First try markdown code block
    patch = extract_code_block(text, "diff")
    if patch:
        return patch

    # Try raw unified diff format
    lines = text.split("\n")
    patch_lines = []
    in_patch = False

    for line in lines:
        if line.startswith("diff --git") or line.startswith("--- "):
            in_patch = True
        if in_patch:
            patch_lines.append(line)

    # Trim trailing non-diff lines (e.g., LLM explanations after the patch)
    while patch_lines and (
        not patch_lines[-1].strip() or not _is_diff_line(patch_lines[-1])
    ):
        patch_lines.pop()

    return "\n".join(patch_lines) if patch_lines else None
