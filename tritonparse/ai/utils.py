#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Utility functions for AI module.

These utilities handle common operations like message formatting
and context management for LLM interactions.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from tritonparse.ai.client import Message


def format_messages(messages: List["Message"]) -> str:
    """Format a list of messages into a single prompt string.

    This is useful for debugging, logging, or when using APIs
    that expect a single prompt string rather than message arrays.

    Args:
        messages: List of Message objects

    Returns:
        Formatted string with role prefixes

    Example:
        >>> msgs = [Message("system", "You are helpful."),
        ...         Message("user", "Hello")]
        >>> print(format_messages(msgs))
        [system]
        You are helpful.

        [user]
        Hello
    """
    if not messages:
        return ""

    parts = []
    for msg in messages:
        parts.append(f"[{msg.role}]")
        parts.append(msg.content)
        parts.append("")  # Empty line between messages
    return "\n".join(parts).rstrip()


def truncate_context(
    text: str,
    max_chars: int,
    strategy: str = "tail",
    ellipsis: str = "... [truncated]",
) -> str:
    """Truncate text to fit within a character limit.

    This is useful for managing context window limits when
    sending large content to LLMs.

    Args:
        text: Text to truncate
        max_chars: Maximum number of characters allowed
        strategy: Truncation strategy - "head", "tail", or "middle"
            - "head": Keep the beginning, truncate end
            - "tail": Keep the end, truncate beginning
            - "middle": Keep beginning and end, truncate middle
        ellipsis: String to indicate truncation

    Returns:
        Truncated text if over limit, original text otherwise

    Raises:
        ValueError: If strategy is not one of "head", "tail", "middle"
    """
    if not text or len(text) <= max_chars:
        return text

    if strategy not in ("head", "tail", "middle"):
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be 'head', 'tail', or 'middle'"
        )

    ellipsis_len = len(ellipsis)
    available = max_chars - ellipsis_len

    if available <= 0:
        return text[:max_chars]

    if strategy == "head":
        # Keep the beginning
        return text[:available] + ellipsis
    elif strategy == "tail":
        # Keep the end
        return ellipsis + text[-available:]
    else:  # middle
        # Keep beginning and end
        half = available // 2
        return text[:half] + ellipsis + text[-half:]
