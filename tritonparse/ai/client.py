#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LLM Client abstractions for AI-powered analysis.

This module provides:
- Data structures for LLM communication (Message, Response, ToolCall)
- Abstract base class LLMClient for different LLM providers
- MockClient for testing without actual LLM calls
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional


@dataclass
class Message:
    """A message in a conversation with an LLM.

    Attributes:
        role: The role of the message sender ("system", "user", or "assistant")
        content: The text content of the message
    """

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCall:
    """A tool/function call requested by the LLM.

    Attributes:
        name: The name of the tool/function to call
        arguments: The arguments to pass to the tool/function
    """

    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class Response:
    """Response from an LLM.

    Attributes:
        content: The text content of the response
        session_id: Optional session ID for multi-turn conversations
        cost_usd: Optional cost of the API call in USD
        tool_calls: Optional list of tool calls requested by the LLM
        raw: Optional raw response data for debugging
    """

    content: str
    session_id: Optional[str] = None
    cost_usd: Optional[float] = None
    tool_calls: Optional[List[ToolCall]] = None
    raw: Optional[Any] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients.

    This class defines the interface for interacting with LLM providers.
    Concrete implementations should handle provider-specific details.
    """

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Send messages to the LLM and get a response.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in the response

        Returns:
            Response from the LLM
        """
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Send messages to the LLM and stream the response.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0.0 = deterministic)

        Yields:
            Content chunks as they are received
        """
        pass


class MockClient(LLMClient):
    """Mock LLM client for testing.

    This client returns predefined responses and tracks call history,
    making it useful for unit testing without actual LLM calls.

    Attributes:
        responses: List of responses to return in order
        call_count: Number of times chat() has been called
        last_messages: Messages from the most recent chat() call
    """

    def __init__(self, responses: Optional[List[str]] = None):
        """Initialize MockClient.

        Args:
            responses: List of response strings to return in order.
                      After exhausting this list, returns "Mock response".
        """
        self.responses: List[str] = responses or []
        self.call_count: int = 0
        self.last_messages: Optional[List[Message]] = None

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Return a predefined response.

        Args:
            messages: Messages (saved to last_messages for verification)
            temperature: Ignored
            max_tokens: Ignored

        Returns:
            Response with next predefined content or "Mock response"
        """
        self.last_messages = messages

        if self.call_count < len(self.responses):
            content = self.responses[self.call_count]
            self.call_count += 1
            return Response(content=content)

        self.call_count += 1
        return Response(content="Mock response")

    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield the response content as a single chunk.

        Args:
            messages: Messages to send
            temperature: Ignored

        Yields:
            The full response content as a single chunk
        """
        response = self.chat(messages, temperature)
        yield response.content
