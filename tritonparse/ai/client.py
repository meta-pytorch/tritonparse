#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LLM Client abstractions for AI-powered analysis.

This module provides:
- Data structures for LLM communication (Message, Response, ToolCall)
- Abstract base class LLMClient for different LLM providers
- MockClient for testing without actual LLM calls
"""

import json
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple


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


class ClaudeCodeClient(LLMClient):
    """LLM client using Claude Code CLI.

    Claude Code is an AI programming assistant CLI tool by Anthropic.

    Prerequisites:
    - claude command must be available in PATH
    - Proper authentication configured

    Attributes:
        allowed_tools: List of tools Claude is allowed to use
        retry_count: Number of retry attempts on failure
        timeout: Timeout in seconds for CLI calls
        model: Model name or alias (default: "opus")
        cwd: Working directory for CLI execution
        session_id: Session ID for multi-turn conversations
    """

    DEFAULT_MODEL = "opus"

    def __init__(
        self,
        allowed_tools: Optional[List[str]] = None,
        retry_count: int = 3,
        timeout: int = 600,
        model: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        """Initialize ClaudeCodeClient.

        Args:
            allowed_tools: Tools Claude can use (e.g., ["Read", "Write", "Bash(git*)"])
            retry_count: Number of retry attempts on failure
            timeout: Timeout in seconds (default: 600 = 10 minutes)
            model: Model name or alias, default "opus" (Claude Opus 4.5)
            cwd: Working directory for running in external repos
        """
        self.allowed_tools = allowed_tools or ["Read", "Grep", "Glob"]
        self.retry_count = retry_count
        self.timeout = timeout
        self.model = model or self.DEFAULT_MODEL
        self.cwd = cwd
        self.session_id: Optional[str] = None

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Call Claude Code CLI.

        Args:
            messages: List of messages (system/user/assistant)
            temperature: Temperature parameter (ignored by Claude Code CLI)
            max_tokens: Max tokens (ignored by Claude Code CLI)

        Returns:
            Response containing content, session_id, and cost
        """
        system_prompt, user_prompt = self._extract_prompts(messages)

        # Use temp files to avoid shell escaping issues
        # Using mkstemp() instead of NamedTemporaryFile for explicit lifecycle control
        # and consistent behavior across Python versions (especially 3.12+)
        fd_user, user_file_path = tempfile.mkstemp(suffix=".txt", prefix="claude_user_")
        with os.fdopen(fd_user, "w") as user_file:
            user_file.write(user_prompt)

        fd_system, system_file_path = tempfile.mkstemp(
            suffix=".txt", prefix="claude_system_"
        )
        with os.fdopen(fd_system, "w") as system_file:
            system_file.write(system_prompt)

        try:
            # Build command using shell to read file contents
            cmd = (
                f'SYSTEM_PROMPT=$(cat "{system_file_path}") && '
                f'cat "{user_file_path}" | claude --system-prompt "$SYSTEM_PROMPT" -p'
            )

            # Add allowed tools
            if self.allowed_tools:
                tools_str = ",".join(self.allowed_tools)
                cmd += f' --allowedTools "{tools_str}"'

            # Resume session if available
            if self.session_id:
                cmd += f' --resume "{self.session_id}"'

            # Model selection
            cmd += f' --model "{self.model}"'

            # JSON output
            cmd += " --output-format json"

            # Execute with retries
            result = None
            for _attempt in range(self.retry_count):
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.cwd,
                )

                if result.returncode == 0:
                    return self._parse_response(result.stdout)

            # All retries failed
            raise RuntimeError(
                f"Claude Code CLI failed after {self.retry_count} attempts. "
                f"Return code: {result.returncode}, Error: {result.stderr}"
            )

        finally:
            # Clean up temp files
            os.unlink(user_file_path)
            os.unlink(system_file_path)

    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Stream response from Claude Code CLI.

        Uses the same temp file + shell command pattern as chat() to avoid
        command-line length limits for long system prompts.

        Args:
            messages: List of messages
            temperature: Temperature parameter (ignored)

        Yields:
            Response content chunks
        """
        system_prompt, user_prompt = self._extract_prompts(messages)

        # Use temp files to avoid shell escaping issues and command-line length limits
        # Same pattern as chat() for consistency
        fd_user, user_file_path = tempfile.mkstemp(
            suffix=".txt", prefix="claude_stream_user_"
        )
        with os.fdopen(fd_user, "w") as user_file:
            user_file.write(user_prompt)

        fd_system, system_file_path = tempfile.mkstemp(
            suffix=".txt", prefix="claude_stream_system_"
        )
        with os.fdopen(fd_system, "w") as system_file:
            system_file.write(system_prompt)

        try:
            # Build command using shell to read file contents (same as chat())
            cmd = (
                f'SYSTEM_PROMPT=$(cat "{system_file_path}") && '
                f'cat "{user_file_path}" | claude --system-prompt "$SYSTEM_PROMPT"'
            )

            # Add allowed tools
            if self.allowed_tools:
                tools_str = ",".join(self.allowed_tools)
                cmd += f' --allowedTools "{tools_str}"'

            # Resume session if available
            if self.session_id:
                cmd += f' --resume "{self.session_id}"'

            # Model selection
            cmd += f' --model "{self.model}"'

            # Stream JSON output
            cmd += " --output-format stream-json"

            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
            )

            try:
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        event_type = event.get("type")

                        if event_type == "assistant":
                            message = event.get("message", {})
                            content = message.get("content", [])
                            for block in content:
                                if isinstance(block, dict) and "text" in block:
                                    yield block["text"]

                        elif event_type == "result":
                            if "session_id" in event:
                                self.session_id = event["session_id"]

                    except json.JSONDecodeError:
                        continue

                process.wait(timeout=self.timeout)

            except subprocess.TimeoutExpired:
                process.kill()
                raise RuntimeError(f"Claude Code CLI timed out after {self.timeout}s")

        finally:
            os.unlink(user_file_path)
            os.unlink(system_file_path)

    def _extract_prompts(self, messages: List[Message]) -> Tuple[str, str]:
        """Extract system and user prompts from messages.

        Args:
            messages: List of messages

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_prompt = msg.content  # Use last user message

        return system_prompt, user_prompt

    def _parse_response(self, stdout: str) -> Response:
        """Parse Claude Code CLI JSON output.

        Args:
            stdout: Raw stdout from CLI

        Returns:
            Parsed Response object
        """
        try:
            data = json.loads(stdout)
            self.session_id = data.get("session_id")
            return Response(
                content=data.get("result", stdout),
                session_id=self.session_id,
                cost_usd=data.get("total_cost_usd"),
                raw=data,
            )
        except json.JSONDecodeError:
            # Non-JSON output, return as-is
            return Response(content=stdout.strip())
