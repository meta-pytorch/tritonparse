#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LLM Client abstractions for AI-powered analysis.

This module provides:
- Data structures for LLM communication (Message, Response, ToolCall)
- Abstract base class LLMClient for different LLM providers
- CLI clients for Claude Code and Codex
- MockClient for testing without actual LLM calls
"""

import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

from tritonparse._json_compat import JSONDecodeError, loads

try:
    from security.frameworks.python.exec.subprocess import TrustedSubprocessWithList
except ModuleNotFoundError as error:  # pragma: no cover - OSS source fallback
    _missing_module = error.name or ""
    if _missing_module != "security" and not _missing_module.startswith("security."):
        raise

    class TrustedSubprocessWithList:
        """Portable fallback when Meta's trusted subprocess wrapper is unavailable."""

        @staticmethod
        def run(
            *, executable: str, cmd_args: List[str], **kwargs: Any
        ) -> "subprocess.CompletedProcess[Any]":
            # The executable is fixed by the caller and dynamic values remain
            # separate argv entries, so the OSS fallback never invokes a shell.
            return subprocess.run([executable, *cmd_args], **kwargs)  # noqa: P204


logger: logging.Logger = logging.getLogger(__name__)

_CODEX_TRANSIENT_FAILURE_MARKERS: Tuple[str, ...] = (
    "stream disconnected before completion",
    "no_capacity",
    "no capacity",
)


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


class CodexClient(LLMClient):
    """LLM client using the Codex CLI's non-interactive ``exec`` mode.

    The client is intentionally read-only and ephemeral. System messages map to
    invocation-scoped Codex developer instructions, while the user message is
    supplied through stdin to avoid shell quoting and command-line length limits.
    """

    def __init__(
        self,
        retry_count: int = 3,
        timeout: int = 600,
        model: Optional[str] = None,
        cwd: Optional[str] = None,
    ) -> None:
        if retry_count < 1:
            raise ValueError("retry_count must be at least 1")
        self.retry_count = retry_count
        self.timeout = timeout
        self.model = model
        self.cwd = os.path.abspath(cwd) if cwd else cwd

    def chat(
        self,
        messages: List[Message],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> Response:
        """Run one stateless Codex turn and return its final agent message."""
        del temperature, max_tokens
        system_prompt, user_prompt = self._extract_prompts(messages)

        for attempt in range(1, self.retry_count + 1):
            try:
                result, content = self._run_once(system_prompt, user_prompt)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    f"Codex CLI timed out after {self.timeout}s"
                ) from error
            except OSError as error:
                raise RuntimeError(f"Failed to launch Codex CLI: {error}") from error

            if result.returncode == 0:
                if not content:
                    raise RuntimeError("Codex CLI returned no final response")
                return Response(
                    content=content,
                    raw={
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    },
                )

            failure = self._format_failure(result.stdout, result.stderr)
            if attempt == self.retry_count or not self._is_transient_failure(failure):
                raise RuntimeError(
                    f"Codex CLI failed on attempt {attempt}/{self.retry_count}: {failure}"
                )

        raise AssertionError("Codex retry loop exited unexpectedly")

    def chat_stream(
        self,
        messages: List[Message],
        temperature: float = 0.0,
    ) -> Iterator[str]:
        """Yield the final response as one chunk.

        CUTracer uses one-shot reasoning today. A future consumer that needs
        progress events can add ``codex exec --json`` parsing without changing
        the synchronous ``chat`` contract.
        """
        yield self.chat(messages, temperature).content

    def _run_once(
        self, system_prompt: str, user_prompt: str
    ) -> Tuple["subprocess.CompletedProcess[Any]", str]:
        output_fd, output_path = tempfile.mkstemp(
            suffix=".md", prefix="codex_last_message_"
        )
        os.close(output_fd)
        try:
            result = TrustedSubprocessWithList.run(
                executable="codex",
                cmd_args=self._build_args(system_prompt, output_path),
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
                check=False,
            )
            content = Path(output_path).read_text(encoding="utf-8").strip()
            return result, content
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass

    def _build_args(self, system_prompt: str, output_path: str) -> List[str]:
        args = [
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "-s",
            "read-only",
            "--disable",
            "hooks",
            "--disable",
            "multi_agent",
        ]
        if system_prompt:
            args.extend(
                [
                    "-c",
                    f"developer_instructions={json.dumps(system_prompt, ensure_ascii=False)}",
                ]
            )
        if self.cwd:
            args.extend(["-C", self.cwd])
        if self.model:
            args.extend(["-m", self.model])
        args.extend(
            [
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--color",
                "never",
                "--output-last-message",
                output_path,
                "-",
            ]
        )
        return args

    @staticmethod
    def _extract_prompts(messages: List[Message]) -> Tuple[str, str]:
        system_prompts: List[str] = []
        user_prompts: List[str] = []
        for message in messages:
            if message.role == "system":
                system_prompts.append(message.content)
            elif message.role == "user":
                user_prompts.append(message.content)
            else:
                raise ValueError(
                    "CodexClient supports at most one system message, exactly one "
                    "user message, and no other roles"
                )

        if len(system_prompts) > 1 or len(user_prompts) != 1:
            raise ValueError(
                "CodexClient supports at most one system message, exactly one user "
                "message, and no other roles"
            )

        system_prompt = system_prompts[0] if system_prompts else ""
        return system_prompt, user_prompts[0]

    @staticmethod
    def _format_failure(stdout: str, stderr: str) -> str:
        stdout_tail = stdout[-500:].strip()
        stderr_tail = stderr[-500:].strip()
        if stdout_tail and stderr_tail:
            return f"stdout: {stdout_tail}\nstderr: {stderr_tail}"
        if stdout_tail:
            return f"stdout: {stdout_tail}"
        if stderr_tail:
            return f"stderr: {stderr_tail}"
        return "no stdout or stderr"

    @staticmethod
    def _is_transient_failure(failure: str) -> bool:
        normalized = failure.lower()
        return any(marker in normalized for marker in _CODEX_TRANSIENT_FAILURE_MARKERS)


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
        model: Model name or alias, or None to let Claude CLI auto-select
        cwd: Working directory for CLI execution
        session_id: Session ID for multi-turn conversations
    """

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
            model: Model name or alias, or None to let Claude CLI auto-select (default: None)
            cwd: Working directory for running in external repos
        """
        self.allowed_tools = allowed_tools or ["Read", "Grep", "Glob"]
        self.retry_count = retry_count
        self.timeout = timeout
        self.model = model
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

            # Model selection (only if explicitly specified)
            if self.model:
                cmd += f' --model "{self.model}"'

            # JSON output
            cmd += " --output-format json"

            # Execute with retries
            logger.info(f"Claude CLI command: {cmd}")
            logger.info(f"Claude CLI cwd: {self.cwd}")
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

                logger.info(
                    f"Claude CLI attempt {_attempt + 1}: "
                    f"returncode={result.returncode}, "
                    f"stdout_len={len(result.stdout)}, "
                    f"stderr_len={len(result.stderr)}"
                )
                if result.stderr:
                    logger.info(f"Claude CLI stderr: {result.stderr[:1000]}")

                if result.returncode == 0:
                    return self._parse_response(result.stdout)

            # All retries failed — include both stderr and stdout for diagnostics.
            # The actual error reason is often in stdout (JSON "result" field),
            # while stderr only contains service warnings (e.g., SEV notices).
            stdout_snippet = ""
            if result and result.stdout:
                try:
                    data = loads(result.stdout)
                    stdout_snippet = data.get("result", result.stdout[:500])
                except (JSONDecodeError, AttributeError):
                    stdout_snippet = result.stdout[:500]
            stderr_snippet = (
                result.stderr[:500] if result and result.stderr else "(empty)"
            )
            raise RuntimeError(
                f"Claude Code CLI failed after {self.retry_count} attempts. "
                f"Return code: {result.returncode}\n"
                f"Error: {stdout_snippet}\n"
                f"Stderr: {stderr_snippet}"
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

            # Model selection (only if explicitly specified)
            if self.model:
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
                        event = loads(line)
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

                    except JSONDecodeError:
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
            data = loads(stdout)
            self.session_id = data.get("session_id")
            return Response(
                content=data.get("result", stdout),
                session_id=self.session_id,
                cost_usd=data.get("total_cost_usd"),
                raw=data,
            )
        except JSONDecodeError:
            # Non-JSON output, return as-is
            return Response(content=stdout.strip())
