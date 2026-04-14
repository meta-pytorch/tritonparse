# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for compat_builder.ai_fixer module."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

from tritonparse.ai import MockClient
from tritonparse.compat_builder.ai_fixer import AICompatFixer


@dataclass
class _FakeResult:
    """Minimal stand-in for CommandResult."""

    success: bool
    stdout: str
    stderr: str = ""
    output: str = ""


class _FakeExecutor:
    """Fake ShellExecutor that returns canned responses."""

    def __init__(self, responses: dict[str, _FakeResult] | None = None) -> None:
        self.responses: dict[str, _FakeResult] = responses or {}
        self.calls: list[tuple[list[str], str]] = []

    def run_command(
        self,
        cmd: list[str],
        cwd: str = "",
        env: dict[str, str] | None = None,
    ) -> _FakeResult:
        self.calls.append((cmd, cwd))
        key = " ".join(cmd)
        for pattern, result in self.responses.items():
            if pattern in key:
                return result
        return _FakeResult(success=False, stdout="", stderr="not found")


class AICompatFixerTest(unittest.TestCase):
    def _make_fixer(
        self,
        client: MockClient,
        executor: _FakeExecutor | None = None,
    ) -> AICompatFixer:
        if executor is None:
            executor = _FakeExecutor()
        bisect_logger = MagicMock()
        bisect_logger.info = MagicMock()
        bisect_logger.warning = MagicMock()
        # pyre-ignore[6]: _FakeExecutor is not ShellExecutor
        return AICompatFixer(
            triton_dir="/fake/triton",
            executor=executor,
            bisect_logger=bisect_logger,
            client=client,
        )

    def test_attempt_fix_calls_client(self) -> None:
        client = MockClient(responses=["Fixed the file successfully"])
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(success=True, stdout="diff content"),
                "git show": _FakeResult(success=True, stdout="show content"),
                "git log": _FakeResult(
                    success=True,
                    stdout="abc123 compat fix: renamed getFoo to getBar",
                ),
            }
        )
        fixer = self._make_fixer(client, executor)

        result = fixer.attempt_fix(
            build_error="/src/Foo.cpp:42: error: no member 'getBar'",
            incompatible_llvm="llvm_abc123",
            llvm_bump_commit="bump_def456",
        )

        self.assertEqual(client.call_count, 1)
        self.assertIsNotNone(client.last_messages)
        # System prompt should be first message
        self.assertEqual(client.last_messages[0].role, "system")
        self.assertIn("LLVM", client.last_messages[0].content)
        # User prompt should contain the LLVM commit hash
        self.assertEqual(client.last_messages[1].role, "user")
        self.assertIn("llvm_abc123", client.last_messages[1].content)
        # Should find the compat fix commit
        self.assertEqual(result, "abc123")

    def test_returns_none_when_no_commit_created(self) -> None:
        client = MockClient(responses=["I couldn't fix the issue"])
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(success=True, stdout=""),
                "git show": _FakeResult(success=True, stdout=""),
                # git log returns a non-compat-fix commit
                "git log": _FakeResult(
                    success=True,
                    stdout="abc123 some other commit message",
                ),
            }
        )
        fixer = self._make_fixer(client, executor)

        result = fixer.attempt_fix(
            build_error="error",
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
        )

        self.assertIsNone(result)
        self.assertEqual(client.call_count, 1)

    def test_returns_none_on_client_error(self) -> None:
        client = MockClient()
        # Make client.chat raise
        client.chat = MagicMock(side_effect=RuntimeError("timeout"))  # pyre-ignore[8]
        fixer = self._make_fixer(client)

        result = fixer.attempt_fix(
            build_error="error",
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
        )

        self.assertIsNone(result)

    def test_handles_none_build_error(self) -> None:
        client = MockClient(responses=["done"])
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(success=True, stdout=""),
                "git log": _FakeResult(success=True, stdout="x no match"),
            }
        )
        fixer = self._make_fixer(client, executor)

        result = fixer.attempt_fix(
            build_error=None,
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
        )

        self.assertIsNone(result)
        # Should still call the client
        self.assertEqual(client.call_count, 1)

    def test_user_prompt_contains_instructions(self) -> None:
        client = MockClient(responses=["done"])
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(success=True, stdout=""),
                "git log": _FakeResult(success=True, stdout="x no match"),
            }
        )
        fixer = self._make_fixer(client, executor)

        fixer.attempt_fix(
            build_error="error",
            incompatible_llvm="abc123def456",
            llvm_bump_commit="xyz789",
        )

        self.assertIsNotNone(client.last_messages)
        user_msg = client.last_messages[1].content
        self.assertIn("compat fix:", user_msg)
        self.assertIn("abc123def456"[:12], user_msg)
