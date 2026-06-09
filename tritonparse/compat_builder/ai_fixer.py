# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
AI-powered Triton/LLVM compatibility fixer.

Two-phase approach (following CUTracer's AIDeadlockAnalyzer pattern):
- Phase 1 (deterministic): Extract build error + LLVM API change context
- Phase 2 (AI): Claude modifies Triton code to fix incompatibility

Key difference from CUTracer: This fixer needs WRITE permissions
(Edit, Write, Bash) to actually modify code, not just analyze it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tritonparse.ai import ClaudeCodeClient, LLMClient, Message
from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.compat_builder.context_builder import build_fix_context
from tritonparse.compat_builder.prompts import COMPAT_FIX_SYSTEM_PROMPT

logger: logging.Logger = logging.getLogger(__name__)

# Timeout for AI fix attempts (30 minutes, same as CUTracer)
_DEFAULT_TIMEOUT: int = 1800


class AICompatFixer:
    """AI-powered Triton/LLVM compatibility fixer.

    Usage::

        fixer = AICompatFixer(triton_dir="/path/to/worktree", ...)
        fix_commit = fixer.attempt_fix(
            build_error_log=Path("build_error.log"),
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
        )
        if fix_commit:
            print(f"AI generated fix: {fix_commit}")
    """

    def __init__(
        self,
        triton_dir: str,
        executor: ShellExecutor,
        bisect_logger: BisectLogger,
        model: str | None = None,
        timeout: int | None = None,
        client: LLMClient | None = None,
    ) -> None:
        """Initialize AICompatFixer.

        Args:
            triton_dir: Path to Triton repository (compat worktree).
                Used as cwd for Claude so edits stay in the worktree.
            executor: ShellExecutor for git commands.
            bisect_logger: BisectLogger for structured logging.
            model: LLM model name/alias. None = auto-select.
            timeout: Timeout in seconds. Default: 1800.
            client: Optional LLMClient override (for testing with MockClient).
        """
        self.triton_dir: Path = Path(triton_dir).resolve()
        self.llvm_dir: Path = self.triton_dir / "llvm-project"
        self.executor: ShellExecutor = executor
        self.bisect_logger: BisectLogger = bisect_logger

        self.client: LLMClient = client or ClaudeCodeClient(
            allowed_tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
            model=model,
            timeout=timeout or _DEFAULT_TIMEOUT,
            retry_count=2,
            cwd=str(self.triton_dir),
        )

    def attempt_fix(
        self,
        build_error_log: Path | None,
        incompatible_llvm: str,
        llvm_bump_commit: str,
    ) -> str | None:
        """Attempt to fix Triton for compatibility with an LLVM commit.

        Phase 1: Build structured context deterministically.
        Phase 2: Send context to Claude for code modification.

        Args:
            build_error_log: Path to the raw build error log file, or None.
            incompatible_llvm: First incompatible LLVM commit hash.
            llvm_bump_commit: The LLVM bump commit in Triton repo.

        Returns:
            Fix commit hash if successful, None if AI fix failed.
        """
        self.bisect_logger.info(
            f"AI Fix: attempting fix for LLVM {incompatible_llvm[:12]}"
        )

        head_before = self._get_head_commit()

        # Phase 1: Build structured context (deterministic)
        context = build_fix_context(
            build_error_log=build_error_log,
            incompatible_llvm=incompatible_llvm,
            llvm_bump_commit=llvm_bump_commit,
            triton_dir=self.triton_dir,
            llvm_dir=self.llvm_dir,
            executor=self.executor,
        )

        # Phase 2: AI fix (send to Claude)
        user_prompt = self._build_user_prompt(incompatible_llvm, context)
        messages = [
            Message(role="system", content=COMPAT_FIX_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        try:
            response = self.client.chat(messages)
            self.bisect_logger.info(
                f"AI response received ({len(response.content)} chars)"
            )
            self.bisect_logger.debug(f"AI response content:\n{response.content}")
            if response.raw:
                self.bisect_logger.debug(
                    f"AI response raw JSON keys: {list(response.raw.keys())}"
                )
        except RuntimeError as e:
            self.bisect_logger.warning(f"AI client error: {e}")
            return None

        # Check if Claude created a NEW commit (not a stale one from a prior round)
        fix_commit = self._check_for_commit(head_before)
        if fix_commit is not None:
            self.bisect_logger.info(f"AI fix committed: {fix_commit[:12]}")
            return fix_commit

        self.bisect_logger.warning("AI did not create a commit")
        return None

    def _build_user_prompt(
        self,
        incompatible_llvm: str,
        context: str,
    ) -> str:
        """Build user prompt for AI fixer."""
        return (
            f"The files in your current working directory do NOT compile against "
            f"LLVM commit `{incompatible_llvm[:12]}`. Edit them until they do.\n\n"
            "Follow the two-stage process:\n\n"
            "**Stage 1 — Analysis:**\n"
            "1. Read Section 1 (LLVM diff) to identify every API that changed\n"
            "2. Read the build error log file referenced in Section 2 — it "
            "contains the complete compiler output including `note:` and "
            "`candidate:` lines that show the correct new API signatures\n"
            "3. For EACH changed API, use `grep -rn` to search for ALL call sites in "
            "your current working directory\n"
            "4. List every file:line in the working directory that needs editing\n\n"
            "**Stage 2 — Fix (MANDATORY):**\n"
            "After your analysis, you MUST edit the files and commit.\n"
            "1. Edit every identified call site in the working directory\n"
            "2. Verify each file after editing\n"
            "3. Commit with: `compat fix: <description>`\n\n"
            "IMPORTANT: The build error log has the full untruncated output. "
            "Read it carefully for `note:` lines — they show the new function "
            "signatures you need to match. Then use grep to find call sites "
            "beyond what the error shows.\n\n"
            "---\n\n"
            f"{context}"
        )

    def _get_head_commit(self) -> str | None:
        """Return the current HEAD commit hash."""
        result = self.executor.run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=str(self.triton_dir),
        )
        if not result.success:
            return None
        return result.stdout.strip()

    def _check_for_commit(self, head_before: str | None) -> str | None:
        """Check if a NEW commit with 'compat fix:' was created by Claude.

        Compares current HEAD against head_before to avoid returning a
        stale fix commit from a previous iteration.
        """
        result = self.executor.run_command(
            ["git", "log", "-1", "--format=%H %s"],
            cwd=str(self.triton_dir),
        )
        if not result.success:
            return None
        line = result.stdout.strip()
        if "compat fix:" not in line:
            return None
        commit_hash = line.split()[0]
        if head_before is not None and commit_hash == head_before:
            self.bisect_logger.warning(
                "HEAD unchanged after AI fix — stale commit detected"
            )
            return None
        return commit_hash
