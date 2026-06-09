# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for compat_builder.context_builder module."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from tritonparse.compat_builder.context_builder import (
    build_fix_context,
    extract_failing_files,
)


@dataclass
class _FakeResult:
    """Minimal stand-in for CommandResult."""

    success: bool
    stdout: str
    stderr: str = ""
    output: str = ""


class _FakeExecutor:
    """Fake ShellExecutor that returns canned responses keyed by command."""

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


class ExtractFailingFilesTest(unittest.TestCase):
    def test_extracts_cpp_files(self) -> None:
        error = (
            "/src/triton/lib/Foo.cpp:42:10: error: no member named 'getBar'\n"
            "/src/triton/lib/Baz.h:99: error: undeclared identifier\n"
        )
        files = extract_failing_files(error)
        self.assertEqual(files, ["/src/triton/lib/Foo.cpp", "/src/triton/lib/Baz.h"])

    def test_deduplicates_files(self) -> None:
        error = "/a/b.cpp:1:1: error: x\n/a/b.cpp:2:1: error: y\n/a/c.h:3:1: error: z\n"
        files = extract_failing_files(error)
        self.assertEqual(files, ["/a/b.cpp", "/a/c.h"])

    def test_empty_on_no_errors(self) -> None:
        self.assertEqual(extract_failing_files("all good"), [])

    def test_handles_cc_and_hpp(self) -> None:
        error = "/x/y.cc:10: error: foo\n/x/z.hpp:20: error: bar\n"
        files = extract_failing_files(error)
        self.assertEqual(files, ["/x/y.cc", "/x/z.hpp"])


class BuildFixContextTest(unittest.TestCase):
    def test_all_sections_present(self) -> None:
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(
                    success=True, stdout="diff --git a/llvm/include/foo.h"
                ),
                "git show": _FakeResult(
                    success=True, stdout="--- a/lib/Foo.cpp\n+++ b/lib/Foo.cpp"
                ),
            }
        )
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("/src/lib/Foo.cpp:42:10: error: no member named 'getBar'")
            error_log = Path(f.name)

        try:
            # pyre-ignore[6]: _FakeExecutor is not ShellExecutor
            context = build_fix_context(
                build_error_log=error_log,
                incompatible_llvm="abc123def456",
                llvm_bump_commit="def456abc123",
                triton_dir=Path("/fake/triton"),
                llvm_dir=Path("/fake/triton/llvm-project"),
                executor=executor,
            )

            self.assertIn("Section 1: LLVM API Change", context)
            self.assertIn("Section 2: Build Error Log", context)
            self.assertIn("Section 3: Reference Fix", context)
            self.assertIn(str(error_log), context)
        finally:
            error_log.unlink(missing_ok=True)

    def test_llvm_diff_appears_before_error_log(self) -> None:
        executor = _FakeExecutor(
            {
                "git diff": _FakeResult(success=True, stdout="diff content"),
            }
        )

        # pyre-ignore[6]: _FakeExecutor is not ShellExecutor
        context = build_fix_context(
            build_error_log=Path("/tmp/fake_error.log"),
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
            triton_dir=Path("/fake"),
            llvm_dir=Path("/fake/llvm-project"),
            executor=executor,
        )

        llvm_pos = context.find("Section 1: LLVM API Change")
        error_pos = context.find("Section 2: Build Error Log")
        self.assertLess(llvm_pos, error_pos)

    def test_handles_none_error_log(self) -> None:
        executor = _FakeExecutor()  # all commands fail

        # pyre-ignore[6]: _FakeExecutor is not ShellExecutor
        context = build_fix_context(
            build_error_log=None,
            incompatible_llvm="abc123",
            llvm_bump_commit="def456",
            triton_dir=Path("/fake"),
            llvm_dir=Path("/fake/llvm-project"),
            executor=executor,
        )

        self.assertIn("Section 1: LLVM API Change", context)
        self.assertIn("No build error log available", context)
        self.assertIn("failed to get LLVM diff", context)
