# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Build structured LLM context for AI-powered compatibility fixing.

Converts LLVM API changes, build error log references, and reference fixes into
structured text suitable for LLM consumption. The context is organized by
causality:

1. LLVM API Change (the cause — what changed in LLVM headers)
2. Build Error Log (the symptom — pointer to log file for AI to read)
3. Reference Fix from llvm_bump (guidance — how it was eventually fixed)
"""

from __future__ import annotations

import re
from pathlib import Path

from tritonparse.ai.utils import truncate_context
from tritonparse.bisect.executor import ShellExecutor


def build_fix_context(
    build_error_log: Path | None,
    incompatible_llvm: str,
    llvm_bump_commit: str,
    triton_dir: Path,
    llvm_dir: Path,
    executor: ShellExecutor,
    max_total_chars: int = 80000,
) -> str:
    """Build structured context for the AI fixer.

    Organized by causality — LLVM diff first (cause), build error log
    path second (symptom), reference fix third (guidance).
    The build error is NOT embedded — the AI reads the full log file
    directly, preserving note/candidate lines and ordering.

    Args:
        build_error_log: Path to the raw build error log file, or None.
        incompatible_llvm: First incompatible LLVM commit hash.
        llvm_bump_commit: The LLVM bump commit in Triton repo.
        triton_dir: Path to Triton repository (compat worktree).
        llvm_dir: Path to LLVM repository (llvm-project).
        executor: ShellExecutor for running git commands.
        max_total_chars: Maximum total context size in characters.

    Returns:
        Structured markdown context string.
    """
    sections: list[str] = []
    budget = max_total_chars

    # Section 1: LLVM API Change (PRIMARY — the cause)
    section1 = _llvm_change_section(incompatible_llvm, llvm_dir, executor, budget // 3)
    sections.append(section1)
    budget -= len(section1)

    # Section 2: Build Error Log (pointer, not content)
    section2 = _build_error_log_section(build_error_log)
    sections.append(section2)

    # Section 3: Reference Fix from llvm_bump
    error_text = _read_error_log(build_error_log)
    section3 = _reference_fix_section(
        error_text, llvm_bump_commit, triton_dir, executor, budget // 2
    )
    sections.append(section3)

    return "\n\n".join(sections)


def _build_error_log_section(error_log: Path | None) -> str:
    """Section 2: Point the AI to the full build error log file."""
    if error_log is None:
        return "## Section 2: Build Error Log\n\nNo build error log available."
    return (
        "## Section 2: Build Error Log\n\n"
        f"Read the build error log at: `{error_log}`\n\n"
        "This file contains the complete, untruncated compiler output — "
        "including `note:` and `candidate:` lines that show the new API "
        "signatures. Read it to confirm which APIs break the build, then "
        "use grep to find additional call sites the error may not cover."
    )


def _read_error_log(error_log: Path | None) -> str:
    """Read error log file contents, returning empty string if unavailable."""
    if error_log is None or not error_log.exists():
        return ""
    try:
        return error_log.read_text()
    except OSError:
        return ""


def _llvm_change_section(
    incompatible_llvm: str,
    llvm_dir: Path,
    executor: ShellExecutor,
    max_chars: int,
) -> str:
    """Section 1: LLVM API change diff (headers only)."""
    result = executor.run_command(
        [
            "git",
            "diff",
            f"{incompatible_llvm}~1",
            incompatible_llvm,
            "--",
            "llvm/include/",
            "mlir/include/",
        ],
        cwd=str(llvm_dir),
    )

    diff_text = result.stdout if result.success else "(failed to get LLVM diff)"
    truncated = truncate_context(diff_text, max_chars, strategy="head")

    return (
        "## Section 1: LLVM API Change\n\n"
        f"Diff of LLVM headers at commit `{incompatible_llvm[:12]}`:\n\n"
        f"```diff\n{truncated}\n```"
    )


def _reference_fix_section(
    build_error: str,
    llvm_bump_commit: str,
    triton_dir: Path,
    executor: ShellExecutor,
    max_chars: int,
) -> str:
    """Section 3: How the llvm_bump commit fixed the same files."""
    failing_files = extract_failing_files(build_error)

    if not failing_files:
        return (
            "## Section 3: Reference Fix\n\n"
            "(Could not extract failing file paths from build error)"
        )

    diffs: list[str] = []
    for f in failing_files[:5]:
        result = executor.run_command(
            ["git", "show", llvm_bump_commit, "--", f],
            cwd=str(triton_dir),
        )
        if result.success and result.stdout.strip():
            diffs.append(result.stdout)

    if not diffs:
        return (
            "## Section 3: Reference Fix\n\n"
            "(No matching changes found in llvm_bump commit for failing files)"
        )

    combined = "\n\n".join(diffs)
    truncated = truncate_context(combined, max_chars, strategy="head")

    return (
        "## Section 3: Reference Fix (guidance only)\n\n"
        "A later commit eventually fixed these same files. These changes are "
        "**NOT applied to your working directory** — the files on disk still "
        "have the old broken code. Apply the relevant subset of changes to "
        "the files in your working directory.\n\n"
        "Note: This diff may include fixes for multiple unrelated breakpoints. "
        "Only apply changes that fix the build errors you see in the error "
        "log — ignore changes that address errors you don't see.\n\n"
        f"```diff\n{truncated}\n```"
    )


def extract_failing_files(build_error: str) -> list[str]:
    """Extract file paths from C++ build errors.

    Looks for patterns like:
    - /path/to/file.cpp:123:45: error: ...
    - /path/to/file.h:67: error: ...

    Args:
        build_error: Build error output text.

    Returns:
        Deduplicated list of file paths preserving order.
    """
    pattern = r"(/[^\s:]+\.(?:cpp|h|cc|hpp)):(\d+)"
    matches = re.findall(pattern, build_error)
    seen: set[str] = set()
    files: list[str] = []
    for path, _ in matches:
        if path not in seen and "/llvm-project/" not in path:
            seen.add(path)
            files.append(path)
    return files
