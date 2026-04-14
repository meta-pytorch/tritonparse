# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
Build structured LLM context for AI-powered compatibility fixing.

Converts build errors, LLVM API changes, and reference fixes into structured
text suitable for LLM consumption. The context is organized in priority order:

1. Build Error (highest — what's broken)
2. LLVM API Change (what changed in LLVM headers)
3. Reference Fix from llvm_bump (how it was eventually fixed)
4. Current Triton Source (files that failed to compile)
"""

from __future__ import annotations

import re
from pathlib import Path

from tritonparse.ai.utils import truncate_context
from tritonparse.bisect.executor import ShellExecutor


def build_fix_context(
    build_error: str,
    incompatible_llvm: str,
    llvm_bump_commit: str,
    triton_dir: Path,
    llvm_dir: Path,
    executor: ShellExecutor,
    max_total_chars: int = 80000,
) -> str:
    """Build structured context for the AI fixer.

    Organized by priority (most important first), following CUTracer's
    build_llm_context() pattern.

    Args:
        build_error: Triton build error output.
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

    # Section 1: Build Error (highest priority)
    section1 = _build_error_section(build_error, budget // 4)
    sections.append(section1)
    budget -= len(section1)

    # Section 2: LLVM API Change
    section2 = _llvm_change_section(incompatible_llvm, llvm_dir, executor, budget // 3)
    sections.append(section2)
    budget -= len(section2)

    # Section 3: Reference Fix from llvm_bump
    section3 = _reference_fix_section(
        build_error, llvm_bump_commit, triton_dir, executor, budget // 2
    )
    sections.append(section3)
    budget -= len(section3)

    # Section 4: Current Triton Source (failing files)
    section4 = _failing_source_section(build_error, triton_dir, executor, budget)
    if section4:
        sections.append(section4)

    return "\n\n".join(sections)


def _build_error_section(build_error: str, max_chars: int) -> str:
    """Section 1: Build error output."""
    truncated = truncate_context(build_error, max_chars, strategy="tail")
    return (
        "## Section 1: Build Error\n\n"
        "The following Triton build error occurred when compiling against "
        "the incompatible LLVM commit:\n\n"
        f"```\n{truncated}\n```"
    )


def _llvm_change_section(
    incompatible_llvm: str,
    llvm_dir: Path,
    executor: ShellExecutor,
    max_chars: int,
) -> str:
    """Section 2: LLVM API change diff (headers only)."""
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
        "## Section 2: LLVM API Change\n\n"
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
        "## Section 3: Reference Fix from llvm_bump Commit\n\n"
        f"The llvm_bump commit `{llvm_bump_commit[:12]}` contains these fixes "
        "for the same files. Use as reference (may need adaptation):\n\n"
        f"```diff\n{truncated}\n```"
    )


def _failing_source_section(
    build_error: str,
    triton_dir: Path,
    executor: ShellExecutor,
    max_chars: int,
) -> str:
    """Section 4: Current source of failing files."""
    failing_files = extract_failing_files(build_error)

    if not failing_files:
        return ""

    sources: list[str] = []
    for f in failing_files[:3]:
        result = executor.run_command(
            ["cat", f],
            cwd=str(triton_dir),
        )
        if result.success:
            sources.append(f"### {f}\n\n```cpp\n{result.stdout}\n```")

    if not sources:
        return ""

    combined = "\n\n".join(sources)
    truncated = truncate_context(combined, max_chars, strategy="head")

    return f"## Section 4: Current Triton Source\n\n{truncated}"


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
        if path not in seen:
            seen.add(path)
            files.append(path)
    return files
