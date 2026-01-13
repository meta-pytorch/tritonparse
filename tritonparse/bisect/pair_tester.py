# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pair tester for Triton/LLVM commit pairs.

This module provides the PairTester class which implements Phase 3 of the
Triton/LLVM bisect workflow. It tests (Triton, LLVM) commit pairs sequentially
to find the first failing pair.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class PairTesterError(Exception):
    """Exception raised for pair testing errors."""

    pass


@dataclass
class CommitPair:
    """
    A (Triton, LLVM) commit pair.

    Attributes:
        triton_commit: The Triton commit hash.
        llvm_commit: The LLVM commit hash.
        index: The pair index (0-based) in the CSV file.
    """

    triton_commit: str
    llvm_commit: str
    index: int


@dataclass
class PairTestResult:
    """
    Result of pair testing.

    Attributes:
        found_failing: Whether a failing pair was found.
        failing_index: Index of the first failing pair (0-based), or -1 if none.
        good_llvm: LLVM commit from the last passing pair (for bisect).
        bad_llvm: LLVM commit from the first failing pair (for bisect).
        triton_commit: Triton commit of the failing pair.
        total_pairs: Total number of pairs tested.
        all_passed: True if all pairs passed (no failing pair found).
        error_message: Error message if testing failed.
    """

    found_failing: bool
    failing_index: int = -1
    good_llvm: Optional[str] = None
    bad_llvm: Optional[str] = None
    triton_commit: Optional[str] = None
    total_pairs: int = 0
    all_passed: bool = False
    error_message: Optional[str] = None


class PairTester:
    """
    Tests (Triton, LLVM) commit pairs to find the first failing pair.

    This class implements Phase 3 of the bisect workflow - testing commit pairs
    sequentially to find the first pair that causes test failure. This is used
    when Triton bisect identifies an LLVM bump commit.

    Example:
        >>> logger = BisectLogger("./logs")
        >>> executor = ShellExecutor(logger)
        >>> tester = PairTester(
        ...     triton_dir=Path("/path/to/triton"),
        ...     test_script=Path("/path/to/test.py"),
        ...     executor=executor,
        ...     logger=logger,
        ... )
        >>> result = tester.test_from_csv(csv_path=Path("commits.csv"))
        >>> if result.found_failing:
        ...     print(f"First failing pair at index {result.failing_index}")
        ...     print(f"LLVM bisect range: {result.good_llvm} -> {result.bad_llvm}")
    """

    def __init__(
        self,
        triton_dir: Path,
        test_script: Path,
        executor: ShellExecutor,
        logger: BisectLogger,
        conda_env: str = "triton_bisect",
        build_command: Optional[str] = None,
    ) -> None:
        """
        Initialize the pair tester.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script.
            executor: ShellExecutor instance for running commands.
            logger: BisectLogger instance for logging.
            conda_env: Conda environment name for testing.
            build_command: Custom build command template. Use {TRITON_COMMIT}
                and {LLVM_COMMIT} as placeholders.
        """
        self.triton_dir = triton_dir
        self.test_script = test_script
        self.executor = executor
        self.logger = logger
        self.conda_env = conda_env
        self.build_command = build_command

    def _load_pairs_from_csv(self, csv_path: Path) -> List[CommitPair]:
        """
        Load commit pairs from a CSV file.

        The CSV file format:
        - Two columns: triton_commit, llvm_commit
        - Optional header row (auto-detected and skipped)
        - Empty lines are ignored
        - Comment lines starting with # are ignored

        Args:
            csv_path: Path to the CSV file.

        Returns:
            List of CommitPair objects.

        Raises:
            PairTesterError: If CSV file cannot be read.
        """
        if not csv_path.exists():
            raise PairTesterError(f"CSV file not found: {csv_path}")

        pairs: List[CommitPair] = []
        index = 0

        try:
            with open(csv_path, "r") as f:
                # Read all lines and filter out comments for header detection
                all_lines = f.readlines()
                non_comment_lines = [
                    line for line in all_lines if not line.strip().startswith("#")
                ]

                # Use first few non-comment lines for header detection
                sample = "".join(non_comment_lines[:10])
                has_header = False
                if sample.strip():
                    try:
                        has_header = csv.Sniffer().has_header(sample)
                    except csv.Error:
                        # Sniffer failed, assume no header
                        has_header = False

                # Reset and read with csv reader
                f.seek(0)
                reader = csv.reader(f)

                header_skipped = False

                for row in reader:
                    if len(row) < 2:
                        continue

                    triton_commit = row[0].strip().strip('"')
                    llvm_commit = row[1].strip().strip('"')

                    # Skip empty rows
                    if not triton_commit or not llvm_commit:
                        continue

                    # Skip comment lines (starting with #)
                    if triton_commit.startswith("#"):
                        continue

                    # Skip header row (only once)
                    if triton_commit.lower() in ("triton", "triton_commit"):
                        continue

                    # Skip detected header (first non-comment data row if has_header)
                    if has_header and not header_skipped:
                        header_skipped = True
                        continue

                    pairs.append(
                        CommitPair(
                            triton_commit=triton_commit,
                            llvm_commit=llvm_commit,
                            index=index,
                        )
                    )
                    index += 1

        except Exception as e:
            raise PairTesterError(f"Failed to read CSV file {csv_path}: {e}")

        return pairs
