# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Pair tester for Triton/LLVM commit pairs.

This module provides the PairTester class which implements Phase 3 of the
Triton/LLVM bisect workflow. It tests (Triton, LLVM) commit pairs sequentially
to find the first failing pair.
"""

import csv
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.scripts import get_script_path


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

    def test_from_csv(
        self,
        csv_path: Path,
        test_args: Optional[str] = None,
        good_llvm: Optional[str] = None,
        bad_llvm: Optional[str] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> PairTestResult:
        """
        Test commit pairs from a CSV file.

        The CSV file should have two columns: triton_commit, llvm_commit.
        An optional header row is auto-detected and skipped.

        Args:
            csv_path: Path to the CSV file containing commit pairs.
            test_args: Additional arguments to pass to the test script.
            good_llvm: If provided, filter pairs to start from this LLVM commit.
            bad_llvm: If provided, filter pairs to end at this LLVM commit.
                      Both good_llvm and bad_llvm must be provided together.

        Returns:
            PairTestResult with testing results.

        Raises:
            PairTesterError: If CSV file is invalid or testing fails critically.
        """
        # Load and validate pairs
        pairs = self._load_pairs_from_csv(csv_path)

        if not pairs:
            raise PairTesterError(f"No valid commit pairs found in {csv_path}")

        self.logger.info(f"Loaded {len(pairs)} commit pairs from {csv_path}")

        # Initialize filter indices (will be set if filtering is requested)
        self._filter_start_idx = None
        self._filter_end_idx = None

        # Validate LLVM range if specified (does not filter the list)
        if good_llvm and bad_llvm:
            pairs = self._filter_pairs_by_llvm_range(pairs, good_llvm, bad_llvm)
            if not pairs:
                raise PairTesterError(
                    f"No pairs found in LLVM range [{good_llvm}, {bad_llvm}]"
                )
            self.logger.info(
                f"Filtered to {len(pairs)} pairs in LLVM range "
                f"[{good_llvm[:7]}..{bad_llvm[:7]}]"
            )

        # Run the test script
        return self._run_pair_test(
            pairs, csv_path, test_args, good_llvm, bad_llvm, output_callback
        )

    def test_pairs(
        self,
        pairs: List[Tuple[str, str]],
        test_args: Optional[str] = None,
    ) -> PairTestResult:
        """
        Test commit pairs from a list.

        Args:
            pairs: List of (triton_commit, llvm_commit) tuples.
            test_args: Additional arguments to pass to the test script.

        Returns:
            PairTestResult with testing results.

        Raises:
            PairTesterError: If testing fails critically.
        """
        if not pairs:
            raise PairTesterError("No commit pairs provided")

        # Convert to CommitPair objects
        commit_pairs = [
            CommitPair(triton_commit=p[0], llvm_commit=p[1], index=i)
            for i, p in enumerate(pairs)
        ]

        self.logger.info(f"Testing {len(commit_pairs)} commit pairs")

        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_csv:
            tmp_csv.write("triton_commit,llvm_commit\n")
            for pair in commit_pairs:
                tmp_csv.write(f"{pair.triton_commit},{pair.llvm_commit}\n")
            csv_path = Path(tmp_csv.name)

        try:
            return self._run_pair_test(commit_pairs, csv_path, test_args)
        finally:
            # Clean up temporary file
            csv_path.unlink(missing_ok=True)

    def _run_pair_test(
        self,
        pairs: List[CommitPair],
        csv_path: Path,
        test_args: Optional[str] = None,
        good_llvm: Optional[str] = None,
        bad_llvm: Optional[str] = None,
        output_callback: Optional[Callable[[str], None]] = None,
    ) -> PairTestResult:
        """
        Run the pair testing script.

        Args:
            pairs: List of commit pairs.
            csv_path: Path to the CSV file.
            test_args: Additional test arguments.
            good_llvm: Start LLVM commit for filtering (passed to script).
            bad_llvm: End LLVM commit for filtering (passed to script).

        Returns:
            PairTestResult with testing results.
        """
        self.logger.info("Starting pair testing")
        self.logger.info(f"  Triton dir: {self.triton_dir}")
        self.logger.info(f"  Test script: {self.test_script}")
        self.logger.info(f"  CSV file: {csv_path}")
        self.logger.info(f"  Total pairs: {len(pairs)}")

        # Get the embedded test script
        script_path = get_script_path("test_commit_pairs.sh")
        if not script_path.exists():
            raise PairTesterError("test_commit_pairs.sh script not found")

        # Set up environment variables
        env = {
            "TRITON_DIR": str(self.triton_dir),
            "TEST_SCRIPT": str(self.test_script),
            "COMMITS_CSV": str(csv_path),
            "CONDA_ENV": self.conda_env,
            "LOG_DIR": str(self.logger.log_dir),
            # Pass log file path to use consistent naming with BisectLogger
            "PAIR_TEST_LOG_FILE": str(
                self.logger.log_dir / f"{self.logger.session_name}_bisect.log"
            ),
        }

        # Pass LLVM filter range to script if specified
        if good_llvm:
            env["FILTER_GOOD_LLVM"] = good_llvm
        if bad_llvm:
            env["FILTER_BAD_LLVM"] = bad_llvm

        if test_args:
            env["TEST_ARGS"] = test_args

        if self.build_command:
            env["BUILD_COMMAND"] = self.build_command

        # Run the script
        result = self.executor.run_command_streaming(
            ["bash", str(script_path)],
            cwd=str(self.triton_dir),
            env=env,
            output_callback=output_callback,
        )

        # Parse the output
        return self._parse_test_output(result.stdout, result.exit_code, pairs)

    def _filter_pairs_by_llvm_range(
        self,
        pairs: List[CommitPair],
        good_llvm: str,
        bad_llvm: str,
    ) -> List[CommitPair]:
        """
        Validate that the LLVM range exists in the pairs list.

        This method validates the range but does NOT filter the list.
        The actual filtering is done by the shell script using environment
        variables FILTER_GOOD_LLVM and FILTER_BAD_LLVM.

        Args:
            pairs: List of all commit pairs.
            good_llvm: Start LLVM commit (inclusive).
            bad_llvm: End LLVM commit (inclusive).

        Returns:
            The original pairs list (unmodified).

        Raises:
            PairTesterError: If the range cannot be found in the pairs.
        """
        start_idx = None
        end_idx = None

        for i, pair in enumerate(pairs):
            # Match by prefix to allow short hashes
            if start_idx is None:
                if pair.llvm_commit.startswith(good_llvm) or good_llvm.startswith(
                    pair.llvm_commit
                ):
                    start_idx = i

            if pair.llvm_commit.startswith(bad_llvm) or bad_llvm.startswith(
                pair.llvm_commit
            ):
                end_idx = i
                # Don't break - continue to find the last match in case of duplicates

        if start_idx is None:
            raise PairTesterError(
                f"Could not find good_llvm '{good_llvm}' in CSV pairs"
            )

        if end_idx is None:
            raise PairTesterError(f"Could not find bad_llvm '{bad_llvm}' in CSV pairs")

        if start_idx > end_idx:
            raise PairTesterError(
                f"good_llvm '{good_llvm}' appears after bad_llvm '{bad_llvm}' in CSV. "
                f"Expected good_llvm to come before bad_llvm."
            )

        # Store the range for later use but return the original list
        self._filter_start_idx = start_idx
        self._filter_end_idx = end_idx

        # Calculate how many pairs are in the range
        range_count = end_idx - start_idx + 1
        self.logger.info(
            f"LLVM range validated: {range_count} pairs "
            f"(indices {start_idx} to {end_idx})"
        )

        # Return the original list - filtering is done by the script
        return pairs

    def _parse_test_output(
        self,
        output: str,
        exit_code: int,
        pairs: List[CommitPair],
    ) -> PairTestResult:
        """
        Parse the test script output to extract results.

        Args:
            output: Script output.
            exit_code: Script exit code.
            pairs: List of commit pairs (for reference).

        Returns:
            PairTestResult with parsed results.
        """
        total_pairs = len(pairs)

        # Exit code 0 can mean:
        # 1. All pairs passed
        # 2. Test failed and first failing pair was found
        # Exit code 1 means build failed or critical error

        if exit_code == 1:
            # Build failure or critical error
            error_match = re.search(
                r"Status: Build Failed.*?Position: Pair (\d+) of (\d+)",
                output,
                re.DOTALL,
            )
            if error_match:
                failing_index = int(error_match.group(1)) - 1  # Convert to 0-based
                return PairTestResult(
                    found_failing=False,
                    failing_index=failing_index,
                    total_pairs=total_pairs,
                    error_message=f"Build failed at pair {failing_index + 1}",
                )

            # Generic error
            return PairTestResult(
                found_failing=False,
                total_pairs=total_pairs,
                error_message="Pair testing failed with exit code 1",
            )

        # Exit code 0: Check if all passed or test failure found
        if "All Passed" in output or "All commit pairs tested successfully" in output:
            self.logger.info("All commit pairs passed")
            return PairTestResult(
                found_failing=False,
                total_pairs=total_pairs,
                all_passed=True,
            )

        # Look for test failure
        test_failed_match = re.search(
            r"TEST FAILED.*?Position: Pair (\d+) of (\d+)",
            output,
            re.DOTALL,
        )

        if test_failed_match:
            failing_index = int(test_failed_match.group(1)) - 1  # Convert to 0-based

            # Extract commit info
            triton_match = re.search(
                r"Triton Commit: ([a-fA-F0-9]+)",
                output[test_failed_match.start() :],
            )
            llvm_match = re.search(
                r"LLVM Commit: ([a-fA-F0-9]+)",
                output[test_failed_match.start() :],
            )

            triton_commit = triton_match.group(1) if triton_match else None
            bad_llvm = llvm_match.group(1) if llvm_match else None

            # Get good_llvm from previous pair
            good_llvm = None
            if failing_index > 0 and failing_index <= len(pairs):
                good_llvm = pairs[failing_index - 1].llvm_commit

            self.logger.info(f"Found first failing pair at index {failing_index}")
            if good_llvm and bad_llvm:
                self.logger.info(
                    f"  LLVM bisect range: {good_llvm[:7]} -> {bad_llvm[:7]}"
                )

            return PairTestResult(
                found_failing=True,
                failing_index=failing_index,
                good_llvm=good_llvm,
                bad_llvm=bad_llvm,
                triton_commit=triton_commit,
                total_pairs=total_pairs,
            )

        # Could not parse output - check result file as fallback
        return self._parse_result_file(output, total_pairs, pairs)

    def _parse_result_file(
        self,
        output: str,
        total_pairs: int,
        pairs: List[CommitPair],
    ) -> PairTestResult:
        """
        Fallback: Try to parse from result file if output parsing failed.

        Args:
            output: Script output (may contain result file path).
            total_pairs: Total number of pairs.
            pairs: List of commit pairs.

        Returns:
            PairTestResult with parsed results.
        """
        # Look for result file path in output
        result_file_match = re.search(r"Result file: (.+\.txt)", output)
        if not result_file_match:
            # Unable to parse
            self.logger.warning("Could not parse pair test output")
            return PairTestResult(
                found_failing=False,
                total_pairs=total_pairs,
                error_message="Could not parse test output",
            )

        result_file = Path(result_file_match.group(1))
        if not result_file.exists():
            return PairTestResult(
                found_failing=False,
                total_pairs=total_pairs,
                error_message=f"Result file not found: {result_file}",
            )

        try:
            content = result_file.read_text()
            lines = content.strip().split("\n")

            if len(lines) < 2:
                return PairTestResult(
                    found_failing=False,
                    total_pairs=total_pairs,
                    error_message="Invalid result file format",
                )

            # First line is triton_commit,llvm_commit
            first_line = lines[0].split(",")
            triton_commit = first_line[0].strip()
            bad_llvm = first_line[1].strip() if len(first_line) > 1 else None

            # Look for position
            position_match = re.search(r"Position: Pair (\d+) of (\d+)", content)
            if position_match:
                failing_index = int(position_match.group(1)) - 1

                good_llvm = None
                if failing_index > 0 and failing_index <= len(pairs):
                    good_llvm = pairs[failing_index - 1].llvm_commit

                return PairTestResult(
                    found_failing=True,
                    failing_index=failing_index,
                    good_llvm=good_llvm,
                    bad_llvm=bad_llvm,
                    triton_commit=triton_commit,
                    total_pairs=total_pairs,
                )

        except Exception as e:
            self.logger.warning(f"Failed to parse result file: {e}")

        return PairTestResult(
            found_failing=False,
            total_pairs=total_pairs,
            error_message="Could not parse result file",
        )
