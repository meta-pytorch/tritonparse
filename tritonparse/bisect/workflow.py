# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Main workflow controller for bisect operations.

This module provides the BisectWorkflow class which orchestrates the complete
4-phase Triton/LLVM bisect workflow with automatic LLVM bump detection and
state persistence for checkpoint/resume functionality.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from tritonparse.bisect.commit_detector import CommitDetector
from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.llvm_bisector import LLVMBisector
from tritonparse.bisect.logger import BisectLogger
from tritonparse.bisect.pair_tester import PairTester
from tritonparse.bisect.state import BisectPhase, BisectState, StateManager
from tritonparse.bisect.triton_bisector import TritonBisector
from tritonparse.bisect.ui import BisectUI


class BisectWorkflowError(Exception):
    """Exception raised for workflow errors."""

    pass


class BisectWorkflow:
    """
    Main workflow controller for Triton/LLVM bisect.

    Orchestrates the complete 4-phase workflow:
    1. Triton bisect: Find culprit Triton commit
    2. Type check: Detect if culprit is an LLVM bump
    3. Pair test: Test commit pairs to find LLVM range (if LLVM bump)
    4. LLVM bisect: Find culprit LLVM commit (if LLVM bump)

    The workflow supports checkpoint/resume through state persistence.

    Example:
        >>> # Full workflow
        >>> workflow = BisectWorkflow(
        ...     triton_dir="/path/to/triton",
        ...     test_script="/path/to/test.py",
        ...     good_commit="v2.0.0",
        ...     bad_commit="HEAD",
        ...     commits_csv="/path/to/commits.csv",
        ... )
        >>> result = workflow.run()
        >>> print(result["triton_culprit"])
        >>> if result["is_llvm_bump"]:
        ...     print(result["llvm_culprit"])

        >>> # Resume from saved state
        >>> workflow = BisectWorkflow.resume("./bisect_logs/state.json")
        >>> result = workflow.run()
    """

    def __init__(
        self,
        triton_dir: str,
        test_script: str,
        good_commit: str,
        bad_commit: str,
        commits_csv: Optional[str] = None,
        conda_env: str = "triton_bisect",
        log_dir: str = "./bisect_logs",
        build_command: Optional[str] = None,
    ) -> None:
        """
        Initialize the bisect workflow.

        Args:
            triton_dir: Path to the Triton repository.
            test_script: Path to the test script.
            good_commit: Known good Triton commit.
            bad_commit: Known bad Triton commit.
            commits_csv: Path to CSV file with commit pairs (for full workflow).
            conda_env: Conda environment name.
            log_dir: Directory for log files.
            build_command: Custom build command (optional).
        """
        self.logger = BisectLogger(log_dir)
        self.executor = ShellExecutor(self.logger)

        self.state = BisectState(
            triton_dir=str(Path(triton_dir).resolve()),
            test_script=str(Path(test_script).resolve()),
            good_commit=good_commit,
            bad_commit=bad_commit,
            commits_csv=str(Path(commits_csv).resolve()) if commits_csv else None,
            conda_env=conda_env,
            log_dir=str(Path(log_dir).resolve()),
            build_command=build_command,
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete bisect workflow.

        Runs through all phases sequentially, starting from the current phase.
        State is saved after each phase completes.

        Returns:
            Dictionary containing workflow results:
            - status: Final phase status
            - triton_culprit: Culprit Triton commit
            - is_llvm_bump: Whether culprit is an LLVM bump
            - llvm_culprit: Culprit LLVM commit (if applicable)
            - llvm_range: LLVM commit range (if applicable)

        Raises:
            BisectWorkflowError: If workflow fails.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Triton/LLVM Bisect Workflow")
        self.logger.info("=" * 60)

        try:
            # Phase 1: Triton Bisect
            if self.state.phase == BisectPhase.TRITON_BISECT:
                self._run_triton_bisect()

            # Phase 2: Type Check
            if self.state.phase == BisectPhase.TYPE_CHECK:
                self._run_type_check()

            # Phase 3: Pair Test (if LLVM bump)
            if self.state.phase == BisectPhase.PAIR_TEST:
                self._run_pair_test()

            # Phase 4: LLVM Bisect (if needed)
            if self.state.phase == BisectPhase.LLVM_BISECT:
                self._run_llvm_bisect()

            # Generate and return report
            return self.state.to_report()

        except Exception as e:
            self._handle_failure(str(e))
            raise BisectWorkflowError(f"Workflow failed: {e}") from e

    def run_triton_only(self) -> str:
        """
        Execute Triton bisect only.

        This is a shortcut for running just Phase 1 without the full workflow.

        Returns:
            Culprit Triton commit hash.
        """
        self.logger.info("Running Triton bisect only")

        bisector = TritonBisector(
            triton_dir=self.state.triton_dir,
            test_script=self.state.test_script,
            conda_env=self.state.conda_env,
            logger=self.logger,
            build_command=self.state.build_command,
        )

        culprit = bisector.run(
            good_commit=self.state.good_commit,
            bad_commit=self.state.bad_commit,
        )

        self.state.triton_culprit = culprit
        self.state.phase = BisectPhase.COMPLETED
        self._save_state()

        return culprit

    def run_llvm_only(
        self,
        triton_commit: str,
        good_llvm: str,
        bad_llvm: str,
    ) -> str:
        """
        Execute LLVM bisect only.

        This is a shortcut for running just Phase 4 without the full workflow.

        Args:
            triton_commit: Triton commit to use (or None for current HEAD).
            good_llvm: Known good LLVM commit.
            bad_llvm: Known bad LLVM commit.

        Returns:
            Culprit LLVM commit hash.
        """
        self.logger.info("Running LLVM bisect only")

        bisector = LLVMBisector(
            triton_dir=self.state.triton_dir,
            test_script=self.state.test_script,
            conda_env=self.state.conda_env,
            logger=self.logger,
            build_command=self.state.build_command,
        )

        culprit = bisector.run(
            triton_commit=triton_commit,
            good_llvm=good_llvm,
            bad_llvm=bad_llvm,
        )

        self.state.llvm_culprit = culprit
        self.state.phase = BisectPhase.COMPLETED
        self._save_state()

        return culprit

    @classmethod
    def resume(cls, state_path: str) -> "BisectWorkflow":
        """
        Resume workflow from saved state.

        Args:
            state_path: Path to saved state file.

        Returns:
            BisectWorkflow instance with restored state.

        Raises:
            FileNotFoundError: If state file doesn't exist.
        """
        state = StateManager.load(state_path)

        workflow = cls(
            triton_dir=state.triton_dir,
            test_script=state.test_script,
            good_commit=state.good_commit,
            bad_commit=state.bad_commit,
            commits_csv=state.commits_csv,
            conda_env=state.conda_env,
            log_dir=state.log_dir,
            build_command=state.build_command,
        )

        # Restore full state (including results)
        workflow.state = state

        workflow.logger.info(f"Resumed from state: {state_path}")
        workflow.logger.info(f"Current phase: {state.phase.value}")

        return workflow

    def _run_triton_bisect(self) -> None:
        """Execute Phase 1: Triton Bisect."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Phase 1: Triton Bisect")
        self.logger.info("=" * 60)

        bisector = TritonBisector(
            triton_dir=self.state.triton_dir,
            test_script=self.state.test_script,
            conda_env=self.state.conda_env,
            logger=self.logger,
            build_command=self.state.build_command,
        )

        self.state.triton_culprit = bisector.run(
            good_commit=self.state.good_commit,
            bad_commit=self.state.bad_commit,
        )

        self.logger.info(f"Triton culprit found: {self.state.triton_culprit}")

        self.state.phase = BisectPhase.TYPE_CHECK
        self._save_state()

    def _run_type_check(self) -> None:
        """Execute Phase 2: Commit Type Detection."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Phase 2: Commit Type Detection")
        self.logger.info("=" * 60)

        detector = CommitDetector(
            triton_dir=Path(self.state.triton_dir),
            executor=self.executor,
            logger=self.logger,
        )

        bump_info = detector.detect(self.state.triton_culprit)

        self.state.is_llvm_bump = bump_info.is_llvm_bump
        self.state.old_llvm_hash = bump_info.old_hash
        self.state.new_llvm_hash = bump_info.new_hash

        if not bump_info.is_llvm_bump:
            self.logger.info("Commit is NOT an LLVM bump. Workflow complete.")
            self.state.phase = BisectPhase.COMPLETED
            self._save_state()
            return

        self.logger.info("Commit IS an LLVM bump. Proceeding to pair testing.")

        # Check if commits_csv is provided
        if not self.state.commits_csv:
            raise BisectWorkflowError(
                "LLVM bump detected but no commits CSV provided. "
                "Use --commits-csv to specify the compatibility CSV file, "
                "or run with --llvm-only to bisect LLVM directly."
            )

        self.state.phase = BisectPhase.PAIR_TEST
        self._save_state()

    def _run_pair_test(self) -> None:
        """Execute Phase 3: Pair Testing."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Phase 3: Pair Testing")
        self.logger.info("=" * 60)

        tester = PairTester(
            triton_dir=Path(self.state.triton_dir),
            test_script=Path(self.state.test_script),
            executor=self.executor,
            logger=self.logger,
            conda_env=self.state.conda_env,
            build_command=self.state.build_command,
        )

        result = tester.test_from_csv(csv_path=Path(self.state.commits_csv))

        if result.error_message and not result.found_failing:
            raise BisectWorkflowError(f"Pair testing failed: {result.error_message}")

        if result.all_passed:
            self.logger.info("All pairs passed. No LLVM regression found.")
            self.state.phase = BisectPhase.COMPLETED
            self._save_state()
            return

        if not result.found_failing:
            raise BisectWorkflowError("Pair testing did not find a failing pair.")

        self.state.failing_pair_index = result.failing_index
        self.state.good_llvm = result.good_llvm
        self.state.bad_llvm = result.bad_llvm
        self.state.triton_commit_for_llvm = result.triton_commit

        self.logger.info(f"First failing pair found at index {result.failing_index}")
        self.logger.info(f"LLVM range: {result.good_llvm} -> {result.bad_llvm}")

        self.state.phase = BisectPhase.LLVM_BISECT
        self._save_state()

    def _run_llvm_bisect(self) -> None:
        """Execute Phase 4: LLVM Bisect."""
        self.logger.info("")
        self.logger.info("=" * 60)
        self.logger.info("Phase 4: LLVM Bisect")
        self.logger.info("=" * 60)

        bisector = LLVMBisector(
            triton_dir=self.state.triton_dir,
            test_script=self.state.test_script,
            conda_env=self.state.conda_env,
            logger=self.logger,
            build_command=self.state.build_command,
        )

        # Use the Triton commit from pair testing, or fall back to culprit
        triton_commit = (
            self.state.triton_commit_for_llvm or self.state.triton_culprit
        )

        self.state.llvm_culprit = bisector.run(
            triton_commit=triton_commit,
            good_llvm=self.state.good_llvm,
            bad_llvm=self.state.bad_llvm,
        )

        self.logger.info(f"LLVM culprit found: {self.state.llvm_culprit}")

        self.state.phase = BisectPhase.COMPLETED
        self._save_state()

    def _save_state(self) -> None:
        """Save current state to file using logger's session_name."""
        state_path = StateManager.save(self.state, session_name=self.logger.session_name)
        self.logger.debug(f"State saved to: {state_path}")

    def _handle_failure(self, error: str) -> None:
        """Handle workflow failure."""
        self.state.phase = BisectPhase.FAILED
        self.state.error_message = error
        self._save_state()

        self.logger.error(f"Bisect failed: {error}")
        self.logger.info("")
        self.logger.info("State has been saved. To resume, run:")
        state_path = StateManager.get_state_path(
            self.state.log_dir, self.logger.session_name
        )
        self.logger.info(f"  tritonparse bisect --resume --state {state_path}")

        # Cleanup: reset git bisect state
        self._cleanup_on_failure()

    def _cleanup_on_failure(self) -> None:
        """Clean up git bisect state on failure."""
        # Reset Triton repo bisect state
        self.executor.run_command(
            ["git", "bisect", "reset"],
            cwd=self.state.triton_dir,
        )

        # Reset LLVM repo bisect state (if it exists)
        llvm_dir = Path(self.state.triton_dir) / "llvm-project"
        if llvm_dir.exists():
            self.executor.run_command(
                ["git", "bisect", "reset"],
                cwd=str(llvm_dir),
            )
