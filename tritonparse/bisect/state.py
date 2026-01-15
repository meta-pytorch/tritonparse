# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
State management for bisect workflow.

This module provides state persistence for the bisect workflow, enabling
checkpoint/resume functionality. The state is saved as JSON and can be
loaded to continue from where the workflow left off.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class BisectPhase(Enum):
    """
    Bisect workflow phases.

    The workflow progresses through these phases sequentially:
    1. TRITON_BISECT: Find culprit Triton commit
    2. TYPE_CHECK: Detect if culprit is an LLVM bump
    3. PAIR_TEST: Test commit pairs to find LLVM range (if LLVM bump)
    4. LLVM_BISECT: Find culprit LLVM commit (if LLVM bump)
    5. COMPLETED: Workflow finished successfully
    6. FAILED: Workflow failed with error
    """

    TRITON_BISECT = "triton_bisect"
    TYPE_CHECK = "type_check"
    PAIR_TEST = "pair_test"
    LLVM_BISECT = "llvm_bisect"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BisectState:
    """
    Complete bisect workflow state.

    This dataclass holds all configuration and progress information needed
    to run or resume a bisect workflow.

    Attributes:
        triton_dir: Path to the Triton repository.
        test_script: Path to the test script.
        good_commit: Known good Triton commit.
        bad_commit: Known bad Triton commit.
        commits_csv: Path to CSV file with commit pairs (for full workflow).
        conda_env: Conda environment name.
        log_dir: Directory for log files.
        build_command: Custom build command (optional).
        phase: Current workflow phase.
        started_at: ISO timestamp when workflow started.
        updated_at: ISO timestamp of last state update.
        triton_culprit: Culprit Triton commit (Phase 1 result).
        is_llvm_bump: Whether culprit is an LLVM bump (Phase 2 result).
        old_llvm_hash: Old LLVM hash before bump (if LLVM bump).
        new_llvm_hash: New LLVM hash after bump (if LLVM bump).
        failing_pair_index: Index of first failing pair (Phase 3 result).
        good_llvm: Good LLVM commit for bisect (Phase 3 result).
        bad_llvm: Bad LLVM commit for bisect (Phase 3 result).
        triton_commit_for_llvm: Triton commit to use for LLVM bisect.
        llvm_culprit: Culprit LLVM commit (Phase 4 result).
        error_message: Error message if workflow failed.
    """

    # Configuration
    triton_dir: str
    test_script: str
    good_commit: str
    bad_commit: str
    commits_csv: Optional[str] = None
    conda_env: str = "triton_bisect"
    log_dir: str = "./bisect_logs"
    build_command: Optional[str] = None
    session_name: Optional[str] = None  # Links state file to log files

    # Progress
    phase: BisectPhase = BisectPhase.TRITON_BISECT
    started_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Phase 1 results (Triton bisect)
    triton_culprit: Optional[str] = None

    # Phase 2 results (Type check)
    is_llvm_bump: Optional[bool] = None
    old_llvm_hash: Optional[str] = None
    new_llvm_hash: Optional[str] = None

    # Phase 3 results (Pair test)
    failing_pair_index: Optional[int] = None
    good_llvm: Optional[str] = None
    bad_llvm: Optional[str] = None
    triton_commit_for_llvm: Optional[str] = None

    # Phase 4 results (LLVM bisect)
    llvm_culprit: Optional[str] = None

    # Error handling
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        data = asdict(self)
        data["phase"] = self.phase.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BisectState":
        """Create state from dictionary."""
        data = data.copy()
        data["phase"] = BisectPhase(data["phase"])
        return cls(**data)

    # ========== Persistence Methods ==========

    def save(
        self, path: Optional[Path] = None, session_name: Optional[str] = None
    ) -> Path:
        """
        Save state to JSON file.

        Updates timestamps before saving. This is a convenience method that
        delegates to StateManager.save().

        Args:
            path: Optional explicit path. If provided, overrides session_name.
            session_name: Session identifier for file naming.

        Returns:
            Path where state was saved.
        """
        return StateManager.save(
            self, session_name=session_name, path=str(path) if path else None
        )

    @classmethod
    def load(cls, path: Path) -> "BisectState":
        """
        Load state from JSON file.

        Args:
            path: Path to state file.

        Returns:
            Loaded BisectState instance.

        Raises:
            FileNotFoundError: If state file doesn't exist.
        """
        return StateManager.load(str(path))

    @classmethod
    def load_or_create(
        cls,
        log_dir: str,
        session_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "BisectState":
        """
        Load existing state or create new one.

        If a state file exists (matching session_name or most recent), loads it.
        Otherwise, creates a new state with the provided arguments.

        Args:
            log_dir: Log directory path.
            session_name: Optional session identifier. If provided, looks for
                         that specific state file.
            **kwargs: Arguments for creating new state (triton_dir, test_script, etc.)

        Returns:
            BisectState instance (loaded or newly created).
        """
        state_path = None

        if session_name:
            # Look for specific session state
            state_path = StateManager.get_state_path(log_dir, session_name)
            if not state_path.exists():
                state_path = None
        else:
            # Look for most recent state
            state_path = StateManager.find_latest_state(log_dir)

        if state_path and state_path.exists():
            return cls.load(state_path)

        return cls(log_dir=log_dir, session_name=session_name, **kwargs)

    # ========== Report Generation ==========

    def to_report(self) -> Dict[str, Any]:
        """
        Generate a report dictionary for final output.

        Returns:
            Dictionary containing workflow results suitable for display.
        """
        report: Dict[str, Any] = {
            "status": self.phase.value,
            "triton_culprit": self.triton_culprit,
            "is_llvm_bump": self.is_llvm_bump,
        }

        if self.is_llvm_bump:
            report["llvm_culprit"] = self.llvm_culprit
            # Original LLVM bump info (from Type Check phase)
            report["llvm_bump"] = {
                "old": self.old_llvm_hash,
                "new": self.new_llvm_hash,
            }
            # LLVM bisect range (from Pair Test phase)
            report["llvm_range"] = {
                "good": self.good_llvm,
                "bad": self.bad_llvm,
            }
            report["failing_pair_index"] = self.failing_pair_index
            report["triton_commit_for_llvm"] = self.triton_commit_for_llvm

        if self.error_message:
            report["error"] = self.error_message

        return report


class StateManager:
    """
    Manages bisect state persistence.

    Provides methods to save, load, and display bisect state.

    State files are named with a session_name (typically a timestamp) to
    correlate with log files from the same run:
    - Log files: {session_name}_bisect.log, {session_name}_bisect_commands.log
    - State file: {session_name}_state.json

    Example:
        >>> state = BisectState(
        ...     triton_dir="/path/to/triton",
        ...     test_script="/path/to/test.py",
        ...     good_commit="v2.0.0",
        ...     bad_commit="HEAD",
        ... )
        >>> # Save with session name (correlates with logs)
        >>> path = StateManager.save(state, session_name="20251212_120643")
        >>> # Result: {log_dir}/20251212_120643_state.json
        >>>
        >>> # Load from file
        >>> loaded = StateManager.load(str(path))
        >>> StateManager.print_status(loaded)
    """

    STATE_SUFFIX = "_state.json"

    @staticmethod
    def get_state_path(log_dir: str, session_name: str) -> Path:
        """
        Get state file path for a session.

        Args:
            log_dir: Log directory path.
            session_name: Session identifier (typically timestamp like "20251212_120643").

        Returns:
            Path to {session_name}_state.json in the log directory.
        """
        return Path(log_dir) / f"{session_name}{StateManager.STATE_SUFFIX}"

    @staticmethod
    def save(
        state: BisectState,
        session_name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> Path:
        """
        Save state to JSON file.

        Updates the timestamps before saving. The session_name is stored in
        the state for later reference.

        Args:
            state: BisectState to save.
            session_name: Session identifier for file naming. If not provided,
                         uses state.session_name or generates a timestamp.
            path: Explicit file path. If provided, overrides session_name.

        Returns:
            Path where state was saved.
        """
        # Update timestamps
        now = datetime.now().isoformat()
        state.updated_at = now
        if state.started_at is None:
            state.started_at = now

        # Determine path
        if path is not None:
            save_path = Path(path)
        else:
            # Use provided session_name, or state's session_name, or generate one
            if session_name is None:
                session_name = getattr(state, "session_name", None)
            if session_name is None:
                session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store session_name in state for reference
            state.session_name = session_name
            save_path = StateManager.get_state_path(state.log_dir, session_name)

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(save_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        return save_path

    @staticmethod
    def load(path: str) -> BisectState:
        """
        Load state from JSON file.

        Args:
            path: Path to state file.

        Returns:
            Loaded BisectState.

        Raises:
            FileNotFoundError: If state file doesn't exist.
            json.JSONDecodeError: If file is not valid JSON.
            ValueError: If state data is invalid.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return BisectState.from_dict(data)

    @staticmethod
    def exists(path: str) -> bool:
        """Check if state file exists."""
        return Path(path).exists()

    @staticmethod
    def find_latest_state(log_dir: str) -> Optional[Path]:
        """
        Find the most recent state file in a directory.

        Args:
            log_dir: Log directory path.

        Returns:
            Path to the most recent state file, or None if no state files exist.
        """
        log_path = Path(log_dir)
        if not log_path.exists():
            return None

        state_files = list(log_path.glob(f"*{StateManager.STATE_SUFFIX}"))
        if not state_files:
            return None

        # Sort by modification time, most recent first
        state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return state_files[0]

    @staticmethod
    def print_status(state: BisectState) -> None:
        """
        Print human-readable status summary.

        Used by --status CLI mode to display current bisect state.

        Args:
            state: BisectState to display.
        """
        print("=" * 60)
        print("Bisect Status")
        print("=" * 60)
        print()

        # Phase and timing
        phase_display = state.phase.value.replace("_", " ").title()
        print(f"Phase:       {phase_display}")
        print(f"Started:     {state.started_at or 'N/A'}")
        print(f"Updated:     {state.updated_at or 'N/A'}")
        print()

        # Configuration
        print("Configuration:")
        print(f"  Triton Dir:   {state.triton_dir}")
        print(f"  Test Script:  {state.test_script}")
        print(f"  Good Commit:  {state.good_commit}")
        print(f"  Bad Commit:   {state.bad_commit}")
        if state.commits_csv:
            print(f"  Commits CSV:  {state.commits_csv}")
        print(f"  Conda Env:    {state.conda_env}")
        print(f"  Log Dir:      {state.log_dir}")
        print()

        # Results
        print("Results:")
        print(f"  Triton Culprit:  {state.triton_culprit or 'N/A'}")

        if state.is_llvm_bump is not None:
            bump_str = "Yes" if state.is_llvm_bump else "No"
            print(f"  Is LLVM Bump:    {bump_str}")

            if state.is_llvm_bump:
                if state.old_llvm_hash and state.new_llvm_hash:
                    print(
                        f"  LLVM Change:     {state.old_llvm_hash[:12]} -> "
                        f"{state.new_llvm_hash[:12]}"
                    )
                if state.failing_pair_index is not None:
                    print(f"  Failing Pair:    Index {state.failing_pair_index}")
                if state.good_llvm and state.bad_llvm:
                    print(
                        f"  LLVM Range:      {state.good_llvm[:12]} -> "
                        f"{state.bad_llvm[:12]}"
                    )
                print(f"  LLVM Culprit:    {state.llvm_culprit or 'N/A'}")

        # Error
        if state.error_message:
            print()
            print(f"Error: {state.error_message}")

        print()
        print("=" * 60)
