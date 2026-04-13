# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
State management for compat_builder workflow.

Provides state persistence for the compat-build workflow, enabling
checkpoint/resume functionality. The state is saved as JSON and can be
loaded to continue from where the workflow left off.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from tritonparse._json_compat import dumps, loads


class CompatBuildPhase(Enum):
    """
    Compat-build workflow phases.

    The workflow progresses through these phases:
    1. INITIALIZING: Setting up worktree and compat branch.
    2. FINDING_INCOMPATIBLE: Running compat probe to find next incompatible LLVM.
    3. AI_FIXING: AI is attempting to fix the incompatibility.
    4. WAITING_FOR_FIX: Paused, waiting for user to provide a manual fix commit.
    5. APPLYING_FIX: Verifying and committing a fix.
    6. COMPLETED: All pairs recorded, CSV generated.
    7. FAILED: Workflow failed with an unrecoverable error.
    """

    INITIALIZING = "initializing"
    FINDING_INCOMPATIBLE = "finding_incompatible"
    AI_FIXING = "ai_fixing"
    WAITING_FOR_FIX = "waiting_for_fix"
    APPLYING_FIX = "applying_fix"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CompatBuildState:
    """
    Complete compat-build workflow state.

    Attributes:
        triton_dir: Path to the source Triton repository (read-only).
        llvm_bump_commit: The Triton commit that bumped LLVM.
        output_csv: Path where the output commits.csv will be written.
        conda_env: Conda environment name for running Triton.
        log_dir: Directory for log files.
        worktree_root: Root directory for auto-created worktrees.
        worktree_path: Path to the dedicated compat worktree (resolved during initialize).
        session_name: Links state file to log files.
        old_llvm: LLVM commit hash before the bump (good boundary).
        new_llvm: LLVM commit hash after the bump (bad boundary, terminal).
        current_triton: Current Triton commit being tested in worktree.
        current_llvm_good: Last known-good LLVM commit for the current Triton commit.
        pairs: List of (triton_commit, llvm_last_compatible) pairs recorded so far.
        phase: Current workflow phase.
        started_at: ISO timestamp when workflow started.
        updated_at: ISO timestamp of last state update.
        last_incompatible_llvm: Most recently found incompatible LLVM commit.
        last_build_error: Build error output from last compat probe failure.
        ai_fix_attempted: Whether AI fix was already attempted for the current
            incompatibility. Reset to False after each apply_fix().
        error_message: Error message if workflow failed.
    """

    # Configuration
    triton_dir: str
    llvm_bump_commit: str
    output_csv: str
    conda_env: str = "triton_bisect"
    log_dir: str = "./compat_build_logs"
    worktree_root: str | None = None
    worktree_path: str | None = None
    session_name: str | None = None

    # LLVM range (set during initialize)
    old_llvm: str | None = None
    new_llvm: str | None = None

    # Progress tracking
    current_triton: str | None = None
    current_llvm_good: str | None = None
    pairs: list[tuple[str, str]] = field(default_factory=list)

    # Workflow state
    phase: CompatBuildPhase = CompatBuildPhase.INITIALIZING
    started_at: str | None = None
    updated_at: str | None = None

    # Fix assistance
    last_incompatible_llvm: str | None = None
    last_build_error: str | None = None
    ai_fix_attempted: bool = False
    error_message: str | None = None

    def add_pair(self, triton_commit: str, llvm_last_compatible: str) -> None:
        """Add a (triton_commit, llvm_last_compatible) pair to the recorded list."""
        self.pairs.append((triton_commit, llvm_last_compatible))

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        data = asdict(self)
        data["phase"] = self.phase.value
        # tuples serialize as lists in JSON; keep consistent
        data["pairs"] = [list(p) for p in self.pairs]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CompatBuildState:
        """Create state from dictionary (e.g. loaded from JSON)."""
        data = data.copy()
        data["phase"] = CompatBuildPhase(data["phase"])
        data["pairs"] = [tuple(p) for p in data.get("pairs", [])]
        return cls(**data)

    def save(
        self,
        path: Path | None = None,
        session_name: str | None = None,
    ) -> Path:
        """Save state to JSON file."""
        return CompatStateManager.save(
            self, session_name=session_name, path=str(path) if path else None
        )

    @classmethod
    def load(cls, path: Path) -> CompatBuildState:
        """Load state from JSON file."""
        return CompatStateManager.load(str(path))


class CompatStateManager:
    """
    Manages compat-build state persistence.

    State files are named with a session_name (typically a timestamp) to
    correlate with log files from the same run:
    - Log files: {session_name}_bisect.log
    - State file: {session_name}_compat_state.json
    """

    STATE_SUFFIX = "_compat_state.json"

    @staticmethod
    def get_state_path(log_dir: str, session_name: str) -> Path:
        """Get the state file path for a given session."""
        return Path(log_dir) / f"{session_name}{CompatStateManager.STATE_SUFFIX}"

    @staticmethod
    def save(
        state: CompatBuildState,
        session_name: str | None = None,
        path: str | None = None,
    ) -> Path:
        """
        Save state to JSON file, updating timestamps.

        Args:
            state: CompatBuildState to persist.
            session_name: Session identifier. Falls back to state.session_name
                or a generated timestamp.
            path: Explicit file path. Overrides session_name if provided.

        Returns:
            Path where state was saved.
        """
        now = datetime.now().isoformat()
        state.updated_at = now
        if state.started_at is None:
            state.started_at = now

        if path is not None:
            save_path = Path(path)
        else:
            if session_name is None:
                session_name = state.session_name
            if session_name is None:
                session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            state.session_name = session_name
            save_path = CompatStateManager.get_state_path(state.log_dir, session_name)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(dumps(state.to_dict(), indent=True))
        return save_path

    @staticmethod
    def load(path: str) -> CompatBuildState:
        """
        Load state from a JSON file.

        Raises:
            FileNotFoundError: If the state file does not exist.
        """
        with open(path) as f:
            data = loads(f.read())
        return CompatBuildState.from_dict(data)

    @staticmethod
    def find_latest_state(log_dir: str) -> Path | None:
        """Find the most recent state file in log_dir, or None if absent."""
        log_path = Path(log_dir)
        if not log_path.exists():
            return None
        state_files = list(log_path.glob(f"*{CompatStateManager.STATE_SUFFIX}"))
        if not state_files:
            return None
        state_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return state_files[0]
