# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
State management for bisect workflow.

This module provides state persistence for the bisect workflow, enabling
checkpoint/resume functionality. The state is saved as JSON and can be
loaded to continue from where the workflow left off.
"""

from dataclasses import asdict, dataclass
from enum import Enum
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
