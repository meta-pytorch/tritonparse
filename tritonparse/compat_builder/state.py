# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
State management for compat_builder.

This module provides data classes for managing the state of the compat-build
workflow and representing LLVM bump blocks in commits.csv.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class CompatBuildPhase(Enum):
    """compat-build workflow phases."""

    INITIALIZING = "initializing"
    FINDING_INCOMPATIBLE = "finding_incompatible"
    WAITING_FOR_FIX = "waiting_for_fix"
    APPLYING_FIX = "applying_fix"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BumpBlock:
    """
    A single LLVM bump block (minimum granularity, cannot be split).

    The order of pairs within a block is determined by the bisect process
    and cannot be changed. Blocks can be reordered based on LLVM timestamps.

    Attributes:
        bump_commit: Triton's LLVM bump commit hash
        range_start: old_llvm (LLVM hash before bump)
        range_end: new_llvm (LLVM hash after bump)
        generated_at: Timestamp when this block was generated
        description: Optional description of this bump
        pairs: List of (triton_commit, llvm_commit) pairs, order preserved
        range_start_timestamp: Unix timestamp for sorting (populated at runtime)
    """

    bump_commit: str
    range_start: str
    range_end: str
    generated_at: str
    pairs: List[Tuple[str, str]]
    description: Optional[str] = None
    range_start_timestamp: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "bump_commit": self.bump_commit,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "generated_at": self.generated_at,
            "description": self.description,
            "pairs": self.pairs,
            "range_start_timestamp": self.range_start_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BumpBlock:
        """Create BumpBlock from dictionary."""
        return cls(
            bump_commit=data["bump_commit"],
            range_start=data["range_start"],
            range_end=data["range_end"],
            generated_at=data["generated_at"],
            description=data.get("description"),
            pairs=[tuple(p) for p in data.get("pairs", [])],
            range_start_timestamp=data.get("range_start_timestamp"),
        )

    def format_csv_block(self) -> str:
        """
        Format this block as CSV content with structured comments.

        Returns:
            Formatted CSV block string including header comments and pairs.
        """
        lines = []
        lines.append("# === LLVM_BUMP_START ===")
        lines.append(f"# bump_commit: {self.bump_commit}")
        lines.append(f"# range_start: {self.range_start}")
        lines.append(f"# range_end: {self.range_end}")
        lines.append(f"# generated_at: {self.generated_at}")
        if self.description:
            lines.append(f"# description: {self.description}")

        for triton_commit, llvm_commit in self.pairs:
            lines.append(f"{triton_commit},{llvm_commit}")

        lines.append("# === LLVM_BUMP_END ===")
        return "\n".join(lines)


@dataclass
class CompatBuildState:
    """
    State for compat-build workflow.

    Attributes:
        triton_dir: Path to Triton repository
        llvm_bump_commit: Target LLVM bump commit hash
        output_csv: Output CSV file path
        conda_env: Conda environment name
        log_dir: Log directory path
        session_name: Unique session identifier
        old_llvm: Starting LLVM hash (from llvm_bump~1)
        new_llvm: Target LLVM hash (from llvm_bump)
        current_triton: Current Triton commit on fork branch
        current_llvm_good: Current confirmed compatible LLVM
        pairs: Generated (triton_commit, llvm_commit) pairs
        phase: Current workflow phase
        started_at: Workflow start timestamp
        updated_at: Last update timestamp
        last_incompatible_llvm: Most recently found incompatible LLVM
        last_build_error: Most recent build error (for fix assistance)
        error_message: Error message if workflow failed
    """

    triton_dir: str
    llvm_bump_commit: str
    output_csv: str
    conda_env: str = "triton_bisect"
    log_dir: str = "./compat_build_logs"
    session_name: Optional[str] = None

    old_llvm: Optional[str] = None
    new_llvm: Optional[str] = None

    current_triton: Optional[str] = None
    current_llvm_good: Optional[str] = None

    pairs: List[Tuple[str, str]] = field(default_factory=list)

    phase: CompatBuildPhase = CompatBuildPhase.INITIALIZING
    started_at: Optional[str] = None
    updated_at: Optional[str] = None

    last_incompatible_llvm: Optional[str] = None
    last_build_error: Optional[str] = None

    error_message: Optional[str] = None

    def add_pair(self, triton_commit: str, llvm_commit: str) -> None:
        """Add a (Triton, LLVM) pair."""
        self.pairs.append((triton_commit, llvm_commit))

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "triton_dir": self.triton_dir,
            "llvm_bump_commit": self.llvm_bump_commit,
            "output_csv": self.output_csv,
            "conda_env": self.conda_env,
            "log_dir": self.log_dir,
            "session_name": self.session_name,
            "old_llvm": self.old_llvm,
            "new_llvm": self.new_llvm,
            "current_triton": self.current_triton,
            "current_llvm_good": self.current_llvm_good,
            "pairs": self.pairs,
            "phase": self.phase.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "last_incompatible_llvm": self.last_incompatible_llvm,
            "last_build_error": self.last_build_error,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CompatBuildState:
        """Create CompatBuildState from dictionary."""
        data = data.copy()
        data["phase"] = CompatBuildPhase(data["phase"])
        data["pairs"] = [tuple(p) for p in data.get("pairs", [])]
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save state to file.

        Args:
            path: Optional path to save to. If not provided, uses default location.

        Returns:
            Path where state was saved.
        """
        self.updated_at = datetime.now().isoformat()
        if self.started_at is None:
            self.started_at = self.updated_at

        if path is None:
            if self.session_name is None:
                self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(self.log_dir) / f"{self.session_name}_compat_state.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> CompatBuildState:
        """
        Load state from file.

        Args:
            path: Path to state file.

        Returns:
            Loaded CompatBuildState instance.

        Raises:
            FileNotFoundError: If state file does not exist.
            json.JSONDecodeError: If state file is not valid JSON.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_state_file_path(self) -> Path:
        """Get the default state file path for this session."""
        if self.session_name is None:
            self.session_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.log_dir) / f"{self.session_name}_compat_state.json"

    def create_bump_block(self, description: Optional[str] = None) -> BumpBlock:
        """
        Create a BumpBlock from the current state.

        Args:
            description: Optional description for the bump block.

        Returns:
            BumpBlock instance with current state data.
        """
        return BumpBlock(
            bump_commit=self.llvm_bump_commit,
            range_start=self.old_llvm,
            range_end=self.new_llvm,
            generated_at=datetime.now().strftime("%Y-%m-%d"),
            description=description,
            pairs=list(self.pairs),
        )
