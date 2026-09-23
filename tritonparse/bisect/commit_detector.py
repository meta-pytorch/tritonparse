# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compare the LLVM source and artifact descriptors used by Triton revisions."""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from tritonparse.bisect.executor import ShellExecutor
from tritonparse.bisect.logger import BisectLogger


class CommitDetectorError(Exception):
    """The LLVM descriptors could not be read or compared reliably."""


@dataclass
class LLVMDescriptor:
    """LLVM source and optional prebuilt-artifact metadata at one Git revision."""

    revision: str
    llvm_hash: str
    source_file: str
    build_number: Optional[int] = None
    sha256sum: dict[str, str] = field(default_factory=dict)


@dataclass
class LLVMBumpInfo:
    """A source bump is distinct from an artifact-only update.

    ``artifact_changed=None`` means artifact equality is unknown, for example
    when a legacy descriptor has no checksums or the platform is unspecified.
    Only ``is_llvm_bump`` permits proceeding to LLVM source pair testing.
    """

    is_llvm_bump: bool
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None
    triton_commit: Optional[str] = None
    old_descriptor: Optional[LLVMDescriptor] = None
    new_descriptor: Optional[LLVMDescriptor] = None
    artifact_changed: Optional[bool] = None
    artifact_platform: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        if self.is_llvm_bump:
            return f"LLVM source changed: {self.old_hash} -> {self.new_hash}"
        if self.artifact_changed:
            return "LLVM source is unchanged; prebuilt artifact metadata changed."
        if self.artifact_changed is None:
            return (
                "LLVM source is unchanged; artifact equality could not be determined."
            )
        return "LLVM source and known artifact checksums are unchanged."


class CommitDetector:
    """Read both descriptor formats and compare against the first parent.

    A caller may select another parent explicitly with ``detect(parent=...)``.
    ``artifact_platform`` selects a key in ``sha256sum`` and defaults to the
    explicit ``TRITON_LLVM_SYSTEM_SUFFIX`` override, when set. Without a selected
    platform, differing checksum maps do not prove the local artifact changed.
    """

    LLVM_HASH_FILE = "cmake/llvm-hash.txt"
    LLVM_INFO_FILE = "cmake/llvm-info.json"

    def __init__(
        self,
        triton_dir: Path,
        executor: ShellExecutor,
        logger: BisectLogger,
        artifact_platform: Optional[str] = None,
    ) -> None:
        self.triton_dir = triton_dir
        self.executor = executor
        self.logger = logger
        self.artifact_platform = artifact_platform or os.environ.get(
            "TRITON_LLVM_SYSTEM_SUFFIX"
        )

    def _git(self, *args: str) -> str:
        result = self.executor.run_command(["git", *args], cwd=str(self.triton_dir))
        if not result.success:
            raise CommitDetectorError(
                f"Cannot read LLVM descriptor (git {' '.join(args)}): {result.stderr}"
            )
        return result.stdout

    def read_llvm_descriptor(self, commit: str) -> LLVMDescriptor:
        """Read modern JSON first, falling back only when that file is absent.

        Invalid revisions, unreadable objects and malformed JSON are errors,
        never evidence that LLVM is unchanged. Reading the tree distinguishes
        an absent JSON file from a failed ``git show``.
        """
        revision = self._git(
            "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}"
        ).strip()
        files = self._git(
            "ls-tree",
            "--name-only",
            revision,
            "--",
            self.LLVM_INFO_FILE,
            self.LLVM_HASH_FILE,
        ).splitlines()
        if self.LLVM_INFO_FILE in files:
            content = self._git("show", f"{revision}:{self.LLVM_INFO_FILE}")
            try:
                info = json.loads(content)
            except ValueError as error:
                raise CommitDetectorError(
                    f"Invalid {self.LLVM_INFO_FILE} at {revision}: {error}"
                ) from error
            if not isinstance(info, dict):
                raise CommitDetectorError(
                    f"LLVM descriptor at {revision} must be an object"
                )
            llvm_hash = info.get("llvm_hash")
            if not isinstance(llvm_hash, str) or not re.fullmatch(
                r"[0-9a-fA-F]{40}", llvm_hash
            ):
                raise CommitDetectorError(f"Invalid LLVM source hash at {revision}")
            build_number = info.get("build_number")
            if build_number is not None and (
                type(build_number) is not int or build_number < 0
            ):
                raise CommitDetectorError(f"Invalid LLVM build_number at {revision}")
            checksums = info.get("sha256sum", {})
            if not isinstance(checksums, dict) or any(
                not isinstance(platform, str)
                or not platform
                or not isinstance(checksum, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", checksum)
                for platform, checksum in checksums.items()
            ):
                raise CommitDetectorError(f"Invalid LLVM sha256sum at {revision}")
            return LLVMDescriptor(
                revision=revision,
                llvm_hash=llvm_hash.lower(),
                source_file=self.LLVM_INFO_FILE,
                build_number=build_number,
                sha256sum={key: value.lower() for key, value in checksums.items()},
            )
        if self.LLVM_HASH_FILE in files:
            return LLVMDescriptor(
                revision=revision,
                llvm_hash=self._extract_hash_from_content(
                    self._git("show", f"{revision}:{self.LLVM_HASH_FILE}")
                ),
                source_file=self.LLVM_HASH_FILE,
            )
        raise CommitDetectorError(
            f"No {self.LLVM_INFO_FILE} or {self.LLVM_HASH_FILE} at {revision}"
        )

    def detect(self, commit: str, *, parent: Optional[str] = None) -> LLVMBumpInfo:
        """Compare descriptors; only a source-hash change is an LLVM bump."""
        new = self.read_llvm_descriptor(commit)
        old = self.read_llvm_descriptor(parent or f"{new.revision}^")
        if len(old.llvm_hash) != 40 or len(new.llvm_hash) != 40:
            raise CommitDetectorError(
                "Comparing LLVM revisions requires full 40-character source hashes; "
                "abbreviated hashes cannot establish source equality."
            )
        info = LLVMBumpInfo(
            is_llvm_bump=old.llvm_hash != new.llvm_hash,
            old_hash=old.llvm_hash,
            new_hash=new.llvm_hash,
            triton_commit=new.revision,
            old_descriptor=old,
            new_descriptor=new,
            artifact_changed=self._artifact_changed(old, new),
            artifact_platform=self.artifact_platform,
        )
        self.logger.info(info.describe())
        return info

    def _artifact_changed(
        self, old: LLVMDescriptor, new: LLVMDescriptor
    ) -> Optional[bool]:
        if (
            old.build_number is not None
            and new.build_number is not None
            and old.build_number != new.build_number
        ):
            return True
        if self.artifact_platform:
            old_sum = old.sha256sum.get(self.artifact_platform)
            new_sum = new.sha256sum.get(self.artifact_platform)
            if old_sum is not None and new_sum is not None:
                return old_sum != new_sum
            return None
        if old.sha256sum and old.sha256sum == new.sha256sum:
            return False
        return None

    def _extract_hash_from_content(self, content: str) -> str:
        """Read legacy hashes, preserving support for abbreviated lookup results."""
        for line in content.splitlines():
            line = line.strip()
            if re.fullmatch(r"[0-9a-fA-F]{7,40}", line):
                return line.lower()
        raise CommitDetectorError(f"No valid hash found in content: {content[:100]}")

    def get_llvm_hash_at_commit(self, commit: str) -> str:
        """Return the LLVM source hash from either supported descriptor format."""
        return self.read_llvm_descriptor(commit).llvm_hash
