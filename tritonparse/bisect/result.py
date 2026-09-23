# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Structured outcomes of a Git bisect attempt, including incomplete searches."""

import re
import signal
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


@dataclass
class BisectResult:
    """Only `found` identifies a unique culprit.

    An ambiguous result retains every candidate reported by Git. The exit code
    describes the command result or interruption, not each per-commit test.
    It is unset for errors caught before a command result was available.
    Artifact paths link to the replay log and the complete build/test output.
    """

    status: Literal["found", "ambiguous", "aborted", "error"]
    exit_code: Optional[int] = None
    culprit: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    message: str = ""
    repository: Optional[str] = None
    good_commit: Optional[str] = None
    bad_commit: Optional[str] = None
    git_bisect_log: Optional[str] = None
    command_log: Optional[str] = None
    result_file: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_output(cls, output: str, exit_code: int = 0) -> "BisectResult":
        abort = re.search(r"bisect run failed: exit code (-?\d+)", output)
        if (
            (abort and (int(abort.group(1)) < 0 or int(abort.group(1)) >= 128))
            or exit_code in (130, 143)
            or exit_code in (-signal.SIGINT, -signal.SIGTERM)
        ):
            return cls(
                status="aborted",
                exit_code=exit_code,
                message=f"Git bisect aborted (exit {exit_code}):\n{output[-1000:]}",
            )

        candidates = cls.parse_skip_candidates(output)
        if candidates:
            return cls(
                status="ambiguous",
                exit_code=exit_code,
                candidates=candidates,
                message=(
                    "Skipped commits prevent a unique first-bad result. "
                    f"Candidates ({len(candidates)}):\n" + "\n".join(candidates)
                ),
            )

        # A first-bad line printed before an execution failure is not success.
        if exit_code != 0:
            return cls(
                status="error",
                exit_code=exit_code,
                message=f"Git bisect failed (exit {exit_code}):\n{output[-1000:]}",
            )

        matches = list(
            dict.fromkeys(
                re.findall(
                    r"^([a-f0-9]{7,40}) is the first bad commit\s*$",
                    output,
                    flags=re.MULTILINE,
                )
            )
        )
        if len(matches) == 1:
            return cls(
                status="found",
                exit_code=exit_code,
                culprit=matches[0],
                message=f"Unique first bad commit: {matches[0]}",
            )
        return cls(
            status="error",
            exit_code=exit_code,
            candidates=matches,
            message=f"Cannot identify a unique bisect result:\n{output[-1000:]}",
        )

    @staticmethod
    def parse_skip_candidates(output: str) -> list[str]:
        markers = list(
            re.finditer(
                r"^The first bad commit could be any of:\s*$",
                output,
                flags=re.MULTILINE,
            )
        )
        if not markers:
            return []
        candidates = []
        for line in output[markers[-1].end() :].splitlines():
            line = line.strip()
            if not line and not candidates:
                continue
            if not re.fullmatch(r"[a-f0-9]{40}", line):
                break
            if line not in candidates:
                candidates.append(line)
        return candidates
