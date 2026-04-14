# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""
CSV management for the compat_builder workflow.

CSVManager reads, validates, and writes the single-bump commits.csv
produced by CompatBuilder.generate_csv(). The CSV format is:

    # schema_version=1
    # llvm_bump_commit=<hash>
    # old_llvm=<hash>
    # new_llvm=<hash>
    # final_bad_triton_commit=<hash>
    # final_bad_llvm=<hash>
    triton_commit,llvm_commit_last_compatible
    <triton_hash>,<llvm_hash>
    ...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Required metadata keys present in the CSV header (in canonical order)
_METADATA_KEYS: tuple[str, ...] = (
    "schema_version",
    "llvm_bump_commit",
    "old_llvm",
    "new_llvm",
    "final_bad_triton_commit",
    "final_bad_llvm",
)

_COLUMN_HEADER = "triton_commit,llvm_commit_last_compatible"
_SCHEMA_VERSION = "1"


@dataclass
class BumpBlock:
    """
    Parsed metadata from a single-bump commits.csv header.

    Each field corresponds to a ``# key=value`` line in the CSV header.
    The terminal bad boundary (final_bad_triton_commit, final_bad_llvm)
    is stored in metadata — not as a regular data row — per design decision 8.7.
    """

    schema_version: str
    llvm_bump_commit: str
    old_llvm: str
    new_llvm: str
    final_bad_triton_commit: str
    final_bad_llvm: str


class CSVManager:
    """
    Reads, validates, and writes single-bump commits.csv files.

    Usage (reading and validating)::

        mgr = CSVManager(Path("commits.csv"))
        meta = mgr.load_metadata()
        pairs = mgr.load_pairs()
        errors = mgr.validate_monotonic_pairs() + mgr.validate_terminal_boundary()
        if errors:
            raise ValueError("\\n".join(errors))

    Usage (writing)::

        CSVManager.write_csv(
            output_path=Path("commits.csv"),
            metadata=BumpBlock(...),
            pairs=[("triton_aaa", "llvm_aaa"), ...],
        )
    """

    def __init__(self, csv_path: Path) -> None:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self._path = csv_path

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def load_metadata(self) -> BumpBlock:
        """
        Parse ``# key=value`` metadata lines from the CSV header.

        Stops reading at the first non-comment line. Only lines that
        contain ``=`` are treated as key-value pairs; other comment lines
        are skipped.

        Raises:
            ValueError: If a required metadata key is absent or the file
                is syntactically invalid.
        """
        raw: dict[str, str] = {}
        with open(self._path) as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped.startswith("#"):
                    break
                content = stripped[1:].strip()
                if "=" not in content:
                    continue
                key, _, value = content.partition("=")
                raw[key.strip()] = value.strip()

        missing = [k for k in _METADATA_KEYS if k not in raw]
        if missing:
            raise ValueError(
                f"CSV missing required metadata keys: {missing} in {self._path}"
            )

        return BumpBlock(
            schema_version=raw["schema_version"],
            llvm_bump_commit=raw["llvm_bump_commit"],
            old_llvm=raw["old_llvm"],
            new_llvm=raw["new_llvm"],
            final_bad_triton_commit=raw["final_bad_triton_commit"],
            final_bad_llvm=raw["final_bad_llvm"],
        )

    def load_pairs(self) -> list[tuple[str, str]]:
        """
        Parse data rows from the CSV (skipping comment and header lines).

        Returns:
            List of (triton_commit, llvm_last_compatible) tuples in file order.

        Raises:
            ValueError: If a data row does not have exactly 2
                comma-separated fields.
        """
        pairs: list[tuple[str, str]] = []
        past_column_header = False
        with open(self._path) as f:
            for lineno, raw_line in enumerate(f, 1):
                line = raw_line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                if not past_column_header:
                    past_column_header = True
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    raise ValueError(
                        f"Line {lineno}: expected 2 comma-separated fields, "
                        f"got {len(parts)}: {line!r}"
                    )
                pairs.append((parts[0].strip(), parts[1].strip()))
        return pairs

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_monotonic_pairs(self) -> list[str]:
        """
        Structural validation of data rows (no git access required).

        Checks:
        - Each row has a non-empty triton_commit.
        - Each row has a non-empty llvm_commit_last_compatible.
        - No duplicate triton commits appear in the list.

        Returns:
            List of error strings describing each violation.
            An empty list means the pairs are structurally valid.
        """
        errors: list[str] = []
        try:
            pairs = self.load_pairs()
        except ValueError as e:
            return [str(e)]

        seen_triton: set[str] = set()
        for i, (triton, llvm) in enumerate(pairs):
            if not triton:
                errors.append(f"Row {i}: empty triton_commit")
            if not llvm:
                errors.append(f"Row {i}: empty llvm_commit_last_compatible")
            if triton in seen_triton:
                errors.append(f"Row {i}: duplicate triton_commit {triton!r}")
            seen_triton.add(triton)
        return errors

    def validate_terminal_boundary(self) -> list[str]:
        """
        Validate terminal boundary metadata self-consistency.

        Per design decision 8.7, the terminal bad boundary is stored in
        metadata as:
        - final_bad_triton_commit == llvm_bump_commit
        - final_bad_llvm == new_llvm

        Returns:
            List of error strings describing each violation.
            An empty list means the terminal boundary is consistent.
        """
        errors: list[str] = []
        try:
            meta = self.load_metadata()
        except (ValueError, FileNotFoundError) as e:
            return [str(e)]

        if meta.final_bad_triton_commit != meta.llvm_bump_commit:
            errors.append(
                f"final_bad_triton_commit {meta.final_bad_triton_commit!r} "
                f"!= llvm_bump_commit {meta.llvm_bump_commit!r}"
            )
        if meta.final_bad_llvm != meta.new_llvm:
            errors.append(
                f"final_bad_llvm {meta.final_bad_llvm!r} != new_llvm {meta.new_llvm!r}"
            )
        return errors

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def write_csv(
        output_path: Path,
        metadata: BumpBlock,
        pairs: list[tuple[str, str]],
    ) -> Path:
        """
        Write a single-bump commits.csv.

        Produces the canonical format: ``# key=value`` metadata header,
        then the column header row, then one data row per pair.

        Args:
            output_path: Destination path (parent dirs are created as needed).
            metadata: BumpBlock containing all required header fields.
            pairs: Ordered list of (triton_commit, llvm_last_compatible) tuples.

        Returns:
            The path where the file was written (same as ``output_path``).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = [
            f"# schema_version={metadata.schema_version}",
            f"# llvm_bump_commit={metadata.llvm_bump_commit}",
            f"# old_llvm={metadata.old_llvm}",
            f"# new_llvm={metadata.new_llvm}",
            f"# final_bad_triton_commit={metadata.final_bad_triton_commit}",
            f"# final_bad_llvm={metadata.final_bad_llvm}",
            _COLUMN_HEADER,
        ]
        for triton_commit, llvm_last_compat in pairs:
            lines.append(f"{triton_commit},{llvm_last_compat}")

        output_path.write_text("\n".join(lines) + "\n")
        return output_path
