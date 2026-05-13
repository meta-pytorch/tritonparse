# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Tests for `parse_trace_filename_metadata` in `tritonparse.parse.common`.

Covers the right-to-left suffix peeling contract:

- Canonical filename with all three suffixes (rank, pid, host).
- Each legacy combination (no host / no pid+host / no rank / no metadata).
- Every extension the logger emits (`.ndjson`, `.bin.ndjson`,
  `.bin.ndjson.gz`, `.ndjson.gz`, `.ndjson.zst`, `.clp`).
- Adversarial usernames that contain look-alike `_rank_N_` / `_pid_N_` /
  `_host_X_` substrings — the helper MUST land on the rightmost real
  suffix, not on the username decoy.
- lg's `--store-by-source` prefix doesn't break peeling.
"""

import unittest

from tritonparse.parse.common import (
    parse_trace_filename_metadata,
    TraceFilenameMetadata,
)

_PFX = "dedicated_log_triton_trace_"


class CanonicalAndLegacyShapesTest(unittest.TestCase):
    """All 5 combinations of present/absent suffixes the logger can emit."""

    def test_canonical_full_suffix(self) -> None:
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}user_rank_0_pid_12345_host_devgpu001_.ndjson"
            ),
            TraceFilenameMetadata(rank=0, pid=12345, host="devgpu001"),
        )

    def test_no_host_legacy(self) -> None:
        self.assertEqual(
            parse_trace_filename_metadata(f"{_PFX}user_rank_0_pid_12345_.ndjson"),
            TraceFilenameMetadata(rank=0, pid=12345, host=None),
        )

    def test_no_pid_no_host_legacy(self) -> None:
        self.assertEqual(
            parse_trace_filename_metadata(f"{_PFX}user_rank_0_.ndjson"),
            TraceFilenameMetadata(rank=0, pid=None, host=None),
        )

    def test_no_rank_only_pid_host(self) -> None:
        """A no-rank file carrying both pid+host (pre-init in new format)."""
        self.assertEqual(
            parse_trace_filename_metadata(f"{_PFX}user_pid_12345_host_devgpu_.ndjson"),
            TraceFilenameMetadata(rank=None, pid=12345, host="devgpu"),
        )

    def test_no_metadata_at_all(self) -> None:
        """Pre-D104316194 legacy file with bare USER suffix."""
        self.assertEqual(
            parse_trace_filename_metadata(f"{_PFX}user_.ndjson"),
            TraceFilenameMetadata(rank=None, pid=None, host=None),
        )


class ExtensionVariantsTest(unittest.TestCase):
    """Every extension the structured logger / compress_single_file emits."""

    def _expect_full_canonical(self, name: str) -> None:
        self.assertEqual(
            parse_trace_filename_metadata(name),
            TraceFilenameMetadata(rank=0, pid=12345, host="devgpu"),
        )

    def test_ndjson(self) -> None:
        self._expect_full_canonical(f"{_PFX}user_rank_0_pid_12345_host_devgpu_.ndjson")

    def test_bin_ndjson(self) -> None:
        """Logger writes `.bin.ndjson` for gzip/zstd before compression."""
        self._expect_full_canonical(
            f"{_PFX}user_rank_0_pid_12345_host_devgpu_.bin.ndjson"
        )

    def test_bin_ndjson_gz(self) -> None:
        """`.bin.ndjson.gz` — outer .gz first, then inner .bin.ndjson."""
        self._expect_full_canonical(
            f"{_PFX}user_rank_0_pid_12345_host_devgpu_.bin.ndjson.gz"
        )

    def test_bin_ndjson_zst(self) -> None:
        self._expect_full_canonical(
            f"{_PFX}user_rank_0_pid_12345_host_devgpu_.bin.ndjson.zst"
        )

    def test_ndjson_gz(self) -> None:
        """`.ndjson.gz` — `compress_single_file` post-processing format."""
        self._expect_full_canonical(
            f"{_PFX}user_rank_0_pid_12345_host_devgpu_.ndjson.gz"
        )

    def test_clp(self) -> None:
        self._expect_full_canonical(f"{_PFX}user_rank_0_pid_12345_host_devgpu_.clp")

    def test_no_extension(self) -> None:
        """Defensive — peeling still works without an extension."""
        self._expect_full_canonical(f"{_PFX}user_rank_0_pid_12345_host_devgpu_")


class AdversarialUsernameTest(unittest.TestCase):
    """The reason this helper exists: USER substrings that look like suffixes."""

    def test_username_with_all_three_decoy_suffixes(self) -> None:
        """USER = `team_host_fake_pid_999_rank_7` — every fake suffix present.

        Old `re.search()` would have returned (rank=7, pid=999, host="fake").
        Right-to-left peeling correctly lands on the real rightmost values.
        """
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}team_host_fake_pid_999_rank_7"
                f"_rank_0_pid_12345_host_devgpu001_.ndjson"
            ),
            TraceFilenameMetadata(rank=0, pid=12345, host="devgpu001"),
        )

    def test_username_with_pid_decoy_in_middle(self) -> None:
        """USER ends in `_pid_99_` — real pid follows further right.

        With no host suffix, this also exercises the case where peeling
        must distinguish the username's `_pid_99_` from the real
        `_pid_555_`.
        """
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}team_pid_99_user_rank_2_pid_555_.ndjson"
            ),
            TraceFilenameMetadata(rank=2, pid=555, host=None),
        )

    def test_username_with_rank_decoy(self) -> None:
        """USER contains `_rank_9_` — real rank suffix is rank=0."""
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}group_rank_9_user_rank_0_pid_12_.ndjson"
            ),
            TraceFilenameMetadata(rank=0, pid=12, host=None),
        )

    def test_username_with_host_decoy(self) -> None:
        """USER contains `_host_eu_` — real host is `devgpu001`."""
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}team_host_eu_user_rank_0_pid_12345_host_devgpu001_.ndjson"
            ),
            TraceFilenameMetadata(rank=0, pid=12345, host="devgpu001"),
        )


class StoreBySourcePrefixTest(unittest.TestCase):
    """lg `--store-by-source` adds a `log.<source>.` prefix; suffix peeling unaffected."""

    def test_store_by_source_prefix(self) -> None:
        self.assertEqual(
            parse_trace_filename_metadata(
                f"log.source.{_PFX}user_rank_0_pid_12345_host_devgpu_.ndjson"
            ),
            TraceFilenameMetadata(rank=0, pid=12345, host="devgpu"),
        )


class HostnameEdgeCaseTest(unittest.TestCase):
    """Logger sanitizes hostname to [a-zA-Z0-9-]; helper must agree."""

    def test_host_with_dashes(self) -> None:
        """Sanitized hostnames may contain dashes (FQDN dots → dashes)."""
        self.assertEqual(
            parse_trace_filename_metadata(
                f"{_PFX}user_rank_0_pid_1_host_dev-gpu-001_.ndjson"
            ).host,
            "dev-gpu-001",
        )

    def test_empty_host_returns_none(self) -> None:
        """`_host__` (empty) doesn't match — regex requires at least 1 char."""
        # Logger guarantees `host_unknown_` fallback, so this shape is
        # malformed in practice. We just want the helper to degrade
        # gracefully (host=None) instead of yielding an empty string.
        self.assertIsNone(
            parse_trace_filename_metadata(f"{_PFX}user_rank_0_pid_1_host__.ndjson").host
        )
