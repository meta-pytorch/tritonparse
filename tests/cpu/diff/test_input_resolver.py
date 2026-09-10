#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for resolving diff inputs to local files."""

import unittest

# Any fetchable URL works here; these tests never dereference it.
REMOTE_URL = "https://example.com/logs/t.ndjson.gz"


class TestResolveInput(unittest.TestCase):
    def test_local_path_passes_through_unlinkable(self) -> None:
        """Local inputs are untouched and yield no shareable URL."""
        from tritonparse.diff.core.input_resolver import resolve_input

        resolved = resolve_input("/tmp/trace.ndjson")
        self.assertEqual(resolved.local_path, "/tmp/trace.ndjson")
        self.assertEqual(resolved.display, "/tmp/trace.ndjson")
        self.assertIsNone(resolved.json_url)


class TestWantsLinks(unittest.TestCase):
    """Uploading is gated on a link actually being printed."""

    def test_gating(self) -> None:
        from tritonparse.diff.cli import _wants_links

        self.assertTrue(_wants_links(no_url=False, quiet=False))
        self.assertFalse(_wants_links(no_url=True, quiet=False))
        self.assertFalse(_wants_links(no_url=False, quiet=True))


class TestGenerateOutputPath(unittest.TestCase):
    def test_local_input_writes_beside_input(self) -> None:
        from tritonparse.diff.cli import _generate_output_path
        from tritonparse.diff.core.input_resolver import ResolvedInput

        local = ResolvedInput(
            local_path="/a/b/trace.ndjson", display="/a/b/trace.ndjson"
        )
        self.assertEqual(_generate_output_path(local), "/a/b/trace_diff.ndjson")

    def test_gz_suffix_preserved(self) -> None:
        from tritonparse.diff.cli import _generate_output_path
        from tritonparse.diff.core.input_resolver import ResolvedInput

        local = ResolvedInput(local_path="/a/t.ndjson.gz", display="/a/t.ndjson.gz")
        self.assertEqual(_generate_output_path(local), "/a/t_diff.ndjson.gz")

    def test_remote_input_writes_to_cwd_not_tempdir(self) -> None:
        """Output must not land in the temp dir, which is deleted on exit."""
        from tritonparse.diff.cli import _generate_output_path
        from tritonparse.diff.core.input_resolver import ResolvedInput

        remote = ResolvedInput(
            local_path="/tmp/tritonparse_diff_xyz/bucket/tree/logs/t.ndjson.gz",
            display="remote://bucket/tree/logs/t.ndjson.gz",
            json_url=REMOTE_URL,
        )
        self.assertEqual(_generate_output_path(remote), "t_diff.ndjson.gz")
