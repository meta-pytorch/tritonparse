# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests that every init() keyword is reachable through TritonParseManager.

The context manager historically exposed an arbitrary subset of init()'s
keywords, so callers who wanted the rest had to drop down to init() directly and
give up the manager's parse/cleanup lifecycle. These tests pin the two halves
together: a signature-parity test that fails when init() grows a keyword the
manager does not forward, plus explicit coverage of the keywords themselves.
"""

import inspect
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tritonparse import context_manager
from tritonparse.structured_logging import init

# init()'s trace_folder is the one keyword the manager deliberately does not
# mirror by name: it owns the trace directory lifecycle, taking `log_dir` (or
# creating a temporary directory) and passing the result to init() positionally.
TRACE_FOLDER_PARAM = "trace_folder"


def _init_keywords():
    return [
        name
        for name in inspect.signature(init).parameters
        if name != TRACE_FOLDER_PARAM
    ]


class TritonParseManagerInitParityTest(unittest.TestCase):
    """Every init() keyword must be both accepted and forwarded."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def test_manager_accepts_every_init_keyword(self):
        manager_params = set(
            inspect.signature(context_manager.TritonParseManager.__init__).parameters
        )
        # **parse_kwargs would swallow an unknown keyword silently instead of
        # forwarding it to init(), so require a real named parameter for each.
        missing = [name for name in _init_keywords() if name not in manager_params]
        self.assertEqual(
            missing,
            [],
            f"TritonParseManager does not expose init() keyword(s): {missing}",
        )

    def test_manager_forwards_every_init_keyword(self):
        # Sentinels rather than realistic values: init() is mocked, so this
        # asserts plumbing only. Distinct objects catch a keyword wired to the
        # wrong attribute, and being non-None means the keywords the manager
        # forwards conditionally are exercised too.
        sentinels = {name: object() for name in _init_keywords()}

        with patch.object(context_manager, "init") as mock_init:
            manager = context_manager.TritonParseManager(
                log_dir=self.tmp_dir, **sentinels
            )
            manager.__enter__()

        mock_init.assert_called_once()
        forwarded = mock_init.call_args.kwargs
        for name, sentinel in sentinels.items():
            self.assertIn(name, forwarded, f"{name} was not forwarded to init()")
            self.assertIs(forwarded[name], sentinel, f"{name} was forwarded altered")
        # The trace directory is still passed positionally, as before.
        self.assertEqual(mock_init.call_args.args, (self.tmp_dir,))


class TritonParseManagerKeywordForwardingTest(unittest.TestCase):
    """Explicit per-keyword coverage on top of the parity test above.

    The parity test proves every init() keyword is forwarded with the value it
    was given; these pin the semantics that make forwarding correct for the
    keywords whose defaults are load-bearing.
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def _forwarded_kwargs(self, **manager_kwargs):
        with patch.object(context_manager, "init") as mock_init:
            manager = context_manager.TritonParseManager(
                log_dir=self.tmp_dir, **manager_kwargs
            )
            manager.__enter__()
        mock_init.assert_called_once()
        return mock_init.call_args.kwargs

    def test_defaults_match_init_defaults(self):
        # Forwarding the defaults unconditionally is only safe because each one
        # is already init()'s own default, i.e. equivalent to omitting it.
        kwargs = self._forwarded_kwargs()
        self.assertIs(kwargs["enable_trace_launch_within_profiling"], False)
        self.assertIs(kwargs["enable_more_tensor_information"], False)
        self.assertIsNone(kwargs["compression"])
        self.assertIsNone(kwargs["enable_full_python_source"])
        self.assertIsNone(kwargs["enable_sass_dump"])

    def test_full_python_source_true_is_forwarded(self):
        kwargs = self._forwarded_kwargs(enable_full_python_source=True)
        self.assertTrue(kwargs["enable_full_python_source"])

    def test_full_python_source_false_is_forwarded(self):
        # Forwarded unchanged so init() can apply its force-off semantics.
        kwargs = self._forwarded_kwargs(enable_full_python_source=False)
        self.assertIs(kwargs["enable_full_python_source"], False)

    def test_sass_dump_true_is_forwarded(self):
        kwargs = self._forwarded_kwargs(enable_sass_dump=True)
        self.assertTrue(kwargs["enable_sass_dump"])

    def test_trace_launch_within_profiling_is_forwarded(self):
        kwargs = self._forwarded_kwargs(enable_trace_launch_within_profiling=True)
        self.assertIs(kwargs["enable_trace_launch_within_profiling"], True)
        # Mutual exclusion with enable_trace_launch is init()'s business; the
        # manager must not pre-empt it by suppressing either value.
        self.assertIs(kwargs["enable_trace_launch"], False)

    def test_more_tensor_information_is_forwarded(self):
        kwargs = self._forwarded_kwargs(
            enable_trace_launch=True, enable_more_tensor_information=True
        )
        self.assertIs(kwargs["enable_more_tensor_information"], True)
        self.assertIs(kwargs["enable_trace_launch"], True)

    def test_compression_is_forwarded(self):
        kwargs = self._forwarded_kwargs(compression="gzip")
        self.assertEqual(kwargs["compression"], "gzip")

    def test_compression_is_not_consumed_by_parse_kwargs(self):
        # compression is an init() keyword, not a unified_parse() one; it must
        # not leak into the **parse_kwargs bag.
        manager = context_manager.TritonParseManager(
            log_dir=self.tmp_dir, compression="gzip"
        )
        self.assertNotIn("compression", manager.parse_kwargs)
        self.assertEqual(manager.compression, "gzip")


if __name__ == "__main__":
    unittest.main()
