# Copyright (c) Meta Platforms, Inc. and affiliates.

import inspect
import sys
import tempfile
import unittest
from unittest import mock

from tritonparse import cli, context_manager
from tritonparse.parse import utils as parse_utils


class ParseKernelFilterCliTest(unittest.TestCase):
    def test_cli_dispatches_the_raw_allowlist(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                [
                    "tritonparseoss",
                    "parse",
                    "/tmp/logs",
                    "--kernel-allowlist",
                    "matmul*,*attention*",
                ],
            ),
            mock.patch.object(cli, "is_fbcode", return_value=False),
            mock.patch.object(parse_utils, "is_fbcode", return_value=False),
            mock.patch.object(cli, "unified_parse") as parse_mock,
        ):
            cli.main()

        self.assertEqual(parse_mock.call_args.kwargs["source"], "/tmp/logs")
        self.assertEqual(
            parse_mock.call_args.kwargs["kernel_allowlist"],
            "matmul*,*attention*",
        )
        self.assertTrue(parse_mock.call_args.kwargs["skip_logger"])

    def test_cli_defaults_to_no_filter(self) -> None:
        with (
            mock.patch.object(
                sys,
                "argv",
                ["tritonparseoss", "parse", "/tmp/logs"],
            ),
            mock.patch.object(cli, "is_fbcode", return_value=False),
            mock.patch.object(parse_utils, "is_fbcode", return_value=False),
            mock.patch.object(cli, "unified_parse") as parse_mock,
        ):
            cli.main()

        self.assertIsNone(parse_mock.call_args.kwargs["kernel_allowlist"])


class UnifiedParseKernelFilterTest(unittest.TestCase):
    def test_allowlist_is_an_explicit_public_parameter(self) -> None:
        parameter = inspect.signature(parse_utils.unified_parse).parameters[
            "kernel_allowlist"
        ]

        self.assertIsNone(parameter.default)

    def test_forwards_allowlist_to_oss_run(self) -> None:
        raw_allowlist = " target*, *attention* "
        with (
            mock.patch.object(parse_utils, "is_fbcode", return_value=False),
            mock.patch.object(parse_utils, "oss_run", return_value="oss") as run_mock,
        ):
            result = parse_utils.unified_parse(
                source="logs",
                kernel_allowlist=raw_allowlist,
                skip_logger=True,
            )

        self.assertEqual(result, "oss")
        self.assertEqual(run_mock.call_args.kwargs["kernel_allowlist"], raw_allowlist)


class ContextManagerKernelFilterTest(unittest.TestCase):
    def test_forwards_allowlist_as_a_parse_keyword(self) -> None:
        with tempfile.TemporaryDirectory() as log_dir:
            manager = context_manager.TritonParseManager(
                log_dir=log_dir,
                kernel_allowlist="target*",
            )

            with (
                mock.patch.object(context_manager, "init") as init_mock,
                mock.patch.object(
                    context_manager,
                    "unified_parse",
                    return_value="parsed",
                ) as parse_mock,
                mock.patch.object(context_manager, "clear_logging_config"),
            ):
                manager.__enter__()
                manager.__exit__(None, None, None)

        self.assertEqual(manager.output_link, "parsed")
        self.assertNotIn("kernel_allowlist", init_mock.call_args.kwargs)
        self.assertEqual(parse_mock.call_args.kwargs["kernel_allowlist"], "target*")
