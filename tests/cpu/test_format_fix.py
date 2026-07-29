# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import unittest
from unittest.mock import MagicMock, patch

from tritonparse.tools import format_fix


class FormatFixTest(unittest.TestCase):
    @patch("tritonparse.tools.format_fix.subprocess.run")
    def test_ruff_check_reports_all_lint_issues(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        self.assertTrue(format_fix.run_ruff_check(check_only=True))

        mock_run.assert_called_once_with(
            ["ruff", "check", "."],
            capture_output=True,
            text=True,
            check=False,
        )

    @patch("tritonparse.tools.format_fix.run_ruff_check")
    @patch("tritonparse.tools.format_fix.run_ufmt", return_value=True)
    def test_format_only_skips_linter(
        self, mock_ufmt: MagicMock, mock_ruff: MagicMock
    ) -> None:
        with patch.object(sys, "argv", ["format_fix", "--format-only", "--check-only"]):
            self.assertEqual(format_fix.main(), 0)

        mock_ufmt.assert_called_once_with(True, False)
        mock_ruff.assert_not_called()

    @patch("tritonparse.tools.format_fix.run_ruff_check", return_value=True)
    @patch("tritonparse.tools.format_fix.run_ufmt")
    def test_lint_only_skips_formatter(
        self, mock_ufmt: MagicMock, mock_ruff: MagicMock
    ) -> None:
        with patch.object(sys, "argv", ["format_fix", "--lint-only", "--check-only"]):
            self.assertEqual(format_fix.main(), 0)

        mock_ufmt.assert_not_called()
        mock_ruff.assert_called_once_with(True, False)


if __name__ == "__main__":
    unittest.main()
