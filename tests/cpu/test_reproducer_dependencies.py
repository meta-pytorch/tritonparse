# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Run the OSS dependency-analysis suites with the internal CPU test runner."""

import unittest

from tritonparse.reproducer.tests import test_ast_analyzer, test_multi_file_analyzer


# Testpilot discovers TestCase classes directly rather than invoking load_tests.
class HigherOrderFunctionDetectionTest(
    test_ast_analyzer.TestHigherOrderFunctionDetection
):
    pass


class ForwardHelperDependenciesTest(test_ast_analyzer.TestForwardHelperDependencies):
    pass


class MultiFileCallGraphAnalyzerTest(
    test_multi_file_analyzer.TestMultiFileCallGraphAnalyzer
):
    pass


class ForwardHelperCopyTest(test_multi_file_analyzer.TestForwardHelperCopy):
    pass


if __name__ == "__main__":
    unittest.main()
