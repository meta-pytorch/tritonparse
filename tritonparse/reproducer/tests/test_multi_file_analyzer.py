# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
Tests for MultiFileCallGraphAnalyzer.

This test suite validates the multi-file call graph analysis functionality,
including cross-file dependency tracking, import resolution, and result consolidation.
"""

import ast
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tritonparse.reproducer.multi_file_analyzer import MultiFileCallGraphAnalyzer
from tritonparse.reproducer.placeholder_replacer import get_dependent_source_map


class TestMultiFileCallGraphAnalyzer(unittest.TestCase):
    """Test the MultiFileCallGraphAnalyzer with real Triton test files."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Get the directory containing the tritonparse package.
        # The analyzer canonicalizes file paths with `Path.resolve()`, so the
        # fixtures must use canonical paths too; under Buck, `__file__` names
        # a link-tree path that resolves to the real source file.
        test_file = Path(__file__).resolve()
        self.code_root = str(test_file.parents[3])

        # Test artifacts directory
        self.artifacts_dir = test_file.parent / "artifacts"
        self.entry_file = str(self.artifacts_dir / "triton_fused_kernel.py")
        self.preprocess_file = str(self.artifacts_dir / "triton_preprocess.py")
        self.utils_file = str(self.artifacts_dir / "triton_utils.py")

    def test_single_file_analysis(self) -> None:
        """Test analysis of a single file without following imports."""
        # Setup: Analyze only triton_utils.py (which has no imports to internal files)
        analyzer = MultiFileCallGraphAnalyzer(
            entry_file=self.utils_file,
            entry_function="add_values",
            code_roots=self.code_root,
        )

        # Execute: Run analysis
        result = analyzer.analyze()

        # Assert: Only one file analyzed (triton_utils.py)
        self.assertEqual(result.stats.total_files_analyzed, 1)
        self.assertIn(self.utils_file, result.analyzed_files)

        # Assert: add_values is the backend, should not be in dependent functions
        # (dependent functions excludes the backend itself)
        self.assertNotIn(
            "tritonparse.reproducer.tests.artifacts.triton_utils.add_values",
            result.functions,
        )

        # Assert: All imports should be external (triton, tl)
        self.assertTrue(all(imp.is_external for imp in result.imports))

    def test_multi_file_traversal(self) -> None:
        """Test that analyzer follows imports across multiple files."""
        # Setup: Start from main_kernel in triton_fused_kernel.py
        analyzer = MultiFileCallGraphAnalyzer(
            entry_file=self.entry_file,
            entry_function="main_kernel",
            code_roots=self.code_root,
        )

        # Execute: Run analysis
        result = analyzer.analyze()

        # Assert: Should analyze all 3 files in the call chain
        # main_kernel -> scale_kernel -> add_values
        self.assertEqual(result.stats.total_files_analyzed, 3)

        # Assert: All 3 kernel files should be included
        self.assertIn(self.entry_file, result.analyzed_files)
        self.assertIn(self.preprocess_file, result.analyzed_files)
        self.assertIn(self.utils_file, result.analyzed_files)

        # Assert: Verify scale_kernel and add_values are in result.functions
        function_names = set(result.functions.keys())
        self.assertEqual(len(result.functions.items()), 2)
        self.assertTrue(
            any("scale_kernel" in name for name in function_names),
            f"scale_kernel should be in functions. Found: {function_names}",
        )
        self.assertTrue(
            any("add_values" in name for name in function_names),
            f"add_values should be in functions. Found: {function_names}",
        )

        # Assert: Verify source code was properly extracted with exact lengths
        # Find the full qualified names for these functions
        scale_kernel_name = next(
            name for name in function_names if "scale_kernel" in name
        )
        add_values_name = next(name for name in function_names if "add_values" in name)

        # Verify source code has exact expected lengths
        scale_kernel_source = result.functions[scale_kernel_name]
        add_values_source = result.functions[add_values_name]

        self.assertGreaterEqual(
            len(scale_kernel_source),
            50,
            f"scale_kernel source code should be exactly 50 chars. Got: {len(scale_kernel_source)} chars",
        )
        self.assertGreaterEqual(
            len(add_values_source),
            50,
            f"add_values source code should be exactly 50 chars. Got: {len(add_values_source)} chars",
        )

        # Verify short names are correctly extracted
        self.assertEqual(result.function_short_names[scale_kernel_name], "scale_kernel")
        self.assertEqual(result.function_short_names[add_values_name], "add_values")

    def test_symlink_path_resolution(self) -> None:
        """Test that analyzer works correctly when entry_file is accessed via symlink.

        This simulates the buck-out link-tree scenario where kernel source files
        are symlinked from buck-out to the real fbcode source.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink to the utils artifact
            symlink_path = os.path.join(tmpdir, "triton_utils.py")
            os.symlink(self.utils_file, symlink_path)

            analyzer = MultiFileCallGraphAnalyzer(
                entry_file=symlink_path,
                entry_function="add_values",
            )

            # entry_file should be resolved to the real path, not the symlink
            self.assertEqual(analyzer.entry_file, self.utils_file)

            result = analyzer.analyze()

            # analyzed_files should contain canonical (resolved) paths
            self.assertEqual(result.stats.total_files_analyzed, 1)
            self.assertIn(self.utils_file, result.analyzed_files)
            # The symlink path should NOT appear in analyzed_files
            self.assertNotIn(symlink_path, result.analyzed_files)

    def test_symlink_multi_file_traversal(self) -> None:
        """Test multi-file analysis when starting from a symlinked entry file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a symlink to the entry kernel
            symlink_path = os.path.join(tmpdir, "triton_fused_kernel.py")
            os.symlink(self.entry_file, symlink_path)

            analyzer = MultiFileCallGraphAnalyzer(
                entry_file=symlink_path,
                entry_function="main_kernel",
            )

            result = analyzer.analyze()

            # Should still traverse all 3 files
            self.assertEqual(result.stats.total_files_analyzed, 3)

            # All paths should be canonical (resolved)
            self.assertIn(self.entry_file, result.analyzed_files)
            self.assertIn(self.preprocess_file, result.analyzed_files)
            self.assertIn(self.utils_file, result.analyzed_files)

            # Dependent functions should still be found
            function_names = set(result.functions.keys())
            self.assertTrue(any("scale_kernel" in name for name in function_names))
            self.assertTrue(any("add_values" in name for name in function_names))

    def test_embedded_functions_filtered_from_imports(self) -> None:
        """Test that get_dependent_source_map filters out imports for embedded functions.

        When a function (e.g., scale_kernel) is extracted and embedded in the
        reproducer, the corresponding 'from module import scale_kernel' should
        not appear in import_statements.
        """
        from tritonparse.reproducer.placeholder_replacer import get_dependent_source_map

        dep_result = get_dependent_source_map(
            function_name="main_kernel",
            file_path=self.entry_file,
        )

        self.assertIsNotNone(dep_result)
        assert dep_result is not None

        # scale_kernel and add_values should be in embedded functions
        embedded_short_names = {
            name.split(".")[-1] for name in dep_result.functions.keys()
        }
        self.assertIn("scale_kernel", embedded_short_names)
        self.assertIn("add_values", embedded_short_names)

        # Import statements should NOT contain imports for embedded functions
        for stmt in dep_result.import_statements:
            self.assertNotIn(
                "scale_kernel",
                stmt,
                f"Embedded function 'scale_kernel' should not appear in import: {stmt}",
            )
            self.assertNotIn(
                "add_values",
                stmt,
                f"Embedded function 'add_values' should not appear in import: {stmt}",
            )


class TestForwardHelperCopy(unittest.TestCase):
    def test_copy_and_regenerate_preserves_helpers_and_their_ast(self) -> None:
        kernel = "def kernel(x):\n    return helper(x) + 1\n"
        helpers = (
            "def helper(x):\n    return leaf(x) * 2\n\ndef leaf(x):\n    return x + 3\n"
        )
        expected = {node.name: ast.dump(node) for node in ast.parse(helpers).body}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "path", sys.path.copy()),
        ):
            root = Path(directory)
            (root / "pyproject.toml").write_text("")
            source = root / "original.py"
            source.write_text(kernel + "\n" + helpers)
            # The copy path emits the entry before its helpers. Treat that
            # output as the next capture's source, then generate a second copy.
            for generation in range(2):
                with self.subTest(generation=generation):
                    result = get_dependent_source_map("kernel", str(source))
                    self.assertIsNotNone(result)
                    extracted = {
                        ast.parse(code).body[0].name: ast.dump(ast.parse(code).body[0])
                        for code in result.functions.values()
                    }
                    self.assertEqual(extracted, expected)
                    generated = kernel + "\n" + "\n\n".join(result.functions.values())
                    namespace = {}
                    exec(compile(generated, str(source), "exec"), namespace)
                    self.assertEqual(namespace["kernel"](4), 15)
                    source = root / f"copy_{generation}.py"
                    source.write_text(generated)

    def test_imported_helper_can_call_another_later_helper(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(sys, "path", sys.path.copy()),
        ):
            root = Path(directory)
            (root / "forward_helpers_fixture.py").write_text(
                "def helper(x):\n    return leaf(x)\n\ndef leaf(x):\n    return x * 2\n"
            )
            source = root / "entry.py"
            source.write_text(
                "from forward_helpers_fixture import helper\n\n"
                "def kernel(x):\n    return helper(x)\n"
            )
            result = MultiFileCallGraphAnalyzer(
                entry_file=str(source),
                entry_function="kernel",
                code_roots=str(root),
            ).analyze()
            self.assertEqual(
                set(result.functions),
                {"forward_helpers_fixture.helper", "forward_helpers_fixture.leaf"},
            )
            self.assertEqual(result.stats.total_files_analyzed, 2)


if __name__ == "__main__":
    unittest.main()
