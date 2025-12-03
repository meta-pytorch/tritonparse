#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for ImportResolver.

Tests the ImportResolver implementation to ensure it correctly:
1. Resolves internal imports
2. Detects external modules
3. Handles non-existent modules gracefully
"""

import unittest
from pathlib import Path

from tritonparse.reproducer.import_resolver import ImportResolver


class ImportResolverTest(unittest.TestCase):
    """Unit tests for ImportResolver class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Use the tritonparse directory as project root for testing
        test_dir = Path(__file__).resolve().parent
        reproducer_dir = test_dir.parent
        self.project_root = str(reproducer_dir.parent)
        self.resolver = ImportResolver(project_root=self.project_root)

    def test_resolve_tritonparse_module(self) -> None:
        """Test resolving a module within tritonparse."""
        # Setup: module that exists in tritonparse
        module_name = "reproducer.ast_analyzer"

        # Execute: resolve the module
        path, is_external = self.resolver.resolve_import(module_name)

        # Assert: should resolve to tritonparse path
        self.assertIsNotNone(path)
        assert path is not None  # For type checker
        self.assertTrue(path.startswith(self.project_root))
        self.assertTrue(Path(path).exists())
        self.assertFalse(is_external)

    def test_external_module_torch(self) -> None:
        """Test that torch is correctly identified as external."""
        # Setup: torch is a known external module
        module_name = "torch"

        # Execute: resolve the module
        path, is_external = self.resolver.resolve_import(module_name)

        # Assert: should be external
        self.assertIsNone(path)
        self.assertTrue(is_external)

    def test_nonexistent_module(self) -> None:
        """Test handling of non-existent module."""
        # Setup: a module that doesn't exist
        module_name = "this_module_does_not_exist_anywhere"

        # Execute: resolve the module
        path, is_external = self.resolver.resolve_import(module_name)

        # Assert: should handle gracefully as external
        self.assertIsNone(path)
        self.assertTrue(is_external)

    def test_is_external_module_torch(self) -> None:
        """Test is_external_module() for torch."""
        # Setup: torch module name
        module_name = "torch"

        # Execute: check if external
        result = self.resolver.is_external_module(module_name)

        # Assert: should be true
        self.assertTrue(result)

    def test_is_external_module_internal(self) -> None:
        """Test is_external_module() for internal module."""
        # Setup: internal module name
        module_name = "reproducer.ast_analyzer"

        # Execute: check if external
        result = self.resolver.is_external_module(module_name)

        # Assert: should be false
        self.assertFalse(result)
