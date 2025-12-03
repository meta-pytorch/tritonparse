# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import ast
import unittest
from pathlib import Path

from tritonparse.reproducer.import_parser import ImportParser
from tritonparse.reproducer.import_resolver import ImportResolver


class TestImportParser(unittest.TestCase):
    """Test ImportParser functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        test_dir = Path(__file__).resolve().parent
        reproducer_dir = test_dir.parent
        self.project_root = str(reproducer_dir.parent)
        self.resolver = ImportResolver(self.project_root)
        self.parser = ImportParser(self.resolver)

    def test_parse_simple_import(self) -> None:
        """Test parsing simple 'import X' statement."""
        code = """
import os
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].import_type, "import")
        self.assertEqual(imports[0].module, "os")
        self.assertEqual(imports[0].names, ["os"])
        self.assertTrue(imports[0].is_external)

    def test_parse_import_with_alias(self) -> None:
        """Test parsing 'import X as Y' statement."""
        code = """
import numpy as np
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "numpy")
        self.assertIn("np", imports[0].aliases)
        self.assertEqual(imports[0].aliases["np"], "numpy")

    def test_parse_from_import(self) -> None:
        """Test parsing 'from X import Y' statement."""
        code = """
from typing import List, Dict
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 2)
        self.assertEqual(imports[0].import_type, "from_import")
        self.assertEqual(imports[0].module, "typing")
        self.assertIn("List", imports[0].names)
        self.assertIn("Dict", imports[1].names)

    def test_parse_from_import_with_alias(self) -> None:
        """Test parsing 'from X import Y as Z' statement."""
        code = """
from collections import OrderedDict as OD
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].names, ["OrderedDict"])
        self.assertIn("OD", imports[0].aliases)
        self.assertEqual(imports[0].aliases["OD"], "OrderedDict")

    def test_parse_relative_import_level_1(self) -> None:
        """Test parsing relative import 'from . import X'.

        When level=1 (from . import X), we're importing from the current package.
        With package="pytorch.tritonparse.reproducer", level=1 means current package,
        so the full module is "pytorch.tritonparse.reproducer".
        """
        code = """
from . import utils
"""
        tree = ast.parse(code, filename="test.py")
        package = "tritonparse.reproducer"
        imports = self.parser.parse_imports(tree, "test.py", package)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].level, 1)
        self.assertEqual(imports[0].module, "tritonparse.reproducer")
        self.assertEqual(imports[0].names, ["utils"])

    def test_parse_relative_import_level_2(self) -> None:
        """Test parsing relative import 'from .. import X'.

        When level=2 (from ..utils import X), we're importing from the parent package.
        With package="pytorch.tritonparse.reproducer.submodule", level=2 removes 1 component,
        giving us "pytorch.tritonparse.reproducer", then we append "utils" to get
        "pytorch.tritonparse.reproducer.utils".
        """
        code = """
from ..utils import helper
"""
        tree = ast.parse(code, filename="test.py")
        package = "tritonparse.reproducer.submodule"
        imports = self.parser.parse_imports(tree, "test.py", package)

        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].level, 2)
        self.assertEqual(imports[0].module, "tritonparse.reproducer.utils")
        self.assertEqual(imports[0].names, ["helper"])

    def test_parse_multiple_imports(self) -> None:
        """Test parsing multiple import statements."""
        code = """
import os
import sys
from typing import List
from collections import defaultdict
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 4)
        self.assertEqual(imports[0].module, "os")
        self.assertEqual(imports[1].module, "sys")
        self.assertEqual(imports[2].module, "typing")
        self.assertEqual(imports[3].module, "collections")

    def test_parse_project_internal_import(self) -> None:
        """Test parsing imports from within the project."""
        code = """
from tritonparse.reproducer import ast_analyzer
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 1)
        self.assertFalse(imports[0].is_external)
        self.assertIsNotNone(imports[0].resolved_path)
        if imports[0].resolved_path:
            self.assertTrue(imports[0].resolved_path.startswith(self.project_root))

    def test_parse_lineno_tracking(self) -> None:
        """Test that line numbers are correctly tracked."""
        code = """
import os

from typing import List


import sys
"""
        tree = ast.parse(code, filename="test.py")
        imports = self.parser.parse_imports(tree, "test.py")

        self.assertEqual(len(imports), 3)
        self.assertEqual(imports[0].lineno, 2)  # import os
        self.assertEqual(imports[1].lineno, 4)  # from typing import List
        self.assertEqual(imports[2].lineno, 7)  # import sys
