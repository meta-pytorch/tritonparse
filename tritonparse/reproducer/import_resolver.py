#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
ImportResolver: Resolves Python import statements to file paths.

This module provides functionality to resolve import statements to their
corresponding file paths using Python's importlib system, without actually
importing the modules (no side effects).
"""

import importlib.util
import logging
import sys
from importlib.machinery import ModuleSpec

logger = logging.getLogger(__name__)


class ImportResolver:
    """
    Resolves import statements to absolute file paths using importlib.

    Uses Python's import resolution system (importlib.util.find_spec) to
    locate module files without actually importing them, avoiding side effects.
    """

    def __init__(self, project_root: str) -> None:
        """
        Initialize the ImportResolver.

        Args:
            project_root: Absolute path to project root directory
                         (e.g., /path/to/project/root)
        """
        self.project_root = project_root

        # Ensure project_root is in sys.path for importlib to find modules
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        # Common third-party modules to exclude from analysis
        self.external_modules: set[str] = {
            "torch",
            "triton",
            "numpy",
            "pandas",
            "transformers",
            "tqdm",
            "typing_extensions",
            "pydantic",
            "requests",
            "pytest",
            "unittest",
        }

    def resolve_import(
        self,
        module_name: str,
        package: str | None = None,
    ) -> tuple[str | None, bool]:
        """
        Resolve import to absolute file path using importlib.

        Args:
            module_name: Module to import (e.g., "torch" or "pytorch.tritonparse")
            package: Package context for relative imports (e.g., "pytorch.tritonparse")

        Returns:
            Tuple of (file_path, is_external):
            - file_path: Absolute path to .py file, or None if not resolvable
            - is_external: True if third-party/built-in module

        Examples:
            >>> resolver = ImportResolver("/data/.../fbcode")
            >>> path, is_ext = resolver.resolve_import("pytorch.tritonparse.module")
            >>> # path: "/data/.../fbcode/pytorch/tritonparse/module.py"
            >>> # is_ext: False

            >>> path, is_ext = resolver.resolve_import("torch")
            >>> # path: None
            >>> # is_ext: True
        """
        # Check if it's a known external module
        base_module = module_name.split(".")[0]
        if base_module in self.external_modules:
            logger.debug(
                "Import '%s' marked as external (known third-party module)",
                module_name,
            )
            return None, True

        try:
            # Use importlib to find the module (without importing it)
            spec: ModuleSpec | None = importlib.util.find_spec(module_name, package)

            if spec is None or spec.origin is None:
                # Module not found or built-in (has no file)
                logger.debug(
                    "Import '%s' not resolvable (spec=%s, origin=%s)",
                    module_name,
                    spec,
                    spec.origin if spec else None,
                )
                return None, True

            origin = spec.origin

            # Skip special cases like frozen/built-in modules
            if origin == "frozen" or origin == "built-in":
                logger.debug("Import '%s' is a built-in module", module_name)
                return None, True

            # Check if the module is within project root
            is_internal = origin.startswith(self.project_root)

            if is_internal:
                logger.debug(
                    "Import '%s' resolved to INTERNAL file: %s",
                    module_name,
                    origin,
                )
                return origin, False
            else:
                # External module (outside project)
                logger.warning(
                    "Import '%s' resolved to file '%s' which is OUTSIDE project_root '%s'. "
                    "This import will be skipped. If this is unexpected, verify your --code-root parameter matches "
                    "the directory containing your source files.",
                    module_name,
                    origin,
                    self.project_root,
                )
                return None, True

        except (ImportError, ValueError, AttributeError) as e:
            # Module doesn't exist or can't be resolved
            logger.debug("Import '%s' failed to resolve: %s", module_name, str(e))
            return None, True

    def is_external_module(self, module_name: str) -> bool:
        """
        Check if a module is external (third-party or built-in).

        Args:
            module_name: Name of the module to check

        Returns:
            True if module is external, False if it's internal (project)
        """
        base_module = module_name.split(".")[0]
        return base_module in self.external_modules
