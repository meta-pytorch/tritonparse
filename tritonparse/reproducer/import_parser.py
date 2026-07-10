# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

import ast
import os

from tritonparse.reproducer.import_info import ImportInfo
from tritonparse.reproducer.import_resolver import ImportResolver


class ImportParser:
    """
    Parse import statements from Python AST.

    Extracts all import statements from Python source code using AST,
    resolves them to file paths, and returns structured ImportInfo objects.
    """

    def __init__(self, import_resolver: ImportResolver) -> None:
        """
        Initialize the import parser.

        Args:
            import_resolver: ImportResolver instance for resolving import paths
        """
        self.import_resolver = import_resolver

    def parse_imports(
        self, tree: ast.Module, source_file: str, package: str | None = None
    ) -> list[ImportInfo]:
        """
        Extract all import statements from AST.

        Handles:
        - import X
        - import X as Y
        - from X import Y
        - from X import Y as Z
        - from . import X  (relative)
        - from .. import X (relative)

        Args:
            tree: Parsed AST module
            source_file: File path containing the imports
            package: Package context for relative imports (e.g., "pytorch.tritonparse")

        Returns:
            List of ImportInfo objects with resolved paths
        """
        imports: list[ImportInfo] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handle: import X, Y as Z
                imports.extend(self._parse_import_node(node, source_file))
            elif isinstance(node, ast.ImportFrom):
                # Handle: from X import Y, Z as W
                imports.extend(self._parse_import_from_node(node, source_file, package))

        return imports

    def _parse_import_node(
        self, node: ast.Import, source_file: str
    ) -> list[ImportInfo]:
        """
        Parse 'import X' statements.

        Args:
            node: ast.Import node
            source_file: File containing this import

        Returns:
            List of ImportInfo objects for each imported module
        """
        imports: list[ImportInfo] = []

        for alias in node.names:
            # Resolve the import to a file path
            resolved_path, is_external = self.import_resolver.resolve_import(alias.name)

            # Build aliases dict if alias is used
            aliases: dict[str, str] = {}
            if alias.asname:
                aliases[alias.asname] = alias.name

            imports.append(
                ImportInfo(
                    import_type="import",
                    module=alias.name,
                    names=[alias.name.split(".")[-1]],  # Last component as name
                    aliases=aliases,
                    source_file=source_file,
                    resolved_path=resolved_path,
                    is_external=is_external,
                    lineno=node.lineno,
                    level=0,  # Absolute import
                )
            )

        return imports

    def _parse_import_from_node(
        self,
        node: ast.ImportFrom,
        source_file: str,
        package: str | None = None,
    ) -> list[ImportInfo]:
        """
        Parse 'from X import Y' statements.

        Args:
            node: ast.ImportFrom node
            source_file: File containing this import
            package: Package context for relative imports

        Returns:
            List of ImportInfo objects for each imported name
        """
        imports: list[ImportInfo] = []

        # Get module name (may be None for relative imports like "from . import X")
        module = node.module or ""
        level = node.level  # 0 = absolute, 1 = ".", 2 = "..", etc.

        # For relative imports, construct the full module name
        if level > 0 and package:
            # Build relative module name
            # level=1: current package (from . import X)
            # level=2: parent package (from .. import X)
            # Formula: Remove (level-1) components from package, then append module
            # Examples:
            # - package="a.b.c", level=1, module=None -> "a.b.c" (current)
            # - package="a.b.c", level=1, module="d" -> "a.b.c.d"
            # - package="a.b.c", level=2, module=None -> "a.b" (parent)
            # - package="a.b.c", level=2, module="d" -> "a.b.d"
            package_parts = package.split(".")

            # Remove (level-1) components
            if level == 1:
                parent_package = package
            elif level <= len(package_parts):
                parent_package = ".".join(package_parts[: -(level - 1)])
            else:
                parent_package = ""

            # Append module if specified
            if module:
                full_module = f"{parent_package}.{module}" if parent_package else module
            else:
                full_module = parent_package
        else:
            full_module = module

        # Resolve the module to a file path
        resolved_path, is_external = self.import_resolver.resolve_import(full_module)

        # Relative imports can be mis-resolved by importlib to a stdlib or
        # third-party module when the package context is unknown (e.g., files
        # analyzed from a build/run-tree) or when a sibling module shadows a
        # stdlib name (e.g., a local "math.py" shadowing the stdlib "math").
        # Resolve such imports against the source file's directory on disk,
        # which is unambiguous, so their definitions can be embedded.
        if level > 0:
            disk_path = self._resolve_relative_on_disk(source_file, module, level)
            if disk_path is not None:
                resolved_path, is_external = disk_path, False

        # Parse each imported name
        for alias in node.names:
            # Build aliases dict if alias is used
            aliases: dict[str, str] = {}
            if alias.asname:
                aliases[alias.asname] = alias.name

            imports.append(
                ImportInfo(
                    import_type="from_import",
                    module=full_module,
                    names=[alias.name],
                    aliases=aliases,
                    source_file=source_file,
                    resolved_path=resolved_path,
                    is_external=is_external,
                    lineno=node.lineno,
                    level=level,
                )
            )

        return imports

    @staticmethod
    def _resolve_relative_on_disk(
        source_file: str, module: str, level: int
    ) -> str | None:
        """Resolve a relative import to a sibling module path on disk.

        importlib (find_spec) can incorrectly resolve a relative import to a
        stdlib/third-party module when the package context is unknown, or when a
        local module shadows a stdlib name (e.g., a sibling ``math.py``). This
        resolves the target directly from the importing file's directory.

        Args:
            source_file: Absolute path of the file containing the import.
            module: Module part of ``from .<module> import ...`` ("" for
                ``from . import ...``).
            level: Relative-import level (1 = ".", 2 = "..", ...).

        Returns:
            Absolute path to the target ``.py`` (or package ``__init__.py``),
            or None if it cannot be found on disk.
        """
        base_dir = os.path.dirname(os.path.abspath(source_file))
        # level=1 -> current package directory; each extra level goes up one.
        for _ in range(level - 1):
            base_dir = os.path.dirname(base_dir)
        if module:
            rel = os.path.join(*module.split("."))
            module_file = os.path.join(base_dir, rel + ".py")
            if os.path.isfile(module_file):
                return module_file
            package_init = os.path.join(base_dir, rel, "__init__.py")
            if os.path.isfile(package_init):
                return package_init
            return None
        package_init = os.path.join(base_dir, "__init__.py")
        if os.path.isfile(package_init):
            return package_init
        return None
