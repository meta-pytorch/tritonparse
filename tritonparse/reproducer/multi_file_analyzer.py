# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""
MultiFileCallGraphAnalyzer: Orchestrates multi-file static call graph analysis.

This module provides the main analyzer that coordinates traversal across multiple
Python files, following imports to extract all transitively-called functions.
"""

import argparse
import ast
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Set

from tritonparse.reproducer.ast_analyzer import CallGraph, Edge
from tritonparse.reproducer.consolidated_result import AnalysisStats, ConsolidatedResult
from tritonparse.reproducer.import_info import ImportInfo
from tritonparse.reproducer.import_parser import ImportParser
from tritonparse.reproducer.import_resolver import ImportResolver

logger = logging.getLogger(__name__)


def _auto_detect_code_root(entry_file: str) -> str:
    """
    Auto-detect code root from entry file path.

    Walks up the directory tree from the entry file until it finds a common
    code root indicator. Currently supports:
    - "fbcode" directories (Meta's monorepo structure)
    - Directories containing "setup.py", "pyproject.toml" (Python projects)
    - Git repository roots (directories containing ".git")

    Args:
        entry_file: Absolute path to the entry file

    Returns:
        Absolute path to the code root directory

    Raises:
        ValueError: If code root cannot be detected from the entry file path
    """
    path = Path(entry_file).resolve()
    logger.debug("Auto-detecting code root from: %s", path)

    # Walk up the directory tree
    for parent in path.parents:
        # Check for fbcode (Meta's monorepo)
        if parent.name == "fbcode" or parent.name == "fbsource":
            logger.info("Auto-detected code root (fbcode): %s", parent)
            return str(parent)

        # Check for Python project markers
        if (parent / "setup.py").exists() or (parent / "pyproject.toml").exists():
            logger.info("Auto-detected code root (Python project): %s", parent)
            return str(parent)

        # Check for Git repository root
        if (parent / ".git").exists():
            logger.info("Auto-detected code root (Git repository): %s", parent)
            return str(parent)

    raise ValueError(
        f"Could not auto-detect code root from entry file: {entry_file}. "
        "Please specify --code-root explicitly."
    )


class MultiFileCallGraphAnalyzer:
    """
    Multi-file static call graph analyzer for creating reproducers.

    This analyzer orchestrates the analysis across multiple Python files by:
    1. Starting with an entry file and function
    2. Using CallGraph to extract dependencies within each file
    3. Following imports to analyze dependent files
    4. Consolidating results from all analyzed files
    """

    def __init__(
        self,
        entry_file: str,
        entry_function: str,
        code_roots: Optional[str] = None,
    ) -> None:
        """
        Initialize multi-file analyzer.

        The analyzer automatically computes the qualified backend name from
        entry_file + entry_function, removing the need for manual backend specification.

        Args:
            entry_file: Path to file containing entry function
            entry_function: Name of entry function (short name, e.g., "main_kernel")
            code_roots: Absolute path to default code root directory (auto-detected if None)

        Example:
            >>> analyzer = MultiFileCallGraphAnalyzer(
            ...     entry_file="/path/to/kernel.py",
            ...     entry_function="main_kernel",
            ...     code_roots="/path/to/fbcode",  # Optional
            ... )
        """
        self.entry_file = entry_file
        self.entry_function = entry_function

        # Auto-detect code_roots if not provided
        if code_roots is None:
            code_roots = _auto_detect_code_root(entry_file)
        self.code_roots = code_roots

        # Track multiple code roots for files from different projects
        self.file_to_code_root: dict[str, str] = {entry_file: code_roots}

        entry_module = self._file_to_module_name(entry_file)
        qualified_backend = f"{entry_module}.{entry_function}"
        self.backends = [entry_function]
        self.qualified_backend = qualified_backend

        self.import_resolver = ImportResolver(code_roots)
        self.import_parser = ImportParser(self.import_resolver)

        self.visited_files: Set[str] = set()
        self.pending_files: list[str] = []
        self.file_analyzers: dict[str, CallGraph] = {}
        self.all_imports: dict[str, list[ImportInfo]] = {}
        self.used_imports: dict[str, list[ImportInfo]] = {}

    def analyze(self) -> ConsolidatedResult:
        """
        Perform multi-file analysis.

        This is the main entry point that:
        1. Analyzes entry file
        2. Recursively analyzes imported files (breadth-first)
        3. Consolidates results

        Returns:
            ConsolidatedResult with functions, imports, and statistics
        """
        self._analyze_file(self.entry_file)

        while self.pending_files:
            pending_item = self.pending_files.pop(0)

            if isinstance(pending_item, tuple):
                file_path, backends_for_file = pending_item
            else:
                file_path = pending_item
                backends_for_file = None

            if file_path not in self.visited_files:
                self._analyze_file(file_path, backends_for_file)

        return self._consolidate_results()

    def _analyze_file(
        self, file_path: str, backends_for_file: Optional[list[str]] = None
    ) -> None:
        """
        Analyze a single file with CallGraph.

        This method:
        1. Marks file as visited
        2. Creates CallGraph analyzer for the file
        3. Extracts dependent functions
        4. Parses imports from AST
        5. Identifies which imports are used by dependent functions
        6. Adds imported files to pending_files queue

        Args:
            file_path: Absolute path to the Python file to analyze
            backends_for_file: Specific backends for this file (defaults to self.backends for entry file)
        """
        self.visited_files.add(file_path)
        logger.info("Analyzing %s", file_path)

        with open(file_path) as f:
            source_code = f.read()
        tree = ast.parse(source_code, filename=file_path)

        file_backends = (
            backends_for_file if backends_for_file is not None else self.backends
        )

        module_name = self._file_to_module_name(file_path)
        analyzer = CallGraph(
            filename=file_path,
            module_name=module_name,
            backends=file_backends,
            transitive_closure=True,
        )
        analyzer.visit(tree)

        self.file_analyzers[file_path] = analyzer

        module_name = self._file_to_module_name(file_path)
        package_name = (
            ".".join(module_name.split(".")[:-1]) if "." in module_name else None
        )
        imports = self.import_parser.parse_imports(
            tree, file_path, package=package_name
        )
        self.all_imports[file_path] = imports

        used_imports_list = self._identify_used_imports(imports, analyzer)
        self.used_imports[file_path] = used_imports_list

        imports_by_file: dict[str, list[str]] = {}
        for import_info in used_imports_list:
            if (
                import_info.resolved_path
                and not import_info.is_external
                and import_info.resolved_path not in self.visited_files
            ):
                if import_info.resolved_path not in imports_by_file:
                    imports_by_file[import_info.resolved_path] = []
                imports_by_file[import_info.resolved_path].extend(import_info.names)

        if imports_by_file:
            logger.info(
                "Found %d internal import file(s) to analyze from %s",
                len(imports_by_file),
                file_path,
            )
            for resolved_path, imported_names in imports_by_file.items():
                logger.debug(
                    "  → Will analyze %s for functions: %s",
                    resolved_path,
                    imported_names,
                )
        else:
            logger.debug(
                "No internal imports found in %s (all imports are external or already visited)",
                file_path,
            )

        for resolved_path, imported_names in imports_by_file.items():
            already_pending = any(
                (
                    pending_item[0] == resolved_path
                    if isinstance(pending_item, tuple)
                    else pending_item == resolved_path
                )
                for pending_item in self.pending_files
            )
            if not already_pending:
                unique_names = list(dict.fromkeys(imported_names))
                self.pending_files.append((resolved_path, unique_names))
                logger.info(
                    "Added %s to pending analysis queue with backends: %s",
                    resolved_path,
                    unique_names,
                )

    def _identify_used_imports(
        self,
        imports: list[ImportInfo],
        analyzer: CallGraph,
    ) -> list[ImportInfo]:
        """
        Identify which imports are actually used by dependent functions.

        Strategy:
        1. Get all callees from dependent functions (from call graph edges)
        2. Match callees to import statements
        3. Return only imports that are actually used

        Args:
            imports: All imports in the file
            analyzer: CallGraph analyzer for this file

        Returns:
            List of ImportInfo objects for imports that are used
        """
        used_symbols = self._extract_used_symbols(analyzer)
        logger.debug("Used symbols: %s", used_symbols)

        matching_imports = self._find_matching_imports(imports, used_symbols)
        import_groups = self._group_imports(matching_imports)
        used_imports_list = self._select_best_imports(import_groups)

        return used_imports_list

    def _extract_used_symbols(self, analyzer: CallGraph) -> Set[str]:
        """Extract all symbols used in the call graph."""
        used_symbols: Set[str] = set()
        for edge in analyzer.edges:
            used_symbols.add(edge.callee)
            callee_parts = edge.callee.split(".")
            if callee_parts:
                used_symbols.add(callee_parts[0])
        return used_symbols

    def _find_matching_imports(
        self, imports: list[ImportInfo], used_symbols: Set[str]
    ) -> list[tuple[ImportInfo, bool]]:
        """Find imports that match the used symbols."""
        matching_imports: list[tuple[ImportInfo, bool]] = []

        for import_info in imports:
            logger.debug(
                "Checking import: %s %s -> %s (external: %s)",
                import_info.import_type,
                import_info.module,
                import_info.names,
                import_info.is_external,
            )

            is_used = self._is_import_used(import_info, used_symbols)
            if is_used:
                is_internal_match = not import_info.is_external
                matching_imports.append((import_info, is_internal_match))

        return matching_imports

    def _is_import_used(self, import_info: ImportInfo, used_symbols: Set[str]) -> bool:
        """Check if an import is used based on the used symbols."""
        if self._matches_qualified_name(import_info, used_symbols):
            return True
        if self._matches_short_name(import_info, used_symbols):
            return True
        if self._matches_name(import_info, used_symbols):
            return True
        if self._matches_alias(import_info, used_symbols):
            return True
        if self._matches_module_prefix(import_info, used_symbols):
            return True
        return False

    def _matches_qualified_name(
        self, import_info: ImportInfo, used_symbols: Set[str]
    ) -> bool:
        """Check if import matches on qualified name."""
        if import_info.import_type != "from_import" or not import_info.module:
            return False

        for name in import_info.names:
            qualified_name = f"{import_info.module}.{name}"
            for symbol in used_symbols:
                if symbol == qualified_name or symbol.startswith(qualified_name + "."):
                    logger.debug(
                        "  Matched on qualified name: %s == %s",
                        qualified_name,
                        symbol,
                    )
                    return True
        return False

    def _matches_short_name(
        self, import_info: ImportInfo, used_symbols: Set[str]
    ) -> bool:
        """Check if import matches on short module name."""
        if import_info.import_type != "from_import" or not import_info.module:
            return False

        module_short_name = import_info.module.split(".")[-1]
        for name in import_info.names:
            short_qualified = f"{module_short_name}.{name}"
            if short_qualified in used_symbols:
                logger.debug(
                    "  Matched on short name: %s in used_symbols",
                    short_qualified,
                )
                return True
        return False

    def _matches_name(self, import_info: ImportInfo, used_symbols: Set[str]) -> bool:
        """Check if import matches on name."""
        for name in import_info.names:
            if name in used_symbols:
                logger.debug("  Matched on name: %s in used_symbols", name)
                return True
        return False

    def _matches_alias(self, import_info: ImportInfo, used_symbols: Set[str]) -> bool:
        """Check if import matches on alias."""
        for alias in import_info.aliases:
            if alias in used_symbols:
                logger.debug("  Matched on alias: %s in used_symbols", alias)
                return True
        return False

    def _matches_module_prefix(
        self, import_info: ImportInfo, used_symbols: Set[str]
    ) -> bool:
        """Check if import matches on module prefix."""
        if import_info.import_type != "import":
            return False

        module_prefix = import_info.module + "."
        for symbol in used_symbols:
            if symbol.startswith(module_prefix):
                logger.debug("  Matched on module prefix: %s", module_prefix)
                return True
        return False

    def _group_imports(
        self, matching_imports: list[tuple[ImportInfo, bool]]
    ) -> dict[tuple[str, tuple[str, ...]], list[tuple[ImportInfo, bool]]]:
        """Group imports by module and names."""
        import_groups: dict[
            tuple[str, tuple[str, ...]], list[tuple[ImportInfo, bool]]
        ] = {}

        for import_info, is_internal_match in matching_imports:
            module_short = (
                import_info.module.split(".")[-1] if import_info.module else ""
            )
            names_key = tuple(sorted(import_info.names))
            key = (module_short, names_key)

            if key not in import_groups:
                import_groups[key] = []
            import_groups[key].append((import_info, is_internal_match))

        return import_groups

    def _select_best_imports(
        self,
        import_groups: dict[tuple[str, tuple[str, ...]], list[tuple[ImportInfo, bool]]],
    ) -> list[ImportInfo]:
        """Select the best import from each group."""
        used_imports_list: list[ImportInfo] = []

        for group_imports in import_groups.values():
            group_imports.sort(key=lambda x: (not x[1], x[0].is_external))
            used_imports_list.append(group_imports[0][0])

        return used_imports_list

    def _get_code_root_for_file(self, file_path: str) -> str:
        """
        Get or auto-detect the code root for a specific file.

        This allows handling files from different project roots within the same analysis.

        Args:
            file_path: Absolute path to Python file

        Returns:
            Code root for this file (from cache or auto-detected)
        """
        if file_path in self.file_to_code_root:
            return self.file_to_code_root[file_path]

        # Auto-detect code root for this file
        try:
            code_root = _auto_detect_code_root(file_path)
            self.file_to_code_root[file_path] = code_root
            logger.debug("Detected code root %s for file %s", code_root, file_path)
            return code_root
        except ValueError:
            # Fall back to the analyzer's default code root
            logger.warning(
                "Could not auto-detect code root for %s, using default: %s",
                file_path,
                self.code_roots,
            )
            self.file_to_code_root[file_path] = self.code_roots
            return self.code_roots

    def _file_to_module_name(self, file_path: str) -> str:
        """
        Convert file path to Python module name.

        Example:
            /data/users/wychi/fbsource/fbcode/pytorch/tritonparse/module.py
            -> pytorch.tritonparse.module

        Args:
            file_path: Absolute path to Python file

        Returns:
            Module name as a dotted string
        """
        code_root = self._get_code_root_for_file(file_path)
        code_root_path = Path(code_root)
        file = Path(file_path)

        try:
            rel_path = file.relative_to(code_root_path)
            module_path = str(rel_path).replace("/", ".").removesuffix(".py")
            return module_path
        except ValueError:
            # File is not under the detected code root
            # Fall back to using the file's stem as the module name
            logger.warning(
                "File %s is not under code root %s, using file stem as module name",
                file_path,
                code_root,
            )
            return file.stem

    def _consolidate_results(self) -> ConsolidatedResult:
        """
        Consolidate results from all file analyzers.

        This method:
        1. Collects all functions and their source code from all files
        2. Tracks function locations
        3. Collects and deduplicates imports
        4. Collects all call graph edges
        5. Builds statistics

        Returns:
            ConsolidatedResult with all analysis results
        """
        all_functions: dict[str, str] = {}
        function_to_file: dict[str, str] = {}
        all_edges: List[Edge] = []

        functions_to_extract = self._collect_functions_to_extract(all_edges)
        self._extract_function_sources(
            functions_to_extract, all_functions, function_to_file
        )
        all_imports_list = self._collect_all_imports()
        unique_imports = self._deduplicate_imports(all_imports_list)

        stats = AnalysisStats(
            total_files_analyzed=len(self.visited_files),
            total_functions_found=len(all_functions),
            total_imports=len(unique_imports),
            external_imports=sum(1 for imp in unique_imports if imp.is_external),
            internal_imports=sum(1 for imp in unique_imports if not imp.is_external),
        )

        return ConsolidatedResult(
            functions=all_functions,
            function_locations=function_to_file,
            function_short_names={
                qualified: qualified.split(".")[-1]
                for qualified in all_functions.keys()
            },
            imports=unique_imports,
            edges=all_edges,
            analyzed_files=self.visited_files.copy(),
            stats=stats,
        )

    def _collect_functions_to_extract(self, all_edges: List[Edge]) -> set[str]:
        """Collect all functions that need to be extracted from analyzers."""
        functions_to_extract: set[str] = set()

        for _file_path, analyzer in self.file_analyzers.items():
            dependent_funcs = analyzer.get_dependent_functions()
            functions_to_extract.update(dependent_funcs)

            for backend in analyzer.backends:
                for local_func in analyzer.local_functions:
                    if local_func.split(".")[-1] == backend:
                        # Skip the primary entry function - it's not a dependency
                        if local_func == self.qualified_backend:
                            logger.debug(
                                "Skipping entry function %s (not a dependency)",
                                local_func,
                            )
                            continue

                        logger.debug(
                            "Adding backend function %s from file %s (backend: %s)",
                            local_func,
                            _file_path,
                            backend,
                        )
                        functions_to_extract.add(local_func)

            all_edges.extend(analyzer.edges)

        return functions_to_extract

    def _extract_function_sources(
        self,
        functions_to_extract: set[str],
        all_functions: dict[str, str],
        function_to_file: dict[str, str],
    ) -> None:
        """Extract source code for all functions from file analyzers."""
        for file_path, analyzer in self.file_analyzers.items():
            for func_name in functions_to_extract:
                if func_name in analyzer.func_nodes:
                    self._extract_single_function_source(
                        func_name, file_path, analyzer, all_functions, function_to_file
                    )

    def _extract_single_function_source(
        self,
        func_name: str,
        file_path: str,
        analyzer: CallGraph,
        all_functions: dict[str, str],
        function_to_file: dict[str, str],
    ) -> None:
        """Extract source code for a single function with source location comment."""
        source_code_map = analyzer.get_dependent_functions_source_code()

        if func_name in source_code_map:
            # Source location comment is already added by get_dependent_functions_source_code
            all_functions[func_name] = source_code_map[func_name]
            function_to_file[func_name] = file_path
        else:
            self._extract_function_from_ast(
                func_name, file_path, analyzer, all_functions, function_to_file
            )

    def _extract_function_from_ast(
        self,
        func_name: str,
        file_path: str,
        analyzer: CallGraph,
        all_functions: dict[str, str],
        function_to_file: dict[str, str],
    ) -> None:
        """Extract function source code directly from AST node with source location comment."""
        node = analyzer.func_nodes[func_name]

        if not analyzer.source_code:
            with open(file_path) as f:
                analyzer.source_code = f.read()

        source_lines = analyzer.source_code.splitlines(keepends=True)

        if node.decorator_list:
            start_line = node.decorator_list[0].lineno
        else:
            start_line = node.lineno

        end_line = node.end_lineno

        if start_line is not None and end_line is not None:
            func_source = "".join(source_lines[start_line - 1 : end_line])
            # Add source location comment
            source_comment = f"# Source: {file_path}:{start_line}-{end_line}\n"
            all_functions[func_name] = source_comment + func_source
            function_to_file[func_name] = file_path

    def _collect_all_imports(self) -> list[ImportInfo]:
        """Collect all imports from visited files."""
        all_imports_list: list[ImportInfo] = []
        for file_path in self.visited_files:
            if file_path in self.used_imports:
                all_imports_list.extend(self.used_imports[file_path])
        return all_imports_list

    def _deduplicate_imports(self, imports: list[ImportInfo]) -> list[ImportInfo]:
        """
        Deduplicate imports while preserving order.

        Merges duplicate imports:
        - from X import A
        - from X import B
        -> from X import A, B

        Args:
            imports: List of ImportInfo objects (may contain duplicates)

        Returns:
            Deduplicated list of ImportInfo objects
        """
        import_groups: dict[tuple[str, str], ImportInfo] = {}

        for imp in imports:
            key = (imp.import_type, imp.module)
            if key in import_groups:
                existing = import_groups[key]
                for name in imp.names:
                    if name not in existing.names:
                        existing.names.append(name)
                existing.aliases.update(imp.aliases)
            else:
                import_groups[key] = ImportInfo(
                    import_type=imp.import_type,
                    module=imp.module,
                    names=imp.names.copy(),
                    source_file=imp.source_file,
                    resolved_path=imp.resolved_path,
                    is_external=imp.is_external,
                    lineno=imp.lineno,
                    aliases=imp.aliases.copy(),
                    level=imp.level,
                )

        return list(import_groups.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-file call graph analyzer for Python code. "
        "Analyzes a Python function and its dependencies across multiple files, "
        "extracting all transitively-called functions and their imports."
    )
    parser.add_argument(
        "--entry-file",
        "-f",
        required=True,
        help="Path to the entry file containing the function to analyze",
    )
    parser.add_argument(
        "--entry-function",
        "-F",
        required=True,
        help="Name of the entry function to analyze",
    )
    parser.add_argument(
        "--code-roots",
        "-r",
        default="",
        help="Path to default code root directory (default: auto-detect from entry file path)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON file path (default: creates a temp file in /tmp/)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Auto-detect code_roots if not provided
    code_roots = args.code_roots
    if not code_roots:
        try:
            code_roots = _auto_detect_code_root(args.entry_file)
        except ValueError as e:
            logger.error("%s", e)
            exit(1)

    analyzer = MultiFileCallGraphAnalyzer(
        entry_file=args.entry_file,
        entry_function=args.entry_function,
        code_roots=code_roots,
    )

    logger.info("Starting analysis of %s in %s", args.entry_function, args.entry_file)
    result = analyzer.analyze()

    logger.info("Analysis complete:")
    logger.info("  Files analyzed: %d", result.stats.total_files_analyzed)
    logger.info("  Functions found: %d", result.stats.total_functions_found)
    logger.info("  Total imports: %d", result.stats.total_imports)
    logger.info("  External imports: %d", result.stats.external_imports)
    logger.info("  Internal imports: %d", result.stats.internal_imports)

    logger.info("\nDependent functions (short names):")
    for func_name in sorted(result.function_short_names.keys()):
        short_name = result.function_short_names[func_name]
        if short_name != args.entry_function:
            logger.info(
                "  - %s. code size: %d", short_name, len(result.functions[func_name])
            )

    output_data = {
        "entry_file": args.entry_file,
        "entry_function": args.entry_function,
        "qualified_backend": analyzer.qualified_backend,
        "stats": {
            "total_files_analyzed": result.stats.total_files_analyzed,
            "total_functions_found": result.stats.total_functions_found,
            "total_imports": result.stats.total_imports,
            "external_imports": result.stats.external_imports,
            "internal_imports": result.stats.internal_imports,
        },
        "analyzed_files": sorted(result.analyzed_files),
        "functions": {
            func_name: {
                "source": source,
                "file": result.function_locations.get(func_name, "unknown"),
                "short_name": result.function_short_names.get(func_name, func_name),
            }
            for func_name, source in result.functions.items()
        },
        "imports": [
            {
                "import_type": imp.import_type,
                "module": imp.module,
                "names": imp.names,
                "source_file": imp.source_file,
                "resolved_path": imp.resolved_path,
                "is_external": imp.is_external,
                "lineno": imp.lineno,
                "aliases": imp.aliases,
                "level": imp.level,
            }
            for imp in result.imports
        ],
        "edges": [
            {"caller": edge.caller, "callee": edge.callee} for edge in result.edges
        ],
    }

    if args.output:
        output_path = args.output
    else:
        # Create a temp file with a descriptive name
        import os

        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix=f"multi_file_analysis_{args.entry_function}_",
            dir="/tmp",
            text=True,
        )
        os.close(temp_fd)  # Close the file descriptor
        output_path = temp_path

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Detailed results written to: %s", output_path)
