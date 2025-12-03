#  Copyright (c) Meta Platforms, Inc. and affiliates.

import ast
import builtins
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


# Default built-in Python functions to filter out from call graph
DEFAULT_BUILTIN_FILTERS = [
    name
    for name in dir(builtins)
    if callable(getattr(builtins, name)) and not name.startswith("_")
]


@dataclass(frozen=True)
class Site:
    filename: str
    lineno: int
    col: int


@dataclass
class FuncDescriptor:
    name: str
    decorators: List[str]
    site: Site


@dataclass
class Edge:
    caller: str
    callee: str
    site: Site
    call_type: str
    callee_descriptor: Optional[FuncDescriptor]


def split_by_the_last_dot(s: str) -> Tuple[Optional[str], Optional[str]]:
    if s is None:
        return None, None
    if "." in s:
        return tuple(s.rsplit(".", 1))  # pyre-ignore[7]
    else:
        return None, s


class CallGraph(ast.NodeVisitor):
    """
    AST visitor that builds a call graph by tracking function calls and definitions.

    This class traverses an AST and records:
    - Function definitions and their decorators
    - Function calls and their call sites
    - Import statements and name bindings
    - Lambda expressions
    """

    def __init__(
        self,
        filename: str = "<string>",
        module_name: str = "<module>",
        backends: Optional[List[str]] = None,
        transitive_closure: bool = True,
        callee_prefix_filters: Optional[List[str]] = None,
        callee_name_filters: Optional[List[str]] = None,
    ):
        self.filename = filename

        self.edges: List[Edge] = []
        self.decorator_edges: List[Edge] = []
        assert backends is not None, "Backends must not be None"
        self.backends: Dict[str, List[Any]] = {}
        for backend in backends:
            self.backends[backend] = []

        self.scope_stack: List[str] = []
        self.module_name = module_name

        self.bindings_stack: List[Dict[str, str]] = [dict()]
        self.local_functions: Set[str] = set()

        # Track functions in the call chain for transitive closure
        self.transitive_closure = transitive_closure
        # Note: backends are provided as short names (e.g., "_attn_fwd_base_opt")
        # but we'll need to match them against fully qualified names later
        # We store both the short name and will add the qualified name when we see the function definition
        self.tracked_functions: Set[str] = (
            set(backends) if transitive_closure else set()
        )

        # Prefix filters to exclude certain callees (e.g., "triton.", "tl.")
        self.callee_prefix_filters = callee_prefix_filters or []

        # Name filters to exclude specific built-in function names
        # Combine user-provided filters with default built-ins
        self.callee_name_filters = set(
            (callee_name_filters or []) + DEFAULT_BUILTIN_FILTERS
        )

        # lambda node -> synthetic id (stable within this pass)
        self._lambda_ids: Dict[ast.Lambda, str] = {}

        # Store function AST nodes and source code for extraction
        self.func_nodes: Dict[str, ast.FunctionDef] = {}
        self.source_code: str = ""

    # ---------- helpers ----------
    def _cur_scope(self) -> str:
        return ".".join([self.module_name] + self.scope_stack).strip(".")

    def _push_scope(self, name: str) -> None:
        self.scope_stack.append(name)
        self.bindings_stack.append({})

    def _pop_scope(self) -> None:
        self.scope_stack.pop()
        self.bindings_stack.pop()

    def _bind(self, name: str, target: str) -> None:
        self.bindings_stack[-1][name] = target

    def _bind_func_descriptor(self, node, decorators: List[str]) -> None:
        name = node.name
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )
        self.bindings_stack[-1][f"__{name}_descriptor__"] = FuncDescriptor(
            name, decorators, site
        )

    def _resolve_name(self, id_: str) -> str:
        for env in reversed(self.bindings_stack):
            if id_ in env:
                return env[id_]
        return id_

    def _resolve_func_descriptor(self, id_: str) -> Optional[FuncDescriptor]:
        for env in reversed(self.bindings_stack):
            decorator_constant = f"__{id_}_descriptor__"
            if decorator_constant in env:
                return env[decorator_constant]
        return None

    def _resolve_attr(self, node: ast.AST) -> str:
        parts: List[str] = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            head = self._resolve_name(cur.id)
        else:
            return "<dynamic_attr>"
        return ".".join([head] + list(reversed(parts)))

    def _lambda_id(self, node: ast.Lambda) -> str:
        lid = self._lambda_ids.get(node)
        if lid is None:
            scope = self._cur_scope() or "<module>"
            lid = f"{scope}.<lambda>@{getattr(node,'lineno',-1)}:{getattr(node,'col_offset',-1)}"
            self._lambda_ids[node] = lid
        return lid

    def _record_call(
        self, callee: str, node: ast.AST, maybe_triton: bool = False, caller=None
    ) -> None:
        if caller is None:
            caller = self._cur_scope() or "<module>"
        # replace callee with caller class name if it is "self." call
        if "." in caller and callee.startswith("self."):
            caller_prefix, _ = split_by_the_last_dot(caller)
            # remove the "self." prefix
            callee_name = callee[5:]
            callee = caller_prefix + "." + callee_name
        site = Site(
            self.filename, getattr(node, "lineno", -1), getattr(node, "col_offset", -1)
        )

        # Check if callee should be filtered out based on prefix filters
        is_filtered = any(
            callee.startswith(prefix) for prefix in self.callee_prefix_filters
        )
        if is_filtered:
            return

        # Check if callee should be filtered out based on exact name match
        # Extract the function name from qualified name (e.g., "module.func" -> "func")
        callee_name = callee.split(".")[-1] if "." in callee else callee
        if callee_name in self.callee_name_filters:
            return

        # Determine if the caller should be tracked based on the transitive_closure flag
        if self.transitive_closure:
            # Transitive mode: track calls from functions in tracked_functions or matching backends
            is_tracked = caller in self.tracked_functions or any(
                backend in caller for backend in self.backends
            )
        else:
            # Backend-only mode: only track calls from functions matching backend patterns
            is_tracked = any(backend in caller for backend in self.backends)

        # In transitive closure mode, record all edges during AST traversal
        # We'll filter them afterwards based on reachability from backends
        if self.transitive_closure:
            callee_descriptor = self._resolve_func_descriptor(callee)
            self.edges.append(
                Edge(
                    caller,
                    callee,
                    callee_descriptor=callee_descriptor,
                    site=site,
                    call_type="regular",
                )
            )
        elif is_tracked:
            # Backend-only mode: only record if caller matches a backend
            callee_descriptor = self._resolve_func_descriptor(callee)
            self.edges.append(
                Edge(
                    caller,
                    callee,
                    callee_descriptor=callee_descriptor,
                    site=site,
                    call_type="regular",
                )
            )

    def _filter_edges_by_reachability(self) -> None:
        """Filter edges to keep only those reachable from backend functions.

        This implements the transitive closure: starting from backend functions,
        we iteratively add all callees until no new functions are added.
        """
        if not self.transitive_closure:
            return

        # Build caller -> callees mapping from all recorded edges
        call_graph: Dict[str, Set[str]] = {}
        for edge in self.edges:
            if edge.caller not in call_graph:
                call_graph[edge.caller] = set()
            call_graph[edge.caller].add(edge.callee)

        # Initialize reachable functions with backends
        reachable: Set[str] = set()
        for backend in self.backends:
            # Add both the short name and any fully qualified names that match
            reachable.add(backend)
            for func in self.local_functions:
                if backend in func:
                    reachable.add(func)

        # Iteratively add callees until no new functions are added
        changed = True
        while changed:
            changed = False
            new_reachable = set(reachable)
            for func in reachable:
                if func in call_graph:
                    for callee in call_graph[func]:
                        if callee not in new_reachable:
                            new_reachable.add(callee)
                            changed = True
            reachable = new_reachable

        # Filter edges to keep only those where caller is reachable
        self.edges = [edge for edge in self.edges if edge.caller in reachable]

        # Update tracked_functions to reflect reachable functions
        self.tracked_functions = reachable

    def get_unique_edges(self) -> List[Edge]:
        """Return deduplicated edges, keeping only unique (caller, callee) pairs.

        When a function calls another function from multiple locations,
        this returns only one edge representing that relationship.
        """
        seen: Set[Tuple[str, str]] = set()
        unique_edges: List[Edge] = []

        for edge in self.edges:
            key = (edge.caller, edge.callee)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)

        return unique_edges

    def get_dependent_functions(self) -> Set[str]:
        """Return all functions that are transitively called from backend functions.

        This returns the set of all functions reachable from the specified backend
        functions through the call graph. In transitive closure mode, this includes
        all functions in the call chain. In backend-only mode, this only includes
        direct callees of backend functions.

        Returns:
            Set of qualified function names that are dependencies of the backend functions.
            Excludes the backend functions themselves.
        """
        # Get all callees from the edges (functions that are called)
        dependent_funcs: Set[str] = set()
        for edge in self.edges:
            dependent_funcs.add(edge.callee)

        # Remove backend functions from the result if they appear as callees
        for backend in self.backends:
            dependent_funcs.discard(backend)
            # Also check for fully qualified backend names
            for func in list(dependent_funcs):
                if backend in func and func in self.tracked_functions:
                    # This is a backend function, remove it
                    backend_qualified = None
                    for tracked in self.tracked_functions:
                        if backend in tracked and tracked in self.local_functions:
                            backend_qualified = tracked
                            break
                    if backend_qualified and func == backend_qualified:
                        dependent_funcs.discard(func)

        return dependent_funcs

    def visit(self, node: ast.AST) -> Any:
        """Override visit to filter edges and store source code after traversal."""
        # Store source code if this is the module node
        if isinstance(node, ast.Module) and not self.source_code:
            # Read source code for later extraction
            with open(self.filename, "r") as f:
                self.source_code = f.read()

        result = super().visit(node)

        # Only filter edges when visiting the top-level Module node
        if isinstance(node, ast.Module):
            self._filter_edges_by_reachability()
        return result

    def get_dependent_functions_source_code(self) -> Dict[str, str]:
        """Return source code for all dependent functions with source location comments.

        Extracts the source code of all functions that are transitively
        called from the backend functions. Useful for creating standalone
        reproducers or understanding function dependencies.

        Returns:
            Dictionary mapping function qualified names to their source code.
            Only includes functions that are defined in the analyzed file.
            Each function's source code is prefixed with a comment indicating
            the source file and line numbers.
        """
        dependent_funcs = self.get_dependent_functions()
        result: Dict[str, str] = {}

        if not self.source_code:
            # Source code wasn't stored during visit
            with open(self.filename, "r") as f:
                self.source_code = f.read()

        source_lines = self.source_code.splitlines(keepends=True)

        for func_name in dependent_funcs:
            if func_name in self.func_nodes:
                node = self.func_nodes[func_name]

                # Determine starting line: if there are decorators, start from the first decorator
                # Otherwise start from the function definition
                if node.decorator_list:
                    # Get the first decorator's line number (1-indexed)
                    start_line = node.decorator_list[0].lineno
                else:
                    # No decorators, start from the function definition
                    start_line = node.lineno

                # Get the end line of the function (1-indexed)
                end_line = node.end_lineno

                if start_line is not None and end_line is not None:
                    # Convert to 0-indexed and extract source lines
                    func_source = "".join(source_lines[start_line - 1 : end_line])
                    # Add source location comment
                    source_comment = (
                        f"# Source: {self.filename}:{start_line}-{end_line}\n"
                    )
                    result[func_name] = source_comment + func_source

        return result

    # ---------- imports / aliases ----------
    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self._bind(name, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        for alias in node.names:
            local = alias.asname or alias.name
            target = f"{mod}.{alias.name}" if mod else alias.name
            self._bind(local, target)
        self.generic_visit(node)

    # ---------- defs ----------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return self._visit_function_like(node)

    def _visit_function_like(self, node) -> None:
        qual = (self._cur_scope() + "." if self._cur_scope() else "") + node.name
        self._bind(node.name, qual)
        self.local_functions.add(qual)

        # Store the AST node for later source code extraction
        self.func_nodes[qual] = node

        # If this function matches any backend (by name substring),
        # add its qualified name to tracked_functions for transitive tracking
        if self.transitive_closure:
            for backend in self.backends:
                if backend in qual:
                    self.tracked_functions.add(qual)
                    break

        decorators = []
        if node.decorator_list:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    callee = self._resolve_name(dec.id)
                elif isinstance(dec, ast.Attribute):
                    callee = self._resolve_attr(dec)
                elif isinstance(dec, ast.Call):
                    # best effort to guess the structure of the decorator
                    if isinstance(dec.func, ast.Name):
                        callee = f"<dynamic_decorator_{dec.func.id}>"
                    elif isinstance(dec.func, ast.Attribute):
                        if isinstance(dec.func.value, ast.Name):
                            callee = f"<dynamic_decorator_{dec.func.value.id}.{dec.func.attr}>"
                        elif isinstance(dec.func.value, ast.Attribute):
                            callee = f"<dynamic_decorator_{dec.func.value.value.id}.{dec.func.attr}>"
                        else:
                            callee = "<dynamic_decorator>"
                    else:
                        callee = "<dynamic_decorator>"
                else:
                    callee = "<dynamic_decorator>"
                decorators.append(callee)

        self._bind_func_descriptor(node, decorators)

        self._push_scope(node.name)
        if node.name in self.backends:
            self._record_call(node.name, node, maybe_triton=False, caller=node.name)
        self.generic_visit(node)
        self._pop_scope()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qual = (self._cur_scope() + "." if self._cur_scope() else "") + node.name
        self._bind(node.name, qual)
        self._push_scope(node.name)
        self.generic_visit(node)
        self._pop_scope()

    # ---------- lambda support ----------
    def visit_Lambda(self, node: ast.Lambda) -> None:
        """
        Give each lambda a synthetic qualified name and traverse its body in that scope
        so we can record calls made inside lambda bodies.
        """
        lid = self._lambda_id(node)
        # Enter a readable, stable scope name
        scope_name = lid.split(".")[-1]  # "<lambda>@line:col"
        self._push_scope(scope_name)
        # The lambda body is a single expression; visit it so nested Calls are captured
        self.visit(node.body)
        self._pop_scope()
        # Do not call generic_visit (we already visited body)

    def visit_Assign(self, node: ast.Assign) -> None:
        def rhs_symbol(n: ast.AST) -> Optional[str]:
            if isinstance(n, ast.Name):
                return self._resolve_name(n.id)
            if isinstance(n, ast.Attribute):
                return self._resolve_attr(n)
            if isinstance(n, ast.Lambda):
                return self._lambda_id(n)
            return None

        sym = rhs_symbol(node.value)
        if sym:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self._bind(t.id, sym)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # a: T = lambda ...
        if node.value is not None:
            if isinstance(node.target, ast.Name):
                if isinstance(node.value, ast.Lambda):
                    self._bind(node.target.id, self._lambda_id(node.value))
                elif isinstance(node.value, ast.Name):
                    self._bind(node.target.id, self._resolve_name(node.value.id))
                elif isinstance(node.value, ast.Attribute):
                    self._bind(node.target.id, self._resolve_attr(node.value))
        self.generic_visit(node)

    # ---------- call sites ----------
    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        maybe_triton = False
        if isinstance(fn, ast.Name):
            callee = self._resolve_name(fn.id)
        elif isinstance(fn, ast.Attribute):
            callee = self._resolve_attr(fn)
        elif isinstance(fn, ast.Lambda):
            callee = self._lambda_id(fn)  # inline IIFE-style lambda
        elif isinstance(fn, ast.Subscript):
            # Likely a Triton kernel call with subscript syntax
            if isinstance(fn.value, ast.Name):
                callee = fn.value.id
            elif isinstance(fn.value, ast.Attribute):
                callee = fn.value.value.id  # pyre-ignore[16]
                if hasattr(fn.value, "attr"):
                    callee = callee + "." + fn.value.attr
            else:
                callee = "<dynamic_call>"
            maybe_triton = True
        else:
            callee = "<dynamic_call>"

        self._record_call(callee, node, maybe_triton=maybe_triton)
        self.generic_visit(node)


def test_call_graph_analysis(
    function_name: str, module_name: str, file_path: str
) -> None:
    import ast

    print(f"Analyzing call graph for: {function_name}")
    print(f"File: {file_path}")
    print("=" * 80)

    with open(file_path, "r") as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)

    # Analyze with prefix filters
    analyzer = CallGraph(
        filename=file_path,
        module_name=module_name,
        backends=[f"{module_name}.{function_name}"],
        transitive_closure=True,
        callee_prefix_filters=["triton.", "tl."],
    )
    analyzer.visit(tree)

    print(f"\nTotal edges found: {len(analyzer.edges)}")
    print(f"Total tracked functions: {len(analyzer.tracked_functions)}")
    print(f"Total local functions: {len(analyzer.local_functions)}")

    print("\n--- Tracked Functions (all dependencies) ---")
    for func in sorted(analyzer.tracked_functions):
        print(f"  - {func}")

    print("\n--- Sample of Local Functions ---")
    for func in sorted(list(analyzer.local_functions)[:20]):
        print(f"  - {func}")

    print("\n--- Call Graph Edges (triton.* and tl.* filtered out) ---")
    unique_edges = analyzer.get_unique_edges()
    print(
        f"Unique edges: {len(unique_edges)} (total edges with duplicates: {len(analyzer.edges)})"
    )
    for i, edge in enumerate(unique_edges, 1):
        print(f"{i}. {edge.caller} -> {edge.callee} (line {edge.site.lineno})")

    print("\n--- Dependent Functions (transitively called from backend) ---")
    dependent_funcs = analyzer.get_dependent_functions()
    print(f"Total dependent functions: {len(dependent_funcs)}")
    for func in sorted(dependent_funcs):
        print(f"  - {func}")

    print("\n--- Dependent Functions Source Code ---")
    source_code_map = analyzer.get_dependent_functions_source_code()
    print(f"Total functions with source code: {len(source_code_map)}")
    for func_name in sorted(source_code_map.keys()):
        source = source_code_map[func_name]
        lines = source.split("\n")
        print(f"\n{func_name}:")
        print(f"  Lines of code: {len(lines)}")
        print("  First 3 lines:")
        for line in lines[:3]:
            print(f"    {line}")

    print("\n" + "=" * 80)
    print("Analysis complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze call graph for a specific function in a Python file."
    )
    parser.add_argument(
        "--file-path",
        "-f",
        required=True,
        help="Path to the source file containing the function",
    )
    parser.add_argument(
        "--module-name",
        "-m",
        required=True,
        help="Fully qualified module name (e.g., module.submodule.file)",
    )
    parser.add_argument(
        "--function-name",
        "-n",
        required=True,
        help="Name of the function to analyze",
    )

    args = parser.parse_args()

    test_call_graph_analysis(
        function_name=args.function_name,
        module_name=args.module_name,
        file_path=args.file_path,
    )
