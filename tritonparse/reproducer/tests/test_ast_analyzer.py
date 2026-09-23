# Copyright (c) Meta Platforms, Inc. and affiliates.


"""
Tests for CallGraph AST analyzer.

This test suite validates call graph analysis, including detection of
function references passed as arguments to higher-order functions
(e.g., tl.map_elementwise(_mask_scalar, ...)).
"""

import ast
import itertools
import os
import tempfile
import textwrap
import unittest

from tritonparse.reproducer.ast_analyzer import CallGraph


class TestHigherOrderFunctionDetection(unittest.TestCase):
    """Test that function references passed as call arguments are detected."""

    def _analyze_source(
        self,
        source: str,
        backend: str,
        callee_prefix_filters: list[str] | None = None,
    ) -> CallGraph:
        """Helper: write source to a temp file, parse, and analyze."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()
            tmp_path = tmp.name

        try:
            tree = ast.parse(source, filename=tmp_path)
            analyzer = CallGraph(
                filename=tmp_path,
                module_name="test_module",
                backends=[f"test_module.{backend}"],
                transitive_closure=True,
                callee_prefix_filters=callee_prefix_filters or [],
            )
            analyzer.visit(tree)
            return analyzer
        finally:
            os.unlink(tmp_path)

    def test_function_ref_in_positional_arg(self) -> None:
        """Test that a function passed as a positional argument is detected."""
        source = """\
def helper(x):
    return x * 2

def apply_fn(fn, data):
    return fn(data)

def main_kernel(data):
    return apply_fn(helper, data)
"""
        analyzer = self._analyze_source(source, "main_kernel")
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        # Assert: both apply_fn (direct call) and helper (passed as arg) are dependencies
        self.assertIn("apply_fn", dep_short_names)
        self.assertIn("helper", dep_short_names)

    def test_function_ref_in_keyword_arg(self) -> None:
        """Test that a function passed as a keyword argument is detected."""
        source = """\
def helper(x):
    return x + 1

def apply_fn(data, fn=None):
    return fn(data)

def main_kernel(data):
    return apply_fn(data, fn=helper)
"""
        analyzer = self._analyze_source(source, "main_kernel")
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        self.assertIn("apply_fn", dep_short_names)
        self.assertIn("helper", dep_short_names)

    def test_map_elementwise_pattern(self) -> None:
        """Test the tl.map_elementwise(_mask_scalar, ...) pattern.

        This reproduces the exact bug: _mask_scalar is passed as a function
        reference to tl.map_elementwise, which is filtered out by prefix
        filters. The function reference should still be detected.
        """
        source = """\
import triton
import triton.language as tl

@triton.jit
def _mask_scalar(qk, col_limit_right, s, i):
    return qk

@triton.jit
def _apply_causal_mask(qk, col_limit_right, BLOCK_N: tl.constexpr):
    offs_n = tl.arange(0, BLOCK_N)
    s = offs_n & ~15
    i = offs_n & 15
    return tl.map_elementwise(_mask_scalar, qk, col_limit_right, s, i)

@triton.jit
def _softmax_inner_loop(qk):
    return _apply_causal_mask(qk, 0, 128)

@triton.jit
def _attn_fwd_ws(data):
    return _softmax_inner_loop(data)
"""
        analyzer = self._analyze_source(
            source, "_attn_fwd_ws", callee_prefix_filters=["triton.", "tl."]
        )
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        # Assert: the full call chain is detected, including _mask_scalar
        self.assertIn("_softmax_inner_loop", dep_short_names)
        self.assertIn("_apply_causal_mask", dep_short_names)
        self.assertIn(
            "_mask_scalar",
            dep_short_names,
            "_mask_scalar should be detected even though it is passed as an "
            "argument to tl.map_elementwise (which is filtered by prefix)",
        )

    def test_non_function_args_are_not_detected(self) -> None:
        """Test that regular variable arguments do not create false edges."""
        source = """\
def helper():
    return 42

def main_kernel(data):
    x = 10
    return helper() + x
"""
        analyzer = self._analyze_source(source, "main_kernel")
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        # Assert: helper is detected (direct call), but no extra functions
        self.assertIn("helper", dep_short_names)
        self.assertEqual(len(dep_short_names), 1)

    def test_transitive_function_ref_detection(self) -> None:
        """Test transitive closure through function references.

        If A calls B(C) where C is passed by reference, and C calls D,
        then D should also be in the dependency set.
        """
        source = """\
def leaf_fn(x):
    return x

def mid_fn(x):
    return leaf_fn(x)

def apply_fn(fn, data):
    return fn(data)

def main_kernel(data):
    return apply_fn(mid_fn, data)
"""
        analyzer = self._analyze_source(source, "main_kernel")
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        # Assert: full chain is detected
        self.assertIn("apply_fn", dep_short_names)
        self.assertIn("mid_fn", dep_short_names)
        self.assertIn("leaf_fn", dep_short_names)

    def test_multiple_function_refs_in_one_call(self) -> None:
        """Test that multiple function references in a single call are all detected."""
        source = """\
def fn_a(x):
    return x + 1

def fn_b(x):
    return x + 2

def combine(f1, f2, data):
    return f1(data) + f2(data)

def main_kernel(data):
    return combine(fn_a, fn_b, data)
"""
        analyzer = self._analyze_source(source, "main_kernel")
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        self.assertIn("fn_a", dep_short_names)
        self.assertIn("fn_b", dep_short_names)
        self.assertIn("combine", dep_short_names)

    def test_function_ref_with_filtered_caller(self) -> None:
        """Test function ref detection works even when the enclosing call is filtered.

        When tl.map_elementwise is filtered by callee_prefix_filters,
        the function reference in its arguments should still be recorded.
        """
        source = """\
def _scalar_op(x):
    return x * 2

def kernel(data):
    return tl.map_elementwise(_scalar_op, data)
"""
        analyzer = self._analyze_source(source, "kernel", callee_prefix_filters=["tl."])
        dependent = analyzer.get_dependent_functions()
        dep_short_names = {name.split(".")[-1] for name in dependent}

        # Assert: _scalar_op should still be detected despite tl. being filtered
        self.assertIn("_scalar_op", dep_short_names)


class TestForwardHelperDependencies(unittest.TestCase):
    def _analyze(self, source: str) -> CallGraph:
        source = textwrap.dedent(source)
        graph = CallGraph(
            filename="example.py",
            module_name="example",
            backends=["kernel"],
            callee_prefix_filters=["triton.", "tl."],
        )
        graph.source_code = source
        graph.visit(ast.parse(source))
        return graph

    def test_transitive_helpers_are_independent_of_definition_order(self) -> None:
        functions = [
            "@triton.jit\ndef kernel(x):\n    return helper(x)\n",
            "@triton.jit\ndef helper(x):\n    return leaf(x) + 1\n",
            "@triton.jit\ndef leaf(x):\n    return x * 2\n",
        ]
        expected = {
            "example.helper": ast.dump(ast.parse(functions[1])),
            "example.leaf": ast.dump(ast.parse(functions[2])),
        }
        for ordered in itertools.permutations(functions):
            with self.subTest(order=[ast.parse(f).body[0].name for f in ordered]):
                graph = self._analyze("\n".join(ordered))
                extracted = graph.get_dependent_functions_source_code()
                self.assertEqual(
                    {
                        name: ast.dump(ast.parse(code))
                        for name, code in extracted.items()
                    },
                    expected,
                )

    def test_later_helper_as_higher_order_argument(self) -> None:
        for call in (
            "tl.map_elementwise(helper, x)",
            "tl.map_elementwise(x, fn=helper)",
        ):
            with self.subTest(call=call):
                graph = self._analyze(
                    f"def kernel(x):\n    return {call}\n\n"
                    "def helper(x):\n    return leaf(x)\n\n"
                    "def leaf(x):\n    return x\n"
                )
                self.assertEqual(
                    set(graph.get_dependent_functions_source_code()),
                    {"example.helper", "example.leaf"},
                )

    def test_nested_helpers_do_not_resolve_to_same_named_outer_function(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                def invoke():
                    return helper(x)
                def helper(y):
                    return leaf(y)
                return invoke()

            def helper(x):
                return unrelated(x)

            def leaf(x):
                return x

            def unrelated(x):
                return x
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()),
            {"example.kernel.invoke", "example.kernel.helper", "example.leaf"},
        )

    def test_parameters_shadow_module_helpers(self) -> None:
        for parameters in ("helper", "helper, /", "*, helper", "*helper", "**helper"):
            with self.subTest(parameters=parameters):
                graph = self._analyze(
                    f"def kernel({parameters}):\n    return apply(helper)\n\n"
                    "def helper(x):\n    return x\n"
                )
                self.assertEqual(graph.get_dependent_functions_source_code(), {})

    def test_lambda_parameter_shadows_module_helper(self) -> None:
        graph = self._analyze("""
            def kernel(callback):
                invoke = lambda helper: helper(1)
                return invoke(callback)
            def helper(x):
                return x
        """)
        self.assertNotIn("example.helper", graph.get_dependent_functions())

    def test_unknown_assignments_shadow_module_helpers(self) -> None:
        for assignment in (
            "helper = factory()",
            "helper: object = factory()",
            "helper, extra = factory()",
            "helper = None",
        ):
            with self.subTest(assignment=assignment):
                graph = self._analyze(
                    f"def kernel(x):\n    {assignment}\n    return apply(helper, x)\n\n"
                    "def helper(x):\n    return x\n"
                )
                self.assertEqual(graph.get_dependent_functions_source_code(), {})

    def test_explicit_import_binding_takes_precedence(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                from external import helper
                return helper(x)
            def helper(x):
                return x
        """)
        self.assertIn("external.helper", graph.get_dependent_functions())
        self.assertEqual(graph.get_dependent_functions_source_code(), {})

    def test_alias_to_later_helper_is_extracted(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                callback = helper
                return callback(x)
            def helper(x):
                return x
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()), {"example.helper"}
        )

    def test_defaults_resolve_before_parameter_binding(self) -> None:
        graph = self._analyze("""
            def helper(x):
                return x
            def kernel(helper=helper(1)):
                return helper
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()), {"example.helper"}
        )

    def test_recursive_helpers_terminate(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                return helper(x) if x else 0
            def helper(x):
                return kernel(x - 1)
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()), {"example.helper"}
        )

    def test_conditional_definitions_stay_in_the_enclosing_scope(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                return helper(x)
            if enabled:
                def helper(x):
                    return x
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()), {"example.helper"}
        )

    def test_class_definitions_do_not_leak_into_module_declarations(self) -> None:
        graph = self._analyze("""
            def kernel(x):
                return helper(x)
            class Other:
                def helper(self, x):
                    return unrelated(x)
            def helper(x):
                return x
        """)
        self.assertEqual(
            set(graph.get_dependent_functions_source_code()), {"example.helper"}
        )


if __name__ == "__main__":
    unittest.main()
