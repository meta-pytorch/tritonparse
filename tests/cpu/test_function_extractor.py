# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for function extractor functionality."""

import tempfile
import unittest
from pathlib import Path

from tritonparse.reproducer.function_extractor import (
    extract_autotune_config_params,
    extract_function_with_decorators,
    is_constexpr_param,
)


class TestExtractAutotuneConfigParams(unittest.TestCase):
    """Tests for extract_autotune_config_params function."""

    def test_extract_simple_config(self):
        """Test simple triton.Config patterns."""
        source = """
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32}),
    ],
    key=["M", "N"]
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N"})

    def test_extract_quoted_keys(self):
        """Test quoted keys."""
        source = """
triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4})
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_SIZE", "NUM_WARPS"})

    def test_extract_unquoted_keys(self):
        """Test unquoted keys."""
        source = """
triton.Config({BLOCK_SIZE: 128, NUM_STAGES: 2})
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_SIZE", "NUM_STAGES"})

    def test_extract_multiple_configs(self):
        """Test multiple Config instances."""
        source = """
configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_K": 32}),
    triton.Config({"BLOCK_M": 128, "BLOCK_K": 64, "BLOCK_N": 64}),
]
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_K", "BLOCK_N"})

    def test_no_configs(self):
        """Test source with no triton.Config."""
        source = """
@triton.jit
def simple_kernel(a_ptr, n_elements):
    pid = tl.program_id(0)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, set())

    def test_empty_source(self):
        """Test empty source."""
        params = extract_autotune_config_params("")
        self.assertEqual(params, set())

    def test_direct_config_import(self):
        """Test Config imported directly."""
        source = """
from triton import Config

configs = [
    Config({"BLOCK_M": 64, "BLOCK_N": 32}),
    Config({"BLOCK_M": 128, "BLOCK_N": 64}),
]
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N"})

    def test_config_with_keyword_args(self):
        """Test Config with keyword arguments."""
        source = """
triton.Config({"BLOCK_M": 64}, BLOCK_N=32, BLOCK_K=16)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K"})

    def test_config_filters_triton_specific_kwargs(self):
        """Test triton-specific kwargs are filtered."""
        source = """
triton.Config(
    {"BLOCK_M": 64, "BLOCK_N": 32},
    num_warps=4,
    num_stages=2,
    num_ctas=1,
)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N"})
        self.assertNotIn("num_warps", params)
        self.assertNotIn("num_stages", params)
        self.assertNotIn("num_ctas", params)

    def test_config_filters_all_triton_kwargs(self):
        """Test all triton kwargs are filtered."""
        source = """
triton.Config(
    {"BLOCK_SIZE": 128},
    num_warps=4,
    num_stages=3,
    num_ctas=2,
    pre_hook=some_func,
    minRegAutoWS=32,
    maxRegAutoWS=64,
    pingpongAutoWS=1,
    maxnreg=128,
)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_SIZE"})

    def test_syntax_error_handling(self):
        """Test syntax errors return empty set."""
        source = """
def broken_function(
    # Missing closing paren
triton.Config({"BLOCK": 64})
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, set())

    def test_mixed_dict_and_keyword_params(self):
        """Test dict and keyword params combined."""
        source = """
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64}, BLOCK_N=32, num_warps=4),
        triton.Config({"BLOCK_M": 128}, BLOCK_N=64, BLOCK_K=32, num_stages=2),
    ],
    key=["M"]
)
@triton.jit
def kernel():
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K"})

    def test_config_only_keyword_args(self):
        """Test Config with only keyword args."""
        source = """
triton.Config(BLOCK_M=64, BLOCK_N=32, num_warps=4)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N"})

    def test_module_level_config_list_with_direct_configs(self):
        """Test module-level config list used in autotune decorator."""
        source = """
from triton import Config

MATMUL_CONFIGS = [
    Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
        num_stages=3,
        num_warps=8,
    ),
    Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
        num_stages=4,
        num_warps=4,
    ),
]

@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"})

    def test_function_returning_config_list(self):
        """Test function that returns a list of configs."""
        source = """
from triton import Config

def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        Config(
                            {
                                "BLOCK_M": block_m,
                                "BLOCK_N": block_n,
                                "BLOCK_K": block_k,
                                "SPLIT_K": 1,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs

@triton.autotune(
    configs=get_configs_io_bound(),
    key=["M", "N", "K"],
)
@triton.jit
def io_bound_kernel(BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K"})

    def test_list_comprehension_with_inline_config(self):
        """Test list comprehension with Config."""
        source = """
configs = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=4)
    for bm in [64, 128, 256]
    for bn in [32, 64, 128]
]

@triton.autotune(configs=configs, key=["M", "N"])
@triton.jit
def kernel(BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N"})

    def test_helper_function_generates_config_with_dict_literal(self):
        """Test helper function that builds config_kwargs dict."""
        source = """
def make_tile_config(BM, BN, occ, subtile, subtile_p, vectmul, add2reduce):
    config_kwargs = {
        "BLOCK_M": BM,
        "BLOCK_N": BN,
        "occupancy": occ,
        "SUBTILING": subtile,
        "SUBTILING_P": subtile_p,
        "VECT_MUL": vectmul,
        "FADD2_REDUCE": add2reduce,
    }
    extra_kwargs = {"pre_hook": _host_descriptor_pre_hook}
    return triton.Config(config_kwargs, **extra_kwargs)

TILE_CONFIGS = [
    make_tile_config(128, 128, 2, True, False, True, True),
    make_tile_config(128, 64, 3, True, False, True, True),
]

@triton.autotune(configs=TILE_CONFIGS, key=["seqlen_q", "seqlen_k"])
@triton.jit
def attention_kernel(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    occupancy: tl.constexpr,
    SUBTILING: tl.constexpr,
    SUBTILING_P: tl.constexpr,
    VECT_MUL: tl.constexpr,
    FADD2_REDUCE: tl.constexpr,
):
    pass
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(
            params,
            {
                "BLOCK_M",
                "BLOCK_N",
                "occupancy",
                "SUBTILING",
                "SUBTILING_P",
                "VECT_MUL",
                "FADD2_REDUCE",
            },
        )

    def test_config_with_variable_dict_reference(self):
        """Test Config with variable dict reference."""
        source = """
config_dict = {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 16}
triton.Config(config_dict, num_warps=4)
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K"})

    def test_helper_with_standard_config_pattern(self):
        """Test standard config helper pattern."""
        source = """
from triton import Config

def make_standard_config(BM, BN, BK, num_stages, num_warps):
    config = {
        "BLOCK_M": BM,
        "BLOCK_N": BN,
        "BLOCK_K": BK,
    }
    return Config(config, num_stages=num_stages, num_warps=num_warps)

configs = [
    make_standard_config(128, 256, 32, 3, 8),
    make_standard_config(64, 128, 64, 4, 4),
]
"""
        params = extract_autotune_config_params(source)
        self.assertEqual(params, {"BLOCK_M", "BLOCK_N", "BLOCK_K"})


class TestIsConstexprParam(unittest.TestCase):
    """Tests for is_constexpr_param function."""

    def test_constexpr_with_spaces(self):
        """Test constexpr with spaces."""
        source = "def kernel(a_ptr, BLOCK_SIZE: tl.constexpr): pass"
        self.assertTrue(is_constexpr_param("BLOCK_SIZE", source))

    def test_constexpr_without_spaces(self):
        """Test constexpr without spaces."""
        source = "def kernel(a_ptr, BLOCK_SIZE:tl.constexpr): pass"
        self.assertTrue(is_constexpr_param("BLOCK_SIZE", source))

    def test_non_constexpr_param(self):
        """Test non-constexpr params return False."""
        source = "def kernel(a_ptr, BLOCK_SIZE: tl.constexpr): pass"
        self.assertFalse(is_constexpr_param("a_ptr", source))

    def test_param_not_in_source(self):
        """Test params not in source return False."""
        source = "def kernel(a_ptr, n_elements): pass"
        self.assertFalse(is_constexpr_param("BLOCK_SIZE", source))

    def test_multiline_signature(self):
        """Test multiline function signature."""
        source = """
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pass
"""
        self.assertTrue(is_constexpr_param("BLOCK_M", source))
        self.assertTrue(is_constexpr_param("BLOCK_N", source))
        self.assertTrue(is_constexpr_param("BLOCK_K", source))
        self.assertFalse(is_constexpr_param("M", source))
        self.assertFalse(is_constexpr_param("a_ptr", source))


class TestExtractFunctionWithDecorators(unittest.TestCase):
    """Tests for extract_function_with_decorators function."""

    def test_extract_simple_function(self):
        """Test simple function without decorators."""
        source = """
def simple_function(x, y):
    return x + y

def other_function():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()

            result = extract_function_with_decorators(tmp.name, "simple_function")

            self.assertIsNotNone(result)
            self.assertIn("def simple_function", result)
            self.assertIn("return x + y", result)
            # Should not include the other function
            self.assertNotIn("other_function", result)

            # Cleanup
            Path(tmp.name).unlink()

    def test_extract_function_with_decorator(self):
        """Test function with decorators."""
        source = """
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK": 64})],
    key=["n"]
)
@triton.jit
def my_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offsets, 0)

def helper_function():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()

            result = extract_function_with_decorators(tmp.name, "my_kernel")

            self.assertIsNotNone(result)
            # Should include decorators
            self.assertIn("@triton.autotune", result)
            self.assertIn("@triton.jit", result)
            # Should include function body
            self.assertIn("def my_kernel", result)
            self.assertIn("tl.program_id", result)
            # Should not include other function
            self.assertNotIn("helper_function", result)

            # Cleanup
            Path(tmp.name).unlink()

    def test_extract_nonexistent_function(self):
        """Test nonexistent function returns None."""
        source = """
def existing_function():
    pass
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(source)
            tmp.flush()

            result = extract_function_with_decorators(tmp.name, "nonexistent_function")

            self.assertIsNone(result)

            # Cleanup
            Path(tmp.name).unlink()

    def test_extract_from_nonexistent_file(self):
        """Test nonexistent file returns None."""
        result = extract_function_with_decorators(
            "/nonexistent/path/to/file.py", "some_function"
        )
        self.assertIsNone(result)

    def test_extract_with_source_repo_dir(self):
        """Test path resolution with source_repo_dir."""
        source = """
def target_function():
    return 42
"""
        # Create a nested directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file at tmpdir/subdir/kernel.py
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            kernel_file = subdir / "kernel.py"
            kernel_file.write_text(source)

            # Try to extract using a "production" path that doesn't exist
            # but can be resolved using source_repo_dir
            prod_path = "/some/prod/path/subdir/kernel.py"

            result = extract_function_with_decorators(
                prod_path, "target_function", source_repo_dir=tmpdir
            )

            self.assertIsNotNone(result)
            self.assertIn("def target_function", result)
            self.assertIn("return 42", result)


if __name__ == "__main__":
    unittest.main()
