#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tensor value analyzer for the diff module.

This module provides Level 5 comparison of actual tensor values
between two launch events. When tensor blobs are saved
(TRITONPARSE_SAVE_TENSOR_BLOBS=1), loads the saved tensors and
computes comprehensive numeric error metrics. When blobs are not
available but inline statistics are present
(TRITONPARSE_MORE_TENSOR_INFORMATION=1), falls back to comparing
min/max/mean/std summary statistics.
"""

import logging
from typing import Any

from tritonparse.diff.core.diff_types import TensorArgDiff, TensorValueDiff

logger = logging.getLogger(__name__)

_INLINE_STAT_KEYS = ("min", "max", "mean", "std")


def _has_inline_stats(info: dict[str, Any]) -> bool:
    """Check if a tensor arg has all required inline statistics.

    Uses ``is not None`` rather than truthiness because 0.0 is a valid
    stat value that would be falsy.

    Args:
        info: Tensor metadata dictionary from extracted_args.

    Returns:
        True if all of min, max, mean, std are present and non-None.
    """
    return all(info.get(k) is not None for k in _INLINE_STAT_KEYS)


class TensorValueAnalyzer:
    """Analyzer for Level 5 tensor value comparison.

    Compares actual numeric tensor values between two launch events.
    Supports two comparison modes:

    - **blob mode**: Loads saved tensor blobs (requires
      TRITONPARSE_SAVE_TENSOR_BLOBS=1) and computes comprehensive
      element-wise error metrics.
    - **stats mode**: Compares inline summary statistics (min, max,
      mean, std) when blobs are unavailable but
      TRITONPARSE_MORE_TENSOR_INFORMATION=1 was set during tracing.

    Blob mode is preferred when available; stats mode is a fallback.

    Attributes:
        launch_a: First launch event.
        launch_b: Second launch event.
        atol: Absolute tolerance for allclose comparison.
        rtol: Relative tolerance for allclose comparison.
    """

    def __init__(
        self,
        launch_a: dict[str, Any],
        launch_b: dict[str, Any],
        atol: float = 1e-5,
        rtol: float = 1e-3,
    ):
        """Initialize the tensor value analyzer.

        Args:
            launch_a: First launch event dictionary.
            launch_b: Second launch event dictionary.
            atol: Absolute tolerance for allclose comparison.
            rtol: Relative tolerance for allclose comparison.
        """
        self.launch_a = launch_a
        self.launch_b = launch_b
        self.atol = atol
        self.rtol = rtol

    def analyze(self) -> TensorValueDiff:
        """Analyze tensor value differences between two launch events.

        Returns:
            TensorValueDiff containing per-argument comparison results.
        """
        args_a = self.launch_a.get("extracted_args", {})
        args_b = self.launch_b.get("extracted_args", {})

        if not args_a or not args_b:
            return TensorValueDiff(
                status="skipped",
                warning="No extracted_args found in launch events",
            )

        # Find common tensor arguments
        tensor_names_a = {
            name for name, info in args_a.items() if info.get("type") == "tensor"
        }
        tensor_names_b = {
            name for name, info in args_b.items() if info.get("type") == "tensor"
        }
        common_tensors = sorted(tensor_names_a & tensor_names_b)

        if not common_tensors:
            return TensorValueDiff(
                status="skipped",
                warning="No common tensor arguments found between launch events",
            )

        # Check if any tensor has blob_path or inline stats on both sides
        has_any_blobs = any(
            args_a[name].get("blob_path") and args_b[name].get("blob_path")
            for name in common_tensors
        )
        has_any_inline_stats = any(
            _has_inline_stats(args_a[name]) and _has_inline_stats(args_b[name])
            for name in common_tensors
        )

        if not has_any_blobs and not has_any_inline_stats:
            return TensorValueDiff(
                status="skipped",
                warning=(
                    "No tensor data found for comparison. "
                    "Re-run tracing with TRITONPARSE_SAVE_TENSOR_BLOBS=1 "
                    "or TRITONPARSE_MORE_TENSOR_INFORMATION=1"
                ),
            )

        # Compare each common tensor argument
        per_arg: dict[str, TensorArgDiff] = {}
        args_compared = 0
        args_identical = 0
        args_close = 0
        args_divergent = 0

        for name in common_tensors:
            arg_diff = self._compare_arg(name, args_a[name], args_b[name])
            per_arg[name] = arg_diff

            if arg_diff.status == "skipped":
                continue

            args_compared += 1
            if arg_diff.status == "identical":
                args_identical += 1
            elif arg_diff.status == "close":
                args_close += 1
            elif arg_diff.status in (
                "divergent",
                "shape_mismatch",
                "dtype_mismatch",
            ):
                args_divergent += 1

        # Determine overall status
        if args_compared == 0:
            overall_status = "skipped"
        elif args_divergent > 0:
            overall_status = "divergent"
        elif args_identical == args_compared:
            overall_status = "identical"
        else:
            overall_status = "close"

        return TensorValueDiff(
            status=overall_status,
            args_compared=args_compared,
            args_identical=args_identical,
            args_close=args_close,
            args_divergent=args_divergent,
            atol=self.atol,
            rtol=self.rtol,
            per_arg=per_arg,
        )

    def _compare_arg(
        self, name: str, info_a: dict[str, Any], info_b: dict[str, Any]
    ) -> TensorArgDiff:
        """Compare a single tensor argument between two launches.

        Routes to blob-based or stats-based comparison depending on
        what data is available. Blob comparison is preferred.

        Args:
            name: Argument name.
            info_a: Tensor metadata from launch A's extracted_args.
            info_b: Tensor metadata from launch B's extracted_args.

        Returns:
            TensorArgDiff with comparison results.
        """
        blob_path_a = info_a.get("blob_path")
        blob_path_b = info_b.get("blob_path")

        shape_a = info_a.get("shape", [])
        shape_b = info_b.get("shape", [])
        dtype_a = info_a.get("dtype", "")
        dtype_b = info_b.get("dtype", "")

        base_diff = TensorArgDiff(
            arg_name=name,
            shape_a=shape_a,
            shape_b=shape_b,
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            metadata_a=info_a,
            metadata_b=info_b,
        )

        # Route to the appropriate comparison method
        if blob_path_a and blob_path_b:
            return self._compare_arg_blob(name, info_a, info_b, base_diff)
        elif _has_inline_stats(info_a) and _has_inline_stats(info_b):
            return self._compare_arg_stats(info_a, info_b, base_diff)
        else:
            base_diff.status = "skipped"
            return base_diff

    def _compare_arg_blob(
        self,
        name: str,
        info_a: dict[str, Any],
        info_b: dict[str, Any],
        base_diff: TensorArgDiff,
    ) -> TensorArgDiff:
        """Compare a single tensor argument using full blob data.

        Loads saved tensor blobs from disk and computes comprehensive
        element-wise error metrics.

        Args:
            name: Argument name (for logging).
            info_a: Tensor metadata from launch A's extracted_args.
            info_b: Tensor metadata from launch B's extracted_args.
            base_diff: Pre-populated TensorArgDiff with shape/dtype info.

        Returns:
            TensorArgDiff with blob-based comparison results.
        """
        # Check shape compatibility
        if base_diff.shape_a != base_diff.shape_b:
            base_diff.status = "shape_mismatch"
            return base_diff

        # Check dtype compatibility
        if base_diff.dtype_a != base_diff.dtype_b:
            base_diff.status = "dtype_mismatch"
            return base_diff

        # Load tensors
        try:
            from tritonparse.tools.load_tensor import load_tensor

            tensor_a = load_tensor(info_a["blob_path"], device="cpu")
            tensor_b = load_tensor(info_b["blob_path"], device="cpu")
        except Exception as e:
            logger.warning("Failed to load tensors for arg '%s': %s", name, e)
            base_diff.status = "load_error"
            return base_diff

        # Compute metrics
        metrics = self._compute_metrics(tensor_a, tensor_b)
        metrics["comparison_mode"] = "blob"
        base_diff.metrics = metrics

        # Determine per-arg status
        max_abs = metrics.get("max_abs_error")
        allclose = metrics.get("allclose")

        if max_abs is not None and max_abs == 0.0:
            base_diff.status = "identical"
        elif allclose:
            base_diff.status = "close"
        else:
            base_diff.status = "divergent"

        return base_diff

    def _compare_arg_stats(
        self,
        info_a: dict[str, Any],
        info_b: dict[str, Any],
        base_diff: TensorArgDiff,
    ) -> TensorArgDiff:
        """Compare a single tensor argument using inline statistics.

        Uses min/max/mean/std from TRITONPARSE_MORE_TENSOR_INFORMATION=1
        when full tensor blobs are not available. This is a lower-fidelity
        comparison that detects large-scale divergences but cannot compute
        element-wise metrics like allclose or cosine similarity.

        Args:
            info_a: Tensor metadata from launch A's extracted_args.
            info_b: Tensor metadata from launch B's extracted_args.
            base_diff: Pre-populated TensorArgDiff with shape/dtype info.

        Returns:
            TensorArgDiff with stats-based comparison results.
        """
        metrics = self._compute_stats_metrics(info_a, info_b)
        metrics["comparison_mode"] = "stats"
        base_diff.metrics = metrics

        # Determine per-arg status based on stat differences
        min_diff = metrics["min_diff"]
        max_diff = metrics["max_diff"]
        mean_diff = metrics["mean_diff"]
        std_diff = metrics["std_diff"]

        if min_diff == 0.0 and max_diff == 0.0 and mean_diff == 0.0 and std_diff == 0.0:
            base_diff.status = "identical"
        elif (
            min_diff <= self.atol
            and max_diff <= self.atol
            and mean_diff <= self.atol
            and std_diff <= self.atol
        ):
            base_diff.status = "close"
        else:
            base_diff.status = "divergent"

        return base_diff

    def _compute_stats_metrics(
        self, info_a: dict[str, Any], info_b: dict[str, Any]
    ) -> dict[str, float | int | bool | str | None]:
        """Compute comparison metrics from inline tensor statistics.

        Args:
            info_a: Tensor metadata with min/max/mean/std from launch A.
            info_b: Tensor metadata with min/max/mean/std from launch B.

        Returns:
            Dictionary of computed metrics.
        """
        min_a = info_a["min"]
        min_b = info_b["min"]
        max_a = info_a["max"]
        max_b = info_b["max"]
        mean_a = info_a["mean"]
        mean_b = info_b["mean"]
        std_a = info_a["std"]
        std_b = info_b["std"]

        return {
            "min_a": min_a,
            "min_b": min_b,
            "min_diff": abs(min_a - min_b),
            "max_a": max_a,
            "max_b": max_b,
            "max_diff": abs(max_a - max_b),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": abs(mean_a - mean_b),
            "std_a": std_a,
            "std_b": std_b,
            "std_diff": abs(std_a - std_b),
            "atol": self.atol,
        }

    def _compute_metrics(
        self, tensor_a: Any, tensor_b: Any
    ) -> dict[str, float | int | bool | str | None]:
        """Compute comprehensive numeric comparison metrics.

        Args:
            tensor_a: First tensor (torch.Tensor).
            tensor_b: Second tensor (torch.Tensor).

        Returns:
            Dictionary of computed metrics.
        """
        import torch

        # Cast to float for numeric comparison
        a = tensor_a.float()
        b = tensor_b.float()

        diff = (a - b).abs()

        # Basic error metrics
        max_abs_error = diff.max().item()
        mean_abs_error = diff.mean().item()

        # Relative error (avoid division by zero)
        denom = b.abs().clamp(min=1e-12)
        max_rel_error = (diff / denom).max().item()

        # RMSE
        rmse = diff.pow(2).mean().sqrt().item()

        # Cosine similarity
        a_flat = a.flatten().unsqueeze(0)
        b_flat = b.flatten().unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(a_flat, b_flat).item()

        # Allclose
        allclose = torch.allclose(tensor_a, tensor_b, atol=self.atol, rtol=self.rtol)

        # NaN/Inf counts
        nan_count_a = torch.isnan(a).sum().item()
        nan_count_b = torch.isnan(b).sum().item()
        inf_count_a = torch.isinf(a).sum().item()
        inf_count_b = torch.isinf(b).sum().item()

        # Mismatched elements
        not_close = ~torch.isclose(tensor_a, tensor_b, atol=self.atol, rtol=self.rtol)
        num_mismatched = not_close.sum().item()
        total_elements = tensor_a.numel()
        mismatch_ratio = num_mismatched / total_elements if total_elements > 0 else 0.0

        return {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "max_rel_error": max_rel_error,
            "rmse": rmse,
            "cosine_similarity": cos_sim,
            "allclose": allclose,
            "atol": self.atol,
            "rtol": self.rtol,
            "nan_count_a": nan_count_a,
            "nan_count_b": nan_count_b,
            "inf_count_a": inf_count_a,
            "inf_count_b": inf_count_b,
            "num_mismatched_elements": num_mismatched,
            "mismatch_ratio": mismatch_ratio,
        }


def analyze_tensor_values(
    launch_a: dict[str, Any],
    launch_b: dict[str, Any],
    atol: float = 1e-5,
    rtol: float = 1e-3,
) -> TensorValueDiff:
    """Convenience function to analyze tensor value differences.

    Args:
        launch_a: First launch event dictionary.
        launch_b: Second launch event dictionary.
        atol: Absolute tolerance for allclose comparison.
        rtol: Relative tolerance for allclose comparison.

    Returns:
        TensorValueDiff containing per-argument comparison results.
    """
    analyzer = TensorValueAnalyzer(launch_a, launch_b, atol=atol, rtol=rtol)
    return analyzer.analyze()
