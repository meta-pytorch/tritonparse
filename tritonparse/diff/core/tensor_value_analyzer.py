#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tensor value analyzer for the diff module.

This module provides Level 5 comparison of actual tensor values
between two launch events. When tensor blobs are saved
(TRITONPARSE_SAVE_TENSOR_BLOBS=1), loads the saved tensors and
computes comprehensive numeric error metrics.
"""

import logging
from typing import Any

from tritonparse.diff.core.diff_types import TensorArgDiff, TensorValueDiff

logger = logging.getLogger(__name__)


class TensorValueAnalyzer:
    """Analyzer for Level 5 tensor value comparison.

    Compares actual numeric tensor values between two launch events
    by loading saved tensor blobs and computing error metrics.

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

        # Check if any tensor has blob_path on both sides
        has_any_blobs = any(
            args_a[name].get("blob_path") and args_b[name].get("blob_path")
            for name in common_tensors
        )
        if not has_any_blobs:
            return TensorValueDiff(
                status="skipped",
                warning=(
                    "No tensor blobs found. "
                    "Re-run tracing with TRITONPARSE_SAVE_TENSOR_BLOBS=1"
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

        # Skip if either side lacks blob_path
        if not blob_path_a or not blob_path_b:
            base_diff.status = "skipped"
            return base_diff

        # Check shape compatibility
        if shape_a != shape_b:
            base_diff.status = "shape_mismatch"
            return base_diff

        # Check dtype compatibility
        if dtype_a != dtype_b:
            base_diff.status = "dtype_mismatch"
            return base_diff

        # Load tensors
        try:
            from tritonparse.tools.load_tensor import load_tensor

            tensor_a = load_tensor(blob_path_a, device="cpu")
            tensor_b = load_tensor(blob_path_b, device="cpu")
        except Exception as e:
            logger.warning("Failed to load tensors for arg '%s': %s", name, e)
            base_diff.status = "load_error"
            return base_diff

        # Compute metrics
        metrics = self._compute_metrics(tensor_a, tensor_b)
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

    def _compute_metrics(
        self, tensor_a: Any, tensor_b: Any
    ) -> dict[str, float | int | bool | None]:
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
