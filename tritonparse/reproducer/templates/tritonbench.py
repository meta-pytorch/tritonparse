# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from tritonbench.utils.triton_op import (
    BenchmarkOperator,
    register_benchmark,
    REGISTERED_X_VALS,
)


imported_kernel_function: Optional[Callable[[Tuple[int], Dict[str, Any]], None]] = None

# {{IR_OVERRIDE_SETUP_PLACEHOLDER}}

# {{KERNEL_SYSPATH_PLACEHOLDER}}

# {{KERNEL_IMPORT_PLACEHOLDER}}

# {{UTILITY_FUNCTIONS_PLACEHOLDER}}

assert imported_kernel_function is not None, "imported_kernel_function is missing"

KERNEL_NAME = "{{KERNEL_NAME_PLACEHOLDER}}"
REPRO_CONTEXT_FILE_NAME = "{{JSON_FILE_NAME_PLACEHOLDER}}"


def _get_launch_kernel_args() -> Tuple[Tuple[int], Dict[str, Any]]:
    script_dir = Path(__file__).resolve().parent  # noqa: F821
    json_file = script_dir / REPRO_CONTEXT_FILE_NAME

    grid, args_dict = create_args_from_json_file(json_file)  # noqa: F821, F841

    print("Recorded kernel arguments dictionary:")
    for name, arg in args_dict.items():
        if isinstance(arg, torch.Tensor):
            print(
                f"  {name}: Tensor:  {arg.shape} {arg.dtype} stride: {arg.stride()}, is_contiguous: {arg.is_contiguous()}"
            )
        else:
            print(f"  {name}: {arg}")
    print(f"Grid: {grid}")

    return tuple(grid), args_dict


grid, args_dict = _get_launch_kernel_args()


def _launch_kernel(grid: tuple[int], args_dict: dict[str, Any]):
    try:
        assert grid is not None
        assert args_dict is not None

        # {{KERNEL_INVOCATION_PLACEHOLDER}}

    except Exception as e:
        print(f"Error: {e}")
        print("Failed to launch kernel!")


# HACK: @register_x_val doesn't allow us to pass `operator_name`` as a parameter
tensor_args = {k: v for k, v in args_dict.items() if isinstance(v, torch.Tensor)}
x_vals_label = ", ".join(tensor_args.keys())
REGISTERED_X_VALS[KERNEL_NAME] = x_vals_label


class Operator(BenchmarkOperator):
    @register_benchmark(operator_name=KERNEL_NAME)
    def run_kernel(self, grid, args_dict):
        return lambda: _launch_kernel(grid, args_dict)

    def get_input_iter(self):
        yield {"grid": grid, "args_dict": args_dict}

    def get_x_val(self, example_inputs):
        tensors_shapes = [
            tuple(v.shape)
            for v in example_inputs["args_dict"].values()
            if isinstance(v, torch.Tensor)
        ]
        return tuple(tensors_shapes)


if __name__ == "__main__":
    print("do_benchmark...")

    args = [
        "--benchmark-name",
        KERNEL_NAME,
    ]

    from tritonbench.utils.parser import get_parser

    parser = get_parser(args)
    tb_args, extra_args = parser.parse_known_args(args)
    bench = Operator(tb_args, extra_args)
    bench.run()

    print(bench.output)
    print("Benchmark completed successfully!")
