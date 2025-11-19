# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import inspect
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_function_source(
    func: Callable[[], None], with_invocation: bool = True
) -> list[str]:
    """
    Extract function source code and optionally include invocation.

    Args:
        func: Function to extract source code from
        with_invocation: Whether to include function invocation code

    Returns:
        List containing source code and optional invocation statement
    """
    source = inspect.getsource(func).rstrip()
    result = [source]

    if with_invocation:
        result.append("")
        result.append(f"{func.__name__}()")

    return result


def _disable_triton_autotune() -> None:
    """
    Monkey patch the triton.autotune decorator to skip autotuning entirely.
    """
    logger.info("Disabling triton autotune")

    def dummy_autotune(configs, key=None, **kwargs):
        def decorator(func):
            return func  # Just pass through, let @triton.jit handle compilation

        return decorator

    import triton

    triton.autotune = dummy_autotune
    logger.info("Disabled triton autotune")
