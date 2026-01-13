# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Script management for bisect operations.

This module provides utilities for locating embedded shell scripts
used by the bisect functionality.
"""

from pathlib import Path

# Try importlib.resources for Python 3.9+
try:
    from importlib.resources import as_file, files

    _USE_IMPORTLIB = True
except ImportError:
    _USE_IMPORTLIB = False


def get_script_path(script_name: str) -> Path:
    """
    Get the absolute path to an embedded shell script.

    Args:
        script_name: Name of the script file (e.g., "bisect_triton.sh")

    Returns:
        Path object pointing to the script file.

    Raises:
        FileNotFoundError: If the script cannot be found.

    Example:
        >>> path = get_script_path("bisect_triton.sh")
        >>> print(path)
        /path/to/tritonparse/bisect/scripts/bisect_triton.sh
    """
    # Method 1: Use importlib.resources (Python 3.9+)
    if _USE_IMPORTLIB:
        try:
            ref = files("tritonparse.bisect.scripts").joinpath(script_name)
            with as_file(ref) as path:
                if path.exists():
                    return path.resolve()
        except (TypeError, AttributeError, ModuleNotFoundError):
            pass

    # Method 2: Fall back to __file__ relative path
    scripts_dir = Path(__file__).parent
    script_path = scripts_dir / script_name

    if script_path.exists():
        return script_path.resolve()

    raise FileNotFoundError(
        f"Script not found: {script_name}. Searched in: {scripts_dir}"
    )


def get_bisect_triton_script() -> Path:
    """Get the path to bisect_triton.sh script."""
    return get_script_path("bisect_triton.sh")


def get_bisect_llvm_script() -> Path:
    """Get the path to bisect_llvm.sh script."""
    return get_script_path("bisect_llvm.sh")


def get_test_commit_pairs_script() -> Path:
    """Get the path to test_commit_pairs.sh script."""
    return get_script_path("test_commit_pairs.sh")


def list_available_scripts() -> list[str]:
    """
    List all available shell scripts in the scripts directory.

    Returns:
        List of script filenames.
    """
    scripts_dir = Path(__file__).parent
    return [f.name for f in scripts_dir.glob("*.sh")]
