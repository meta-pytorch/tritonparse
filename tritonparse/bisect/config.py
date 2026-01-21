# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Bisect configuration from environment variables.

This module centralizes all environment variable reads for the bisect module,
providing consistent parsing and default values that match the bash scripts.

Environment Variables:
    USE_UV: Use uv package manager instead of pip/conda (default: "0")
            Set to "1" to enable uv mode.

Example:
    >>> from tritonparse.bisect.config import get_config
    >>> config = get_config()
    >>> if config.use_uv:
    ...     print("Using uv package manager")
"""

import os
from dataclasses import dataclass


def _get_bool_env(name: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.

    Treats "1" as True, all other values (including "0", "", None) as False.
    This matches the bash script behavior: [[ "$VAR" == "1" ]].

    Args:
        name: Environment variable name.
        default: Default value if not set (not used, kept for API consistency).

    Returns:
        True if the environment variable is set to "1", False otherwise.
    """
    return os.environ.get(name) == "1"


def _get_str_env(name: str, default: str = "") -> str:
    """
    Get string environment variable with default.

    Args:
        name: Environment variable name.
        default: Default value if not set.

    Returns:
        The environment variable value or default.
    """
    return os.environ.get(name, default)


@dataclass(frozen=True)
class BisectConfig:
    """
    Configuration for bisect operations from environment variables.

    This dataclass holds all configuration values read from environment
    variables. It is frozen (immutable) to prevent accidental modification.

    Attributes:
        use_uv: Whether to use uv package manager instead of pip.
                Corresponds to USE_UV environment variable.
    """

    use_uv: bool = False


def get_config() -> BisectConfig:
    """
    Get bisect configuration from environment variables.

    This function reads all relevant environment variables and returns
    a BisectConfig object with the parsed values.

    Returns:
        BisectConfig object with current configuration.

    Example:
        >>> import os
        >>> os.environ["USE_UV"] = "1"
        >>> config = get_config()
        >>> config.use_uv
        True
        >>> os.environ["USE_UV"] = "0"
        >>> config = get_config()
        >>> config.use_uv
        False
    """
    return BisectConfig(
        use_uv=_get_bool_env("USE_UV"),
    )
