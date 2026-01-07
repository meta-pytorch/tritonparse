#  Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from typing import Optional

# Check if debug mode is enabled via environment variable
TRITONPARSE_DEBUG = os.getenv("TRITONPARSE_DEBUG", None) in ["1", "true", "True"]

# Main logger for tritonparse
logger = logging.getLogger("tritonparse")

# Set default level based on TRITONPARSE_DEBUG environment variable
# This only affects tritonparse.* loggers, not the root logger or other libraries
logger.setLevel(logging.DEBUG if TRITONPARSE_DEBUG else logging.INFO)

# Add a default StreamHandler if no handlers are configured
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
    # Prevent logs from propagating to root logger to avoid duplicate output
    logger.propagate = False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a tritonparse logger instance.

    Args:
        name: Sub-module name. If provided, creates a "tritonparse.{name}" child logger.
              If None, returns the main "tritonparse" logger.

    Returns:
        logging.Logger: Logger instance

    Examples:
        >>> from tritonparse.tp_logger import get_logger
        >>> logger = get_logger()  # Returns "tritonparse" logger
        >>> logger = get_logger("SourceMapping")  # Returns "tritonparse.SourceMapping" logger
    """
    if name:
        return logging.getLogger(f"tritonparse.{name}")
    return logger
