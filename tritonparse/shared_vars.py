#  Copyright (c) Meta Platforms, Inc. and affiliates.
# We'd like to sperate structured logging module and tritonparse module as much as possible. So, put the shared variables here.
import importlib.util
import os

# The compilation information will be stored to /logs/DEFAULT_TRACE_FILE_PREFIX by default
# unless other flags disable or set another store. Add USER to avoid permission issues in shared servers.
DEFAULT_TRACE_FILE_PREFIX = (
    f"dedicated_log_triton_trace_{os.getenv('USER', 'unknown')}_"
)
DEFAULT_TRACE_FILE_PREFIX_WITHOUT_USER = "dedicated_log_triton_trace_"
# Return True if test outputs (e.g., temp dirs) should be preserved.
TEST_KEEP_OUTPUT = os.getenv("TEST_KEEP_OUTPUT", "0") in ["1", "true", "True"]


def is_fbcode():
    """Check if running in fbcode environment.

    Can be overridden via the TRITONPARSE_FB_MODE environment variable:
      TRITONPARSE_FB_MODE=0  — force OSS mode (skip all fb imports)
      TRITONPARSE_FB_MODE=1  — force fbcode mode
    If unset, auto-detects by checking whether tritonparse.fb is importable.

    This is useful when tritonparse is installed from fbcode into an OSS
    environment on a devserver (e.g., a conda/venv setup). In that case,
    the tritonparse.fb package is physically present so auto-detection
    returns True, but Meta-internal dependencies like manifold are not
    installed, causing ImportError at runtime. Set TRITONPARSE_FB_MODE=0
    to force OSS mode in such environments.
    """
    override = os.getenv("TRITONPARSE_FB_MODE")
    if override is not None:
        return override in ("1", "true", "True")
    return importlib.util.find_spec("tritonparse.fb") is not None
