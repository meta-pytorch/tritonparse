#  Copyright (c) Meta Platforms, Inc. and affiliates.
# We'd like to sperate structured logging module and tritonparse module as much as possible. So, put the shared variables here.
#
# Policy: reader-side / CLI / parse environment flags live here;
# writer-side trace-collection flags live in structured_logging.py.
import importlib.util
import logging
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


def get_enabled_analyses() -> set[str] | None:
    """
    Get user-enabled analysis list from the ``TRITONPARSE_ANALYSIS`` env var.

    Returns:
        None: enable all analyses (default).
        set: enabled analysis names (may be empty to disable all).
    """
    logger = logging.getLogger("tritonparse")
    env_value = os.environ.get("TRITONPARSE_ANALYSIS", "all").strip()

    if not env_value or env_value.lower() == "none":
        return set()
    elif env_value.lower() == "all":
        return None

    # Comma-separated analysis list — names are case-insensitive
    raw_names = [n.strip().lower() for n in env_value.split(",") if n.strip()]

    # "all"/"none" mixed into a comma list is likely misuse
    if "all" in raw_names:
        logger.warning(
            "TRITONPARSE_ANALYSIS contains 'all' mixed with other names. "
            "Use 'all' alone to enable everything, or a plain comma list for specific analyses."
        )
        return None
    if "none" in raw_names:
        logger.warning(
            "TRITONPARSE_ANALYSIS contains 'none' mixed with other names. "
            "Use 'none' alone to disable everything."
        )
        return set()

    # Validate against registered analyzers (lazy import to avoid circular deps)
    from tritonparse.parse.ir_analysis import AnalysisRegistry

    known = {name.lower() for name in AnalysisRegistry.list_analyzers()}
    unknown = {n for n in raw_names if n not in known}
    if unknown:
        logger.warning(
            f"TRITONPARSE_ANALYSIS contains unknown analyzer names: {unknown}. "
            f"Available: {sorted(known)}"
        )

    return set(raw_names) & known if known else set(raw_names)
