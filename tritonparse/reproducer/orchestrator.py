from pathlib import Path

from tritonparse.reproducer.ingestion.ndjson import build_context_bundle

from tritonparse.tools.prettify_ndjson import load_ndjson
from tritonparse.tp_logger import logger


def reproducer(
    input_path: str,
    line_index: int,
    out_dir: str,
):
    """
    Generate a reproducer script from NDJSON trace file.

    Args:
        input_path: Path to the NDJSON trace file.
        line_index: Line index of the launch event to reproduce.
        out_dir: Output directory for reproducer files.
    """
    logger.debug(f"Building bundle from {input_path} at line {line_index}")
    events = load_ndjson(Path(input_path), save_irs=True)
    logger.debug(f"Loaded {len(events)} events")

    # Build context bundle from the specified launch event
    context_bundle = build_context_bundle(events, line_index)
    logger.debug(
        f"Built context bundle for kernel: {context_bundle.kernel_info.function_name}"
    )
