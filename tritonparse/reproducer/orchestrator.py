from tritonparse.tp_logger import logger
from tritonparse.tools.prettify_ndjson import load_ndjson
from pathlib import Path

def reproducer(
    input_path: str,
    line_index: int,
    out_dir: str,
):
    logger.debug(f"Building bundle from {input_path} at line {line_index}")
    events = load_ndjson(
        Path(input_path), save_irs=True
    )
