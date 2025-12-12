#  Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import Optional

from tritonparse.common import is_fbcode
from tritonparse.info.kernel_query import find_launch_index_by_kernel
from tritonparse.reproducer.ingestion.ndjson import build_context_bundle
from tritonparse.reproducer.placeholder_replacer import (
    DefaultPlaceholderReplacer,
    PlaceholderReplacer,
)
from tritonparse.reproducer.templates.loader import load_template_code
from tritonparse.reproducer.types import KernelImportMode
from tritonparse.reproducer.utils import determine_output_paths, format_python_code
from tritonparse.tools.prettify_ndjson import load_ndjson, save_prettified_json
from tritonparse.tp_logger import logger


def reproduce(
    input_path: str,
    line_index: int,
    out_dir: str,
    template: str,
    kernel_name: Optional[str] = None,
    launch_id: int = 0,
    replacer: Optional[PlaceholderReplacer] = None,
    kernel_import: KernelImportMode = KernelImportMode.DEFAULT,
    skip_logger: bool = False,
) -> dict[str, str]:
    """
    Generate a reproducer script from NDJSON trace file.

    Must provide either line_index OR (kernel_name + launch_id), not both.
    If kernel_name is provided, the line_index parameter will be ignored and
    recalculated from the kernel lookup.

    Args:
        input_path: Path to ndjson file. Supports uncompressed (.ndjson),
            gzip compressed (.ndjson.gz), and gzip member concatenation (.bin.ndjson) formats.
        line_index: 0-based index in events list. Ignored if kernel_name is provided.
        out_dir: Output directory for reproducer files.
        template: Template name to use for the reproducer.
        kernel_name: Exact kernel name to match (case-sensitive). If provided, line_index will be recalculated.
        launch_id: 0-based launch index for the kernel (default: 0, first launch).
        replacer: Optional custom PlaceholderReplacer instance. If None, uses DefaultPlaceholderReplacer.
        kernel_import: Kernel import mode (DEFAULT or COPY).
        skip_logger: Whether to skip usage logging (default: False).
    """
    if not skip_logger and is_fbcode():
        from tritonparse.fb.utils import usage_report_logger

        usage_report_logger()

    events = load_ndjson(Path(input_path))
    logger.debug(f"Loaded {len(events)} events")

    # If kernel_name is provided, lookup the actual line_index (overrides the parameter)
    if kernel_name is not None:
        logger.debug(
            f"Looking up kernel '{kernel_name}' launch_id={launch_id} in {input_path}"
        )
        line_index = find_launch_index_by_kernel(events, kernel_name, launch_id)
        logger.debug(
            f"Found kernel '{kernel_name}' launch_id={launch_id} at line {line_index}"
        )

    logger.debug(f"Building bundle from {input_path} at line {line_index}")

    # Build context bundle from the specified launch event
    context_bundle = build_context_bundle(events, line_index)
    logger.debug(
        f"Built context bundle for kernel: {context_bundle.kernel_info.function_name}"
    )
    out_py_path, temp_json_path = determine_output_paths(
        out_dir, context_bundle.kernel_info.function_name, template
    )
    save_prettified_json(context_bundle.raw_launch_event, temp_json_path)

    # Save compilation event JSON if using OVERRIDE_TTIR mode
    comp_json_path = None
    if kernel_import == KernelImportMode.OVERRIDE_TTIR:
        comp_json_path = (
            temp_json_path.parent / f"{temp_json_path.stem}_compilation.json"
        )
        save_prettified_json(context_bundle.raw_comp_event, comp_json_path)

    logger.debug("Loading reproducer template.")
    template_code = load_template_code(template)

    # Use PlaceholderReplacer to replace all placeholders
    # If no custom replacer provided, use the default one
    if replacer is None:
        replacer = DefaultPlaceholderReplacer()
    final_code = replacer.replace(
        template_code,
        context_bundle,
        temp_json_path=temp_json_path,
        kernel_import=kernel_import,
        comp_json_filename=comp_json_path.name if comp_json_path else None,
    )

    # Format the generated code
    final_code = format_python_code(final_code)

    out_py_path.write_text(final_code, encoding="utf-8")

    filepath = context_bundle.kernel_info.file_path
    filepath = "/".join(filepath.split("/")[5:])
    ret = {
        "kernel_src_path": filepath,
        "kernel": context_bundle.kernel_info.function_name,
        "repro_script": str(out_py_path.resolve()),
        "repro_context": str(temp_json_path.resolve()),
    }
    logger.info("REPRODUCER_OUTPUT\n%s", ret)

    return ret
