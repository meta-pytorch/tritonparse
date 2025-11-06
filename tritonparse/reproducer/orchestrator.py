#  Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import Optional

from tritonparse.reproducer.ingestion.ndjson import build_context_bundle
from tritonparse.reproducer.placeholder_replacer import (
    DefaultPlaceholderReplacer,
    PlaceholderReplacer,
)
from tritonparse.reproducer.templates.loader import load_template_code
from tritonparse.reproducer.types import KernelImportMode
from tritonparse.reproducer.utils import determine_output_paths
from tritonparse.tools.prettify_ndjson import load_ndjson, save_prettified_json
from tritonparse.tp_logger import logger


def reproduce(
    input_path: str,
    line_index: int,
    out_dir: str,
    template: str,
    replacer: Optional[PlaceholderReplacer] = None,
    kernel_import: KernelImportMode = KernelImportMode.DEFAULT,
) -> dict[str, str]:
    """
    Generate a reproducer script from NDJSON trace file.

    Args:
        input_path: Path to the NDJSON trace file.
        line_index: 0-based index of the launch event to reproduce in the events list.
        out_dir: Output directory for reproducer files.
        template: Template name to use for the reproducer.
        replacer: Optional custom PlaceholderReplacer instance. If None, uses DefaultPlaceholderReplacer.
        kernel_import: Kernel import mode (DEFAULT or COPY).
    """
    logger.debug(f"Building bundle from {input_path} at line {line_index}")
    events = load_ndjson(Path(input_path))
    logger.debug(f"Loaded {len(events)} events")

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
