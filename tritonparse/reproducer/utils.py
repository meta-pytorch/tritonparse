from datetime import datetime
from pathlib import Path


def determine_output_paths(out_dir: str, kernel_name: str):
    """
    Determine output file paths for reproducer script and context data.

    Args:
        out_dir: Output directory path. If empty, uses default location.
        kernel_name: Name of the kernel for default directory naming.

    Returns:
        Tuple of (python_script_path, json_context_path) as Path objects.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_directory = Path(out_dir) / kernel_name
    output_directory.mkdir(parents=True, exist_ok=True)

    out_py_path = output_directory / f"repro_{timestamp}.py"
    temp_json_path = output_directory / f"repro_context_{timestamp}.json"

    return out_py_path, temp_json_path
