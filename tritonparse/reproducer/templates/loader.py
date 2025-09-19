from importlib.resources import files as pkg_files
from pathlib import Path
from typing import List


BUILTIN_TEMPLATES_PACKAGE = "tritonparse.reproducer.templates"


def _is_path_like(template_arg: str) -> bool:
    return "/" in template_arg or "\\" in template_arg or template_arg.endswith(".py")


def _read_file_text(path: Path) -> str:
    p = path.expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Template not found: {p}")
    return p.read_text(encoding="utf-8")


def _read_builtin_template_text(name: str) -> str:
    resource = pkg_files(BUILTIN_TEMPLATES_PACKAGE).joinpath(f"{name}.py")
    # resource may not exist if an invalid name is provided
    try:
        with resource.open("r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        available = ", ".join(list_builtin_templates())
        raise FileNotFoundError(
            f"Builtin template '{name}' not found. Available: {available}"
        ) from exc


def list_builtin_templates() -> List[str]:
    """
    Return the list of available builtin template names (without .py suffix).
    """
    names: List[str] = []
    for entry in pkg_files(BUILTIN_TEMPLATES_PACKAGE).iterdir():
        try:
            if entry.is_file():
                filename = entry.name
                if filename.endswith(".py") and not filename.startswith("__"):
                    names.append(filename[:-3])
        except (OSError, FileNotFoundError):
            # Defensive: in case entry access fails in some environments
            continue
    names.sort()
    return names


def load_template_code(template_arg: str) -> str:
    """
    Load template code by name (builtin, without .py) or by filesystem path.
    """
    if _is_path_like(template_arg):
        return _read_file_text(Path(template_arg))
    return _read_builtin_template_text(template_arg)
