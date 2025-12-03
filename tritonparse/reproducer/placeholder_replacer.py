#  Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC
from typing import Any, Dict, Optional, Protocol

from tritonparse.reproducer.function_extractor import extract_utility_functions
from tritonparse.reproducer.ingestion.ndjson import ContextBundle
from tritonparse.reproducer.templates.utils import (
    _disable_triton_autotune,
    get_function_source,
)
from tritonparse.reproducer.types import KernelImportMode
from tritonparse.reproducer.utils import (
    _generate_import_statements,
    _generate_invocation_snippet,
    _parse_kernel_signature,
)
from tritonparse.tp_logger import logger


class HandlerProtocol(Protocol):
    def __call__(
        self, code: str, context_bundle: ContextBundle, **kwargs: Any
    ) -> str: ...


class PlaceholderReplacer(ABC):
    """
    Abstract base class for template placeholder replacement.

    Subclasses should register replacement handlers in their __init__ method
    by calling self.register(placeholder, handler_function).

    Each handler function should have the signature:
        handler(code: str, context_bundle: ContextBundle, **kwargs) -> str
    """

    def __init__(self):
        # Dictionary mapping placeholder strings to handler functions
        self.handlers: Dict[str, HandlerProtocol] = {}

    def register(self, placeholder: str, handler: HandlerProtocol):
        """
        Register a handler function for a specific placeholder.

        Args:
            placeholder: The placeholder string to replace (e.g., "{{JSON_FILE_NAME_PLACEHOLDER}}")
            handler: A callable that takes (code, context_bundle, **kwargs) and returns modified code
        """
        self.handlers[placeholder] = handler

    def replace(
        self, template_code: str, context_bundle: ContextBundle, **kwargs: Any
    ) -> str:
        """
        Replace all registered placeholders in the template code.

        Args:
            template_code: The template code containing placeholders
            context_bundle: Context information about the kernel
            **kwargs: Additional keyword arguments passed to handler functions

        Returns:
            The code with all placeholders replaced
        """
        code = template_code
        for handler in self.handlers.values():
            code = handler(code, context_bundle, **kwargs)
        return code


class DefaultPlaceholderReplacer(PlaceholderReplacer):
    """
    Default implementation of PlaceholderReplacer.

    Handles the following placeholders:
    - {{JSON_FILE_NAME_PLACEHOLDER}}: Replaced with the JSON file name
    - # {{KERNEL_SYSPATH_PLACEHOLDER}}: Replaced with sys.path setup code
    - # {{KERNEL_IMPORT_PLACEHOLDER}}: Replaced with kernel import statement
    - # {{KERNEL_INVOCATION_PLACEHOLDER}}: Replaced with kernel invocation code
    """

    KERNEL_NAME_PLACEHOLDER = "{{KERNEL_NAME_PLACEHOLDER}}"
    JSON_FILE_NAME_PLACEHOLDER = "{{JSON_FILE_NAME_PLACEHOLDER}}"
    IR_OVERRIDE_SETUP_PLACEHOLDER = "# {{IR_OVERRIDE_SETUP_PLACEHOLDER}}"
    KERNEL_SYSPATH_PLACEHOLDER = "# {{KERNEL_SYSPATH_PLACEHOLDER}}"
    KERNEL_IMPORT_PLACEHOLDER = "# {{KERNEL_IMPORT_PLACEHOLDER}}"
    UTILITY_FUNCTIONS_PLACEHOLDER = "# {{UTILITY_FUNCTIONS_PLACEHOLDER}}"
    KERNEL_INVOCATION_PLACEHOLDER = "# {{KERNEL_INVOCATION_PLACEHOLDER}}"

    def __init__(self):
        super().__init__()
        # Register all default handlers
        self.register(self.JSON_FILE_NAME_PLACEHOLDER, self._replace_json_filename)
        self.register(
            self.IR_OVERRIDE_SETUP_PLACEHOLDER, self._replace_ir_override_setup
        )
        self.register(self.KERNEL_SYSPATH_PLACEHOLDER, self._replace_kernel_syspath)
        self.register(self.KERNEL_IMPORT_PLACEHOLDER, self._replace_kernel_import)
        self.register(
            self.UTILITY_FUNCTIONS_PLACEHOLDER, self._replace_utility_functions
        )
        self.register(
            self.KERNEL_INVOCATION_PLACEHOLDER, self._replace_kernel_invocation
        )
        self.register(self.KERNEL_NAME_PLACEHOLDER, self._replace_kernel_name)

    def _replace_kernel_name(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel name placeholder."""
        kernel_name = context_bundle.kernel_info.function_name
        if not kernel_name:
            raise ValueError("Kernel function name is not available")
        return code.replace(self.KERNEL_NAME_PLACEHOLDER, kernel_name)

    def _replace_json_filename(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the JSON file name placeholder."""
        temp_json_path = kwargs.get("temp_json_path")
        if temp_json_path is None:
            raise ValueError("temp_json_path is required for JSON filename replacement")
        return code.replace(self.JSON_FILE_NAME_PLACEHOLDER, temp_json_path.name)

    def _replace_ir_override_setup(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the IR override setup placeholder."""
        kernel_import = kwargs.get("kernel_import", KernelImportMode.DEFAULT)

        if kernel_import != KernelImportMode.OVERRIDE_TTIR:
            return code.replace(self.IR_OVERRIDE_SETUP_PLACEHOLDER, "")

        comp_json_filename = kwargs.get("comp_json_filename")
        if not comp_json_filename:
            raise ValueError("comp_json_filename is required for OVERRIDE_TTIR mode")

        setup_code = f'''
def create_ttir_tempfile():
    """Extract TTIR from compilation event and create temporary file."""
    script_dir = Path(__file__).resolve().parent
    comp_json_file = script_dir / "{comp_json_filename}"

    with open(comp_json_file, 'r') as f:
        comp_data = json.load(f)

    # Extract TTIR content
    kernel_name = comp_data['payload']['metadata']['name']
    ttir_key = f"{{kernel_name}}.ttir"
    ttir_content = comp_data['payload']['file_content'][ttir_key]

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.ttir',
        delete=False,
        prefix=f'{{kernel_name}}_'
    )
    temp_file.write(ttir_content)
    temp_file.close()
    return temp_file.name


# Monkeypatch triton.autotune to use our TTIR
_ttir_file = create_ttir_tempfile()
_original_autotune = None

def _patched_autotune(configs, key=None, **kwargs):
    """Patched autotune that uses our TTIR file."""
    import triton
    # Replace configs with our single config using ir_override
    new_configs = [triton.Config(kwargs={{}}, ir_override=_ttir_file)]
    # Call original autotune with our config
    return _original_autotune(new_configs, key=[], **kwargs)

# Apply the monkeypatch before importing the kernel
import triton
_original_autotune = triton.autotune
triton.autotune = _patched_autotune
'''

        return code.replace(self.IR_OVERRIDE_SETUP_PLACEHOLDER, setup_code)

    def _replace_kernel_syspath(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel sys.path placeholder."""
        kernel_import = kwargs.get("kernel_import", KernelImportMode.DEFAULT)

        if kernel_import == KernelImportMode.DEFAULT:
            sys_stmt, _ = _generate_import_statements(context_bundle.kernel_info)
            return code.replace(self.KERNEL_SYSPATH_PLACEHOLDER, sys_stmt)
        elif kernel_import == KernelImportMode.COPY:
            comment = (
                "# Kernel sys.path setup skipped - kernel source code embedded below"
            )
            return code.replace(self.KERNEL_SYSPATH_PLACEHOLDER, comment)
        elif kernel_import == KernelImportMode.OVERRIDE_TTIR:
            comment = "# Kernel sys.path setup skipped - using IR override mode"
            return code.replace(self.KERNEL_SYSPATH_PLACEHOLDER, comment)
        else:
            raise ValueError(f"Unknown kernel_import mode: {kernel_import}")

    def _replace_kernel_import(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel import placeholder."""
        kernel_import = kwargs.get("kernel_import", KernelImportMode.DEFAULT)

        if kernel_import == KernelImportMode.DEFAULT:
            _, import_statement = _generate_import_statements(
                context_bundle.kernel_info
            )

            final_stmt = "\n".join(
                [import_statement, ""] + get_function_source(_disable_triton_autotune)
            )
            return code.replace(self.KERNEL_IMPORT_PLACEHOLDER, final_stmt)
        elif kernel_import == KernelImportMode.COPY:
            source_code = context_bundle.kernel_info.source_code
            func_name = context_bundle.kernel_info.function_name

            if not source_code or not source_code.strip():
                raise ValueError("Kernel source code is empty, cannot use 'copy' mode")
            if not func_name:
                raise ValueError(
                    "Cannot determine kernel function name for 'copy' mode"
                )

            if kernel_import == KernelImportMode.COPY:
                dependent_source_map = get_dependent_source_map(
                    context_bundle.kernel_info.function_name,
                    context_bundle.kernel_info.file_path,
                )
                # Only add dependent functions if extraction was successful
                if dependent_source_map:
                    # Add separator, import statements, and dependent functions
                    dependent_code = (
                        "\n\n# Dependent functions extracted from source file\n\n"
                    )
                    dependent_code += "\n\n".join(dependent_source_map.values())
                    source_code += "\n\n" + dependent_code
                    logger.debug("Appended dependent functions to kernel source code")

            # Add common imports needed for most Triton kernels
            import_lines = [
                "import torch",
                "import numpy as np",
                "import triton",
                "import triton.language as tl",
                "from typing import List, Tuple",
                "",
            ] + get_function_source(_disable_triton_autotune)

            # Combine: imports + kernel source code + alias
            embedded_code = "\n".join(import_lines)
            embedded_code += "\n" + source_code
            embedded_code += f"\n\n# Use kernel function directly\nimported_kernel_function = {func_name}"

            return code.replace(self.KERNEL_IMPORT_PLACEHOLDER, embedded_code)
        elif kernel_import == KernelImportMode.OVERRIDE_TTIR:
            comment = "# Kernel import skipped - using IR override mode with TTIR"
            return code.replace(self.KERNEL_IMPORT_PLACEHOLDER, comment)
        else:
            raise ValueError(f"Unknown kernel_import mode: {kernel_import}")

    def _replace_utility_functions(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the utility functions placeholder with extracted functions."""
        utility_code = extract_utility_functions()
        return code.replace(self.UTILITY_FUNCTIONS_PLACEHOLDER, utility_code)

    def _replace_kernel_invocation(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel invocation placeholder."""
        source_code = context_bundle.kernel_info.source_code
        pos_args, kw_args = _parse_kernel_signature(source_code)
        invocation_snippet = _generate_invocation_snippet(pos_args, kw_args)
        return code.replace(self.KERNEL_INVOCATION_PLACEHOLDER, invocation_snippet)


def get_dependent_source_map(
    function_name: str, file_path: str
) -> Optional[dict[str, str]]:
    """
    Extract dependent functions and their required imports.

    Returns:
        A tuple of (functions_dict, import_statements_list) or None if extraction fails.
        - functions_dict: Maps qualified function names to their source code
        - import_statements_list: List of formatted import statements needed by dependent functions
    """
    from pathlib import Path

    from tritonparse.tp_logger import logger

    source_path = Path(file_path)
    if not source_path.exists():
        return None

    try:
        # Use MultiFileCallGraphAnalyzer for multi-file analysis
        from tritonparse.reproducer.multi_file_analyzer import (
            MultiFileCallGraphAnalyzer,
        )

        analyzer = MultiFileCallGraphAnalyzer(
            entry_file=file_path,
            entry_function=function_name,
        )
        result = analyzer.analyze()

        logger.info(
            f"Extracted {result.stats.total_functions_found} dependent functions "
            f"from {result.stats.total_files_analyzed} files with "
            f"{result.stats.total_imports} imports"
        )

        # Print dependent functions' short names
        logger.info("\nDependent functions (short names):")
        for func_name in sorted(result.function_short_names.keys()):
            short_name = result.function_short_names[func_name]
            logger.info(
                "  - %s. %s", short_name, result.functions[func_name].splitlines()[0]
            )

        return result.functions

    except Exception as e:
        # If AST analysis fails, continue without dependent functions
        logger.warning(f"Failed to extract dependent functions: {e}", exc_info=True)
        return None
