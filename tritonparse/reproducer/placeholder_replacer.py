from abc import ABC

from typing import Any, Dict, Protocol

from tritonparse.reproducer.ingestion.ndjson import ContextBundle
from tritonparse.reproducer.types import KernelImportMode
from tritonparse.reproducer.utils import (
    _generate_import_statements,
    _generate_invocation_snippet,
    _parse_kernel_signature,
)


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
        for placeholder, handler in self.handlers.items():
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

    def __init__(self):
        super().__init__()
        # Register all default handlers
        self.register("{{JSON_FILE_NAME_PLACEHOLDER}}", self._replace_json_filename)
        self.register(
            "# {{IR_OVERRIDE_SETUP_PLACEHOLDER}}", self._replace_ir_override_setup
        )
        self.register("# {{KERNEL_SYSPATH_PLACEHOLDER}}", self._replace_kernel_syspath)
        self.register("# {{KERNEL_IMPORT_PLACEHOLDER}}", self._replace_kernel_import)
        self.register(
            "# {{KERNEL_INVOCATION_PLACEHOLDER}}", self._replace_kernel_invocation
        )

    def _replace_json_filename(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the JSON file name placeholder."""
        temp_json_path = kwargs.get("temp_json_path")
        if temp_json_path is None:
            raise ValueError("temp_json_path is required for JSON filename replacement")
        return code.replace("{{JSON_FILE_NAME_PLACEHOLDER}}", temp_json_path.name)

    def _replace_ir_override_setup(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the IR override setup placeholder."""
        kernel_import = kwargs.get("kernel_import", KernelImportMode.DEFAULT)

        if kernel_import != KernelImportMode.OVERRIDE_TTIR:
            return code.replace("# {{IR_OVERRIDE_SETUP_PLACEHOLDER}}", "")

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

        return code.replace("# {{IR_OVERRIDE_SETUP_PLACEHOLDER}}", setup_code)

    def _replace_kernel_syspath(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel sys.path placeholder."""
        kernel_import = kwargs.get("kernel_import", KernelImportMode.DEFAULT)

        if kernel_import == KernelImportMode.DEFAULT:
            sys_stmt, _ = _generate_import_statements(context_bundle.kernel_info)
            return code.replace("# {{KERNEL_SYSPATH_PLACEHOLDER}}", sys_stmt)
        elif kernel_import == KernelImportMode.COPY:
            comment = (
                "# Kernel sys.path setup skipped - kernel source code embedded below"
            )
            return code.replace("# {{KERNEL_SYSPATH_PLACEHOLDER}}", comment)
        elif kernel_import == KernelImportMode.OVERRIDE_TTIR:
            comment = "# Kernel sys.path setup skipped - using IR override mode"
            return code.replace("# {{KERNEL_SYSPATH_PLACEHOLDER}}", comment)
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
            return code.replace("# {{KERNEL_IMPORT_PLACEHOLDER}}", import_statement)
        elif kernel_import == KernelImportMode.COPY:
            source_code = context_bundle.kernel_info.source_code
            func_name = context_bundle.kernel_info.function_name

            if not source_code or not source_code.strip():
                raise ValueError("Kernel source code is empty, cannot use 'copy' mode")
            if not func_name:
                raise ValueError(
                    "Cannot determine kernel function name for 'copy' mode"
                )

            # Add common imports needed for most Triton kernels
            import_lines = [
                "import torch",
                "import numpy as np",
                "import triton",
                "import triton.language as tl",
                "",
            ]

            # Combine: imports + kernel source code + alias
            embedded_code = "\n".join(import_lines)
            embedded_code += "\n" + source_code
            embedded_code += f"\n\n# Use kernel function directly\nimported_kernel_function = {func_name}"

            return code.replace("# {{KERNEL_IMPORT_PLACEHOLDER}}", embedded_code)
        elif kernel_import == KernelImportMode.OVERRIDE_TTIR:
            comment = "# Kernel import skipped - using IR override mode with TTIR"
            return code.replace("# {{KERNEL_IMPORT_PLACEHOLDER}}", comment)
        else:
            raise ValueError(f"Unknown kernel_import mode: {kernel_import}")

    def _replace_kernel_invocation(
        self, code: str, context_bundle: ContextBundle, **kwargs
    ) -> str:
        """Replace the kernel invocation placeholder."""
        source_code = context_bundle.kernel_info.source_code
        pos_args, kw_args = _parse_kernel_signature(source_code)
        invocation_snippet = _generate_invocation_snippet(pos_args, kw_args)
        return code.replace("# {{KERNEL_INVOCATION_PLACEHOLDER}}", invocation_snippet)
