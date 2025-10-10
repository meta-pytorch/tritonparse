from enum import Enum


class KernelImportMode(str, Enum):
    """
    Kernel import strategy for reproducer generation.

    Inherits from str to allow direct string comparison and use in argparse.
    """

    DEFAULT = "default"
    """Import kernel from original file (current behavior)."""

    COPY = "copy"
    """Embed kernel source code directly in reproducer."""

    # Future modes can be added here:
    # OVERRIDE_TTIR = "override-ttir"
    # """Use TTIR from compilation event with monkeypatch."""
