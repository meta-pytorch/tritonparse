#  Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum


class KernelImportMode(str, Enum):
    """
    Kernel import strategy for reproducer generation.

    Inherits from str to allow direct string comparison and use in argparse.

    Attributes:
        DEFAULT: Import kernel from original file (current behavior).
        COPY: Embed kernel source code directly in reproducer.
        OVERRIDE_TTIR: Generate a stub kernel with captured IR override.
    """

    DEFAULT = "default"
    COPY = "copy"
    OVERRIDE_TTIR = "override-ttir"

    def __str__(self) -> str:
        return self.value
