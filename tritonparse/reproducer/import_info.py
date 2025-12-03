# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

from dataclasses import dataclass, field


@dataclass
class ImportInfo:
    """Information about an import statement."""

    # Import statement details
    import_type: str  # "import" or "from_import"
    module: str  # e.g., "torch.nn.functional"
    names: list[str]  # Imported names: ["func1", "func2"]

    # Resolution metadata
    source_file: str  # File containing this import
    resolved_path: str | None  # Resolved file path (None if external)
    is_external: bool  # True for third-party/built-in
    lineno: int  # Line number in source file

    # Fields with defaults (must come after required fields)
    aliases: dict[str, str] = field(default_factory=dict)  # {local_name: original_name}
    level: int = 0  # 0 = absolute, 1 = ".", 2 = "..", etc.
