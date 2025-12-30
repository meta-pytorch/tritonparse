#  Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum
from pathlib import Path
from typing import Tuple


class SourceType(str, Enum):
    """Enumeration of supported source types for OSS only."""

    LOCAL = "local"
    LOCAL_FILE = "local_file"

    @classmethod
    def _missing_(cls, value: object) -> "SourceType":
        """
        Handle unknown source types by raising a ValueError.

        Args:
            value: The unknown value that was attempted to be used as a SourceType

        Returns:
            Never returns, always raises ValueError
        """
        valid_types = [e.value for e in cls]
        raise ValueError(
            f"Invalid source type '{value}'. Valid types are: {', '.join(valid_types)}"
        )


class Source:
    """Represents a source of logs to parse."""

    def __init__(self, source_str: str, verbose: bool = False):
        """
        Initialize a Source object by parsing the source string.

        Args:
            source_str: String representing the source
            verbose: Whether to print verbose information
        """
        self.source_str = source_str
        self.verbose = verbose
        self.type, self.value = self._parse_source()

    def _parse_source(self) -> Tuple[SourceType, str]:
        # Check if it's a local path
        path = Path(self.source_str)
        if path.is_dir():
            return SourceType.LOCAL, str(path.absolute())
        elif path.is_file():
            return SourceType.LOCAL_FILE, str(path.absolute())
        else:
            raise ValueError(
                f"Source '{self.source_str}' is not a valid directory or file"
            )
