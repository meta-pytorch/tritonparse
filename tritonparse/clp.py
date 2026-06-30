#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Interface to interact with yscope_clp_core
"""

import json
from typing import Iterator

from yscope_clp_core import open_archive, ClpArchiveReader


class ClpTextStream(Iterator[str]):
    def __init__(self, archive: ClpArchiveReader) -> None:
        self.archive = archive
        return None

    def __iter__(self) -> "ClpTextStream":
        return self

    def __next__(self) -> str:
        event = next(self.archive)
        return json.dumps(event.get_kv_pairs(), separators=(",", ":")) + "\n"

    def readlines(self) -> list[str]:
        return list(self)


def clp_open(clp_dir: str, open_mode: str):
    assert open_mode in ["r", "w"], "CLP only supports r and w modes."
    return open_archive(clp_dir, open_mode)
