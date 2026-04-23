#  Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Interface to interact with yscope_clp_core
"""

import yscope_clp_core


def clp_open(clp_dir: str, open_mode: str):
    if open_mode not in ["r", "w"]:
        raise ValueError("CLP only supports r and w modes.")
    return yscope_clp_core.open_archive(clp_dir, open_mode)
