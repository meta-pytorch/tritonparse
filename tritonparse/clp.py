"""
Interface to interact with yscope_clp_core
"""

import yscope_clp_core


def clp_open(clp_dir: str, open_mode: str):
    assert open_mode in ["r", "w"], "CLP only supports r and w modes."
    return yscope_clp_core.open_archive(clp_dir, open_mode)
