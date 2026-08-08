# Copyright (c) Meta Platforms, Inc. and affiliates.
"""A platform string safe to publish.

``platform.platform()`` embeds the distro's full kernel build string, which on many hosts
carries a vendor or build-farm tag.  That identifies the machine's build lineage without
saying anything useful about reproducibility, and it has no place in a published
artifact.  Keep the parts a reader might act on -- OS, kernel version, architecture,
libc -- and drop the rest.
"""

from __future__ import annotations

import platform
import re


def neutral_platform() -> str:
    base = re.match(r"[0-9]+(?:\.[0-9]+)*", platform.release())
    libc = "-".join(x for x in platform.libc_ver() if x)
    parts = [platform.system(), base.group(0) if base else platform.release(),
             platform.machine()]
    if libc:
        parts.append(f"with-{libc}")
    return "-".join(parts)
