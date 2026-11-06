# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class TargetProfile:
    key: str
    name: str
    backend: str
    arch: int | str
    warp_size: int
    multiprocessor_count: int
    max_shared_mem: int

    def target(self) -> Any:
        from triton.backends.compiler import GPUTarget

        return GPUTarget(self.backend, self.arch, self.warp_size)

    def properties(self) -> SimpleNamespace:
        values: dict[str, Any] = {
            "name": self.name,
            "multi_processor_count": self.multiprocessor_count,
            "shared_memory_per_block_optin": self.max_shared_mem,
            "warp_size": self.warp_size,
        }
        if self.backend == "cuda":
            arch = int(self.arch)
            values.update(major=arch // 10, minor=arch % 10)
        else:
            values["gcnArchName"] = self.arch
        return SimpleNamespace(**values)

    def metadata(self) -> dict[str, Any]:
        return {
            "profile": self.key,
            "name": self.name,
            "target": {
                "backend": self.backend,
                "arch": self.arch,
                "warp_size": self.warp_size,
            },
            "multiprocessor_count": self.multiprocessor_count,
            "max_shared_mem": self.max_shared_mem,
        }


def target_profiles() -> dict[str, TargetProfile]:
    # These are device profiles, not defaults for every SKU with the same ISA.
    return {
        "cuda:90": TargetProfile("cuda:90", "H100 SXM5", "cuda", 90, 32, 132, 232448),
        "cuda:100": TargetProfile("cuda:100", "B200", "cuda", 100, 32, 148, 232448),
        "hip:gfx942": TargetProfile(
            "hip:gfx942", "MI300X", "hip", "gfx942", 64, 304, 65536
        ),
        "hip:gfx950": TargetProfile(
            "hip:gfx950", "MI350X", "hip", "gfx950", 64, 256, 163840
        ),
    }
