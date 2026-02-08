#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Metadata analyzer for the diff module.

This module provides Level 1 comparison of compilation metadata
(num_stages, num_warps, cluster_dims, target arch, shared memory, etc.)
between two compilation events.
"""

from typing import Any

from tritonparse.diff.core.diff_types import MetadataDiff


# Metadata keys to compare
METADATA_KEYS = [
    # Core compile parameters
    "num_stages",
    "num_warps",
    "num_ctas",
    "maxnreg",
    # Warp specialization parameters (SM90+)
    "num_buffers_warp_spec",
    "num_consumer_groups",
    "reg_dec_producer",
    "reg_inc_consumer",
    # Cluster/target configuration
    "cluster_dims",
    "target",
    "shared",
    # Source-level attributes
    "src_constants",
    "src_attrs",
]


class MetadataAnalyzer:
    """Analyzer for Level 1 metadata comparison.

    Compares compilation configuration parameters between two events.

    Attributes:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        metadata_keys: List of metadata keys to compare.
    """

    def __init__(
        self,
        comp_a: dict[str, Any],
        comp_b: dict[str, Any],
        metadata_keys: list[str] | None = None,
    ):
        """Initialize the metadata analyzer.

        Args:
            comp_a: First compilation event.
            comp_b: Second compilation event.
            metadata_keys: Optional list of metadata keys to compare.
                          Defaults to METADATA_KEYS if not provided.
        """
        self.comp_a = comp_a
        self.comp_b = comp_b
        self.metadata_keys = metadata_keys or METADATA_KEYS

    def analyze(self) -> MetadataDiff:
        """Analyze metadata differences between two compilations.

        Returns:
            MetadataDiff containing sames and diffs dictionaries.
        """
        meta_a = self._get_metadata(self.comp_a)
        meta_b = self._get_metadata(self.comp_b)

        sames: dict[str, Any] = {}
        diffs: dict[str, dict[str, Any]] = {}

        for key in self.metadata_keys:
            val_a = meta_a.get(key)
            val_b = meta_b.get(key)

            if val_a == val_b:
                if val_a is not None:
                    sames[key] = val_a
            else:
                diffs[key] = {"a": val_a, "b": val_b}

        return MetadataDiff(sames=sames, diffs=diffs)

    def _get_metadata(self, comp: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata from a compilation event.

        Tries multiple possible locations for metadata:
        1. compilation_metadata (top-level)
        2. payload.metadata
        3. Merges both with compilation_metadata taking precedence.

        Args:
            comp: Compilation event dictionary.

        Returns:
            Dictionary containing merged metadata.
        """
        compilation_metadata = comp.get("compilation_metadata", {})
        payload_metadata = comp.get("payload", {}).get("metadata", {})

        # Merge with compilation_metadata taking precedence
        return {**payload_metadata, **compilation_metadata}

    def get_highlight_strings(self, metadata_diff: MetadataDiff) -> list[str]:
        """Generate human-readable highlight strings for metadata differences.

        Args:
            metadata_diff: MetadataDiff result from analyze().

        Returns:
            List of strings like "num_stages: 3 → 5".
        """
        highlights = []
        for key, diff in metadata_diff.diffs.items():
            highlights.append(f"{key}: {diff['a']} → {diff['b']}")
        return highlights


def analyze_metadata(
    comp_a: dict[str, Any],
    comp_b: dict[str, Any],
    metadata_keys: list[str] | None = None,
) -> MetadataDiff:
    """Convenience function to analyze metadata differences.

    Args:
        comp_a: First compilation event.
        comp_b: Second compilation event.
        metadata_keys: Optional list of metadata keys to compare.

    Returns:
        MetadataDiff containing sames and diffs dictionaries.
    """
    analyzer = MetadataAnalyzer(comp_a, comp_b, metadata_keys)
    return analyzer.analyze()
