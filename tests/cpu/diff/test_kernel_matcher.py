#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the KernelMatcher multi-strategy matching engine."""

import unittest

from .test_fixtures import (
    create_compilation_event,
    DEFAULT_PYTHON_SOURCE,
    DEFAULT_TTIR,
    DIFFERENT_PYTHON_SOURCE_MATMUL,
    SIMILAR_PYTHON_SOURCE,
)


class TestKernelMatcher(unittest.TestCase):
    """Tests for the KernelMatcher multi-strategy matching engine."""

    def test_exact_hash_match(self) -> None:
        """Same hash, different names -> matched by HASH."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="kernel_v1", kernel_hash="shared_hash"
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="kernel_v2", kernel_hash="shared_hash"
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.HASH)
        self.assertEqual(matched[0].match_confidence, 1.0)
        self.assertEqual(len(unmatched_a), 0)
        self.assertEqual(len(unmatched_b), 0)

    def test_exact_name_match(self) -> None:
        """Same name, different hashes -> matched by NAME."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel", kernel_hash="hash_a"
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel", kernel_hash="hash_b"
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.NAME)
        self.assertEqual(len(unmatched_a), 0)

    def test_source_similarity_match(self) -> None:
        """Different names, >80% similar source -> matched by SOURCE."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel_old",
                    kernel_hash="unique_hash_a",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel_new",
                    kernel_hash="unique_hash_b",
                    python_source=SIMILAR_PYTHON_SOURCE,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.SOURCE)
        self.assertGreaterEqual(matched[0].match_confidence, 0.8)

    def test_fuzzy_name_match(self) -> None:
        """Similar names (>70%), dissimilar sources -> matched by FUZZY_NAME.

        Uses very different sources so source matching doesn't kick in first.
        """
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="softmax_v2_kernel",
                    kernel_hash="unique_hash_c",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="softmax_v3_kernel",
                    kernel_hash="unique_hash_d",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.FUZZY_NAME)
        self.assertGreaterEqual(matched[0].match_confidence, 0.7)

    def test_priority_hash_over_name(self) -> None:
        """Kernel matchable by both hash and name -> hash wins."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel",
                    kernel_hash="shared_hash",
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="add_kernel",
                    kernel_hash="shared_hash",
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, _, _, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.HASH)

    def test_unknown_name_skipped(self) -> None:
        """'unknown' names don't match by name, but fall through to source."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="unknown",
                    kernel_hash="hash_e",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="unknown",
                    kernel_hash="hash_f",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        # "unknown" is skipped by name strategy, but identical source
        # means they match by source
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].match_method, MatchMethod.SOURCE)

    def test_empty_source_skipped(self) -> None:
        """Empty source -> similarity 0.0, no source match.

        Must construct events manually since create_compilation_event
        uses `or` fallback for empty strings.
        """
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        event_a = {
            "event_type": "compilation",
            "kernel_name": "alpha_reduction_kernel",
            "kernel_hash": "hash_g",
            "payload": {
                "python_source": {"content": ""},
                "ttir": DEFAULT_TTIR,
                "source_mappings": {},
            },
        }
        event_b = {
            "event_type": "compilation",
            "kernel_name": "zeta_scatter_transform",
            "kernel_hash": "hash_h",
            "payload": {
                "python_source": {"content": ""},
                "ttir": DEFAULT_TTIR,
                "source_mappings": {},
            },
        }
        events_a = [(0, event_a)]
        events_b = [(0, event_b)]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        # No hash match, different names, empty source, dissimilar names
        # -> all strategies fail, events remain unmatched
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(unmatched_a), 1)

    def test_greedy_best_match(self) -> None:
        """When A1 and A2 both can match B1, the better similarity wins."""
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        # A1 has very similar source to B1, A2 has somewhat similar source to B1
        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="k_x_alpha",
                    kernel_hash="hash_i",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
            (
                1,
                create_compilation_event(
                    kernel_name="k_x_beta",
                    kernel_hash="hash_j",
                    python_source=SIMILAR_PYTHON_SOURCE,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="k_y_gamma",
                    kernel_hash="hash_k",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, _, _, _ = matcher.match()
        self.assertEqual(len(matched), 1)
        # A1 (index 0) should match B1 because it has perfect similarity
        self.assertEqual(matched[0].event_index_a, 0)
        self.assertEqual(len(unmatched_a), 1)

    def test_completely_disjoint(self) -> None:
        """No overlap -> empty matched, all unmatched.

        Use very different names and sources to avoid fuzzy matching.
        """
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            (
                0,
                create_compilation_event(
                    kernel_name="alpha_compute_kernel",
                    kernel_hash="hash_n",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="zeta_transform_kernel",
                    kernel_hash="hash_o",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(unmatched_a), 1)
        self.assertEqual(len(unmatched_b), 1)

    def test_multiple_kernels_mixed_strategies(self) -> None:
        """3 kernels matched by hash, name, and source respectively."""
        from tritonparse.diff.core.diff_types import MatchMethod
        from tritonparse.diff.core.kernel_matcher import KernelMatcher

        events_a = [
            # Will match by hash (same hash, different name)
            (
                0,
                create_compilation_event(
                    kernel_name="old_name",
                    kernel_hash="same_hash_1",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
            # Will match by name (same name, different hash)
            (
                1,
                create_compilation_event(
                    kernel_name="stable_kernel",
                    kernel_hash="hash_p",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
            # Will match by source (different name, different hash, similar source)
            (
                2,
                create_compilation_event(
                    kernel_name="renamed_kernel_old",
                    kernel_hash="hash_q",
                    python_source=DEFAULT_PYTHON_SOURCE,
                ),
            ),
        ]
        events_b = [
            (
                0,
                create_compilation_event(
                    kernel_name="new_name",
                    kernel_hash="same_hash_1",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
            (
                1,
                create_compilation_event(
                    kernel_name="stable_kernel",
                    kernel_hash="hash_r",
                    python_source=DIFFERENT_PYTHON_SOURCE_MATMUL,
                ),
            ),
            (
                2,
                create_compilation_event(
                    kernel_name="renamed_kernel_new",
                    kernel_hash="hash_s",
                    python_source=SIMILAR_PYTHON_SOURCE,
                ),
            ),
        ]
        matcher = KernelMatcher(events_a, events_b)
        matched, unmatched_a, unmatched_b, _, _ = matcher.match()
        self.assertEqual(len(matched), 3)
        self.assertEqual(len(unmatched_a), 0)
        self.assertEqual(len(unmatched_b), 0)

        methods = {m.match_method for m in matched}
        self.assertIn(MatchMethod.HASH, methods)
        self.assertIn(MatchMethod.NAME, methods)
        self.assertIn(MatchMethod.SOURCE, methods)


if __name__ == "__main__":
    unittest.main()
