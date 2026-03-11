# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for MAST CLI JSON format compatibility helpers.

These tests verify that tritonparse correctly handles all known MAST CLI
JSON output formats:
- Old format: state as string ("RUNNING"), transition keys as strings ("RUNNING")
- Intermediate format (D95159275): state as int (1), transition keys as "1"
- Reverted format (D95874972): state as string ("RUNNING"), transition keys as strings
"""

import unittest

from tritonparse.fb.utils import get_transition_timestamp, HpcJobState, is_job_state


class IsJobStateTest(unittest.TestCase):
    """Tests for is_job_state() compatibility helper."""

    def test_integer_state_matches(self):
        """Integer state values should match the corresponding HpcJobState."""
        self.assertTrue(is_job_state(0, HpcJobState.PENDING))
        self.assertTrue(is_job_state(1, HpcJobState.RUNNING))
        self.assertTrue(is_job_state(2, HpcJobState.COMPLETE))
        self.assertTrue(is_job_state(3, HpcJobState.FAILED))
        self.assertTrue(is_job_state(4, HpcJobState.DEAD))
        self.assertTrue(is_job_state(5, HpcJobState.SHUTTING_DOWN))

    def test_integer_state_no_match(self):
        """Integer state values should not match a different HpcJobState."""
        self.assertFalse(is_job_state(0, HpcJobState.RUNNING))
        self.assertFalse(is_job_state(1, HpcJobState.COMPLETE))
        self.assertFalse(is_job_state(2, HpcJobState.RUNNING))

    def test_string_state_matches(self):
        """String state names should match the corresponding HpcJobState."""
        self.assertTrue(is_job_state("PENDING", HpcJobState.PENDING))
        self.assertTrue(is_job_state("RUNNING", HpcJobState.RUNNING))
        self.assertTrue(is_job_state("COMPLETE", HpcJobState.COMPLETE))
        self.assertTrue(is_job_state("FAILED", HpcJobState.FAILED))
        self.assertTrue(is_job_state("DEAD", HpcJobState.DEAD))
        self.assertTrue(is_job_state("SHUTTING_DOWN", HpcJobState.SHUTTING_DOWN))

    def test_string_state_no_match(self):
        """String state names should not match a different HpcJobState."""
        self.assertFalse(is_job_state("PENDING", HpcJobState.RUNNING))
        self.assertFalse(is_job_state("RUNNING", HpcJobState.COMPLETE))
        self.assertFalse(is_job_state("COMPLETE", HpcJobState.RUNNING))

    def test_invalid_types_return_false(self):
        """Non-int/non-str types should return False."""
        self.assertFalse(is_job_state(None, HpcJobState.RUNNING))
        self.assertFalse(is_job_state(1.0, HpcJobState.RUNNING))
        self.assertFalse(is_job_state([], HpcJobState.RUNNING))


class GetTransitionTimestampTest(unittest.TestCase):
    """Tests for get_transition_timestamp() compatibility helper."""

    def test_numeric_string_keys(self):
        """Intermediate format: keys are stringified integers like '1', '2'."""
        transitions = {"0": 100, "1": 200, "2": 300}
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.PENDING), 100
        )
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.RUNNING), 200
        )
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.COMPLETE), 300
        )

    def test_state_name_string_keys(self):
        """Old/reverted format: keys are state names like 'RUNNING', 'COMPLETE'."""
        transitions = {"PENDING": 100, "RUNNING": 200, "COMPLETE": 300}
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.PENDING), 100
        )
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.RUNNING), 200
        )
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.COMPLETE), 300
        )

    def test_missing_key_returns_default(self):
        """Missing keys should return the default value."""
        transitions = {"RUNNING": 200}
        self.assertIsNone(get_transition_timestamp(transitions, HpcJobState.FAILED))
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.FAILED, 0), 0
        )
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.FAILED, 999), 999
        )

    def test_numeric_keys_take_precedence(self):
        """If both formats exist, numeric keys should take precedence."""
        transitions = {"1": 111, "RUNNING": 222}
        self.assertEqual(
            get_transition_timestamp(transitions, HpcJobState.RUNNING), 111
        )

    def test_real_world_fbpkg_response(self):
        """Simulate the actual fbpkg MAST response that caused the bug."""
        transitions = {
            "COMPLETE": 1773078370,
            "RUNNING": 1773078136,
            "PENDING": 1773078084,
        }
        start_time = get_transition_timestamp(transitions, HpcJobState.RUNNING, 0)
        end_time = (
            get_transition_timestamp(transitions, HpcJobState.FAILED)
            or get_transition_timestamp(transitions, HpcJobState.COMPLETE)
            or get_transition_timestamp(transitions, HpcJobState.DEAD)
        )
        self.assertEqual(start_time, 1773078136)
        self.assertEqual(end_time, 1773078370)

    def test_real_world_buck_run_response(self):
        """Simulate the actual buck run MAST response that worked correctly."""
        transitions = {
            "2": 1773078370,
            "1": 1773078136,
            "0": 1773078084,
        }
        start_time = get_transition_timestamp(transitions, HpcJobState.RUNNING, 0)
        end_time = (
            get_transition_timestamp(transitions, HpcJobState.FAILED)
            or get_transition_timestamp(transitions, HpcJobState.COMPLETE)
            or get_transition_timestamp(transitions, HpcJobState.DEAD)
        )
        self.assertEqual(start_time, 1773078136)
        self.assertEqual(end_time, 1773078370)


if __name__ == "__main__":
    unittest.main()
