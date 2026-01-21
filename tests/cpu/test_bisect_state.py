# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for bisect state module (CPU-only, no GPU required)."""

import shutil
import tempfile
import time
import unittest
from pathlib import Path

from tritonparse.bisect import BisectPhase, BisectState, StateManager


class BisectPhaseTest(unittest.TestCase):
    """Tests for BisectPhase enum."""

    def test_phase_values(self):
        """Test that all phases have expected string values."""
        self.assertEqual(BisectPhase.TRITON_BISECT.value, "triton_bisect")
        self.assertEqual(BisectPhase.TYPE_CHECK.value, "type_check")
        self.assertEqual(BisectPhase.PAIR_TEST.value, "pair_test")
        self.assertEqual(BisectPhase.LLVM_BISECT.value, "llvm_bisect")
        self.assertEqual(BisectPhase.COMPLETED.value, "completed")
        self.assertEqual(BisectPhase.FAILED.value, "failed")

    def test_phase_from_string(self):
        """Test creating phase from string value."""
        self.assertEqual(BisectPhase("triton_bisect"), BisectPhase.TRITON_BISECT)
        self.assertEqual(BisectPhase("completed"), BisectPhase.COMPLETED)
        with self.assertRaises(ValueError):
            BisectPhase("invalid_phase")


class BisectStateTest(unittest.TestCase):
    """Tests for BisectState dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
        )
        self.assertEqual(state.phase, BisectPhase.TRITON_BISECT)
        self.assertEqual(state.conda_env, "triton_bisect")
        self.assertEqual(state.log_dir, "./bisect_logs")
        self.assertIsNone(state.triton_culprit)
        self.assertIsNone(state.is_llvm_bump)

    def test_to_dict_serializes_phase_as_string(self):
        """Test to_dict converts BisectPhase enum to string."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            phase=BisectPhase.TYPE_CHECK,
        )
        data = state.to_dict()
        self.assertEqual(data["phase"], "type_check")
        self.assertIsInstance(data["phase"], str)

    def test_from_dict_parses_phase_string(self):
        """Test from_dict converts string back to BisectPhase."""
        data = {
            "triton_dir": "/path",
            "test_script": "/test.py",
            "good_commit": "v1",
            "bad_commit": "v2",
            "phase": "llvm_bisect",
            "commits_csv": None,
            "conda_env": "triton_bisect",
            "log_dir": "./bisect_logs",
            "build_command": None,
            "session_name": None,
            "started_at": None,
            "updated_at": None,
            "triton_culprit": None,
            "is_llvm_bump": None,
            "old_llvm_hash": None,
            "new_llvm_hash": None,
            "failing_pair_index": None,
            "good_llvm": None,
            "bad_llvm": None,
            "triton_commit_for_llvm": None,
            "llvm_culprit": None,
            "error_message": None,
        }
        state = BisectState.from_dict(data)
        self.assertEqual(state.phase, BisectPhase.LLVM_BISECT)
        self.assertIsInstance(state.phase, BisectPhase)

    def test_round_trip_serialization(self):
        """Test state survives to_dict -> from_dict round trip."""
        original = BisectState(
            triton_dir="/path/to/triton",
            test_script="/path/to/test.py",
            good_commit="abc123",
            bad_commit="def456",
            commits_csv="/path/to/commits.csv",
            conda_env="my_env",
            phase=BisectPhase.PAIR_TEST,
            triton_culprit="culprit123",
            is_llvm_bump=True,
            old_llvm_hash="old_hash",
            new_llvm_hash="new_hash",
        )
        data = original.to_dict()
        restored = BisectState.from_dict(data)
        self.assertEqual(restored.triton_dir, original.triton_dir)
        self.assertEqual(restored.phase, original.phase)
        self.assertEqual(restored.triton_culprit, original.triton_culprit)
        self.assertEqual(restored.is_llvm_bump, original.is_llvm_bump)

    def test_to_report_includes_llvm_info_when_bump(self):
        """Test report includes LLVM details when is_llvm_bump=True."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            phase=BisectPhase.COMPLETED,
            triton_culprit="abc123",
            is_llvm_bump=True,
            old_llvm_hash="old_llvm",
            new_llvm_hash="new_llvm",
            good_llvm="good_llvm",
            bad_llvm="bad_llvm",
            llvm_culprit="llvm_culprit_hash",
            failing_pair_index=5,
        )
        report = state.to_report()
        self.assertTrue(report["is_llvm_bump"])
        self.assertEqual(report["llvm_culprit"], "llvm_culprit_hash")
        self.assertIn("llvm_bump", report)
        self.assertEqual(report["llvm_bump"]["old"], "old_llvm")
        self.assertIn("llvm_range", report)

    def test_to_report_excludes_llvm_info_when_not_bump(self):
        """Test report excludes LLVM details when is_llvm_bump=False."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            phase=BisectPhase.COMPLETED,
            triton_culprit="abc123",
            is_llvm_bump=False,
        )
        report = state.to_report()
        self.assertFalse(report["is_llvm_bump"])
        self.assertNotIn("llvm_culprit", report)
        self.assertNotIn("llvm_bump", report)

    def test_to_report_includes_error_when_failed(self):
        """Test report includes error message when workflow failed."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            phase=BisectPhase.FAILED,
            error_message="Something went wrong",
        )
        report = state.to_report()
        self.assertEqual(report["status"], "failed")
        self.assertIn("error", report)
        self.assertEqual(report["error"], "Something went wrong")


class StateManagerTest(unittest.TestCase):
    """Tests for StateManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_get_state_path_format(self):
        """Test state path follows {session_name}_state.json pattern."""
        path = StateManager.get_state_path(self.temp_dir, "test_session")
        self.assertEqual(path.name, "test_session_state.json")
        self.assertEqual(path.parent, Path(self.temp_dir))

    def test_save_creates_directory_if_not_exists(self):
        """Test save creates parent directories."""
        nested_dir = Path(self.temp_dir) / "nested" / "logs"
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=str(nested_dir),
        )
        path = StateManager.save(state, session_name="test_session")
        self.assertTrue(path.exists())
        self.assertTrue(nested_dir.exists())

    def test_save_updates_timestamps(self):
        """Test save sets started_at and updated_at."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=self.temp_dir,
        )
        self.assertIsNone(state.started_at)
        self.assertIsNone(state.updated_at)
        StateManager.save(state, session_name="test_session")
        self.assertIsNotNone(state.started_at)
        self.assertIsNotNone(state.updated_at)

    def test_load_reads_saved_state(self):
        """Test load correctly reads state saved by save."""
        original = BisectState(
            triton_dir="/path/to/triton",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=self.temp_dir,
            phase=BisectPhase.PAIR_TEST,
            triton_culprit="abc123",
        )
        path = StateManager.save(original, session_name="test_session")
        loaded = StateManager.load(str(path))
        self.assertEqual(loaded.triton_dir, original.triton_dir)
        self.assertEqual(loaded.phase, original.phase)
        self.assertEqual(loaded.triton_culprit, original.triton_culprit)

    def test_load_raises_file_not_found(self):
        """Test load raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            StateManager.load("/nonexistent/path/state.json")

    def test_exists_returns_correct_boolean(self):
        """Test exists returns True for existing file, False otherwise."""
        state = BisectState(
            triton_dir="/path",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=self.temp_dir,
        )
        path = StateManager.save(state, session_name="test_session")
        self.assertTrue(StateManager.exists(str(path)))
        self.assertFalse(StateManager.exists("/nonexistent/path.json"))

    def test_find_latest_state_returns_most_recent(self):
        """Test find_latest_state returns file with latest mtime."""
        # Create first state
        state1 = BisectState(
            triton_dir="/path1",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=self.temp_dir,
        )
        StateManager.save(state1, session_name="session_old")
        time.sleep(0.1)  # Ensure different mtime
        # Create second state
        state2 = BisectState(
            triton_dir="/path2",
            test_script="/test.py",
            good_commit="v1",
            bad_commit="v2",
            log_dir=self.temp_dir,
        )
        StateManager.save(state2, session_name="session_new")
        # Find latest
        latest = StateManager.find_latest_state(self.temp_dir)
        self.assertIsNotNone(latest)
        self.assertIn("session_new", str(latest))

    def test_find_latest_state_returns_none_for_empty_dir(self):
        """Test find_latest_state returns None when no state files."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        result = StateManager.find_latest_state(str(empty_dir))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
