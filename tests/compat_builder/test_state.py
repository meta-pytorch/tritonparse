# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict

"""Tests for tritonparse.compat_builder.state."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from tritonparse.compat_builder.state import (
    CompatBuildPhase,
    CompatBuildState,
    CompatStateManager,
)


class CompatBuildPhaseTest(unittest.TestCase):
    def test_enum_values(self) -> None:
        self.assertEqual(CompatBuildPhase.INITIALIZING.value, "initializing")
        self.assertEqual(
            CompatBuildPhase.FINDING_INCOMPATIBLE.value, "finding_incompatible"
        )
        self.assertEqual(CompatBuildPhase.AI_FIXING.value, "ai_fixing")
        self.assertEqual(CompatBuildPhase.WAITING_FOR_FIX.value, "waiting_for_fix")
        self.assertEqual(CompatBuildPhase.APPLYING_FIX.value, "applying_fix")
        self.assertEqual(CompatBuildPhase.COMPLETED.value, "completed")
        self.assertEqual(CompatBuildPhase.FAILED.value, "failed")

    def test_all_phases_present(self) -> None:
        expected = {
            "INITIALIZING",
            "FINDING_INCOMPATIBLE",
            "AI_FIXING",
            "WAITING_FOR_FIX",
            "APPLYING_FIX",
            "COMPLETED",
            "FAILED",
        }
        actual = {phase.name for phase in CompatBuildPhase}
        self.assertEqual(actual, expected)

    def test_from_value_roundtrip(self) -> None:
        for phase in CompatBuildPhase:
            self.assertEqual(CompatBuildPhase(phase.value), phase)


class CompatBuildStateTest(unittest.TestCase):
    def _make_state(self) -> CompatBuildState:
        return CompatBuildState(
            triton_dir="/tmp/triton",
            llvm_bump_commit="abc123def456abcd",
            output_csv="/tmp/commits.csv",
        )

    def test_default_phase_is_initializing(self) -> None:
        state = self._make_state()
        self.assertEqual(state.phase, CompatBuildPhase.INITIALIZING)

    def test_default_pairs_empty(self) -> None:
        state = self._make_state()
        self.assertEqual(state.pairs, [])

    def test_add_pair_appends(self) -> None:
        state = self._make_state()
        state.add_pair("triton_aaa", "llvm_bbb")
        self.assertEqual(len(state.pairs), 1)
        self.assertEqual(state.pairs[0], ("triton_aaa", "llvm_bbb"))

    def test_add_pair_multiple(self) -> None:
        state = self._make_state()
        state.add_pair("t1", "l1")
        state.add_pair("t2", "l2")
        self.assertEqual(len(state.pairs), 2)
        self.assertEqual(state.pairs[0], ("t1", "l1"))
        self.assertEqual(state.pairs[1], ("t2", "l2"))

    def test_to_dict_phase_serialized_as_string(self) -> None:
        state = self._make_state()
        d = state.to_dict()
        self.assertEqual(d["phase"], "initializing")
        self.assertIsInstance(d["phase"], str)

    def test_to_dict_pairs_are_lists(self) -> None:
        state = self._make_state()
        state.add_pair("t1", "l1")
        d = state.to_dict()
        self.assertIsInstance(d["pairs"][0], list)
        self.assertEqual(d["pairs"][0], ["t1", "l1"])

    def test_to_dict_contains_required_keys(self) -> None:
        state = self._make_state()
        d = state.to_dict()
        for key in ("triton_dir", "llvm_bump_commit", "output_csv", "phase", "pairs"):
            self.assertIn(key, d)

    def test_from_dict_roundtrip(self) -> None:
        state = self._make_state()
        state.add_pair("t1", "l1")
        state.add_pair("t2", "l2")
        state.phase = CompatBuildPhase.FINDING_INCOMPATIBLE
        state.old_llvm = "old_llvm_hash"
        state.new_llvm = "new_llvm_hash"
        state.current_triton = "current_triton_abc"
        state.current_llvm_good = "good_llvm_xyz"
        state.ai_fix_attempted = True

        d = state.to_dict()
        restored = CompatBuildState.from_dict(d)

        self.assertEqual(restored.triton_dir, "/tmp/triton")
        self.assertEqual(restored.llvm_bump_commit, "abc123def456abcd")
        self.assertEqual(restored.phase, CompatBuildPhase.FINDING_INCOMPATIBLE)
        self.assertEqual(restored.pairs, [("t1", "l1"), ("t2", "l2")])
        self.assertEqual(restored.old_llvm, "old_llvm_hash")
        self.assertEqual(restored.new_llvm, "new_llvm_hash")
        self.assertEqual(restored.current_triton, "current_triton_abc")
        self.assertTrue(restored.ai_fix_attempted)

    def test_from_dict_empty_pairs(self) -> None:
        state = self._make_state()
        d = state.to_dict()
        restored = CompatBuildState.from_dict(d)
        self.assertEqual(restored.pairs, [])

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state()
            state.log_dir = tmp
            state.add_pair("t1", "l1")
            state.phase = CompatBuildPhase.COMPLETED
            state.old_llvm = "old_hash"
            state.new_llvm = "new_hash"

            saved_path = state.save(session_name="test_session")
            self.assertTrue(saved_path.exists())
            self.assertIn("test_session", saved_path.name)

            loaded = CompatBuildState.load(saved_path)
            self.assertEqual(loaded.phase, CompatBuildPhase.COMPLETED)
            self.assertEqual(loaded.pairs, [("t1", "l1")])
            self.assertEqual(loaded.triton_dir, "/tmp/triton")
            self.assertEqual(loaded.old_llvm, "old_hash")

    def test_save_sets_started_at_and_updated_at(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state()
            state.log_dir = tmp
            self.assertIsNone(state.started_at)
            self.assertIsNone(state.updated_at)
            state.save(session_name="ts")
            self.assertIsNotNone(state.started_at)
            self.assertIsNotNone(state.updated_at)

    def test_save_started_at_not_overwritten_on_second_save(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = self._make_state()
            state.log_dir = tmp
            state.save(session_name="ts")
            first_started = state.started_at
            time.sleep(0.01)
            state.save(session_name="ts")
            self.assertEqual(state.started_at, first_started)

    def test_save_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            explicit_path = Path(tmp) / "my_state.json"
            state = self._make_state()
            saved = state.save(path=explicit_path)
            self.assertEqual(saved, explicit_path)
            self.assertTrue(explicit_path.exists())

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            CompatBuildState.load(Path("/nonexistent/path/state.json"))


class CompatStateManagerTest(unittest.TestCase):
    def test_get_state_path(self) -> None:
        path = CompatStateManager.get_state_path("/logs", "20240101_120000")
        self.assertEqual(path, Path("/logs/20240101_120000_compat_state.json"))

    def test_state_suffix(self) -> None:
        self.assertTrue(CompatStateManager.STATE_SUFFIX.endswith(".json"))
        self.assertIn("compat_state", CompatStateManager.STATE_SUFFIX)

    def test_find_latest_state_missing_dir(self) -> None:
        result = CompatStateManager.find_latest_state("/nonexistent/path/abc123xyz")
        self.assertIsNone(result)

    def test_find_latest_state_empty_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = CompatStateManager.find_latest_state(tmp)
            self.assertIsNone(result)

    def test_find_latest_state_returns_most_recent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = CompatBuildState(
                triton_dir="/t",
                llvm_bump_commit="abc123",
                output_csv="/o.csv",
            )
            state.log_dir = tmp
            CompatStateManager.save(state, session_name="session_a")
            time.sleep(0.05)
            path2 = CompatStateManager.save(state, session_name="session_b")

            latest = CompatStateManager.find_latest_state(tmp)
            self.assertEqual(latest, path2)

    def test_find_latest_state_single_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = CompatBuildState(
                triton_dir="/t",
                llvm_bump_commit="abc123",
                output_csv="/o.csv",
            )
            state.log_dir = tmp
            saved = CompatStateManager.save(state, session_name="only_session")

            latest = CompatStateManager.find_latest_state(tmp)
            self.assertEqual(latest, saved)

    def test_save_uses_session_name_in_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = CompatBuildState(
                triton_dir="/t",
                llvm_bump_commit="abc",
                output_csv="/o.csv",
            )
            state.log_dir = tmp
            path = CompatStateManager.save(state, session_name="mysession")
            self.assertIn("mysession", path.name)

    def test_save_generates_session_name_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = CompatBuildState(
                triton_dir="/t",
                llvm_bump_commit="abc",
                output_csv="/o.csv",
                log_dir=tmp,
            )
            path = CompatStateManager.save(state)
            self.assertTrue(path.exists())
            self.assertIsNotNone(state.session_name)

    def test_load_nonexistent_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            CompatStateManager.load("/nonexistent/state_file.json")
