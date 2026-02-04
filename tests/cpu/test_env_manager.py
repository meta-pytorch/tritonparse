# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for EnvironmentManager (CPU-only, no GPU required)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class EnvironmentManagerTest(unittest.TestCase):
    """Tests for EnvironmentManager class."""

    def test_init(self):
        """Test EnvironmentManager initialization."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        triton_dir = Path("/tmp/oss-triton")
        mock_logger = MagicMock()
        manager = EnvironmentManager(triton_dir, mock_logger)
        self.assertEqual(manager.triton_dir, triton_dir)
        self.assertEqual(manager.llvm_dir, triton_dir / "llvm-project")

    def test_repo_urls(self):
        """Test that repository URLs are correctly set."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        mock_logger = MagicMock()
        manager = EnvironmentManager(Path("/tmp/oss-triton"), mock_logger)
        self.assertEqual(manager.TRITON_REPO, "https://github.com/triton-lang/triton")
        self.assertEqual(manager.LLVM_REPO, "https://github.com/llvm/llvm-project")

    def test_check_environment_status_nothing_exists(self):
        """Test status check when nothing exists."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a non-existent subdirectory
            triton_dir = Path(tmpdir) / "nonexistent"

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Mock executor.run_command to return failure for git and success for conda
            def mock_run_command(cmd, cwd=None):
                result = MagicMock()
                if cmd[0] == "conda":
                    result.success = True
                else:
                    result.success = False
                result.stdout = ""
                return result

            manager.executor.run_command = mock_run_command
            status = manager.check_environment_status()

            self.assertFalse(status["triton_exists"])
            self.assertFalse(status["triton_is_valid_repo"])
            self.assertFalse(status["llvm_exists"])
            self.assertFalse(status["llvm_is_valid_repo"])
            # conda_available depends on mock, should be True
            self.assertTrue(status["conda_available"])

    def test_check_environment_status_dirs_exist_but_not_valid_repos(self):
        """Test status check when directories exist but aren't valid git repos."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            triton_dir = Path(tmpdir)
            llvm_dir = triton_dir / "llvm-project"
            llvm_dir.mkdir()

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Mock executor.run_command - git rev-parse fails (not valid repos)
            def mock_run_command(cmd, cwd=None):
                result = MagicMock()
                if cmd[0] == "conda":
                    result.success = True
                elif cmd == ["git", "rev-parse", "HEAD"]:
                    # Not a valid git repo
                    result.success = False
                else:
                    result.success = False
                result.stdout = ""
                return result

            manager.executor.run_command = mock_run_command
            status = manager.check_environment_status()

            self.assertTrue(status["triton_exists"])
            self.assertFalse(status["triton_is_valid_repo"])
            self.assertTrue(status["llvm_exists"])
            self.assertFalse(status["llvm_is_valid_repo"])

    def test_check_environment_status_valid_git_repos(self):
        """Test status check when directories are valid git repos."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            triton_dir = Path(tmpdir)
            llvm_dir = triton_dir / "llvm-project"
            llvm_dir.mkdir()

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Mock executor.run_command - git rev-parse succeeds (valid repos)
            def mock_run_command(cmd, cwd=None):
                result = MagicMock()
                if cmd[0] == "conda":
                    result.success = True
                elif cmd == ["git", "rev-parse", "HEAD"]:
                    # Valid git repo
                    result.success = True
                    result.stdout = "abc123def456\n"
                else:
                    result.success = True
                result.stdout = getattr(result, "stdout", "")
                return result

            manager.executor.run_command = mock_run_command
            status = manager.check_environment_status()

            self.assertTrue(status["triton_exists"])
            self.assertTrue(status["triton_is_valid_repo"])
            self.assertTrue(status["llvm_exists"])
            self.assertTrue(status["llvm_is_valid_repo"])

    def test_ensure_triton_repo_clone(self):
        """Test cloning Triton when directory doesn't exist."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            triton_dir = Path(tmpdir) / "nonexistent"

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Track calls to run_command
            calls = []

            def mock_run_command(cmd, cwd=None):
                calls.append((cmd, cwd))
                result = MagicMock()
                result.success = True
                result.stdout = "abc123\n"
                result.stderr = ""
                return result

            manager.executor.run_command = mock_run_command
            manager._ensure_triton_repo()

            # ensure_git_repo calls: git clone, then git rev-parse HEAD (verify)
            self.assertGreaterEqual(len(calls), 2)
            # First call: git clone
            cmd, _ = calls[0]
            self.assertEqual(cmd[0], "git")
            self.assertEqual(cmd[1], "clone")
            self.assertIn("triton-lang/triton", cmd[2])
            # Second call: git rev-parse HEAD (verification)
            self.assertEqual(calls[1][0], ["git", "rev-parse", "HEAD"])

    def test_ensure_triton_repo_fetch(self):
        """Test fetching updates when Triton directory exists."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            triton_dir = Path(tmpdir)

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Track calls to run_command
            calls = []

            def mock_run_command(cmd, cwd=None):
                calls.append((cmd, cwd))
                result = MagicMock()
                result.success = True
                result.stdout = "abc123\n"
                result.stderr = ""
                return result

            manager.executor.run_command = mock_run_command
            manager._ensure_triton_repo()

            # Should have called git rev-parse HEAD (validation) then git fetch
            self.assertEqual(len(calls), 2)
            # First call: git rev-parse HEAD
            self.assertEqual(calls[0][0], ["git", "rev-parse", "HEAD"])
            # Second call: git fetch origin
            self.assertEqual(calls[1][0], ["git", "fetch", "origin"])

    def test_ensure_llvm_repo_clone(self):
        """Test cloning LLVM when directory doesn't exist."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        with tempfile.TemporaryDirectory() as tmpdir:
            triton_dir = Path(tmpdir)
            # triton_dir exists but llvm_dir doesn't

            mock_logger = MagicMock()
            manager = EnvironmentManager(triton_dir, mock_logger)

            # Track calls to run_command
            calls = []

            def mock_run_command(cmd, cwd=None):
                calls.append((cmd, cwd))
                result = MagicMock()
                result.success = True
                result.stdout = "abc123\n"
                result.stderr = ""
                return result

            manager.executor.run_command = mock_run_command
            manager._ensure_llvm_repo()

            # ensure_git_repo calls: git clone, then git rev-parse HEAD (verify)
            self.assertGreaterEqual(len(calls), 2)
            # First call: git clone
            cmd, _ = calls[0]
            self.assertEqual(cmd[0], "git")
            self.assertEqual(cmd[1], "clone")
            self.assertIn("llvm/llvm-project", cmd[2])
            # Second call: git rev-parse HEAD (verification)
            self.assertEqual(calls[1][0], ["git", "rev-parse", "HEAD"])

    def test_ensure_conda_env_not_available(self):
        """Test error when conda is not available."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        mock_logger = MagicMock()
        manager = EnvironmentManager(Path("/tmp/oss-triton"), mock_logger)

        def mock_run_command(cmd, cwd=None):
            result = MagicMock()
            result.success = False
            result.stdout = ""
            result.stderr = "conda: command not found"
            return result

        manager.executor.run_command = mock_run_command

        with self.assertRaises(RuntimeError) as ctx:
            manager._ensure_conda_env("test_env")
        self.assertIn("conda not found", str(ctx.exception))

    def test_ensure_conda_env_exists(self):
        """Test when conda env already exists."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        mock_logger = MagicMock()
        manager = EnvironmentManager(Path("/tmp/oss-triton"), mock_logger)

        calls = []

        def mock_run_command(cmd, cwd=None):
            calls.append(cmd)
            result = MagicMock()
            if cmd == ["conda", "--version"]:
                result.success = True
                result.stdout = "conda 23.1.0"
            elif cmd == ["conda", "env", "list"]:
                result.success = True
                result.stdout = "base\ntriton_bisect    /path/to/env\n"
            else:
                result.success = True
                result.stdout = ""
            result.stderr = ""
            return result

        manager.executor.run_command = mock_run_command
        manager._ensure_conda_env("triton_bisect")

        # Should only call conda --version and conda env list, not create
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ["conda", "--version"])
        self.assertEqual(calls[1], ["conda", "env", "list"])

    def test_ensure_conda_env_create(self):
        """Test creating conda env when it doesn't exist."""
        from tritonparse.bisect.env_manager import EnvironmentManager

        mock_logger = MagicMock()
        manager = EnvironmentManager(Path("/tmp/oss-triton"), mock_logger)

        calls = []

        def mock_run_command(cmd, cwd=None):
            calls.append(cmd)
            result = MagicMock()
            if cmd == ["conda", "--version"]:
                result.success = True
                result.stdout = "conda 23.1.0"
            elif cmd == ["conda", "env", "list"]:
                result.success = True
                result.stdout = "base\nother_env    /path/to/env\n"
            else:
                result.success = True
                result.stdout = ""
            result.stderr = ""
            return result

        manager.executor.run_command = mock_run_command
        manager._ensure_conda_env("triton_bisect")

        # Should call all three commands
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], ["conda", "--version"])
        self.assertEqual(calls[1], ["conda", "env", "list"])
        # Third call should be conda create with python=3.12
        self.assertEqual(calls[2][0], "conda")
        self.assertEqual(calls[2][1], "create")
        self.assertIn("python=3.12", calls[2])


if __name__ == "__main__":
    unittest.main()
