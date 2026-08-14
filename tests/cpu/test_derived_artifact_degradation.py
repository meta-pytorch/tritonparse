# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests that a failing optional derived artifact degrades instead of aborting.

Regression coverage for `nvdisasm` not being on the path: the RuntimeError it
raised escaped `extract_file_content`, propagated out of `maybe_trace_triton`
and aborted the entire trace write, so a run produced no raw_logs at all even
though every stage except SASS had succeeded.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import tritonparse.structured_logging as structured_logging
from tritonparse.backend import DerivedArtifactInfo, get_backend_registry
from tritonparse.shared_vars import set_runtime_sass_dump_override
from tritonparse.structured_logging import extract_file_content

TTIR_BODY = "module { // ttir body }"


class DerivedArtifactFailureDegradesTest(unittest.TestCase):
    def setUp(self):
        # The "log once per kind" cache is process-lifetime; clear it so each
        # test sees a deterministic first-failure warning.
        structured_logging._logged_derived_artifact_failures.clear()
        self.addCleanup(structured_logging._logged_derived_artifact_failures.clear)

        self.original_derived_artifacts_env = os.environ.get(
            "TRITONPARSE_DERIVED_ARTIFACTS"
        )
        os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = "sass"
        self.addCleanup(self._restore_derived_artifacts_env)
        set_runtime_sass_dump_override(None)
        self.addCleanup(set_runtime_sass_dump_override, None)

        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

        # A real text IR file plus a cubin, so we can prove the text stage
        # survives a failing derivation.
        self.ttir_path = os.path.join(self.tmp_dir, "kernel.ttir")
        with open(self.ttir_path, "w") as f:
            f.write(TTIR_BODY)
        self.cubin_path = os.path.join(self.tmp_dir, "kernel.cubin")
        with open(self.cubin_path, "wb") as f:
            f.write(b"\x7fELF fake cubin")

        self.metadata_group = {
            "kernel.ttir": self.ttir_path,
            "kernel.cubin": self.cubin_path,
        }

    def _restore_derived_artifacts_env(self):
        if self.original_derived_artifacts_env is None:
            os.environ.pop("TRITONPARSE_DERIVED_ARTIFACTS", None)
        else:
            os.environ["TRITONPARSE_DERIVED_ARTIFACTS"] = (
                self.original_derived_artifacts_env
            )

    def _new_payload(self):
        return {"metadata": {}, "file_path": {}, "file_content": {}}

    def _failing_sass_info(self, exc):
        """A sass DerivedArtifactInfo whose derive_func always raises `exc`."""

        def raising_derive_func(_source_path):
            raise exc

        return DerivedArtifactInfo(
            target_stage_name="sass",
            source_stage_name="cubin",
            tool_name="nvdisasm",
            derive_func=raising_derive_func,
        )

    def _extract_with_failing_artifact(self, payload, exc):
        """Run adapter-driven extraction with a derive_func that raises.

        The adapter binds derive_func at construction time, so the failure is
        injected by patching the artifact list rather than the disasm module.
        """
        adapter = get_backend_registry().resolve_from_backend_name("cuda")
        with patch.object(
            adapter,
            "list_applicable_derived_artifacts",
            return_value=[self._failing_sass_info(exc)],
        ):
            extract_file_content(payload, self.metadata_group, "cuda")

    def test_runtime_error_from_derive_func_does_not_propagate(self):
        # This is the exact failure mode: triton.knobs raises RuntimeError when
        # nvdisasm is not on the path, and it is neither a subprocess nor an OS
        # error, so the previous narrow except clauses let it escape.
        payload = self._new_payload()
        self._extract_with_failing_artifact(
            payload, RuntimeError("Cannot find nvdisasm")
        )
        # Reaching this line at all is the regression assertion.
        self.assertIn("kernel.sass", payload["file_content"])

    def test_rest_of_trace_survives_failing_derive_func(self):
        payload = self._new_payload()
        self._extract_with_failing_artifact(
            payload, RuntimeError("Cannot find nvdisasm")
        )
        # Every non-derived stage is still captured intact.
        self.assertEqual(payload["file_content"]["kernel.ttir"], TTIR_BODY)
        self.assertEqual(payload["file_path"]["kernel.ttir"], self.ttir_path)
        self.assertEqual(payload["file_path"]["kernel.cubin"], self.cubin_path)

    def test_failure_is_recorded_in_trace_content(self):
        payload = self._new_payload()
        self._extract_with_failing_artifact(
            payload, RuntimeError("Cannot find nvdisasm")
        )
        message = payload["file_content"]["kernel.sass"]
        self.assertIn("nvdisasm", message)
        self.assertIn("Cannot find nvdisasm", message)

    def test_failure_is_logged_with_artifact_and_reason(self):
        payload = self._new_payload()
        with self.assertLogs("tritonparse.structured_logging", level="WARNING") as cm:
            self._extract_with_failing_artifact(
                payload, RuntimeError("Cannot find nvdisasm")
            )
        logged = "\n".join(cm.output)
        self.assertIn("sass", logged)
        self.assertIn("nvdisasm", logged)
        self.assertIn("Cannot find nvdisasm", logged)

    def test_failure_logged_once_per_kind(self):
        with self.assertLogs("tritonparse.structured_logging", level="WARNING") as cm:
            self._extract_with_failing_artifact(
                self._new_payload(), RuntimeError("Cannot find nvdisasm")
            )
        self.assertEqual(len(cm.output), 1)

        # A second compilation hitting the same failure must not log again, but
        # it must still record the placeholder in that event's own trace data.
        second_payload = self._new_payload()
        with patch.object(structured_logging.log, "warning") as mock_warning:
            self._extract_with_failing_artifact(
                second_payload, RuntimeError("Cannot find nvdisasm")
            )
        mock_warning.assert_not_called()
        self.assertIn("nvdisasm", second_payload["file_content"]["kernel.sass"])

    def test_clear_logging_config_resets_the_log_once_cache(self):
        """The dedup is scoped to a tracing session, not to the process.

        `TritonParseManager.__exit__` calls `clear_logging_config()`, so without
        this reset a process that traces more than once would warn about a
        still-missing external tool in its first session only and stay silent in
        every later one, even though each session writes its own trace files.
        """
        with self.assertLogs("tritonparse.structured_logging", level="WARNING"):
            self._extract_with_failing_artifact(
                self._new_payload(), RuntimeError("Cannot find nvdisasm")
            )

        # triton is not a dependency of this test target; clear_logging_config()
        # only reaches for it to null out hooks that were never installed here.
        with patch.dict(sys.modules, {"triton": MagicMock()}):
            structured_logging.clear_logging_config()

        with self.assertLogs("tritonparse.structured_logging", level="WARNING") as cm:
            self._extract_with_failing_artifact(
                self._new_payload(), RuntimeError("Cannot find nvdisasm")
            )
        self.assertEqual(len(cm.output), 1)

    def test_subprocess_failure_message_preserved(self):
        payload = self._new_payload()
        self._extract_with_failing_artifact(
            payload, subprocess.CalledProcessError(1, "nvdisasm")
        )
        self.assertIn("nvdisasm failed:", payload["file_content"]["kernel.sass"])

    def test_os_error_message_preserved(self):
        payload = self._new_payload()
        self._extract_with_failing_artifact(payload, OSError("cubin unreadable"))
        self.assertIn(
            "error dumping derived artifact", payload["file_content"]["kernel.sass"]
        )

    def test_successful_derivation_still_stored(self):
        """The containment must not change the happy path."""
        payload = self._new_payload()
        info = DerivedArtifactInfo(
            target_stage_name="sass",
            source_stage_name="cubin",
            tool_name="nvdisasm",
            derive_func=lambda _path: "sass output",
        )
        adapter = get_backend_registry().resolve_from_backend_name("cuda")
        with patch.object(
            adapter, "list_applicable_derived_artifacts", return_value=[info]
        ):
            extract_file_content(payload, self.metadata_group, "cuda")
        self.assertEqual(payload["file_content"]["kernel.sass"], "sass output")
        self.assertEqual(payload["file_content"]["kernel.ttir"], TTIR_BODY)

    def test_legacy_path_also_degrades(self):
        """The legacy fallback reports the failure the same way.

        Containment is not new on this path -- it always ended in a catch-all,
        so it was never the source of the aborted trace. What is pinned here is
        that it now produces the shared placeholder text and the deduplicated
        warning, so the two paths cannot drift apart.
        """
        payload = self._new_payload()
        with patch(
            "tritonparse.tools.disasm.extract",
            side_effect=RuntimeError("Cannot find nvdisasm"),
        ):
            with self.assertLogs(
                "tritonparse.structured_logging", level="WARNING"
            ) as cm:
                # An empty backend name fails adapter resolution and falls back.
                extract_file_content(payload, self.metadata_group, "")
        self.assertIn("nvdisasm", payload["file_content"]["kernel.sass"])
        self.assertEqual(payload["file_content"]["kernel.ttir"], TTIR_BODY)
        self.assertIn("nvdisasm", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()
