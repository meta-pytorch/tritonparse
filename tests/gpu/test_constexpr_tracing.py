# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Compile, trace, and replay constexpr launch arguments on a real GPU."""

import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import triton  # @manual=//triton:triton
import triton.language as tl  # @manual=//triton:triton
from tests.test_utils import GPUTestBase
from tritonparse import structured_logging
from tritonparse._json_compat import loads
from tritonparse.reproducer.orchestrator import reproduce
from tritonparse.reproducer.types import KernelImportMode


@triton.jit
def constexpr_kernel(
    output,
    INDEX,
    SCALE: tl.constexpr,
    FLAG: tl.constexpr,
    MODE: tl.constexpr,
    OPTIONAL: tl.constexpr,
    DTYPE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # INDEX is deliberately unannotated: replaying a bare int changes its specialization.
    tl.static_assert(INDEX == 4)
    offsets = tl.arange(0, BLOCK_SIZE)
    value = offsets.to(DTYPE) * SCALE + INDEX
    if FLAG:
        value += 1
    if MODE == "4":
        value += 4
    if OPTIONAL is None:
        value += 2
    tl.store(output + offsets, value)


class ConstexprLaunchTracingTest(GPUTestBase):
    def test_scalar_constexpr_capture_and_replay(self):
        self._capture_and_replay(tl.float32)

    def test_dtype_constexpr_capture_and_replay(self):
        self._capture_and_replay(tl.constexpr(tl.float32))

    def _read_trace(self, directory):
        paths = list(directory.glob("*.ndjson"))
        self.assertEqual(len(paths), 1)
        events = [loads(line) for line in paths[0].read_text().splitlines()]
        compilations = [e for e in events if e["event_type"] == "compilation"]
        launches = [e for e in events if e["event_type"] == "launch"]
        return paths[0], compilations, launches

    def _capture_and_replay(self, dtype):
        output = torch.empty(32, device=self.cuda_device, dtype=torch.float32)
        args = {
            "output": output,
            "INDEX": tl.constexpr(4),
            "SCALE": tl.constexpr(0.5),
            "FLAG": tl.constexpr(True),
            "MODE": tl.constexpr("4"),
            "OPTIONAL": tl.constexpr(None),
            "DTYPE": dtype,
            "BLOCK_SIZE": tl.constexpr(32),
        }
        expected = torch.arange(32, device=self.cuda_device) * 0.5 + 11

        constexpr_kernel.device_caches.clear()
        baseline = constexpr_kernel[(1,)](**args)
        torch.testing.assert_close(output, expected)

        # Force the traced run through JIT compilation, including its post-compile hook.
        constexpr_kernel.device_caches.clear()
        constexpr_kernel.launch_metadata = None
        with tempfile.TemporaryDirectory(prefix="tritonparse_constexpr_") as temp:
            root = Path(temp)
            try:
                capture_dir = root / "capture"
                structured_logging.init(
                    str(capture_dir), enable_trace_launch=True, compression="none"
                )
                for _ in range(2):
                    output.fill_(float("nan"))
                    compiled = constexpr_kernel[(1,)](**args)
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(compiled.hash, baseline.hash)
                structured_logging.clear_logging_config()

                trace_path, compilations, launches = self._read_trace(capture_dir)
                self.assertEqual(len(compilations), 1)
                self.assertEqual(len(launches), 2)
                self.assertEqual(
                    compilations[0]["payload"]["metadata"]["hash"], baseline.hash
                )
                expected_args = {
                    "INDEX": {"type": "int", "value": 4, "constexpr": True},
                    "SCALE": {"type": "float", "value": 0.5, "constexpr": True},
                    "FLAG": {"type": "bool", "value": True, "constexpr": True},
                    "MODE": {
                        "type": "str",
                        "value": "4",
                        "length": 1,
                        "constexpr": True,
                    },
                    "OPTIONAL": {
                        "type": "NoneType",
                        "repr": "None",
                        "constexpr": True,
                    },
                    "BLOCK_SIZE": {"type": "int", "value": 32, "constexpr": True},
                    "DTYPE": {"type": "dtype", "repr": "fp32"},
                }
                if isinstance(dtype, tl.constexpr):
                    expected_args["DTYPE"]["constexpr"] = True
                for launch in launches:
                    self.assertEqual(
                        launch["compilation_metadata"]["hash"], baseline.hash
                    )
                    for name, info in expected_args.items():
                        self.assertEqual(launch["extracted_args"][name], info)

                result = reproduce(
                    input_path=str(trace_path),
                    line_index=0,
                    out_dir=str(root / "reproducer"),
                    template="example",
                    kernel_name="constexpr_kernel",
                    kernel_import=KernelImportMode.DEFAULT,
                    embed_context=True,
                    skip_logger=True,
                )
                namespace = runpy.run_path(result["repro_script"])
                constexpr_kernel.device_caches.clear()
                constexpr_kernel.launch_metadata = None
                launch_kernel = namespace["launch_kernel"]
                create_args = namespace["create_args_from_json"]
                replay_args = {}

                def record_replay_args(data):
                    grid, reconstructed = create_args(data)
                    replay_args.update(reconstructed)
                    return grid, reconstructed

                replay_dir = root / "replay"
                structured_logging.init(
                    str(replay_dir), enable_trace_launch=True, compression="none"
                )
                with patch.dict(
                    launch_kernel.__globals__,
                    {"create_args_from_json": record_replay_args},
                ):
                    launch_kernel()
                structured_logging.clear_logging_config()

                torch.testing.assert_close(replay_args["output"], expected)
                for name, value in args.items():
                    if isinstance(value, tl.constexpr):
                        self.assertIsInstance(replay_args[name], tl.constexpr)
                        self.assertIs(type(replay_args[name].value), type(value.value))
                        self.assertEqual(replay_args[name].value, value.value)
                _, replay_compilations, replay_launches = self._read_trace(replay_dir)
                self.assertEqual(len(replay_compilations), 1)
                self.assertEqual(len(replay_launches), 1)
                self.assertEqual(
                    replay_compilations[0]["payload"]["metadata"]["hash"], baseline.hash
                )
                self.assertEqual(
                    replay_launches[0]["compilation_metadata"]["hash"], baseline.hash
                )
            finally:
                structured_logging.clear_logging_config()
                constexpr_kernel.launch_metadata = None
                constexpr_kernel.device_caches.clear()
