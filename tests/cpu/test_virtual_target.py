# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import triton
from triton.compiler import ASTSource
from tritonparse import structured_logging, virtual_target
from tritonparse.capture import _prepare_environment, add_capture_args
from tritonparse.parse.trace_processor import (
    _create_fake_compilation,
    parse_single_file,
)
from tritonparse.target_profiles import target_profiles
from tritonparse.virtual_target import (
    _capture_launch,
    _install_grid_patch,
    _install_kernel_patch,
    _VirtualDriver,
    UnsupportedCaptureError,
)


class VirtualTargetTest(unittest.TestCase):
    def test_profiles_describe_devices_not_the_host(self) -> None:
        for key, backend, arch, warp, processors, smem in (
            ("cuda:90", "cuda", 90, 32, 132, 232448),
            ("cuda:100", "cuda", 100, 32, 148, 232448),
            ("hip:gfx942", "hip", "gfx942", 64, 304, 65536),
            ("hip:gfx950", "hip", "gfx950", 64, 256, 163840),
        ):
            with self.subTest(target=key):
                profile = target_profiles()[key]
                target = profile.target()
                self.assertEqual(
                    (backend, arch, warp),
                    (target.backend, target.arch, target.warp_size),
                )
                props = profile.properties()
                self.assertEqual(processors, props.multi_processor_count)
                self.assertEqual(smem, props.shared_memory_per_block_optin)
                self.assertFalse(hasattr(props, "total_memory"))
                self.assertFalse(hasattr(props, "shared_memory_per_block"))
                if backend == "cuda":
                    self.assertEqual(divmod(int(arch), 10), (props.major, props.minor))
                else:
                    self.assertEqual(arch, props.gcnArchName)

    def test_driver_keeps_host_streams_but_rejects_loading(self) -> None:
        host = SimpleNamespace(get_current_stream=lambda device: 1234)
        driver = _VirtualDriver(host, target_profiles()["hip:gfx950"])
        self.assertEqual(1234, driver.get_current_stream(0))
        self.assertEqual("gfx950", driver.get_current_target().arch)
        self.assertEqual(
            256, driver.utils.get_device_properties(0)["multiprocessor_count"]
        )
        for function in (
            driver.utils.load_binary,
            driver.utils.unload_module,
            driver.launcher_cls,
            driver.get_benchmarker(),
        ):
            with self.subTest(function=function):
                with self.assertRaises(UnsupportedCaptureError):
                    function()

    def test_capture_launch_calls_only_hooks(self) -> None:
        calls = []
        metadata = object()
        _capture_launch(
            2,
            1,
            1,
            0,
            None,
            (),
            metadata,
            lambda value: calls.append(("enter", value)),
            lambda value: calls.append(("exit", value)),
            object(),
        )
        self.assertEqual([("enter", metadata), ("exit", metadata)], calls)
        with self.assertRaisesRegex(UnsupportedCaptureError, "executable"):
            _capture_launch(1, 1, 1, 0, 123, (), metadata, None, None)

    def test_kernel_initialization_never_calls_the_loader(self) -> None:
        profile = target_profiles()["cuda:90"]

        class Kernel(SimpleNamespace):
            def _init_handles(self):
                raise AssertionError("Must not call original initialization")

        _install_kernel_patch(Kernel, profile)
        kernel = Kernel()
        kernel.module = kernel.function = None
        kernel.metadata = SimpleNamespace(target=profile.target())
        for _ in range(2):
            kernel._init_handles()
            self.assertIsNone(kernel.module)
            self.assertIsNone(kernel.function)
            self.assertIsNone(kernel._dispatcher)
            self.assertIsNone(kernel.n_regs)
            self.assertIs(kernel._run, _capture_launch)
        kernel.function = 123
        with self.assertRaisesRegex(UnsupportedCaptureError, "loaded kernel"):
            kernel._init_handles()
        kernel.function = None
        kernel.metadata.target = target_profiles()["hip:gfx942"].target()
        with self.assertRaisesRegex(UnsupportedCaptureError, "different target"):
            kernel._init_handles()

    def test_integer_grid_and_callable_grid_evaluated_once(self) -> None:
        class Function:
            def run(self, value, *, grid, warmup=False):
                return grid({"value": value}) if callable(grid) else grid

        _install_grid_patch(Function)
        function = Function()
        self.assertEqual((3,), function.run(0, grid=3))
        self.assertEqual((3, 2), function.run(0, grid=(3, 2)))
        grid = Mock(side_effect=lambda meta: meta["value"] + 1)
        self.assertEqual((5,), function.run(4, grid=grid))
        grid.assert_called_once_with({"value": 4})

    def test_environment_is_private_and_disables_execution_shortcuts(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TRITON_CACHE_DIR": "/external/cache",
                "TRITON_REMOTE_CACHE_BACKEND": "external:Cache",
                "TRITON_TRACE": "/external/traces",
                "TRITON_COMPILE_IQ_APPLY": "1",
                "TRITONPARSE_SAVE_TENSOR_BLOBS": "1",
            },
        ):
            _prepare_environment("/private/cache")
            self.assertEqual("/private/cache", os.environ["TRITON_CACHE_DIR"])
            self.assertNotIn("TRITON_REMOTE_CACHE_BACKEND", os.environ)
            self.assertNotIn("TRITON_TRACE", os.environ)
            for key in (
                "TRITON_USE_C_DISPATCHER",
                "TRITON_ENABLE_C_CACHE",
                "TRITON_AUTOTUNE_USE_C_CACHE",
                "TRITON_COMPILE_IQ_APPLY",
                "TRITON_COMPILE_IQ_COLLECT",
                "TRITONPARSE_MORE_TENSOR_INFORMATION",
                "TRITONPARSE_SAVE_TENSOR_BLOBS",
                "TRITONPARSE_TRACE_MANIFOLD",
            ):
                self.assertEqual("0", os.environ[key], key)

    def test_capture_requires_a_real_host_gpu(self) -> None:
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(UnsupportedCaptureError, "host GPU"):
                virtual_target.install(target_profiles()["cuda:90"], "/unused")

    def test_install_rejects_a_second_target_before_touching_gpu(self) -> None:
        with patch.object(virtual_target, "_installed", True):
            with self.assertRaisesRegex(UnsupportedCaptureError, "new process"):
                virtual_target.install(target_profiles()["cuda:90"], "/unused")

    def test_cli_preserves_workload_arguments(self) -> None:
        parser = argparse.ArgumentParser()
        add_capture_args(parser)
        args = parser.parse_args(
            ["--target", "hip:gfx950", "--out", "traces", "script.py", "--size", "128"]
        )
        self.assertEqual("hip:gfx950", args.target)
        self.assertEqual(Path("script.py"), args.script)
        self.assertEqual(["--size", "128"], args.script_args)


class VirtualTraceTest(unittest.TestCase):
    def test_compilation_and_launch_metadata_remain_honest_after_logging_reset(
        self,
    ) -> None:
        metadata = target_profiles()["hip:gfx950"].metadata()
        with patch.object(structured_logging, "_virtual_target", metadata):
            structured_logging.clear_logging_config()
            with patch.object(structured_logging.triton_trace_log, "handle") as handle:
                for event_type in ("compilation", "launch"):
                    structured_logging.trace_structured_triton(
                        event_type,
                        metadata_fn=lambda: {
                            "executed": True,
                            "runtime_validated": True,
                        },
                    )
                    event = json.loads(
                        structured_logging.TritonJsonFormatter().format(
                            handle.call_args.args[0]
                        )
                    )
                    self.assertEqual(metadata, event["virtual_target"])
                    self.assertIs(event["runtime_validated"], False)
                    if event_type == "launch":
                        self.assertIs(event["executed"], False)

    def test_ordinary_events_do_not_gain_virtual_metadata(self) -> None:
        with patch.object(structured_logging, "_virtual_target", None):
            with patch.object(structured_logging.triton_trace_log, "handle") as handle:
                structured_logging.trace_structured_triton("launch")
                event = json.loads(
                    structured_logging.TritonJsonFormatter().format(
                        handle.call_args.args[0]
                    )
                )
        self.assertNotIn("virtual_target", event)
        self.assertNotIn("executed", event)
        self.assertNotIn("runtime_validated", event)

    def test_compilation_payload_preserves_virtual_target(self) -> None:
        @triton.jit
        def kernel(pointer):
            pass

        profile = target_profiles()["hip:gfx942"]
        metadata = profile.metadata()
        source = ASTSource(kernel, signature={"pointer": "*fp32"})
        with patch.object(structured_logging, "_virtual_target", metadata):
            with patch.object(structured_logging.triton_trace_log, "handle"):
                trace = structured_logging.maybe_trace_triton(
                    source,
                    {"hash": "h1", "name": "kernel", "target": profile.target()},
                    {},
                    None,
                )
        self.assertEqual(metadata, trace["metadata"]["virtual_target"])
        self.assertIs(trace["metadata"]["runtime_validated"], False)

    def test_compilation_rejects_a_conflicting_target_before_emitting(self) -> None:
        @triton.jit
        def kernel(pointer):
            pass

        source = ASTSource(kernel, signature={"pointer": "*fp32"})
        profile = target_profiles()["cuda:90"].metadata()
        with patch.object(structured_logging, "_virtual_target", profile):
            with patch.object(structured_logging.triton_trace_log, "handle") as handle:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    structured_logging.maybe_trace_triton(
                        source,
                        {
                            "hash": "h1",
                            "name": "kernel",
                            "target": target_profiles()["hip:gfx950"].target(),
                        },
                        {},
                        None,
                    )
                handle.assert_not_called()

    def test_synthetic_compilation_retains_virtual_provenance(self) -> None:
        metadata = target_profiles()["hip:gfx942"].metadata()
        event = _create_fake_compilation(
            {
                "virtual_target": metadata,
                "compilation_metadata": {
                    "name": "kernel",
                    "target": metadata["target"],
                },
            },
            "hash",
        )
        self.assertTrue(event["is_fake"])
        self.assertEqual(metadata, event["payload"]["metadata"]["virtual_target"])
        self.assertEqual(metadata["target"], event["payload"]["metadata"]["target"])
        self.assertIs(event["runtime_validated"], False)

    def test_parser_does_not_invent_an_autotune_winner(self) -> None:
        profile = target_profiles()["cuda:90"].metadata()
        stack = [
            {"filename": "/workload.py", "name": "main", "line": 10},
            {"filename": "/triton/runtime/autotuner.py", "name": "run", "line": 20},
        ]
        for virtual in (False, True):
            with self.subTest(virtual=virtual), tempfile.TemporaryDirectory() as root:
                markers = (
                    {"virtual_target": profile, "runtime_validated": False}
                    if virtual
                    else {}
                )
                compilation = {
                    "event_type": "compilation",
                    "pid": 1,
                    "stack": stack,
                    **markers,
                    "payload": {
                        "metadata": {
                            "hash": "h1",
                            "name": "kernel",
                            "target": profile["target"],
                            **markers,
                        },
                        "file_content": {},
                    },
                }
                launch = {
                    "event_type": "launch",
                    "pid": 1,
                    "name": "kernel",
                    "stack": stack,
                    "compilation_metadata": {"hash": "h1"},
                    "grid": [1, 1, 1],
                    **markers,
                }
                if virtual:
                    launch["executed"] = False
                source, output = Path(root) / "trace.ndjson", Path(root) / "parsed"
                source.write_text(
                    "".join(json.dumps(event) + "\n" for event in (compilation, launch))
                )
                output.mkdir()
                parse_single_file(str(source), str(output))
                events = [
                    json.loads(line)
                    for path in output.iterdir()
                    for line in path.read_text().splitlines()
                ]
                parsed = next(
                    event for event in events if event["event_type"] == "launch"
                )
                if virtual:
                    self.assertNotIn("autotune_launch_type", parsed)
                    self.assertEqual(profile, parsed["virtual_target"])
                    self.assertIs(parsed["executed"], False)
                else:
                    self.assertEqual("cached_winner", parsed["autotune_launch_type"])
