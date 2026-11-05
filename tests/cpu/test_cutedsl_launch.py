# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for CuTeDSL launch tracing.

Covers `PreExecuteEvent` handling (original-args and packed paths),
argument naming, the grid taken from the compile-time launch configuration,
schema validation, grouping of CuTeDSL compilation+launch events by hash, and
setup/teardown wiring of the launch hook. No GPU or cutlass installation
required.
"""

import glob
import json
import os
import shutil
import sys
import tempfile
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from tritonparse import cutedsl_logging, structured_logging
from tritonparse.parse.trace_processor import _prescan_for_fake_compilations
from tritonparse.validation.json_validator import validate_record


# Host function launching one kernel; GRID_DIMS is filled in per test.
HOST_MLIR_TEMPLATE = """func.func @cutlass_mk(%arg0: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.ceildivsi %arg0, %c128_i32 : i32
    %1 = cuda.launch_cfg.create<max_attrs = 17 : i32> (blockDim = (%c128_i32, %c1_i32, %c1_i32), dynamicSmemBytes = %c1_i32, gridDim = (GRID_DIMS), stream = %s) : i32, i32, i32, i64, i32, i32, i32, !cuda.stream -> !cuda.launch_cfg<max_attrs = 17>
    %2 = cuda.launch_ex @kernels::@mkernel<%1> (%arg0) : !cuda.launch_cfg<max_attrs = 17>, (i32) -> !cuda.result
}
"""
STATIC_GRID = "%c8_i32, %c1_i32, %c1_i32"
DYNAMIC_GRID = "%0, %c1_i32, %c1_i32"


def _trace_compilation(grid_dims=None, module_hash="deadbeef0123"):
    artifacts = None
    if grid_dims is not None:
        artifacts = SimpleNamespace(
            MLIR=HOST_MLIR_TEMPLATE.replace("GRID_DIMS", grid_dims),
            MLIR_RAW=None,
            PTX=None,
            CUBIN=None,
        )
    return cutedsl_logging.maybe_trace_cutedsl(
        object(),
        function_name="kernel_cutlass_mk_abc123",
        module_hash=module_hash,
        cache_hit=False,
        artifacts=artifacts,
        compile_options=None,
        func_body=None,
        kernel_info=None,
    )


def _trace_launch(**overrides):
    kwargs = {
        "function_name": "kernel_cutlass_mk_abc123",
        "module_hash": "deadbeef0123",
        "kernel_info": {"mkernel": {}},
        "args": (torch.zeros(4), 128),
        "kwargs": {"flag": True},
        "exe_args": ["packed0", "packed1"],
        "executor": None,
    }
    kwargs.update(overrides)
    return cutedsl_logging.maybe_trace_cutedsl_launch(object(), **kwargs)


def _cute_tensor():
    """A stand-in that extract_arg_info takes for a CuTeDSL runtime tensor."""

    class _Tensor:
        shape = (1024,)
        stride = (1,)
        memspace = "gmem"
        data_ptr = 12345

    _Tensor.__module__ = "cutlass.cute.runtime"
    return _Tensor()


class MaybeTraceCutedslLaunchTest(unittest.TestCase):
    def setUp(self):
        cutedsl_logging._reset_for_tests()
        self.addCleanup(cutedsl_logging._reset_for_tests)

    def test_payload_shape_with_original_args(self):
        trace_data = _trace_launch()
        self.assertEqual(trace_data["name"], "kernel_cutlass_mk_abc123")
        self.assertIsNone(trace_data["function"])
        # No compile-time grid known for this hash: empty + marked.
        self.assertEqual(trace_data["grid"], [])
        self.assertTrue(trace_data["grid_unknown"])
        self.assertNotIn("stream", trace_data)
        self.assertEqual(
            trace_data["compilation_metadata"],
            {"hash": "deadbeef0123", "name": "kernel_cutlass_mk_abc123"},
        )
        extracted = trace_data["extracted_args"]
        self.assertEqual(extracted["arg0"]["type"], "tensor")
        self.assertEqual(extracted["arg0"]["shape"], [4])
        self.assertEqual(extracted["arg1"]["type"], "int")
        self.assertEqual(extracted["arg1"]["value"], 128)
        self.assertEqual(extracted["flag"]["type"], "bool")

    def test_args_named_after_kernel_parameters(self):
        # Upstream `PreExecuteEvent.arg_names` names only the positional
        # runtime arguments; keyword arguments arrive keyed in `kwargs`.
        trace_data = _trace_launch(arg_names=("mSrc", "n"))
        self.assertEqual(list(trace_data["extracted_args"]), ["mSrc", "n", "flag"])
        self.assertEqual(trace_data["extracted_args"]["n"]["value"], 128)

    def test_extra_args_fall_back_to_positional_names(self):
        trace_data = _trace_launch(arg_names=("mSrc",))
        self.assertEqual(list(trace_data["extracted_args"]), ["mSrc", "arg1", "flag"])

    def test_static_grid_from_compilation(self):
        _trace_compilation(STATIC_GRID)
        trace_data = _trace_launch()
        self.assertEqual(trace_data["grid"], [8, 1, 1])
        self.assertNotIn("grid_unknown", trace_data)

    def test_dynamic_grid_stays_unknown(self):
        _trace_compilation(DYNAMIC_GRID)
        trace_data = _trace_launch()
        self.assertEqual(trace_data["grid"], [])
        self.assertTrue(trace_data["grid_unknown"])

    def test_no_grid_remembered_without_module_hash(self):
        # The kernel name that stands in for a missing hash can cover several
        # modules, so a launch without a hash must not pick up their grids.
        _trace_compilation(STATIC_GRID, module_hash=None)
        self.assertEqual(cutedsl_logging._STATIC_GRID_BY_HASH, {})
        trace_data = _trace_launch(module_hash=None)
        self.assertEqual(trace_data["grid"], [])
        self.assertTrue(trace_data["grid_unknown"])

    def test_packed_path(self):
        trace_data = _trace_launch(args=None, kwargs=None)
        self.assertEqual(trace_data["extracted_args"], {})
        self.assertTrue(trace_data["args_packed"])
        self.assertEqual(trace_data["arg_count"], 2)

    def test_missing_hash_falls_back_to_name(self):
        trace_data = _trace_launch(module_hash=None)
        self.assertEqual(
            trace_data["compilation_metadata"]["hash"], "kernel_cutlass_mk_abc123"
        )

    def test_allowlist_filtering(self):
        with patch.object(
            structured_logging, "_KERNEL_ALLOWLIST_PATTERNS", ["other_*"]
        ):
            self.assertEqual(_trace_launch(), {})
        with patch.object(
            structured_logging, "_KERNEL_ALLOWLIST_PATTERNS", ["kernel_cutlass_*"]
        ):
            self.assertIn("name", _trace_launch())

    def test_extraction_failure_degrades_gracefully(self):
        class BadRepr:
            def __str__(self):
                raise RuntimeError("boom")

            __repr__ = __str__

        with self.assertLogs("tritonparse.cutedsl_logging", level="WARNING") as cm:
            trace_data = _trace_launch(args=(BadRepr(),))
        self.assertEqual(trace_data["extracted_args"], {"extraction_error": "boom"})
        self.assertTrue(
            any("arg extraction failed" in line for line in cm.output),
            cm.output,
        )

    def test_cute_tensor_extraction(self):
        trace_data = _trace_launch(args=(_cute_tensor(),))
        extracted = trace_data["extracted_args"]
        self.assertEqual(extracted["arg0"]["type"], "cute.Tensor")
        self.assertEqual(extracted["arg0"]["shape"], [1024])
        self.assertEqual(extracted["arg0"]["stride"], [1])
        self.assertEqual(extracted["arg0"]["memspace"], "gmem")
        # A hex string, as logged for torch tensors.
        self.assertEqual(extracted["arg0"]["data_ptr"], "0x3039")

    def test_cute_tensor_non_int_data_ptr_dropped(self):
        # The launch schema expects a hex string; a data_ptr that is not an
        # int (e.g. a bound method on some other tensor-like type) is dropped
        # rather than recorded in a form the schema rejects.
        tensor = _cute_tensor()
        tensor.data_ptr = lambda: 12345
        trace_data = _trace_launch(args=(tensor,))
        self.assertNotIn("data_ptr", trace_data["extracted_args"]["arg0"])

    def test_cute_tensor_hostile_attrs_degrade(self):
        class _Tensor:
            @property
            def shape(self):
                raise TypeError("nope")

        _Tensor.__module__ = "cutlass.cute.runtime"
        trace_data = _trace_launch(args=(_Tensor(),))
        extracted = trace_data["extracted_args"]
        self.assertEqual(extracted["arg0"]["type"], "cute.Tensor")
        self.assertNotIn("shape", extracted["arg0"])


class FakeDSLWithLaunch:
    """Minimal stand-in for the CuTeDSL singleton."""

    def __init__(self):
        self.envar = SimpleNamespace(
            keep_ir=False,
            keep_ir_clean=False,
            keep_ptx=False,
            keep_cubin=False,
            keep_sass=False,
            lineinfo=False,
            dump_dir="/user/cwd",
            loc_tracebacks=0,
        )


class _HookEvent(Enum):
    POST_COMPILE = "post_compile"
    PRE_EXECUTE = "pre_execute"


class FakeHooks:
    """Stand-in for the upstream `cutlass.hooks` API (#3697 + #3703)."""

    HookEvent = _HookEvent

    def __init__(self):
        self.registrations = []

    def register_hook(self, event, hook, *, dsl):
        self.registrations.append((event, hook, dsl))

    def remove_hook(self, event, hook, *, dsl):
        self.registrations.remove((event, hook, dsl))

    def emit(self, event, payload):
        for registered_event, hook, dsl in list(self.registrations):
            if registered_event == event and dsl is payload.owner:
                hook(payload)


def _fake_cutlass_module(dsl, hooks=None):
    cutlass_mod = MagicMock(name="cutlass")
    cutlass_mod.cutlass_dsl.CuTeDSL._get_dsl.return_value = dsl
    cutlass_mod.hooks = hooks
    return cutlass_mod


class LaunchSetupTeardownTest(unittest.TestCase):
    def setUp(self):
        cutedsl_logging._reset_for_tests()
        self.addCleanup(cutedsl_logging._reset_for_tests)
        self._env_patcher = patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        os.environ.pop("CUTE_DSL_LINEINFO", None)
        os.environ.pop("CUTE_DSL_DUMP_DIR", None)
        self.addCleanup(self._env_patcher.stop)

    def test_setup_with_launch_registers_both(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing(enable_launch=True))
            self.assertEqual(
                [event for event, _, _ in hooks.registrations],
                [_HookEvent.POST_COMPILE, _HookEvent.PRE_EXECUTE],
            )
            cutedsl_logging.teardown_cutedsl_tracing()
            self.assertEqual(hooks.registrations, [])

    def test_setup_without_launch_skips_launch_hook(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing())
            self.assertEqual(
                [event for event, _, _ in hooks.registrations],
                [_HookEvent.POST_COMPILE],
            )
            cutedsl_logging.teardown_cutedsl_tracing()

    def test_public_hooks_preferred_and_teardown_preserves_user_hooks(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        original_envar = vars(dsl.envar).copy()
        user_hook = MagicMock()
        hooks.register_hook(_HookEvent.POST_COMPILE, user_hook, dsl=dsl)
        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            for _ in range(2):
                self.assertTrue(
                    cutedsl_logging.setup_cutedsl_tracing(enable_launch=True)
                )
                registrations = hooks.registrations.copy()
                self.assertEqual(
                    [event for event, _, _ in registrations],
                    [
                        _HookEvent.POST_COMPILE,
                        _HookEvent.POST_COMPILE,
                        _HookEvent.PRE_EXECUTE,
                    ],
                )
                self.assertTrue(all(owner is dsl for _, _, owner in registrations))
                self.assertTrue(dsl.envar.keep_ir)
                temp_dir = dsl.envar.dump_dir
                self.assertTrue(os.path.isdir(temp_dir))
                self.assertTrue(
                    cutedsl_logging.setup_cutedsl_tracing(enable_launch=True)
                )
                self.assertEqual(hooks.registrations, registrations)
                cutedsl_logging.teardown_cutedsl_tracing()
                self.assertEqual(
                    hooks.registrations, [(_HookEvent.POST_COMPILE, user_hook, dsl)]
                )
                self.assertEqual(vars(dsl.envar), original_envar)
                self.assertFalse(os.path.exists(temp_dir))

    def test_public_hooks_without_launch(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing())
        self.assertEqual(
            [event for event, _, _ in hooks.registrations], [_HookEvent.POST_COMPILE]
        )

    def test_compile_only_public_api_warns_and_keeps_compilation_tracing(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        hooks.HookEvent = SimpleNamespace(POST_COMPILE=_HookEvent.POST_COMPILE)
        with (
            patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}),
            self.assertLogs("tritonparse.cutedsl_logging", level="WARNING") as cm,
        ):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing(enable_launch=True))
        self.assertTrue(
            any("without launches" in line for line in cm.output), cm.output
        )
        self.assertEqual(
            [event for event, _, _ in hooks.registrations], [_HookEvent.POST_COMPILE]
        )

    def test_partial_registration_failure_rolls_back_and_can_retry(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        original_envar = vars(dsl.envar).copy()
        register = hooks.register_hook

        def fail_launch(event, hook, *, dsl):
            if event is _HookEvent.PRE_EXECUTE:
                raise RuntimeError("registration failed")
            register(event, hook, dsl=dsl)

        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            with (
                patch.object(hooks, "register_hook", side_effect=fail_launch),
                self.assertRaisesRegex(RuntimeError, "registration failed"),
            ):
                cutedsl_logging.setup_cutedsl_tracing(enable_launch=True)
            self.assertEqual(hooks.registrations, [])
            self.assertEqual(vars(dsl.envar), original_envar)
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing(enable_launch=True))
            self.assertEqual(len(hooks.registrations), 2)

    def test_teardown_continues_after_one_remover_fails(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        original_envar = vars(dsl.envar).copy()
        attempted = []
        remove = hooks.remove_hook

        def fail_launch(event, hook, *, dsl):
            attempted.append(event)
            if event is _HookEvent.PRE_EXECUTE:
                raise RuntimeError("removal failed")
            remove(event, hook, dsl=dsl)

        # Setup captures the remover, so changing it later must not affect teardown.
        with (
            patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}),
            patch.object(hooks, "remove_hook", side_effect=fail_launch),
        ):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing(enable_launch=True))
        cutedsl_logging.teardown_cutedsl_tracing()
        self.assertEqual(attempted, [_HookEvent.PRE_EXECUTE, _HookEvent.POST_COMPILE])
        self.assertEqual(vars(dsl.envar), original_envar)
        self.assertEqual(
            [event for event, _, _ in hooks.registrations], [_HookEvent.PRE_EXECUTE]
        )
        cutedsl_logging.teardown_cutedsl_tracing()
        self.assertEqual(len(attempted), 2)


class CutedslLaunchEndToEndTest(unittest.TestCase):
    """Hooks -> NDJSON file -> schema validation -> hash grouping."""

    def setUp(self):
        cutedsl_logging._reset_for_tests()
        self.addCleanup(cutedsl_logging._reset_for_tests)
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def tearDown(self):
        with patch.dict(sys.modules, {"triton": MagicMock(), "cutlass": MagicMock()}):
            structured_logging.clear_logging_config()

    def _emit_events(self, grid_dims=None, **launch_overrides):
        structured_logging.init_basic(trace_folder=self.tmp_dir)
        # Minimal compilation event so the launch has something to group with.
        _trace_compilation(grid_dims)
        _trace_launch(arg_names=("mSrc", "n"), **launch_overrides)
        return self._read_events()

    def _read_events(self):
        structured_logging.TRITON_TRACE_HANDLER._ensure_stream_closed()
        paths = sorted(
            glob.glob(os.path.join(self.tmp_dir, "dedicated_log_triton_trace_*"))
        )
        self.assertEqual(len(paths), 1)
        with open(paths[0]) as f:
            return [json.loads(line) for line in f.read().split("\n") if line.strip()]

    def test_public_hooks_emit_valid_linked_events(self):
        dsl, hooks = FakeDSLWithLaunch(), FakeHooks()
        structured_logging.init_basic(trace_folder=self.tmp_dir)
        raw_mlir = "module {}\n"
        clean_mlir = HOST_MLIR_TEMPLATE.replace("GRID_DIMS", STATIC_GRID)
        with patch.dict(sys.modules, {"cutlass": _fake_cutlass_module(dsl, hooks)}):
            self.assertTrue(cutedsl_logging.setup_cutedsl_tracing(enable_launch=True))
            hooks.emit(
                _HookEvent.POST_COMPILE,
                SimpleNamespace(
                    owner=dsl,
                    function_name="kernel_cutlass_mk_abc123",
                    module_hash="deadbeef0123",
                    cache_hit=False,
                    artifacts=SimpleNamespace(
                        MLIR=clean_mlir, MLIR_RAW=raw_mlir, PTX=None, CUBIN=None
                    ),
                    compile_options=None,
                    func_body=None,
                    kernel_info={"mkernel": {}},
                    future_field="ignored",
                ),
            )
            hooks.emit(
                _HookEvent.PRE_EXECUTE,
                SimpleNamespace(
                    owner=dsl,
                    function_name="kernel_cutlass_mk_abc123",
                    module_hash="deadbeef0123",
                    kernel_info={"mkernel": {}},
                    args=(_cute_tensor(), 128),
                    kwargs={"flag": True},
                    arg_names=("mSrc", "n"),
                    exe_args=None,
                    executor=None,
                    future_field="ignored",
                ),
            )
        events = self._read_events()
        self.assertEqual(
            [event["event_type"] for event in events], ["compilation", "launch"]
        )
        for event in events:
            ok, errors = validate_record(event)
            self.assertTrue(ok, errors)
        compilation, launch = events
        self.assertEqual(
            compilation["payload"]["file_content"],
            {"kernel.mlir": raw_mlir, "kernel.clean_mlir": clean_mlir},
        )
        self.assertEqual(
            launch["compilation_metadata"]["hash"],
            compilation["payload"]["metadata"]["hash"],
        )
        self.assertEqual(launch["grid"], [8, 1, 1])
        self.assertNotIn("grid_unknown", launch)
        self.assertEqual(list(launch["extracted_args"]), ["mSrc", "n", "flag"])
        self.assertEqual(launch["extracted_args"]["mSrc"]["shape"], [1024])
        self.assertEqual(launch["extracted_args"]["n"]["value"], 128)

    def test_emitted_launch_validates_and_groups(self):
        events = self._emit_events()
        self.assertEqual(len(events), 2)
        for event in events:
            ok, errors = validate_record(event)
            self.assertTrue(ok, errors)
        launch = next(e for e in events if e["event_type"] == "launch")
        self.assertEqual(launch["name"], "kernel_cutlass_mk_abc123")
        self.assertEqual(launch["grid"], [])
        self.assertEqual(list(launch["extracted_args"]), ["mSrc", "n", "flag"])

    def test_emitted_static_grid_launch_validates(self):
        events = self._emit_events(STATIC_GRID)
        for event in events:
            ok, errors = validate_record(event)
            self.assertTrue(ok, errors)
        compilation = next(e for e in events if e["event_type"] == "compilation")
        self.assertEqual(
            compilation["payload"]["metadata"]["launch_configs"][0]["grid"], [8, 1, 1]
        )
        launch = next(e for e in events if e["event_type"] == "launch")
        self.assertEqual(launch["grid"], [8, 1, 1])
        self.assertNotIn("grid_unknown", launch)

    def test_emitted_cute_tensor_launch_validates(self):
        events = self._emit_events(args=(_cute_tensor(), 128))
        launch = next(e for e in events if e["event_type"] == "launch")
        ok, errors = validate_record(launch)
        self.assertTrue(ok, errors)
        self.assertEqual(launch["extracted_args"]["mSrc"]["type"], "cute.Tensor")

    def test_prescan_groups_cutedsl_events_by_hash(self):
        events = self._emit_events()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson", delete=False) as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            path = f.name
        self.addCleanup(os.unlink, path)
        compilation_hashes, first_launch_by_hash = _prescan_for_fake_compilations(path)
        self.assertIn("deadbeef0123", compilation_hashes)
        self.assertIn("deadbeef0123", first_launch_by_hash)


if __name__ == "__main__":
    unittest.main()
