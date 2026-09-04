# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Prototype tests: init() must propagate launch tracing to torch inductor.

torch._inductor gates its JIT post-compile hook simulation on
``torch._inductor.config.run_jit_post_compile_hook`` (read from the env var
only once, at torch import time). Setting tritonparse's own module global is
not enough; init() has to set the torch config attribute (current process)
and os.environ (future subprocess/spawn workers) too.

torch/triton are stubbed: neither is available to CPU tests.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import tritonparse.structured_logging as sl

_HOOK_ENV = "TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK"


def _install_torch_stub(*, has_attr: bool = True):
    """Shadow any real torch with a stub inductor config module."""
    torch_mod = types.ModuleType("torch")
    inductor_pkg = types.ModuleType("torch._inductor")
    config_mod = types.ModuleType("torch._inductor.config")
    if has_attr:
        config_mod.run_jit_post_compile_hook = False
    inductor_pkg.config = config_mod
    torch_mod._inductor = inductor_pkg
    sys.modules["torch"] = torch_mod
    sys.modules["torch._inductor"] = inductor_pkg
    sys.modules["torch._inductor.config"] = config_mod
    return config_mod


def _install_triton_stub():
    knobs = SimpleNamespace(
        compilation=SimpleNamespace(listener=None),
        autotuning=SimpleNamespace(listener=None),
        runtime=SimpleNamespace(
            jit_cache_hook=None,
            jit_post_compile_hook=None,
            launch_enter_hook=None,
        ),
    )
    triton_mod = types.ModuleType("triton")
    triton_mod.knobs = knobs
    sys.modules["triton"] = triton_mod
    return knobs


class InductorHookConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {
            name: sys.modules.get(name)
            for name in ("torch", "torch._inductor", "torch._inductor.config", "triton")
        }
        for name in ("torch", "torch._inductor", "torch._inductor.config", "triton"):
            sys.modules.pop(name, None)
        self._saved_env = os.environ.get(_HOOK_ENV)
        os.environ.pop(_HOOK_ENV, None)
        # Snapshot tritonparse globals mutated by init().
        self._saved_globals = {
            name: getattr(sl, name)
            for name in (
                "TRITON_TRACE_LAUNCH",
                "TRITON_TRACE_LAUNCH_WITHIN_PROFILING",
                "TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK",
                "triton_trace_folder",
            )
        }
        self.trace_dir = tempfile.mkdtemp(prefix="tritonparse_inductor_hook_")

    def tearDown(self):
        try:
            sl.clear_logging_config()
        except Exception:
            pass
        for name in ("torch", "torch._inductor", "torch._inductor.config", "triton"):
            sys.modules.pop(name, None)
        for name, mod in self._saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
        if self._saved_env is None:
            os.environ.pop(_HOOK_ENV, None)
        else:
            os.environ[_HOOK_ENV] = self._saved_env
        for name, value in self._saved_globals.items():
            setattr(sl, name, value)
        sl._trace_launch_enabled = False
        shutil.rmtree(self.trace_dir, ignore_errors=True)

    def test_init_enable_trace_launch_reaches_inductor(self):
        config_mod = _install_torch_stub()
        knobs = _install_triton_stub()
        sl.init(self.trace_dir, enable_trace_launch=True)
        # Current process: inductor's compile-time gate now passes.
        self.assertTrue(config_mod.run_jit_post_compile_hook)
        self.assertIsNotNone(knobs.runtime.jit_post_compile_hook)
        self.assertIsNotNone(knobs.runtime.launch_enter_hook)
        # Future workers: fresh interpreters re-read the env var at import.
        self.assertEqual(os.environ.get(_HOOK_ENV), "1")

    def test_init_without_launch_leaves_inductor_alone(self):
        config_mod = _install_torch_stub()
        _install_triton_stub()
        sl.init(self.trace_dir)
        self.assertFalse(config_mod.run_jit_post_compile_hook)
        self.assertNotIn(_HOOK_ENV, os.environ)

    def test_helper_without_torch_still_sets_env(self):
        # torch not importable at all: env part must still apply, no raise.
        _install_triton_stub()
        sys.modules["torch"] = None
        sl._enable_inductor_jit_post_compile_hook()
        self.assertEqual(os.environ.get(_HOOK_ENV), "1")

    def test_helper_old_torch_without_attr_is_noop(self):
        _install_torch_stub(has_attr=False)
        sl._enable_inductor_jit_post_compile_hook()
        self.assertEqual(os.environ.get(_HOOK_ENV), "1")


if __name__ == "__main__":
    unittest.main()
