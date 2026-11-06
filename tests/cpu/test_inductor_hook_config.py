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
from tritonparse.hook_compat import install_runtime_hook

_HOOK_ENV = "TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK"
_STUBBED_MODULES = (
    "torch",
    "torch._inductor",
    "torch._inductor.config",
    "triton",
    "triton.runtime",
    "triton.runtime.jit",
)


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


class _FakeHookChain:
    """Mirror of upstream Triton's HookChain: always truthy, even when empty."""

    def __init__(self) -> None:
        self.calls = []

    def add(self, func) -> None:
        if func not in self.calls:
            self.calls.append(func)

    def remove(self, func) -> None:
        if func in self.calls:
            self.calls.remove(func)

    def __call__(self, *args, **kwargs) -> None:
        for call in list(self.calls):
            call(*args, **kwargs)


def _install_triton_stub(*, hook_chain: bool = False):
    knobs = SimpleNamespace(
        compilation=SimpleNamespace(listener=None),
        autotuning=SimpleNamespace(listener=None),
        runtime=SimpleNamespace(
            jit_cache_hook=None,
            jit_post_compile_hook=None,
            launch_enter_hook=_FakeHookChain() if hook_chain else None,
        ),
    )
    triton_mod = types.ModuleType("triton")
    triton_mod.knobs = knobs
    runtime_mod = types.ModuleType("triton.runtime")
    jit_mod = types.ModuleType("triton.runtime.jit")
    jit_mod.JITFunction = SimpleNamespace(_call_hook=lambda *args, **kwargs: None)
    runtime_mod.jit = jit_mod
    triton_mod.runtime = runtime_mod
    sys.modules["triton"] = triton_mod
    sys.modules["triton.runtime"] = runtime_mod
    sys.modules["triton.runtime.jit"] = jit_mod
    return knobs


class InductorHookConfigTest(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
        for name in _STUBBED_MODULES:
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
        sl.TRITON_TRACE_LAUNCH = False
        sl.TRITON_TRACE_LAUNCH_WITHIN_PROFILING = False
        sl.TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK = False
        sl.triton_trace_folder = None
        sl._trace_launch_enabled = False
        self.trace_dir = tempfile.mkdtemp(prefix="tritonparse_inductor_hook_")

    def tearDown(self):
        try:
            sl.clear_logging_config()
        except Exception:
            pass
        for name in _STUBBED_MODULES:
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

    def test_init_preserves_foreign_hook_chain_subscribers(self):
        _install_torch_stub()
        knobs = _install_triton_stub(hook_chain=True)
        chain = knobs.runtime.launch_enter_hook
        sentinel = lambda metadata: None  # noqa: E731
        chain.add(sentinel)
        sl.init(self.trace_dir, enable_trace_launch=True)
        # The chain object is untouched, the foreign subscriber stays, and
        # tritonparse's launch hook joins it instead of replacing it.
        self.assertIs(knobs.runtime.launch_enter_hook, chain)
        self.assertIn(sentinel, chain.calls)
        self.assertTrue(
            any(isinstance(call, sl.LaunchHookImpl) for call in chain.calls)
        )
        sl.clear_logging_config()
        self.assertIs(knobs.runtime.launch_enter_hook, chain)
        self.assertEqual(chain.calls, [sentinel])
        self.assertIsNone(knobs.runtime.jit_post_compile_hook)

    def test_init_restores_previous_legacy_slot_hooks(self):
        _install_torch_stub()
        knobs = _install_triton_stub()
        previous_launch = lambda metadata: None  # noqa: E731
        previous_jit = lambda **kwargs: None  # noqa: E731
        previous_listener = lambda *args, **kwargs: None  # noqa: E731
        knobs.runtime.launch_enter_hook = previous_launch
        knobs.runtime.jit_post_compile_hook = previous_jit
        knobs.compilation.listener = previous_listener
        sl.init(self.trace_dir, enable_trace_launch=True)
        self.assertIsInstance(knobs.runtime.launch_enter_hook, sl.LaunchHookImpl)
        self.assertIsInstance(knobs.runtime.jit_post_compile_hook, sl.JITHookImpl)
        self.assertIs(knobs.compilation.listener, sl.maybe_trace_triton)
        sl.clear_logging_config()
        self.assertIs(knobs.runtime.launch_enter_hook, previous_launch)
        self.assertIs(knobs.runtime.jit_post_compile_hook, previous_jit)
        self.assertIs(knobs.compilation.listener, previous_listener)

    def test_restore_survives_external_removal_from_chain(self):
        # A chain whose remove() does not guard membership (unlike Triton's
        # own HookChain): restore must still be a no-op after another tool
        # removed our hook first, and must be safe to call twice.
        class UnguardedChain:
            def __init__(self):
                self.calls = []

            def add(self, func):
                self.calls.append(func)

            def remove(self, func):
                self.calls.remove(func)

        chain = UnguardedChain()
        runtime = SimpleNamespace(launch_enter_hook=chain)
        hook = lambda metadata: None  # noqa: E731
        restore = install_runtime_hook(runtime, "launch_enter_hook", hook)
        self.assertEqual(chain.calls, [hook])
        chain.calls.clear()
        restore()
        restore()
        self.assertEqual(chain.calls, [])

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
