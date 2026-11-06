# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Compatibility helpers for Triton's two runtime-hook shapes.

Triton channels supported by tritonparse expose runtime hooks in two forms:

* legacy single-slot hooks, where ``knobs.runtime.<name>`` holds one callable
  or ``None`` (fbcode 3.5/3.8/3.9 channels);
* hook chains, where the same attribute holds a ``HookChain`` container with
  ``add``/``remove`` and a ``calls`` list, and subscribers register without
  replacing each other (recent upstream 3.9).

The version string cannot distinguish the two forms, so detection is by the
shape of the hook object itself. This module is stdlib-only: it is imported
on Triton's import path and must not pull in Torch or Triton.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def is_hook_chain(hook: Any) -> bool:
    """Return whether ``hook`` is a Triton ``HookChain``-style container."""
    return (
        callable(getattr(hook, "add", None))
        and callable(getattr(hook, "remove", None))
        and isinstance(getattr(hook, "calls", None), (list, tuple))
    )


def hook_is_set(hook: Any) -> bool:
    """Return whether a runtime hook carries any registered callable.

    An empty hook chain is not set even though the container object itself
    is truthy; a legacy slot is set when it holds a callable.
    """
    if hook is None:
        return False
    if is_hook_chain(hook):
        return len(hook.calls) > 0
    return bool(hook)


def install_runtime_hook(owner: Any, name: str, hook: Any) -> Callable[[], None]:
    """Register ``hook`` on ``owner.<name>`` and return a restore callable.

    On a hook chain the hook is added to the chain, leaving other
    subscribers and the chain object itself untouched. On a legacy slot
    the hook replaces the previous value. The returned callable undoes
    exactly this registration: it removes the hook from the chain, or
    restores the previous slot value, and only when the hook is still
    the one installed here, so a later subscriber is never clobbered.
    """
    current = getattr(owner, name)
    if is_hook_chain(current):
        current.add(hook)

        def restore_chain() -> None:
            if hook in current.calls:
                current.remove(hook)

        return restore_chain

    previous = current
    setattr(owner, name, hook)

    def restore_slot() -> None:
        if getattr(owner, name, None) is hook:
            setattr(owner, name, previous)

    return restore_slot
