# Copyright (c) Meta Platforms, Inc. and affiliates.
"""CuTeDSL compilation/launch tracing for tritonparse (no fbcode dependencies).

This module mirrors ``structured_logging.py``'s Triton hooks for NVIDIA CuTeDSL
(``@cute.jit``) kernels. It registers listeners on the CuTeDSL singleton
(``cutlass.cutlass_dsl.CuTeDSL._get_dsl()``) and emits NDJSON events in the same
shape as the Triton ones, so the parse/validation/viewer pipeline works
unchanged:

- compilation events carry ``metadata`` (with ``backend_name="cutedsl"`` and
  the kernel launch configurations found in the MLIR), ``file_content``
  (``kernel.mlir``/``kernel.clean_mlir``/``kernel.ptx``, plus ``kernel.sass``
  when SASS dumping is enabled) and ``python_source``;
- launch events carry ``name``/``grid``/``compilation_metadata``/
  ``extracted_args`` at the top level.

Uses the upstream ``cutlass.hooks`` API: ``HookEvent.POST_COMPILE``
(NVIDIA/cutlass#3697) for compilation events and, when the build provides
it, ``HookEvent.PRE_EXECUTE`` (NVIDIA/cutlass#3703) for launch events.
Builds without ``POST_COMPILE`` are detected and skipped with a warning;
builds with only ``POST_COMPILE`` still trace compilations, without
launches.

Enabling tracing also makes CuTeDSL keep its compiler artifacts with Python
source locations. This sets attributes of the singleton's ``envar`` object,
which CuTeDSL reads on every compile, so it works no matter when cutlass was
imported (the ``CUTE_DSL_*`` environment variables themselves are cached on
first read). Teardown restores the previous values. Unless the user already
keeps artifacts or chose ``CUTE_DSL_DUMP_DIR``, the files go to a private
temporary directory instead of the working directory. Line info stays off when
``CUTE_DSL_LOC_TRACEBACKS`` is set, because combining the two segfaults
(verified on CuTeDSL 4.6.2).
"""

import atexit
import inspect
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import tritonparse.structured_logging as _sl
from tritonparse._json_compat import dumps

log = logging.getLogger(__name__)

CUTEDSL_BACKEND_NAME = "cutedsl"

# ``envar`` switches tracing turns on, in the order of the CUTE_DSL_KEEP tokens
# that set them: raw MLIR before any pass ("ir-debug"), clean MLIR after
# canonicalize+cse ("ir"), PTX and CUBIN. Both MLIR dumps carry source
# locations. SASS is not kept by CuTeDSL: like for Triton, it is derived from
# the CUBIN with nvdisasm when SASS dumping is enabled.
CUTEDSL_KEEP_ATTRS = ("keep_ir", "keep_ir_clean", "keep_ptx", "keep_cubin")
CUTEDSL_KEEP_TOKENS = ("ir-debug", "ir", "ptx", "cubin")

# Owning DSL singleton once setup succeeds; None otherwise. Used by teardown.
_CUTEDSL_DSL_REF: Any = None
# Exact registrations made by setup, including event adapters for the hooks API.
_CUTEDSL_UNREGISTER: List[Callable[[], None]] = []
# The ``envar`` object and the original values of the attributes setup changed.
_CUTEDSL_ENVAR_RESTORE: Optional[Tuple[Any, Dict[str, Any]]] = None
# Temporary dump directory created by setup; removed again on teardown.
_CUTEDSL_TEMP_DUMP_DIR: Optional[str] = None
# Grid per module hash for kernels whose MLIR fixes it at compile time.
# CuTeDSL reports no grid at launch time, so launch events look it up here.
_STATIC_GRID_BY_HASH: Dict[str, List[int]] = {}


def _configure_cutedsl_envar(dsl: Any) -> None:
    """Switch on the CuTeDSL settings tracing needs, remembering the originals."""
    global _CUTEDSL_ENVAR_RESTORE, _CUTEDSL_TEMP_DUMP_DIR
    envar = getattr(dsl, "envar", None)
    required = CUTEDSL_KEEP_ATTRS + ("lineinfo", "dump_dir")
    if envar is None or not all(hasattr(envar, attr) for attr in required):
        log.warning(
            "[tritonparse] Cannot configure CuTeDSL artifact keeping on this "
            "cutlass build; set CUTE_DSL_KEEP=%s and CUTE_DSL_LINEINFO=1 before "
            "importing cutlass to capture IR and source locations.",
            ",".join(CUTEDSL_KEEP_TOKENS),
        )
        return
    user_keeps = any(
        getattr(envar, attr, False) for attr in CUTEDSL_KEEP_ATTRS + ("keep_sass",)
    )
    updates: Dict[str, Any] = {
        attr: True for attr in CUTEDSL_KEEP_ATTRS if not getattr(envar, attr)
    }
    # An explicit CUTE_DSL_LINEINFO (even "0") is the user's choice.
    if not envar.lineinfo and "CUTE_DSL_LINEINFO" not in os.environ:
        if getattr(envar, "loc_tracebacks", 0):
            log.warning(
                "[tritonparse] CUTE_DSL_LOC_TRACEBACKS is set, so CuTeDSL line "
                "info stays off (the two together crash CuTeDSL); IR-to-source "
                "mapping will be unavailable."
            )
        else:
            updates["lineinfo"] = True
    if not user_keeps and "CUTE_DSL_DUMP_DIR" not in os.environ:
        temp_dir = tempfile.mkdtemp(prefix="tritonparse_cutedsl_")
        atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
        _CUTEDSL_TEMP_DUMP_DIR = temp_dir
        updates["dump_dir"] = temp_dir
    _CUTEDSL_ENVAR_RESTORE = (envar, {attr: getattr(envar, attr) for attr in updates})
    for attr, value in updates.items():
        setattr(envar, attr, value)


def _restore_cutedsl_envar() -> None:
    """Undo ``_configure_cutedsl_envar`` and remove its temporary directory."""
    global _CUTEDSL_ENVAR_RESTORE, _CUTEDSL_TEMP_DUMP_DIR
    restore, _CUTEDSL_ENVAR_RESTORE = _CUTEDSL_ENVAR_RESTORE, None
    if restore is not None:
        envar, originals = restore
        for attr, value in originals.items():
            setattr(envar, attr, value)
    temp_dir, _CUTEDSL_TEMP_DUMP_DIR = _CUTEDSL_TEMP_DUMP_DIR, None
    if temp_dir is not None:
        shutil.rmtree(temp_dir, ignore_errors=True)


# The host function of a CuTeDSL module configures and launches its kernels
# with ops that each print on one line, e.g.
#   %c128_i32 = arith.constant 128 : i32
#   %4 = cuda.launch_cfg.create<...> (blockDim = (%c128_i32, %c1_i32, %c1_i32),
#   dynamicSmemBytes = %1, gridDim = (%c8_i32, %c1_i32, %c1_i32), stream = %3)
#   cuda.launch_cfg.cluster_dim[%4] (%c2_i32, %c1_i32, %c1_i32)
#   cuda.launch_ex @kernels::@kernel_name<%4> (...)
# SSA names are only unique within a function, so they are resolved per function.
_SSA_NAME = r"%[\w.$-]+(?:#\d+)?"
_FUNC_START_RE = re.compile(r"^\s*(?:func\.func|llvm\.func|gpu\.func|cuda\.kernel)\b")
_INT_CONSTANT_RE = re.compile(
    rf"({_SSA_NAME}) = arith\.constant (-?\d+) : (?:i\d+|index)\b"
)
_LAUNCH_CFG_CREATE_RE = re.compile(rf"({_SSA_NAME}) = cuda\.launch_cfg\.create\b")
_BLOCK_DIM_RE = re.compile(r"\bblockDim = \(([^)]*)\)")
_GRID_DIM_RE = re.compile(r"\bgridDim = \(([^)]*)\)")
_CLUSTER_DIM_RE = re.compile(
    rf"cuda\.launch_cfg\.cluster_dim\[({_SSA_NAME})\] \(([^)]*)\)"
)
_LAUNCH_RE = re.compile(
    rf"cuda\.launch(?:_ex)? @[\w.$-]+::@\"?([\w.$-]+)\"?<({_SSA_NAME})>"
)


def _resolve_dims(operands: str, constants: Dict[str, int]) -> List[Optional[int]]:
    """Map launch dimension operands to ints; None where not a constant."""
    dims: List[Optional[int]] = []
    for operand in operands.split(","):
        token = operand.strip().split(" ")[0]
        if re.fullmatch(r"-?\d+", token):
            dims.append(int(token))
        else:
            dims.append(constants.get(token))
    return dims


def parse_cutedsl_launch_configs(mlir: str) -> List[Dict[str, Any]]:
    """Extract the kernel launch configurations from CuTeDSL MLIR.

    Returns one entry per launch op, in program order, with the ``kernel``
    name, ``block`` and ``grid`` dimensions and, when set, ``cluster``
    dimensions. A dimension that is not a compile-time constant (e.g. one
    derived from a dynamic tensor shape) is None.
    """
    configs: List[Dict[str, Any]] = []
    constants: Dict[str, int] = {}
    launch_cfgs: Dict[str, Dict[str, Any]] = {}
    for line in mlir.splitlines():
        if _FUNC_START_RE.match(line):
            constants.clear()
            launch_cfgs.clear()
            continue
        match = _INT_CONSTANT_RE.search(line)
        if match:
            constants[match.group(1)] = int(match.group(2))
            continue
        match = _LAUNCH_CFG_CREATE_RE.search(line)
        if match:
            cfg: Dict[str, Any] = {}
            for key, dims_re in (("block", _BLOCK_DIM_RE), ("grid", _GRID_DIM_RE)):
                dims = dims_re.search(line)
                cfg[key] = _resolve_dims(dims.group(1), constants) if dims else None
            launch_cfgs[match.group(1)] = cfg
            continue
        match = _CLUSTER_DIM_RE.search(line)
        if match:
            if match.group(1) in launch_cfgs:
                launch_cfgs[match.group(1)]["cluster"] = _resolve_dims(
                    match.group(2), constants
                )
            continue
        match = _LAUNCH_RE.search(line)
        if match and match.group(2) in launch_cfgs:
            configs.append({"kernel": match.group(1), **launch_cfgs[match.group(2)]})
    return configs


def _is_static(dims: Optional[List[Optional[int]]]) -> bool:
    return bool(dims) and all(isinstance(dim, int) for dim in dims)


def _record_launch_configs(
    metadata: Dict[str, Any], mlir: Optional[str], module_hash: Optional[str]
) -> None:
    """Add the MLIR's launch configurations to the compilation metadata.

    Also sets ``cluster_dims`` (as Triton does) when all launches agree on a
    static cluster, and remembers the grid for launch events when the module
    has a hash and launches a single kernel with a static grid.
    """
    if mlir is None:
        return
    configs = parse_cutedsl_launch_configs(mlir)
    # Only a module hash identifies a single module. The kernel name that
    # stands in for a missing hash can cover several, so it keys no grid.
    if module_hash is not None and len(configs) == 1 and _is_static(configs[0]["grid"]):
        _STATIC_GRID_BY_HASH[module_hash] = configs[0]["grid"]
    if not configs:
        return
    metadata["launch_configs"] = configs
    # A launch without a cluster_dim op runs with the default 1x1x1 cluster.
    clusters = {tuple(config.get("cluster") or (1, 1, 1)) for config in configs}
    if len(clusters) == 1:
        cluster = list(clusters.pop())
        if _is_static(cluster):
            metadata["cluster_dims"] = cluster


def _trace_hash(module_hash: Optional[str], function_name: str) -> str:
    # Schema requires a hash; fall back to the (mangled) kernel name on the
    # paths where CuTeDSL reports none.
    return module_hash if module_hash is not None else function_name


def _artifact_content(value: Any, kind: str) -> Optional[str]:
    """Normalize a JitFunctionArtifacts field to file_content-ready text.

    Returns None when there is nothing usable (missing, binary, or a dangling
    path). CuTeDSL fills these fields with file *contents* when the KEEP files
    exist, but leaves the raw path string otherwise, so paths are resolved
    defensively instead of being recorded as code.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        log.debug(
            "[tritonparse] Skipping binary CuTeDSL %s artifact (%d bytes); "
            "file_content only carries text.",
            kind,
            len(value),
        )
        return None
    if not isinstance(value, str):
        return None
    if "\n" in value:
        return value
    if value and os.path.exists(value):
        try:
            with open(value, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as e:
            log.debug(
                "[tritonparse] Cannot read CuTeDSL %s file %s: %s", kind, value, e
            )
            return None
    return None


def _derive_from_cubin(trace_data: Dict[str, Any], cubin: bytes) -> None:
    """Add the enabled artifacts derived from the CUBIN (e.g. SASS).

    Opt-in exactly like the Triton path (``init(enable_sass_dump=True)`` or
    ``TRITONPARSE_DERIVED_ARTIFACTS``). CuTeDSL passes the CUBIN as bytes, so
    the tools run on a temporary copy. Failures leave a placeholder and a
    deduplicated warning instead of breaking the compile being traced.
    """
    from tritonparse.backend import get_backend_registry
    from tritonparse.shared_vars import get_enabled_derived_artifacts

    enabled = get_enabled_derived_artifacts()
    if enabled is not None and not enabled:
        return
    adapter = get_backend_registry().resolve_from_backend_name(CUTEDSL_BACKEND_NAME)
    for info in adapter.list_applicable_derived_artifacts(enabled):
        target_stage = adapter.get_stage_by_name(info.target_stage_name)
        if info.source_stage_name != "cubin" or target_stage is None:
            continue
        derived_filename = f"kernel{target_stage.extension}"
        try:
            with tempfile.TemporaryDirectory(prefix="tritonparse_cutedsl_") as tmp:
                cubin_path = os.path.join(tmp, "kernel.cubin")
                with open(cubin_path, "wb") as f:
                    f.write(cubin)
                content = info.derive_func(cubin_path)
            if content is not None:
                trace_data["file_content"][derived_filename] = content
        except Exception as e:
            _sl._record_derived_artifact_failure(
                trace_data,
                derived_filename,
                adapter.adapter_name,
                info.target_stage_name,
                info.tool_name,
                e,
            )


def extract_cutedsl_python_source(
    trace_data: Dict[str, Any], func_body: Optional[Callable[..., Any]]
) -> None:
    """Record the `@cute.jit` function source in the compilation payload.

    Output shape matches ``extract_python_source_info`` (function-only by
    default, whole file when ``init(enable_full_python_source=True)`` or
    ``TRITON_FULL_PYTHON_SOURCE=1``), so downstream source mapping works
    unchanged. Failures are non-fatal: the key is simply omitted.
    """
    if func_body is None:
        return
    try:
        python_source_file = inspect.getfile(func_body)
        source_lines, function_start_line = inspect.getsourcelines(func_body)
    except (OSError, TypeError) as e:
        log.debug("[tritonparse] Cannot extract CuTeDSL python source: %s", e)
        return
    function_end_line = function_start_line + len(source_lines) - 1

    if _sl.is_full_python_source_enabled():
        try:
            file_size = os.path.getsize(python_source_file)
        except OSError:
            file_size = _sl.TRITON_MAX_SOURCE_SIZE + 1
        if file_size <= _sl.TRITON_MAX_SOURCE_SIZE:
            try:
                with open(python_source_file, "r", encoding="utf-8") as f:
                    file_content = f.read()
                trace_data["python_source"] = {
                    "file_path": python_source_file,
                    "start_line": 1,
                    "end_line": len(file_content.split("\n")),
                    "code": file_content,
                    "function_start_line": function_start_line,
                    "function_end_line": function_end_line,
                }
                return
            except (OSError, UnicodeDecodeError) as e:
                log.debug(
                    "[tritonparse] Full source read failed for %s: %s",
                    python_source_file,
                    e,
                )

    trace_data["python_source"] = {
        "file_path": python_source_file,
        "start_line": function_start_line,
        "end_line": function_end_line,
        "code": "".join(source_lines),
    }


def maybe_trace_cutedsl(
    owner: Any,
    *,
    function_name: str,
    module_hash: Optional[str],
    cache_hit: bool,
    artifacts: Any,
    compile_options: Any,
    func_body: Optional[Callable[..., Any]],
    kernel_info: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Trace a CuTeDSL compilation event. See module docstring for the shape.

    Called with the fields of a ``cutlass.hooks`` ``PostCompileEvent``.
    Returns the collected trace data even when tracing is disabled,
    mirroring ``maybe_trace_triton``.
    """
    if _sl._KERNEL_ALLOWLIST_PATTERNS is not None:
        if not _sl.should_trace_kernel(function_name, _sl._KERNEL_ALLOWLIST_PATTERNS):
            return {}

    trace_hash = _trace_hash(module_hash, function_name)
    trace_data: Dict[str, Any] = defaultdict(dict)
    metadata = trace_data["metadata"]
    metadata["backend_name"] = CUTEDSL_BACKEND_NAME
    metadata["hash"] = trace_hash
    metadata["name"] = function_name
    metadata["cache_hit"] = bool(cache_hit)
    gpu_arch = getattr(compile_options, "gpu_arch", None)
    if gpu_arch:
        metadata["arch"] = gpu_arch
    if kernel_info:
        # Only the kernel names: values hold CUDA enum-keyed attribute dicts
        # that are not JSON-serializable as-is.
        metadata["kernel_names"] = sorted(kernel_info.keys())

    file_content: Dict[str, str] = trace_data["file_content"]
    if artifacts is not None:
        mlir = _artifact_content(getattr(artifacts, "MLIR", None), "MLIR")
        # Upstream CUTLASS keeps MLIR's original meaning (clean when kept,
        # otherwise raw), and exposes the raw dump separately as MLIR_RAW.
        clean_mlir = mlir
        raw_mlir = _artifact_content(getattr(artifacts, "MLIR_RAW", None), "raw MLIR")
        if raw_mlir is not None:
            mlir = raw_mlir
        if mlir is not None:
            file_content["kernel.mlir"] = mlir
        # Without a raw dump, CuTeDSL reports the clean IR as MLIR too.
        if clean_mlir is not None and clean_mlir != mlir:
            file_content["kernel.clean_mlir"] = clean_mlir
        ptx = _artifact_content(getattr(artifacts, "PTX", None), "PTX")
        if ptx is not None:
            file_content["kernel.ptx"] = ptx
        cubin = getattr(artifacts, "CUBIN", None)
        if isinstance(cubin, bytes):
            metadata["has_cubin"] = True
            _derive_from_cubin(trace_data, cubin)
        # Canonicalization folds more launch dimensions into constants.
        _record_launch_configs(metadata, clean_mlir or mlir, module_hash)
    if not file_content:
        del trace_data["file_content"]

    extract_cutedsl_python_source(trace_data, func_body)

    _sl.trace_structured_triton(
        "compilation",
        payload_fn=lambda: dumps(_sl.convert(trace_data)),
    )
    return trace_data


def maybe_trace_cutedsl_launch(
    owner: Any,
    *,
    function_name: str,
    module_hash: Optional[str],
    kernel_info: Optional[Dict[str, Any]],
    args: Optional[tuple],
    kwargs: Optional[Dict[str, Any]],
    exe_args: Optional[List[Any]],
    executor: Any,
    arg_names: Sequence[str] = (),
) -> Dict[str, Any]:
    """Trace a CuTeDSL launch event. See module docstring for the shape.

    Called with the fields of a ``cutlass.hooks`` ``PreExecuteEvent``.
    CuTeDSL reports no grid at launch time, so
    ``grid`` comes from the launch configuration parsed at compile time when
    the MLIR fixes it and CuTeDSL reports a module hash, and is ``[]`` with a
    ``grid_unknown`` marker otherwise (the launch schema requires ``grid``).
    Arguments are named after the kernel parameters in ``arg_names``, falling
    back to ``arg{i}``. CuTeDSL passes ``args``/``kwargs`` for every launch it
    reports; ``exe_args`` is None on a TVM FFI call to a compiled function.
    Given ``args=None``, the event records no argument values, only
    ``args_packed`` and the number of ``exe_args`` as ``arg_count``. Returns
    the collected trace data.
    """
    if _sl._KERNEL_ALLOWLIST_PATTERNS is not None:
        if not _sl.should_trace_kernel(function_name, _sl._KERNEL_ALLOWLIST_PATTERNS):
            return {}

    trace_hash = _trace_hash(module_hash, function_name)
    trace_data: Dict[str, Any] = defaultdict(dict)
    trace_data["name"] = function_name
    trace_data["function"] = None
    grid = _STATIC_GRID_BY_HASH.get(module_hash) if module_hash is not None else None
    if grid is not None:
        trace_data["grid"] = list(grid)
    else:
        # A dynamic grid is computed inside the compiled host code and never
        # reaches Python, and without a module hash the module is ambiguous.
        # An empty grid plus an explicit marker keeps the schema-required
        # field while staying honest.
        trace_data["grid"] = []
        trace_data["grid_unknown"] = True
    trace_data["compilation_metadata"] = {
        "hash": trace_hash,
        "name": function_name,
    }
    if args is not None:
        names = tuple(arg_names or ())
        arg_dict = {
            names[i] if i < len(names) else f"arg{i}": value
            for i, value in enumerate(args)
        }
        if kwargs:
            arg_dict.update(kwargs)
        try:
            trace_data["extracted_args"] = _sl.extract_arg_info(arg_dict)
        except Exception as e:
            # Tracing must never break the launch it observes.
            log.warning("[tritonparse] CuTeDSL arg extraction failed: %s", e)
            trace_data["extracted_args"] = {"extraction_error": str(e)}
    else:
        trace_data["extracted_args"] = {}
        trace_data["args_packed"] = True
        trace_data["arg_count"] = len(exe_args) if exe_args is not None else 0

    _sl.trace_structured_triton("launch", metadata_fn=lambda: _sl.convert(trace_data))
    return trace_data


def _get_cutedsl_singleton() -> Any:
    """Return the CuTeDSL singleton, or None when unavailable."""
    try:
        import cutlass
    except ImportError:
        return None
    dsl_pkg = getattr(cutlass, "cutlass_dsl", None)
    dsl_cls = getattr(dsl_pkg, "CuTeDSL", None) if dsl_pkg is not None else None
    get_dsl = getattr(dsl_cls, "_get_dsl", None) if dsl_cls is not None else None
    if get_dsl is None:
        return None
    try:
        return get_dsl()
    except Exception as e:  # old/broken installs must not break init()
        log.debug("[tritonparse] Cannot access CuTeDSL singleton: %s", e)
        return None


def _get_cutedsl_hooks() -> Any:
    """Return the public hooks API, or None on older CUTLASS builds."""
    try:
        from cutlass import hooks
    except ImportError:
        return None
    if not all(
        callable(getattr(hooks, name, None))
        for name in ("register_hook", "remove_hook")
    ):
        return None
    return hooks


def _compilation_hook(event: Any) -> None:
    # Upstream wraps hook exceptions as DSLRuntimeError (and a PRE_EXECUTE
    # failure stops the call), so tracing must contain its own failures.
    try:
        maybe_trace_cutedsl(
            event.owner,
            function_name=event.function_name,
            module_hash=event.module_hash,
            cache_hit=event.cache_hit,
            artifacts=event.artifacts,
            compile_options=event.compile_options,
            func_body=event.func_body,
            kernel_info=event.kernel_info,
        )
    except Exception as e:
        log.warning("[tritonparse] CuTeDSL compilation tracing failed: %s", e)


def _launch_hook(event: Any) -> None:
    try:
        maybe_trace_cutedsl_launch(
            event.owner,
            function_name=event.function_name,
            module_hash=event.module_hash,
            kernel_info=event.kernel_info,
            args=event.args,
            kwargs=event.kwargs,
            exe_args=event.exe_args,
            executor=event.executor,
            arg_names=event.arg_names,
        )
    except Exception as e:
        log.warning("[tritonparse] CuTeDSL launch tracing failed: %s", e)


def setup_cutedsl_tracing(*, enable_launch: bool = False) -> bool:
    """Register tritonparse hooks on CuTeDSL. Returns True on success.

    No-op (False) when cutlass is not installed or predates the
    ``cutlass.hooks`` ``POST_COMPILE`` event (NVIDIA/cutlass#3697).
    Otherwise also switches on artifact keeping (see the module docstring).
    Launch tracing additionally requires ``PRE_EXECUTE``
    (NVIDIA/cutlass#3703); when unavailable, compilation tracing still
    proceeds with a warning.
    """
    global _CUTEDSL_DSL_REF
    if _CUTEDSL_DSL_REF is not None:
        return True
    dsl = _get_cutedsl_singleton()
    if dsl is None:
        log.debug("[tritonparse] cutlass not importable; skipping CuTeDSL tracing.")
        return False
    hooks = _get_cutedsl_hooks()
    events = getattr(hooks, "HookEvent", None)
    compilation_event = getattr(events, "POST_COMPILE", None)
    launch_event = getattr(events, "PRE_EXECUTE", None)
    if compilation_event is None:
        log.warning(
            "[tritonparse] Installed cutlass has no cutlass.hooks "
            "POST_COMPILE support (NVIDIA/cutlass#3697); skipping "
            "CuTeDSL tracing."
        )
        return False
    _CUTEDSL_DSL_REF = dsl
    try:
        hooks.register_hook(compilation_event, _compilation_hook, dsl=dsl)
        _CUTEDSL_UNREGISTER.append(
            partial(hooks.remove_hook, compilation_event, _compilation_hook, dsl=dsl)
        )
        launch_supported = launch_event is not None
        if enable_launch and launch_supported:
            hooks.register_hook(launch_event, _launch_hook, dsl=dsl)
            _CUTEDSL_UNREGISTER.append(
                partial(hooks.remove_hook, launch_event, _launch_hook, dsl=dsl)
            )
        _configure_cutedsl_envar(dsl)
        log.debug("[tritonparse] CuTeDSL compilation tracing enabled.")
        if enable_launch and launch_supported:
            log.debug("[tritonparse] CuTeDSL launch tracing enabled.")
        elif enable_launch:
            log.warning(
                "[tritonparse] Installed cutlass has cutlass.hooks "
                "POST_COMPILE but no PRE_EXECUTE (NVIDIA/cutlass#3703); "
                "compilation tracing continues without launches."
            )
    except Exception:
        teardown_cutedsl_tracing()
        raise
    return True


def teardown_cutedsl_tracing() -> None:
    """Remove tritonparse hooks and restore CuTeDSL settings. Never raises."""
    global _CUTEDSL_DSL_REF
    _CUTEDSL_DSL_REF = None
    try:
        _restore_cutedsl_envar()
    except Exception as e:  # must not skip the hook unregistration below
        log.warning("[tritonparse] CuTeDSL envar restore failed: %s", e)
    _STATIC_GRID_BY_HASH.clear()
    unregister = _CUTEDSL_UNREGISTER[:]
    _CUTEDSL_UNREGISTER.clear()
    for remove in reversed(unregister):
        try:
            remove()
        except Exception as e:  # teardown must not break clear_logging_config()
            log.warning("[tritonparse] CuTeDSL teardown failed: %s", e)


def _reset_for_tests() -> None:
    """Drop all module state, restoring CuTeDSL settings (tests only)."""
    teardown_cutedsl_tracing()


# Names re-exported for launch tracing and tests.
__all__ = [
    "CUTEDSL_KEEP_TOKENS",
    "extract_cutedsl_python_source",
    "maybe_trace_cutedsl",
    "maybe_trace_cutedsl_launch",
    "parse_cutedsl_launch_configs",
    "setup_cutedsl_tracing",
    "teardown_cutedsl_tracing",
]
