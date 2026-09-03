#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Regenerate the trace files this repo ships as examples and test fixtures.

The example trace served by the website and the fixtures under
``tests/example_output`` used to be produced by hand, on whichever machine and
whichever Triton happened to be around. They drifted: the published example was
captured on Triton 3.6 and predates the ``roofline`` and ``ir_stages`` payloads
entirely, and it embeds ~9.5k absolute paths from the author's fbsource
checkout. This module makes that regeneration a repeatable command so the next
refresh is a one-liner instead of an archaeology exercise.

Usage::

    python -m tritonparse.tools.generate_examples --list
    python -m tritonparse.tools.generate_examples triton --out-dir /tmp/regen
    python -m tritonparse.tools.generate_examples triton --install

Requires a CUDA GPU: the workloads compile and launch real Triton kernels.
``--install`` overwrites files that are checked in, so it is opt-in.

Three things happen between running the workload and writing the artifacts, and
all three are the point of this module:

**Renaming.** The writer names raw logs
``dedicated_log_triton_trace_{user}_rank_{r}_pid_{p}_host_{h}_.ndjson``. Those
suffixes are machine-specific, and ``host_`` in particular leaks an internal
hostname into a file served from a public site. The raw log is renamed to the
bare ``dedicated_log_triton_trace_{token}_.ndjson`` shape before parsing --
before, because the parser derives the mapped filename from the raw basename
and records it in ``log_file_list.json``, so renaming afterwards desyncs them.
The bare shape is not an invention: ``parse_trace_filename_metadata`` reads it
as ``(rank=None, pid=None, host=None)``, and the ``dedicated_log_triton_trace_``
prefix is required -- ``unified_parse`` rejects anything else outright.

**Sanitizing.** Absolute paths appear in stack frames, in
``python_source.file_path``, and inside the IR itself as ``#loc`` records. They
are rewritten to stable placeholders. This also runs before parsing, so the IR
and the Python source agree on a path and the parser's own source-mapping pass
doubles as a consistency check on the rewrite. Rewriting cannot break the
parser by pointing at files that do not exist: the committed fixtures already
reference a ``/scratch`` tree that exists on nobody's machine and parse fine.

**Verifying.** Every artifact is checked for leftover machine-specific strings
before it is written, so a workload that picks up a new path prefix fails the
run instead of quietly publishing it.
"""

from __future__ import annotations

import argparse
import getpass
import gzip
import os
import re
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from tritonparse.tp_logger import get_logger

logger = get_logger("GenerateExamples")

# Repo root: .../tritonparse/tools/generate_examples.py -> .../
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Where the generated artifacts belong when --install is passed.
WEBSITE_PUBLIC = REPO_ROOT / "website" / "public"
FIXTURE_ROOT = REPO_ROOT / "tests" / "example_output"

# The USER token baked into the example filenames. Kept as the historical value
# so every published URL keeps resolving: README, docs and website/src/App.tsx
# all hard-code dedicated_log_triton_trace_findhao__mapped.ndjson.gz, and the
# deep link into the hosted viewer is already out in the world.
TRITON_TRACE_TOKEN = "findhao"


# =============================================================================
# Path sanitizing
# =============================================================================


@dataclass(frozen=True)
class RewriteRule:
    """One path rewrite. ``is_regex`` is explicit so adding a pattern-based
    rule does not depend on how its text happens to be spelled."""

    pattern: str
    replacement: str
    is_regex: bool = False
    priority: int = 0

    def apply(self, text: str) -> str:
        if self.is_regex:
            return re.sub(self.pattern, lambda _match: self.replacement, text)
        return text.replace(self.pattern, self.replacement)


def _build_rewrite_rules() -> List[RewriteRule]:
    """Absolute-path prefixes to rewrite, longest first.

    Longest-first matters: the repo lives under the fbsource checkout, which
    lives under $HOME, so the broadest rule must be applied last or it swallows
    the specific ones.
    """
    rules: List[RewriteRule] = [RewriteRule(str(REPO_ROOT), "/tritonparse")]

    # The interpreter's stdlib and site-packages show up in every stack frame.
    # Collapse the environment prefix but keep the "pythonX.Y/..." tail, which
    # is the part a reader actually uses.
    prefix = Path(sys.prefix).resolve()
    rules.append(RewriteRule(f"{prefix}/lib/", "/python/lib/"))
    rules.append(RewriteRule(str(prefix), "/python"))

    # Inductor's generated-code cache is per-user.
    rules.append(
        RewriteRule(
            r"/tmp/torchinductor_[^/\"]+",
            "/tmp/torchinductor",
            is_regex=True,
            priority=1,
        )
    )

    home = os.path.expanduser("~")
    if home and home not in {"/", "~"}:
        rules.append(RewriteRule(home, "/home/user"))

    rules.sort(key=lambda rule: (rule.priority, len(rule.pattern)), reverse=True)
    return rules


def sanitize_text(text: str, rules: List[RewriteRule]) -> str:
    """Apply the rewrite rules to one blob of NDJSON.

    Operating on the serialized text rather than on parsed objects is
    deliberate: paths also live inside IR strings (``#loc`` records) and inside
    keys, and a value-only walk would leave those behind. The replacements are
    plain ASCII path text, so they cannot produce invalid JSON.
    """
    for rule in rules:
        text = rule.apply(text)
    return text


def _leak_patterns(
    allowed_trace_token: str = TRITON_TRACE_TOKEN,
) -> List[tuple[str, re.Pattern]]:
    """Substrings that must not survive into a published artifact."""
    patterns = [
        ("fbsource checkout", re.compile(r"/fbsource/")),
        ("fbcode path", re.compile(r"/fbcode/")),
    ]
    home = os.path.expanduser("~")
    if home and home not in {"/", "~"}:
        patterns.append(("home directory", re.compile(re.escape(home))))
    host = socket.gethostname().split(".")[0]
    if host:
        patterns.append(("hostname", re.compile(re.escape(host))))
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except OSError as error:
        raise RuntimeError("could not determine the current username") from error
    if not user:
        raise RuntimeError("could not determine the current username")
    # The trace token is a deliberate, reviewed value; only flag the *running*
    # user when it differs from it.
    if user and user != allowed_trace_token:
        patterns.append(("username", re.compile(re.escape(user))))
    return patterns


def assert_clean(path: Path, allowed_trace_token: str = TRITON_TRACE_TOKEN) -> None:
    """Fail loudly if an artifact still carries machine-specific strings."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    found = [
        f"{label} ({pattern.pattern})"
        for label, pattern in _leak_patterns(allowed_trace_token)
        if pattern.search(text)
    ]
    if found:
        raise RuntimeError(
            f"{path.name} still contains machine-specific strings: "
            f"{', '.join(found)}. Add a rewrite rule in _build_rewrite_rules()."
        )


# =============================================================================
# Workloads
# =============================================================================


@dataclass
class Workload:
    """One example trace: what to run and where its artifacts belong."""

    name: str
    description: str
    run: Callable[[], None]
    """Executes the kernels. Tracing is already initialized when this is called."""

    init_kwargs: Dict[str, object]
    """Keyword arguments for structured_logging.init(), i.e. what to capture."""

    trace_token: str
    """USER token baked into the sanitized filename."""

    website_example: bool = False
    """Whether the mapped trace is published as a website example."""

    split_inductor_compilations: bool = False
    """Whether to split the parsed output by inductor frame.

    Off for every example. Splitting is a tlparse-integration feature: it emits
    one f{frame}_fc{compile}_a{attempt}_cai{id}.ndjson per compile frame so
    tlparse can deep-link each frame. The viewer loads exactly one file, so a
    split example would only ever show a fraction of the run. The split path
    stays covered by tests/gpu/test_context_manager.py.
    """

    fixture_subdir: Optional[str] = None
    """Subdirectory of tests/example_output for the parsed artifacts."""

    raw_fixture_subdir: Optional[str] = None
    """Subdirectory of tests/example_output for the raw log fixture."""

    keep_raw_log: bool = False
    """Whether the sanitized raw log is kept as a fixture too."""

    raw_log_max_launches: int = 20
    """Cap on `launch` records in the kept raw log.

    A traced autotune run emits ~1000 launches and a multi-megabyte raw log,
    which is not something anyone can review in a diff. The raw fixture exists
    to pin the shapes the writer puts on disk, so it keeps every compilation
    and autotune record and every distinct event type, and truncates only the
    repetitive launch tail. The parsed fixture keeps all launches.
    """


def _run_complex_kernels() -> None:
    """Two kernels, one autotuned: the workload behind the published example.

    Mirrors tests/gpu/test_complex_kernels.py. It is duplicated rather than
    imported because the test asserts on counts that are tuned for a fast unit
    test, while the example wants a trace that exercises the viewer: autotune
    sessions, several launch groups per kernel, and a GEMM so the roofline
    payload takes its is_gemm branch as well as its pure-bandwidth one.
    """
    import torch
    import triton  # @manual=//triton:triton
    import triton.language as tl  # @manual=//triton:triton

    @triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 16,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 1,
                },
                num_stages=1,
                num_warps=1,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 16,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 1,
                },
                num_stages=1,
                num_warps=1,
            ),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def matmul_kernel(
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size)
        pid_n = (pid % num_pid_in_group) // group_size

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a_block = tl.load(
                a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
            )
            b_block = tl.load(
                b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
            )
            accumulator += tl.dot(a_block, b_block)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        c_block = accumulator.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c_block, mask=c_mask)

    @triton.jit
    def fused_op_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        output_ptr,
        n_elements,
        scale_factor: float,
        ACTIVATION: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        c = tl.load(c_ptr + offsets, mask=mask)

        result = a * b * scale_factor + c
        if ACTIVATION == "relu":
            result = tl.where(result > 0, result, 0.0)

        tl.store(output_ptr + offsets, result, mask=mask)

    def matmul(a, b):
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        def grid(META):
            return (
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            )

        matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            ACTIVATION=None,
        )
        return c

    def fused_op(a, b, c, scale_factor: float, activation: str):
        n_elements = a.numel()
        output = torch.empty_like(a)
        block_size = 8
        grid = (triton.cdiv(n_elements, block_size),)
        fused_op_kernel[grid](
            a,
            b,
            c,
            output,
            n_elements,
            scale_factor,
            ACTIVATION=activation,
            BLOCK_SIZE=block_size,
        )
        return output

    torch.manual_seed(0)

    for shape_a, shape_b in (((16, 16), (16, 16)), ((32, 16), (16, 32))):
        a = torch.randn(shape_a, device="cuda", dtype=torch.float16)
        b = torch.randn(shape_b, device="cuda", dtype=torch.float16)
        matmul(a, b).sum()

    # Four launches across two ACTIVATION specializations, the last at a
    # different size so launch_diff has a varying grid to report and not just
    # varying scalars.
    x = torch.randn((8,), device="cuda", dtype=torch.float32)
    y = torch.randn((8,), device="cuda", dtype=torch.float32)
    z = torch.randn((8,), device="cuda", dtype=torch.float32)
    fused_op(x, y, z, scale_factor=1.0, activation="none").sum()
    fused_op(x, y, z, scale_factor=2.5, activation="none").sum()
    fused_op(x, y, z, scale_factor=1.0, activation="relu").sum()

    small_x = torch.randn((6,), device="cuda", dtype=torch.float32)
    small_y = torch.randn((6,), device="cuda", dtype=torch.float32)
    small_z = torch.randn((6,), device="cuda", dtype=torch.float32)
    fused_op(small_x, small_y, small_z, scale_factor=1.0, activation="relu").sum()

    torch.cuda.synchronize()


WORKLOADS: Dict[str, Workload] = {
    "triton": Workload(
        name="triton",
        description=(
            "Hand-written Triton: an autotuned matmul plus a fused elementwise "
            "kernel. Published as the website's default example."
        ),
        run=_run_complex_kernels,
        # Capture everything the viewer can render: launches (so launch_diff,
        # roofline and the autotune events have data) and SASS (so the NVIDIA
        # stage list is complete and source_mappings covers sass).
        init_kwargs={"enable_trace_launch": True, "enable_sass_dump": True},
        trace_token=TRITON_TRACE_TOKEN,
        website_example=True,
        fixture_subdir="parsed_output_complex",
        raw_fixture_subdir="logs",
        keep_raw_log=True,
    ),
}


# =============================================================================
# Driver
# =============================================================================


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} in {directory}, found "
            f"{[m.name for m in matches]}"
        )
    return matches[0]


def _sanitize_file_in_place(path: Path, rules: List[RewriteRule]) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(sanitize_text(text, rules), encoding="utf-8")


def _sanitize_gzip_in_place(path: Path, rules: List[RewriteRule]) -> None:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    # mtime=0 keeps the gzip header byte-identical across runs, so an unchanged
    # trace does not show up as a diff purely because it was regenerated.
    with gzip.GzipFile(path, "wb", mtime=0) as fh:
        fh.write(sanitize_text(text, rules).encode("utf-8"))


def _write_capped_raw_log(source: Path, target: Path, max_launches: int) -> None:
    """Copy a raw log, keeping at most ``max_launches`` launch records.

    Every non-launch record is kept, so the fixture still covers each event
    type the writer emits. What was dropped is logged rather than silently
    swallowed -- a fixture that looks complete but is not is worse than a
    smaller one that says so.
    """
    from tritonparse._json_compat import loads

    kept_launches = 0
    dropped = 0
    with (
        source.open("r", encoding="utf-8") as src,
        target.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            if not line.strip():
                dst.write(line)
                continue
            try:
                record = loads(line)
            except ValueError:
                record = None
            event_type = record.get("event_type") if isinstance(record, dict) else None
            if event_type == "launch":
                if kept_launches >= max_launches:
                    dropped += 1
                    continue
                kept_launches += 1
            dst.write(line)

    if dropped:
        logger.info(
            "Raw log %s: kept %d launch records, dropped %d (cap=%d)",
            target.name,
            kept_launches,
            dropped,
            max_launches,
        )


def _describe_environment() -> str:
    import torch
    import triton  # @manual=//triton:triton

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    return (
        f"torch {torch.__version__}, triton {triton.__version__}, "
        f"python {sys.version.split()[0]}, {device}"
    )


@dataclass
class Artifacts:
    """What one generation run produced."""

    mapped: Path
    """The parsed, gzipped trace -- the file the viewer loads."""

    parsed_extras: List[Path]
    """Everything else the parser emitted alongside it (log_file_list.json)."""

    raw_log: Optional[Path] = None
    """The sanitized, launch-capped raw log, when the workload keeps one."""

    def all_paths(self) -> List[Path]:
        paths = [self.mapped, *self.parsed_extras]
        if self.raw_log:
            paths.append(self.raw_log)
        return paths


def generate(workload: Workload, out_dir: Path) -> Artifacts:
    """Run one workload and write sanitized artifacts into out_dir."""
    compression = os.environ.get("TRITON_TRACE_COMPRESSION", "none")
    if compression != "none":
        raise RuntimeError(
            "example generation requires uncompressed trace output; unset "
            f"TRITON_TRACE_COMPRESSION (currently {compression!r})"
        )

    from tritonparse.parse.utils import unified_parse
    from tritonparse.structured_logging import clear_logging_config, init

    out_dir.mkdir(parents=True, exist_ok=True)
    rules = _build_rewrite_rules()

    with tempfile.TemporaryDirectory(prefix="tritonparse_examples_") as scratch:
        log_dir = Path(scratch) / "logs"
        parsed_dir = Path(scratch) / "parsed"
        log_dir.mkdir()
        parsed_dir.mkdir()

        # Tracing is driven directly rather than through TritonParseManager,
        # which parses on exit. Parsing has to happen after the raw log is
        # renamed -- the mapped filename is derived from the raw basename --
        # so the manager's parse would be thrown away.
        logger.info("Running workload %r (%s)", workload.name, _describe_environment())
        init(str(log_dir), **workload.init_kwargs)
        try:
            workload.run()
        finally:
            clear_logging_config()

        raw = _find_one(log_dir, "dedicated_log_triton_trace_*.ndjson")
        renamed = log_dir / f"dedicated_log_triton_trace_{workload.trace_token}_.ndjson"
        raw.rename(renamed)
        _sanitize_file_in_place(renamed, rules)
        assert_clean(renamed, workload.trace_token)
        logger.info("Sanitized raw log -> %s", renamed.name)

        unified_parse(
            source=str(log_dir),
            out=str(parsed_dir),
            overwrite=True,
            skip_logger=True,
            split_inductor_compilations=workload.split_inductor_compilations,
        )

        mapped: Optional[Path] = None
        extras: List[Path] = []
        for produced in sorted(parsed_dir.iterdir()):
            if not produced.is_file():
                continue
            target = out_dir / produced.name
            shutil.copy2(produced, target)
            if target.suffix == ".gz":
                _sanitize_gzip_in_place(target, rules)
            else:
                _sanitize_file_in_place(target, rules)
            assert_clean(target, workload.trace_token)
            if produced.name.endswith("_mapped.ndjson.gz"):
                if mapped is not None:
                    raise RuntimeError(
                        f"workload {workload.name!r} produced more than one mapped "
                        f"trace ({mapped.name}, {target.name}). The viewer loads a "
                        "single file, so such a workload must set "
                        "split_inductor_compilations=False."
                    )
                mapped = target
            else:
                extras.append(target)

        if mapped is None:
            raise RuntimeError(
                f"workload {workload.name!r} produced no *_mapped.ndjson.gz in "
                f"{sorted(p.name for p in parsed_dir.iterdir())}"
            )

        raw_log: Optional[Path] = None
        if workload.keep_raw_log:
            raw_log = out_dir / renamed.name
            _write_capped_raw_log(renamed, raw_log, workload.raw_log_max_launches)
            assert_clean(raw_log, workload.trace_token)

    artifacts = Artifacts(mapped=mapped, parsed_extras=extras, raw_log=raw_log)
    for path in artifacts.all_paths():
        logger.info("Wrote %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    return artifacts


def install(workload: Workload, artifacts: Artifacts) -> None:
    """Copy generated artifacts over the checked-in ones."""
    if workload.website_example:
        WEBSITE_PUBLIC.mkdir(parents=True, exist_ok=True)
        dest = WEBSITE_PUBLIC / artifacts.mapped.name
        shutil.copy2(artifacts.mapped, dest)
        logger.info("Installed %s", dest)

    if workload.fixture_subdir:
        fixture_dir = FIXTURE_ROOT / workload.fixture_subdir
        fixture_dir.mkdir(parents=True, exist_ok=True)
        for path in [artifacts.mapped, *artifacts.parsed_extras]:
            shutil.copy2(path, fixture_dir / path.name)
            logger.info("Installed %s", fixture_dir / path.name)

    if artifacts.raw_log:
        if not workload.raw_fixture_subdir:
            raise ValueError(
                f"workload {workload.name!r} keeps a raw log but does not set "
                "raw_fixture_subdir"
            )
        raw_dir = FIXTURE_ROOT / workload.raw_fixture_subdir
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifacts.raw_log, raw_dir / artifacts.raw_log.name)
        logger.info("Installed %s", raw_dir / artifacts.raw_log.name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tritonparse.tools.generate_examples",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "workload",
        nargs="?",
        choices=sorted(WORKLOADS),
        help="Which example to generate. Omit with --list to see the options.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the available workloads and exit."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("example_output_regen"),
        help="Directory to write the generated artifacts into.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help=(
            "Also copy the artifacts over the checked-in example and fixtures. "
            "Overwrites tracked files."
        ),
    )
    args = parser.parse_args(argv)

    if args.list:
        for name, workload in sorted(WORKLOADS.items()):
            print(f"{name}\n    {workload.description}")
        return 0

    if not args.workload:
        parser.error("a workload is required unless --list is given")

    workload = WORKLOADS[args.workload]
    artifacts = generate(workload, args.out_dir.resolve())
    if args.install:
        install(workload, artifacts)
    else:
        print(
            f"\nGenerated {len(artifacts.all_paths())} file(s) in {args.out_dir}. "
            "Re-run with --install to overwrite the checked-in copies."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
