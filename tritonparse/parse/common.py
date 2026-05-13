#  Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import zstandard as zstd
from tritonparse._json_compat import dumps
from tritonparse.shared_vars import (
    DEFAULT_TRACE_FILE_PREFIX_WITHOUT_USER as LOG_PREFIX,
    is_fbcode,
)
from tritonparse.tp_logger import logger

from .trace_processor import parse_single_rank


class TraceFilenameMetadata(NamedTuple):
    """Structured suffixes extracted from a tritonparse trace log basename.

    Each field is None if the corresponding suffix is absent from the
    basename — legacy logs may omit any combination of rank/pid/host.
    """

    rank: Optional[int]
    pid: Optional[int]
    host: Optional[str]


_OUTER_COMPRESSION_EXTS = (".gz", ".zst")
# Longest first — `.bin.ndjson` MUST be tried before `.ndjson` so we don't
# strip just `.ndjson` and leave a stray `.bin` on the basename.
_INNER_FORMAT_EXTS = (".bin.ndjson", ".ndjson", ".clp")

# Each suffix regex anchors with `fullmatch` to the END of the (already
# inner-stripped) basename, so the greedy `.*` always lands on the
# RIGHTMOST occurrence of the suffix. This is the whole point of the
# right-to-left peeling strategy: substrings inside USER that look like
# `_pid_NNN_` / `_host_X_` / `_rank_N_` cannot be mistaken for the real
# logger-emitted suffix.
_HOST_TAIL = re.compile(r"(?P<rest>.*)_host_(?P<host>[a-zA-Z0-9-]+)_")
_PID_TAIL = re.compile(r"(?P<rest>.*)_pid_(?P<pid>\d+)_")
_RANK_TAIL = re.compile(r"(?P<rest>.*)_rank_(?P<rank>\d+)_")


def _strip_extensions(basename: str) -> str:
    """Strip outer compression + ndjson/clp extension(s) in the right order.

    The logger emits one of:
      .ndjson | .bin.ndjson | .bin.ndjson.gz | .bin.ndjson.zst |
      .ndjson.gz | .clp
    """
    for ext in _OUTER_COMPRESSION_EXTS:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            break
    for ext in _INNER_FORMAT_EXTS:
        if basename.endswith(ext):
            basename = basename[: -len(ext)]
            break
    return basename


def parse_trace_filename_metadata(basename: str) -> TraceFilenameMetadata:
    """Extract (rank, pid, host) from a tritonparse trace log basename.

    The structured logger appends suffixes in fixed order at the END of
    the basename, before the extension::

        {USER}_rank_{N}_pid_{PID}_host_{HOST}_.ndjson[.gz|.zst]
                                              ^- trailing `_` always present

    Some suffixes are absent in legacy files (any combination is allowed).
    USER may itself contain underscores AND substrings that look like
    suffixes (e.g. service account `pid_123_team`). To stay correct under
    such adversarial usernames, this helper strips from the RIGHT —
    `fullmatch` anchored at end-of-string with greedy `.*` — rather than
    `re.search()` from the left, which would mis-attribute substrings
    inside USER as the real metadata.

    Returns a TraceFilenameMetadata; any field is None when its suffix
    is absent.
    """
    stripped = _strip_extensions(basename)

    host: Optional[str] = None
    if (m := _HOST_TAIL.fullmatch(stripped)) is not None:
        host = m["host"]
        # Re-add the trailing `_` so the next peel's anchor still matches.
        stripped = m["rest"] + "_"

    pid: Optional[int] = None
    if (m := _PID_TAIL.fullmatch(stripped)) is not None:
        pid = int(m["pid"])
        stripped = m["rest"] + "_"

    rank: Optional[int] = None
    if (m := _RANK_TAIL.fullmatch(stripped)) is not None:
        rank = int(m["rank"])

    return TraceFilenameMetadata(rank=rank, pid=pid, host=host)


def _collect_and_bucket_files(
    raw_log_dir: str,
    rank_config: "RankConfig",
    enable_pre_init_attribution: bool,
) -> Dict["Rank", List[str]]:
    """Collect tritonparse log files and bucket them by Rank.

    Three-pass algorithm:

    Pass 1 — scan the directory; for each eligible log file extract its
    rank/PID/host via `parse_trace_filename_metadata` (right-to-left
    suffix peeling, immune to look-alike substrings inside USER); build
    raw_ranked + raw_no_rank lists and a host_pid_to_rank lookup keyed
    by (host, pid) tuple. Tuple key is necessary because PIDs are not
    globally unique across hosts in a multi-host distributed job.

    Pass 2 (gated) — when enable_pre_init_attribution is True AND rank_config
    is not "no rank", re-attribute no-rank files whose (host, pid) is in
    host_pid_to_rank into the matching ranked bucket. With the default
    Diff 2 setting (False), this pass is a no-op and behavior is
    identical to v1.

    Pass 3 — apply rank_config filtering: --all-ranks keeps every ranked
    bucket; --rank N keeps only that one; --rank none keeps no ranked
    buckets; default (no flag) keeps rank 0. NO_RANK bucket is added only
    when --all-ranks or --rank none is in effect.

    Each bucket is sorted by (host, pid) for deterministic output
    (legacy no-host/no-PID files sort first; cross-PID occurrence_id
    ordering depends on this).
    """
    # (host, pid) tuple key — PIDs are not globally unique across hosts,
    # so we need both to identify a unique process. Legacy files (no host
    # suffix) get host=None and stay in their own namespace; they cannot
    # be confused with new-format files even when PIDs collide.
    host_pid_to_rank: Dict[Tuple[Optional[str], int], int] = {}
    raw_ranked: Dict[int, List[str]] = defaultdict(list)
    raw_no_rank: List[str] = []

    # Sort listdir for deterministic conflict resolution (first-wins) when
    # the same (host, pid) appears in more than one ranked file.
    for item in sorted(os.listdir(raw_log_dir)):
        path = os.path.join(raw_log_dir, item)
        if not os.path.isfile(path):
            continue
        # Use 'in' instead of 'startswith' to support lg's --store-by-source
        # format which produces filenames like
        # log.source.dedicated_log_triton_trace_xxx_.ndjson.
        if LOG_PREFIX not in item:
            continue
        meta = parse_trace_filename_metadata(item)
        if meta.rank is not None:
            raw_ranked[meta.rank].append(path)
            if meta.pid is not None:
                key = (meta.host, meta.pid)
                existing = host_pid_to_rank.get(key)
                if existing is not None and existing != meta.rank:
                    logger.warning(
                        "Conflicting rank attribution for "
                        "(host=%s, pid=%d): existing rank %d vs new rank %d "
                        "— keeping existing",
                        meta.host,
                        meta.pid,
                        existing,
                        meta.rank,
                    )
                else:
                    host_pid_to_rank[key] = meta.rank
        else:
            raw_no_rank.append(path)

    # Pass 2: re-attribute pre-init no-rank files to their (host, PID)-matched rank.
    # Skipped under --rank none (user explicitly wants the unattributed view).
    truly_no_rank: List[str] = []
    if enable_pre_init_attribution and not rank_config.is_no_rank:
        for path in raw_no_rank:
            meta = parse_trace_filename_metadata(os.path.basename(path))
            key = (meta.host, meta.pid) if meta.pid is not None else None
            if key is not None and key in host_pid_to_rank:
                raw_ranked[host_pid_to_rank[key]].append(path)
            else:
                truly_no_rank.append(path)
    else:
        truly_no_rank = list(raw_no_rank)

    buckets: Dict["Rank", List[str]] = {}
    for rank_value, files in raw_ranked.items():
        rank_obj = Rank(rank_value)
        if rank_config.all_ranks:
            buckets[rank_obj] = files
        elif rank_config.is_no_rank:
            continue
        elif rank_config.rank is not None:
            if rank_config.rank.value == rank_value:
                buckets[rank_obj] = files
        else:
            # Default (no --rank, no --all-ranks): include rank 0 only.
            if rank_value == 0:
                buckets[rank_obj] = files

    if (rank_config.all_ranks or rank_config.is_no_rank) and truly_no_rank:
        buckets[Rank(Rank.NO_RANK)] = truly_no_rank
    elif not buckets and truly_no_rank:
        # No ranked files matched the requested rank config, but no-rank
        # files exist (e.g. single-GPU job without dist.init). Fall back
        # to parsing no-rank files instead of raising an error downstream.
        logger.info(
            "No ranked files found; falling back to %d no-rank file(s).",
            len(truly_no_rank),
        )
        buckets[Rank(Rank.NO_RANK)] = truly_no_rank

    # Sort each bucket by (host, PID, path) for deterministic occurrence_id
    # ordering. Legacy no-host/no-PID files sort first (host="", pid=-1).
    # Single helper call per file vs. the previous two-call approach.
    def _bucket_sort_key(path: str) -> Tuple[str, int, str]:
        meta = parse_trace_filename_metadata(os.path.basename(path))
        return (
            meta.host or "",
            meta.pid if meta.pid is not None else -1,
            path,
        )

    for rank in buckets:
        buckets[rank].sort(key=_bucket_sort_key)

    return buckets


if is_fbcode():
    from tritonparse.fb.source_type import SourceType
else:
    from .source_type import SourceType


class Rank:
    """Class representing a rank in distributed training."""

    NO_RANK = -1  # Special value representing no rank (before torch.distributed init)

    def __init__(self, rank_value: Optional[int] = None):
        """
        Initialize a Rank object.

        Args:
            rank_value: Specific rank value, or None for default rank.
                       Use Rank.NO_RANK (-1) for files without rank suffix.
        """
        if rank_value is not None:
            self.value = rank_value
            self.is_default = False
        else:
            self.value = 0
            self.is_default = True

    @property
    def is_no_rank(self) -> bool:
        """Check if this represents a no-rank file (before torch.distributed init)."""
        return self.value == self.NO_RANK

    def to_string(self, prefix: str = "", suffix: str = "") -> str:
        """
        Convert rank to user-friendly string representation.

        Used for:
        - Output directory names: rank_0, rank_1, rank_none
        - CLI parameter display
        - Log output

        Note: Do NOT use this for building lg filenames. Use to_file_suffix() instead.

        Args:
            prefix: Prefix to add before rank string
            suffix: Suffix to add after rank string

        Returns:
            String representation of the rank
        """
        if self.is_default:
            return ""
        if self.is_no_rank:
            return f"{prefix}rank_none{suffix}"
        return f"{prefix}rank_{self.value}{suffix}"

    def to_file_suffix(self) -> str:
        """
        Return the rank suffix for lg filenames.

        File naming conventions:
        - With rank: dedicated_log_triton_trace_{USER}_rank_{N}_.ndjson.gz
        - Without rank: dedicated_log_triton_trace_{USER}_.ndjson.gz

        Returns:
            "_rank_{N}_" for specific rank, "" for no-rank files
        """
        if self.is_no_rank:
            return ""  # No rank files don't have this suffix
        if self.is_default:
            return "_rank_0_"
        return f"_rank_{self.value}_"

    def to_int(self) -> int:
        """
        Convert rank to integer value.

        Returns:
            Integer value of the rank
        """
        return self.value


class RankConfig:
    """Configuration for handling ranks in log processing."""

    def __init__(
        self,
        rank: Optional[Rank] = None,
        all_ranks: bool = False,
    ):
        """
        Initialize a RankConfig object.

        Args:
            rank: Specific rank to process (use Rank(Rank.NO_RANK) for no-rank files)
            all_ranks: Whether to process all ranks (includes no-rank files)
        """
        self.rank = rank
        self.all_ranks = all_ranks

    @property
    def is_no_rank(self) -> bool:
        """Check if this config is for processing no-rank files only."""
        return self.rank is not None and self.rank.is_no_rank

    @classmethod
    def from_cli_args(
        cls, rank: Optional[int], all_ranks: bool, source_type: SourceType
    ) -> "RankConfig":
        """
        Create a RankConfig from command line arguments.

        Args:
            rank: Specific rank value from CLI (use Rank.NO_RANK for --rank none)
            all_ranks: Whether --all-ranks flag was specified
            source_type: Type of source

        Returns:
            Configured RankConfig object
        """
        if all_ranks:
            if rank is not None:
                raise ValueError("Can't specify both a rank and --all-ranks")
            return cls(all_ranks=True)

        if rank is not None:
            return cls(rank=Rank(rank))
        if source_type in [SourceType.LOCAL, SourceType.LOCAL_FILE]:
            # For local files, default to all_ranks to include no-rank files
            return cls(all_ranks=True)
        elif is_fbcode():
            from tritonparse.fb.utils import rank_config_from_cli_args

            return rank_config_from_cli_args(cls, source_type)
        else:
            return cls(all_ranks=True)

    def to_rank(self) -> Rank:
        """
        Get the rank object from this config.

        Returns:
            Rank object
        """
        if self.rank:
            return self.rank
        return Rank()


def print_parsed_files_summary(parsed_log_dir: str) -> None:
    """
    Print a beautiful summary of all parsed files.

    Args:
        parsed_log_dir: Directory containing parsed files
    """
    # Collect all parsed files
    all_parsed_files = []
    for root, _, files in os.walk(parsed_log_dir):
        for file in files:
            file_path = os.path.join(root, file)
            all_parsed_files.append(file_path)

    # Sort files for consistent output
    all_parsed_files.sort()

    # Print beautiful summary
    print("\n" + "=" * 80)
    print("📁 TRITONPARSE PARSING RESULTS")
    print("=" * 80)

    # Print log file list (required for integration)
    print(f"📂 Parsed files directory: {parsed_log_dir}")
    print(f"📊 Total files generated: {len(all_parsed_files)}")

    if all_parsed_files:
        print("\n📄 Generated files:")
        print("-" * 50)
        for i, file_path in enumerate(all_parsed_files, 1):
            # Get relative path for cleaner display
            rel_path = os.path.relpath(file_path, parsed_log_dir)
            file_size = "N/A"
            try:
                size_bytes = os.path.getsize(file_path)
                if size_bytes < 1024:
                    file_size = f"{size_bytes}B"
                elif size_bytes < 1024 * 1024:
                    file_size = f"{size_bytes / 1024:.1f}KB"
                else:
                    file_size = f"{size_bytes / (1024 * 1024):.1f}MB"
            except OSError:
                pass

            print(f"  {i:2d}. 📝 {rel_path} ({file_size})")

    print("=" * 80)
    print("✅ Parsing completed successfully!")
    print("=" * 80 + "\n")


def compress_single_file(
    file_path: str,
    compression: str = "gzip",
    verbose: bool = False,
) -> str:
    """
    Compress a single file and delete the original file.
    Args:
        file_path: Path to the file to compress
        compression: Compression algorithm to use ("gzip" or "zstd")
        verbose: Whether to print verbose information
    Returns:
        Path to the compressed file
    """
    if compression == "gzip":
        compressed_path = file_path + ".gz"
    elif compression == "zstd":
        compressed_path = file_path + ".zst"
    else:
        raise ValueError(f"Unsupported compression: {compression}")

    if verbose:
        logger.info(f"Compressing {file_path} with {compression}")

    with open(file_path, "rb") as f_in:
        if compression == "gzip":
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            cctx = zstd.ZstdCompressor()
            with open(compressed_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)

    # Delete the original file after successful compression
    if os.path.exists(file_path):
        os.remove(file_path)
    if verbose:
        logger.info(f"Deleted original file {file_path}")

    return compressed_path


def copy_local_to_tmpdir(local_path: str, verbose: bool = False) -> str:
    """
    Copy local log files to a temporary directory.

    Args:
        local_path: Path to local directory or single file containing logs
        verbose: Whether to print verbose information

    Returns:
        Path to temporary directory containing copied logs

    Raises:
        RuntimeError: If the local_path does not exist
    """
    if not os.path.exists(local_path):
        raise RuntimeError(f"Path does not exist: {local_path}")

    temp_dir = tempfile.mkdtemp()

    # Handle single file case
    if os.path.isfile(local_path):
        if os.path.basename(local_path).startswith(LOG_PREFIX):
            if verbose:
                logger.info(f"Copying single file {local_path} to {temp_dir}")
            shutil.copy2(local_path, temp_dir)
            return temp_dir
        else:
            raise RuntimeError(
                f"No eligible trace logs found. File '{local_path}' "
                f"does not start with expected prefix '{LOG_PREFIX}'."
            )

    # Handle directory case
    if not os.path.isdir(local_path):
        raise RuntimeError(f"Path is neither a file nor a directory: {local_path}")

    for item in os.listdir(local_path):
        item_path = os.path.join(local_path, item)
        if os.path.isfile(item_path) and os.path.basename(item_path).startswith(
            LOG_PREFIX
        ):
            if verbose:
                logger.info(f"Copying {item_path} to {temp_dir}")
            shutil.copy2(item_path, temp_dir)
        if os.path.isdir(item_path) and os.path.basename(item_path).startswith(
            LOG_PREFIX
        ):
            dir_name = os.path.basename(item_path)
            if verbose:
                logger.info(f"Copying {item_path} to {temp_dir}/{dir_name}")
            shutil.copytree(item_path, f"{temp_dir}/{dir_name}")

    # Check if any files were copied - fail fast with clear error message
    if not os.listdir(temp_dir):
        raise RuntimeError(
            f"No eligible trace logs found in '{local_path}'. "
            f"Expected files with names starting with '{LOG_PREFIX}'. "
            f"Found files: {os.listdir(local_path)}"
        )

    return temp_dir


def _build_kernel_compile_mapping(
    raw_log_dir: str,
    torch_trace_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build kernel compile mapping from inductor's torch trace logs.

    Searches for torch trace log files and parses them to extract
    kernel_source_path -> CompileInfo mappings. These mappings allow
    attribution of Triton kernels to their originating compilation frame
    when pt_info is missing (multi-process Triton JIT scenarios).

    Args:
        raw_log_dir: Directory containing tritonparse logs (used for auto-discovery).
        torch_trace_dir: Explicit directory containing torch trace logs.
            If None, auto-discovers in raw_log_dir.

    Returns:
        Dict mapping kernel source paths to CompileInfo, or None if no logs found.
    """
    from .torch_trace_parser import discover_torch_trace_files, parse_torch_trace_logs

    # Determine where to look for torch trace logs
    search_dirs = []
    if torch_trace_dir:
        search_dirs.append(torch_trace_dir)
    # Also check the raw log directory (torch trace logs may coexist)
    search_dirs.append(raw_log_dir)

    all_log_paths: List[str] = []
    seen_paths: set = set()
    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue
        torch_files = discover_torch_trace_files(search_dir)
        for rank_files in torch_files.values():
            for path in rank_files:
                if path not in seen_paths:
                    all_log_paths.append(path)
                    seen_paths.add(path)

    if not all_log_paths:
        return None

    mapping = parse_torch_trace_logs(all_log_paths)
    if mapping:
        logger.info(
            f"Built kernel compile mapping with {len(mapping)} entries "
            f"from {len(all_log_paths)} torch trace log(s)"
        )
    return mapping if mapping else None


def parse_logs(
    logs_to_parse: str,
    rank_config: RankConfig,
    verbose: bool = False,
    tritonparse_url_prefix: str = "",
    split_inductor_compilations: bool = True,
    torch_trace_dir: Optional[str] = None,
    procedure_checks: list = None,
    enable_pre_init_attribution: bool = True,
) -> Tuple[str, dict]:
    """
    Parse logs.

    Args:
        logs_to_parse: Path to directory containing logs to parse
        rank_config: Rank configuration
        verbose: Whether to print verbose information
        tritonparse_url_prefix: URL prefix for the generated file mapping
        split_inductor_compilations: Whether to split
            output files by frame_id, compile_id, attempt_id, and compiled_autograd_id.
            Defaults to True. This rule follows tlparse's behavior.
        torch_trace_dir: Optional path to directory containing inductor torch trace
            logs. When provided, kernel compilation attribution will use these logs to
            recover frame_id/compile_id for kernels compiled in multi-process scenarios.
            If None, auto-discovers torch trace files in the same directory as tritonparse logs.
        procedure_checks: List of procedure check configurations for FileCheck-based
            pattern detection. If None, uses default patterns.
        enable_pre_init_attribution: When True (default), no-rank trace files
            whose PID also appears in a ranked file are re-attributed to
            that rank — so kernels compiled before `dist.init_process_group`
            are visible under their owning rank in `--rank N` / `--all-ranks`
            output. Disable via the CLI flag `--no-pre-init-attribution` (or
            this kwarg) to debug the boundary between pre/post-init kernels.
            `--rank none` always shows the unattributed view regardless of
            this flag.
    Returns:
        Tuple of (parsed log directory, file mapping)
    """

    raw_log_dir = logs_to_parse
    parsed_log_dir = tempfile.mkdtemp()

    buckets = _collect_and_bucket_files(
        raw_log_dir, rank_config, enable_pre_init_attribution
    )
    if not buckets:
        # Scan for files of ANY rank to give a helpful error message.
        all_available: dict[int, list[str]] = defaultdict(list)
        for item in os.listdir(raw_log_dir):
            path = os.path.join(raw_log_dir, item)
            if not os.path.isfile(path) or LOG_PREFIX not in item:
                continue
            meta = parse_trace_filename_metadata(item)
            if meta.rank is not None:
                all_available[meta.rank].append(path)
        if all_available:
            raise RuntimeError(
                f"Rank {rank_config.to_rank().to_int()} not found. "
                f"Available ranks: {sorted(all_available.keys())}. "
                f"Use --rank N to select a specific rank, or --all-ranks."
            )
        else:
            raise RuntimeError(
                f"No eligible structured trace logs found in {raw_log_dir}"
            )

    # Build kernel compile mapping from torch trace logs (if available)
    kernel_compile_mapping = _build_kernel_compile_mapping(raw_log_dir, torch_trace_dir)

    file_mapping = {"tritonparse_url_prefix": tritonparse_url_prefix}
    # Process one rank at a time. parse_single_rank merges events across all
    # PID files in the bucket by kernel_hash, so per-frame outputs no longer
    # collide between PIDs (the §3.5 finding A scenario).
    for rank, files in buckets.items():
        if rank.is_default:
            rank_key = "rank_default"
        elif rank.is_no_rank:
            rank_key = "rank_none"
        else:
            rank_key = f"rank_{rank.value}"

        # No-rank files go directly to parsed_log_dir (no subdirectory).
        relative_path = "" if rank.is_no_rank else rank.to_string("")
        output_dir = os.path.join(parsed_log_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        parse_single_rank(
            files,
            output_dir,
            split_inductor_compilations,
            kernel_compile_mapping=kernel_compile_mapping,
            procedure_checks=procedure_checks,
        )

        # Collect generated files and gzip them immediately.
        if os.path.exists(output_dir):
            generated_files = []
            mapped_file = None

            for generated_item in os.listdir(output_dir):
                generated_path = os.path.join(output_dir, generated_item)
                if os.path.isfile(generated_path):
                    gz_file_path = compress_single_file(generated_path, verbose=verbose)
                    gz_filename = os.path.basename(gz_file_path)
                    if "mapped" in generated_item.lower():
                        mapped_file = gz_filename
                    else:
                        generated_files.append(gz_filename)

            if rank_key not in file_mapping:
                file_mapping[rank_key] = {"regular_files": [], "mapped_file": None}
            file_mapping[rank_key]["regular_files"].extend(generated_files)
            file_mapping[rank_key]["rank_suffix"] = rank_config.to_rank().to_string(
                suffix="/"
            )
            if mapped_file:
                file_mapping[rank_key]["mapped_file"] = mapped_file

    # Clean up the file mapping - remove None mapped_files and ensure no duplicates
    for rank_key, rank_data in file_mapping.items():
        if rank_key != "tritonparse_url_prefix":
            # Remove duplicates from regular_files
            rank_data["regular_files"] = list(set(rank_data["regular_files"]))
            # Remove mapped_file if None
            if rank_data["mapped_file"] is None:
                del rank_data["mapped_file"]
    # Save file mapping to parsed_log_dir
    log_file_list_path = os.path.join(parsed_log_dir, "log_file_list.json")
    with open(log_file_list_path, "w") as f:
        f.write(dumps(file_mapping, indent=True))

    # NOTICE: this print is required for tlparser-tritonparse integration
    # DON'T REMOVE THIS PRINT
    print(f"tritonparse log file list: {log_file_list_path}")
    return parsed_log_dir, file_mapping


def save_logs(out_dir: Path, parsed_logs: str, overwrite: bool, verbose: bool) -> None:
    """
    Save logs to a local directory.

    Args:
        out_dir: Path to output directory
        parsed_logs: Path to directory containing parsed logs
        overwrite: Whether to overwrite existing logs
        verbose: Whether to print verbose information
    """
    if not out_dir.is_absolute():
        out_dir = out_dir.resolve()

    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Copying parsed logs from {parsed_logs} to {out_dir}")

    # Copy each item in the parsed_logs directory to the output directory
    for item in os.listdir(parsed_logs):
        src_path = os.path.join(parsed_logs, item)
        dst_path = os.path.join(out_dir, item)

        if os.path.isdir(src_path):
            if verbose:
                logger.info(f"Copying directory {src_path}/ to {dst_path}/")
            shutil.copytree(src_path, dst_path)
        else:
            if verbose:
                logger.info(f"Copying file from {src_path} to {dst_path}")
            shutil.copy2(src_path, dst_path)
