#  Copyright (c) Meta Platforms, Inc. and affiliates.

import gzip
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

from tritonparse.shared_vars import (
    DEFAULT_TRACE_FILE_PREFIX_WITHOUT_USER as LOG_PREFIX,
    is_fbcode,
)
from tritonparse.tp_logger import logger

from .extract_source_mappings import parse_single_file

LOG_RANK_REGEX = re.compile(r"rank_(\d+)")


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


def gzip_single_file(file_path: str, verbose: bool = False) -> str:
    """
    Gzip a single file and delete the original file.
    Args:
        file_path: Path to the file to gzip
        verbose: Whether to print verbose information
    Returns:
        Path to the gzipped file
    """
    if file_path.endswith(".gz"):
        return file_path

    gz_file_path = file_path + ".gz"
    if verbose:
        logger.info(f"Gzipping {file_path}")

    with open(file_path, "rb") as f_in:
        with gzip.open(gz_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Delete the original file after successful compression
    os.remove(file_path)
    if verbose:
        logger.info(f"Deleted original file {file_path}")

    return gz_file_path


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


def parse_logs(
    logs_to_parse: str,
    rank_config: RankConfig,
    verbose: bool = False,
    tritonparse_url_prefix: str = "",
    split_inductor_compilations: bool = True,
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
    Returns:
        Tuple of (parsed log directory, file mapping)
    """

    raw_log_dir = logs_to_parse
    parsed_log_dir = tempfile.mkdtemp()
    # Dictionary to store ranks and their log files
    ranks = defaultdict(list)  # Dict[Rank, List[str]]
    # Find all eligible logs in the raw log directory
    for item in os.listdir(raw_log_dir):
        path = os.path.join(raw_log_dir, item)
        if not os.path.isfile(path):
            continue
        # Check if this is a tritonparse log file
        # Use 'in' instead of 'startswith' to support lg's --store-by-source format
        # which produces filenames like: log.source.dedicated_log_triton_trace_xxx_.ndjson
        if LOG_PREFIX not in item:
            continue

        # Check if the log has a rank in its name
        rank_match = LOG_RANK_REGEX.search(item)
        if rank_match:
            # File has rank suffix: dedicated_log_triton_trace_{USER}_rank_{N}_.ndjson.gz
            rank_value = int(rank_match.group(1))
            if rank_config.all_ranks:
                # --all-ranks: include all ranked files
                ranks[Rank(rank_value)].append(path)
            elif not rank_config.is_no_rank and rank_config.rank is not None:
                # --rank N: only include specified rank
                if rank_config.rank.value == rank_value:
                    ranks[Rank(rank_value)].append(path)
            elif rank_config.rank is None and not rank_config.all_ranks:
                # Default case (no --rank, no --all-ranks): include rank 0
                if rank_value == 0:
                    ranks[Rank(rank_value)].append(path)
        else:
            # File has no rank suffix: dedicated_log_triton_trace_{USER}_.ndjson.gz
            if rank_config.all_ranks or rank_config.is_no_rank:
                # --all-ranks or --rank none: include no-rank files
                ranks[Rank(Rank.NO_RANK)].append(path)
    if not ranks:
        raise RuntimeError(f"No eligible structured trace logs found in {raw_log_dir}")
    file_mapping = {"tritonparse_url_prefix": tritonparse_url_prefix}
    # Parse each eligible log
    for rank, files in ranks.items():
        use_filenames = False
        if len(files) > 1:
            logger.warning(
                "Warning: multiple logs found for the same rank. Using filenames."
            )
            use_filenames = True
        # Determine rank key for file mapping
        if rank.is_default:
            rank_key = "rank_default"
        elif rank.is_no_rank:
            rank_key = "rank_none"
        else:
            rank_key = f"rank_{rank.value}"
        for file_path in files:
            filename = os.path.basename(file_path)
            input_file = os.path.join(raw_log_dir, filename)

            relative_path = ""
            if use_filenames:
                # For no-rank files, don't create a subdirectory (same as default)
                rank_prefix = (
                    ""
                    if (rank.is_default or rank.is_no_rank)
                    else f"{rank.to_string('')}/"
                )
                relative_path = f"{rank_prefix}{filename}"
            else:
                # For no-rank files, output directly to parsed_log_dir (no subdirectory)
                relative_path = "" if rank.is_no_rank else rank.to_string("")
            output_dir = os.path.join(parsed_log_dir, relative_path)
            # Parse the file
            parse_single_file(input_file, output_dir, split_inductor_compilations)
            # Collect generated files after parsing and gzip them immediately
            if os.path.exists(output_dir):
                generated_files = []
                mapped_file = None

                for generated_item in os.listdir(output_dir):
                    generated_path = os.path.join(output_dir, generated_item)
                    if os.path.isfile(generated_path):
                        # Gzip the file immediately after parsing
                        gz_file_path = gzip_single_file(generated_path, verbose)
                        gz_filename = os.path.basename(gz_file_path)
                        # Check if it's a mapped file (assuming files with 'mapped' in name)
                        if "mapped" in generated_item.lower():
                            mapped_file = gz_filename
                        else:
                            generated_files.append(gz_filename)
                # Initialize rank entry if not exists
                if rank_key not in file_mapping:
                    file_mapping[rank_key] = {"regular_files": [], "mapped_file": None}
                # Add files to the mapping (now with .gz extensions)
                file_mapping[rank_key]["regular_files"].extend(generated_files)
                # this is used to generate the tritonparse url
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
        json.dump(file_mapping, f, indent=2)

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
