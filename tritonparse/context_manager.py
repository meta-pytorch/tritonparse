#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import tempfile
from typing import Optional

from .parse.utils import unified_parse
from .shared_vars import TEST_KEEP_OUTPUT
from .structured_logging import clear_logging_config, init


def createUniqueTempDirectory():
    return tempfile.mkdtemp()


class TritonParseManager:
    def __init__(
        self,
        enable_trace_launch=False,
        enable_trace_launch_within_profiling: bool = False,
        enable_more_tensor_information: bool = False,
        split_inductor_compilations=True,
        enable_tensor_blob_storage=False,
        enable_sass_dump: Optional[bool] = None,
        enable_full_python_source: Optional[bool] = None,
        tensor_storage_quota=None,
        compression: Optional[str] = None,
        tensor_save_skip_runs=None,
        tensor_save_max_runs=None,
        log_dir=None,
        keep_logs=False,
        **parse_kwargs,
    ):
        """
        Context manager for tritonparse workflow.

        Args:
            enable_trace_launch: Whether to enable trace launch
            enable_trace_launch_within_profiling: Whether to enable launch tracing only
                during torch.profiler's RECORD phase. This patches
                torch.profiler.schedule. Forwarded to init() unchanged, which applies
                its own semantics: enable_trace_launch and
                enable_trace_launch_within_profiling are mutually exclusive, and if
                both are set enable_trace_launch takes priority and ALL launches will
                be traced.
            enable_more_tensor_information: Whether to enable more tensor information
                logging (min, max, mean, std). Forwarded to init() unchanged; it only
                works when enable_trace_launch/TRITON_TRACE_LAUNCH is True.
            split_inductor_compilations: Whether to split inductor compilations in the output
            enable_tensor_blob_storage: Whether to enable tensor blob storage
            enable_sass_dump: Whether to enable NVIDIA SASS dumping. Forwarded to
                init() unchanged, which applies its own semantics: a truthy value
                forces SASS dumping on, while False/None leave the
                TRITONPARSE_DUMP_SASS environment variable in charge.
            enable_full_python_source: Whether to capture the whole Python source file
                of a kernel instead of only its function definition. Forwarded to
                init() with the same three-state semantics: True forces full-file
                capture, False forces function-only capture, and None (the default)
                leaves the TRITON_FULL_PYTHON_SOURCE environment variable in charge.
                Recommended for nested kernels, where the traced @triton.jit entry
                point is only a thin wrapper around the real kernel.
            tensor_storage_quota: Storage quota in bytes for tensor blobs (default: 100GB)
            compression: Compression format for trace files ("none", "gzip", "zstd", or
                "clp"). Forwarded to init() unchanged, which applies its own semantics:
                None (the default) means no override, so TRITON_TRACE_COMPRESSION
                decides and otherwise the module default "none" applies. Note that
                unlike the boolean switches above, init() lets the environment variable
                win over an explicitly passed value.
            tensor_save_skip_runs: Skip tensor blob saving for the first N kernel runs
            tensor_save_max_runs: Save tensor blobs for at most N kernel runs after skipping
            log_dir: Optional directory path to store raw trace logs. If not provided,
                a temporary directory will be created and cleaned up after parsing.
                If provided, the directory will be created if it doesn't exist and
                will NOT be cleaned up after parsing.
            keep_logs: Whether to keep the log directory after parsing. Only effective
                when log_dir is not provided (i.e., when using a temporary directory).
                When log_dir is provided, logs are always kept.
            **parse_kwargs: Additional keyword arguments to pass to unified_parse
        """
        self.enable_trace_launch = enable_trace_launch
        self.enable_trace_launch_within_profiling = enable_trace_launch_within_profiling
        self.enable_more_tensor_information = enable_more_tensor_information
        self.split_inductor_compilations = split_inductor_compilations
        self.enable_tensor_blob_storage = enable_tensor_blob_storage
        self.enable_sass_dump = enable_sass_dump
        self.enable_full_python_source = enable_full_python_source
        self.tensor_storage_quota = tensor_storage_quota
        self.compression = compression
        self.tensor_save_skip_runs = tensor_save_skip_runs
        self.tensor_save_max_runs = tensor_save_max_runs
        self.user_log_dir = log_dir
        self.keep_logs = keep_logs
        self.parse_kwargs = parse_kwargs
        self.dir_path = None
        self.output_link = None
        self._is_temp_log_dir = False  # Track if we created a temporary directory

    def __enter__(self):
        if self.user_log_dir:
            # User specified a log directory
            self.dir_path = self.user_log_dir
            os.makedirs(self.dir_path, exist_ok=True)
            self._is_temp_log_dir = False
        else:
            # Create a temporary directory
            self.dir_path = createUniqueTempDirectory()
            self._is_temp_log_dir = True

        # These are forwarded unconditionally because each of this manager's
        # defaults is already init()'s own default for that keyword: False for the
        # plain bools, and None -- "no override, leave the environment variable in
        # charge" -- for enable_sass_dump, enable_full_python_source and
        # compression. Passing the default through is therefore equivalent to
        # omitting the keyword, and every value keeps init()'s semantics.
        init_kwargs = {
            "enable_trace_launch": self.enable_trace_launch,
            "enable_trace_launch_within_profiling": self.enable_trace_launch_within_profiling,
            "enable_more_tensor_information": self.enable_more_tensor_information,
            "enable_tensor_blob_storage": self.enable_tensor_blob_storage,
            "enable_sass_dump": self.enable_sass_dump,
            "enable_full_python_source": self.enable_full_python_source,
            "compression": self.compression,
        }
        if self.tensor_storage_quota is not None:
            init_kwargs["tensor_storage_quota"] = self.tensor_storage_quota
        if self.tensor_save_skip_runs is not None:
            init_kwargs["tensor_save_skip_runs"] = self.tensor_save_skip_runs
        if self.tensor_save_max_runs is not None:
            init_kwargs["tensor_save_max_runs"] = self.tensor_save_max_runs

        init(self.dir_path, **init_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.output_link = unified_parse(
            source=self.dir_path,
            overwrite=True,
            split_inductor_compilations=self.split_inductor_compilations,
            **self.parse_kwargs,
        )
        clear_logging_config()

        # Decide whether to clean up the log directory
        # Only clean up if:
        # 1. The directory exists
        # 2. It's a temporary directory we created (not user-specified)
        # 3. TEST_KEEP_OUTPUT is not set
        # 4. User didn't explicitly request to keep logs
        should_cleanup = (
            os.path.exists(self.dir_path)
            and self._is_temp_log_dir
            and not TEST_KEEP_OUTPUT
            and not self.keep_logs
        )
        if should_cleanup:
            shutil.rmtree(self.dir_path)
