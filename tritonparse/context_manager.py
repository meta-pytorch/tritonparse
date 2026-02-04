#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import tempfile

from .parse.utils import unified_parse
from .shared_vars import TEST_KEEP_OUTPUT
from .structured_logging import clear_logging_config, init


def createUniqueTempDirectory():
    return tempfile.mkdtemp()


class TritonParseManager:
    def __init__(
        self,
        enable_trace_launch=False,
        split_inductor_compilations=True,
        enable_tensor_blob_storage=False,
        tensor_storage_quota=None,
        log_dir=None,
        keep_logs=False,
        **parse_kwargs,
    ):
        """
        Context manager for tritonparse workflow.

        Args:
            enable_trace_launch: Whether to enable trace launch
            split_inductor_compilations: Whether to split inductor compilations in the output
            enable_tensor_blob_storage: Whether to enable tensor blob storage
            tensor_storage_quota: Storage quota in bytes for tensor blobs (default: 100GB)
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
        self.split_inductor_compilations = split_inductor_compilations
        self.enable_tensor_blob_storage = enable_tensor_blob_storage
        self.tensor_storage_quota = tensor_storage_quota
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

        init_kwargs = {
            "enable_trace_launch": self.enable_trace_launch,
            "enable_tensor_blob_storage": self.enable_tensor_blob_storage,
        }
        if self.tensor_storage_quota is not None:
            init_kwargs["tensor_storage_quota"] = self.tensor_storage_quota

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
