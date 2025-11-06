#  Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import shutil
import tempfile

from .shared_vars import TEST_KEEP_OUTPUT
from .structured_logging import clear_logging_config, init
from .utils import unified_parse


def createUniqueTempDirectory():
    return tempfile.mkdtemp()


class TritonParseManager:
    def __init__(
        self,
        enable_trace_launch=False,
        split_inductor_compilations=True,
        enable_tensor_blob_storage=False,
        tensor_storage_quota=None,
        **parse_kwargs,
    ):
        """
        Context manager for tritonparse workflow.

        Args:
            enable_trace_launch: Whether to enable trace launch
            split_inductor_compilations: Whether to split inductor compilations in the output
            enable_tensor_blob_storage: Whether to enable tensor blob storage
            tensor_storage_quota: Storage quota in bytes for tensor blobs (default: 100GB)
            **parse_kwargs: Additional keyword arguments to pass to unified_parse
        """
        self.enable_trace_launch = enable_trace_launch
        self.split_inductor_compilations = split_inductor_compilations
        self.enable_tensor_blob_storage = enable_tensor_blob_storage
        self.tensor_storage_quota = tensor_storage_quota
        self.parse_kwargs = parse_kwargs
        self.dir_path = None
        self.output_link = None

    def __enter__(self):
        self.dir_path = createUniqueTempDirectory()
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
        if os.path.exists(self.dir_path) and not TEST_KEEP_OUTPUT:
            shutil.rmtree(self.dir_path)
