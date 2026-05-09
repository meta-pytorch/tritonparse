#  Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class IRStageDescriptor:
    """Describes an IR stage in the compilation pipeline.

    Fields:
        name: Internal stage identifier (e.g., "ttir", "ptx").
        extension: File extension for artifacts (e.g., ".ttir", ".ptx").
        display_name: Human-readable name for UI display.
        display_order: UI display order (lower values appear first).
        is_text: True if format is text, False if binary.
        supports_source_mapping: True if this stage supports source-to-source mapping.
        parser_id: Identifier for the parser to use.
        syntax_id: Syntax highlighting identifier for web display.
    """

    name: str
    extension: str
    display_name: str
    display_order: int
    is_text: bool
    supports_source_mapping: bool
    parser_id: str
    syntax_id: str


@dataclass
class DerivedArtifactInfo:
    """Describes an artifact derived by running tools on another stage's output.

    Fields:
        source_stage_name: Name of the source stage (e.g., "cubin").
        target_stage_name: Name of the target stage (e.g., "sass").
        tool_name: Name of the tool used to generate the artifact (e.g., "nvdisasm").
        adapter_affinity: Name of the adapter this derivation belongs to.
        derive_func: Callable that takes a source file path and returns derived content,
            or None if the tool is unavailable or derivation fails.
    """

    source_stage_name: str
    target_stage_name: str
    tool_name: str
    adapter_affinity: str
    derive_func: Callable[[str], str | None]


class DerivedArtifactRegistry:
    """Registry for managing derived artifact info objects.

    Adapters register their derived artifacts here. The registry supports
    lookup by target stage name and listing all known target stage names
    (useful for env-var validation).
    """

    _registry: dict[str, DerivedArtifactInfo] = {}

    @classmethod
    def register(cls, info: DerivedArtifactInfo) -> None:
        key = info.target_stage_name
        if key in cls._registry:
            import logging

            logging.getLogger("tritonparse").debug(
                f"Overwriting existing derived artifact registration for '{key}'"
            )
        cls._registry[key] = info

    @classmethod
    def get_by_target(cls, target_stage_name: str) -> DerivedArtifactInfo | None:
        return cls._registry.get(target_stage_name)

    @classmethod
    def list_for_adapter(cls, adapter_name: str) -> list[DerivedArtifactInfo]:
        return [
            info
            for info in cls._registry.values()
            if info.adapter_affinity == adapter_name
        ]

    @classmethod
    def list_target_stage_names(cls) -> list[str]:
        return list(cls._registry.keys())


class CompilationPipelineAdapter(ABC):
    """Abstract base class for compilation pipeline adapters.

    Adapters provide backend-specific configuration for different
    compilation pipelines (e.g., CUDA vs HIP). Each adapter defines the
    IR stages, derived artifacts, and analysis passes for its pipeline.
    """

    @property
    @abstractmethod
    def adapter_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def runtime_backend(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def pytorch_module(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_ir_stages(self) -> list[IRStageDescriptor]:
        raise NotImplementedError

    def get_stage_by_name(self, stage_name: str) -> IRStageDescriptor | None:
        for stage in self.get_ir_stages():
            if stage.name == stage_name:
                return stage
        return None

    def get_stage_by_artifact(self, artifact_name: str) -> IRStageDescriptor | None:
        artifact_suffix = Path(artifact_name).suffix
        for stage in self.get_ir_stages():
            if stage.extension == artifact_suffix:
                return stage
        return None

    @property
    def known_stage_extensions(self) -> set[str]:
        return {stage.extension for stage in self.get_ir_stages()}

    def get_derived_artifacts(self) -> list[DerivedArtifactInfo]:
        return DerivedArtifactRegistry.list_for_adapter(self.adapter_name)

    def register_backend_derived_artifact(
        self,
        source_stage_name: str,
        target_stage_name: str,
        tool_name: str,
        derive_func: Callable[[str], str | None],
    ) -> None:
        """Register a backend-specific derived artifact to the registry."""
        DerivedArtifactRegistry.register(
            DerivedArtifactInfo(
                source_stage_name=source_stage_name,
                target_stage_name=target_stage_name,
                tool_name=tool_name,
                adapter_affinity=self.adapter_name,
                derive_func=derive_func,
            )
        )

    def collect_derived_artifact_contents(
        self, source_path: str, info: DerivedArtifactInfo
    ) -> str | None:
        """Run the derivation tool and return the generated artifact contents."""
        return info.derive_func(source_path)

    def get_analysis_passes(self) -> list[str]:
        """
        Return list of analysis pass names for this adapter.

        Base implementation: gets analyzers from AnalysisRegistry
        that match this adapter's affinity or are common (no affinity).

        Can be overridden by subclasses for custom behavior.
        """
        from tritonparse.parse.ir_analysis import AnalysisRegistry

        my_name = self.adapter_name
        analyzer_names = []

        for _, info in AnalysisRegistry.list_analyzer_infos():
            # Include analyzers specific to this adapter or common analyzers
            if info.adapter_affinity in (my_name, None):
                analyzer_names.append(info.name)

        return analyzer_names

    def get_executable_analyzers(
        self,
        file_content: dict[str, str],
        enabled_analyses: set[str] | None = None,
    ) -> list[str]:
        """
        Get list of analyzers that can be executed based on:
        1. User-enabled analyses (from environment variable)
        2. Available intermediate products (file_content)

        Args:
            file_content: Dictionary mapping file keys to file content
            enabled_analyses: User-enabled analysis names (None = all)

        Returns:
            List of executable analyzer names
        """
        from tritonparse.parse.ir_analysis import AnalysisRegistry

        # Get declared analyzers for this adapter (already registered)
        declared_analyzers = self.get_analysis_passes()
        executable = []

        for analyzer_name in declared_analyzers:
            # Check 1: Is it enabled by user?
            if enabled_analyses is not None and analyzer_name not in enabled_analyses:
                continue

            # Check 2: Are required stages available?
            info = AnalysisRegistry.get_analyzer_info(analyzer_name)
            if not info:
                continue

            stages_available = True
            for stage_name in info.required_stages:
                stage = self.get_stage_by_name(stage_name)
                if not stage:
                    stages_available = False
                    break

                # Check if any file in file_content has this stage's extension
                if not any(k.endswith(stage.extension) for k in file_content):
                    stages_available = False
                    break

            if stages_available:
                executable.append(analyzer_name)

        return executable

    def run_analysis_pass(
        self,
        pass_name: str,
        entry: dict,
        procedure_checks: list | None = None,
    ) -> dict[str, Any]:
        """
        Execute the specified analysis pass.

        Args:
            pass_name: Analysis name (e.g., "amd_buffer_ops", "loop_schedules")
            entry: Trace entry (contains payload)
            procedure_checks: Procedure checks configuration

        Returns:
            Analysis result dictionary

        Raises:
            ValueError: If the pass_name is not found in the registry
        """
        from tritonparse.parse.ir_analysis import AnalysisRegistry

        analyzer = AnalysisRegistry.get_analyzer(pass_name)
        if analyzer is None:
            available = AnalysisRegistry.list_analyzers()
            raise ValueError(
                f"Analyzer '{pass_name}' not found. Available analyzers: {available}"
            )

        return analyzer(entry, procedure_checks)

    def register_backend_analyzer(
        self,
        analyzer_id: str,
        analyzer_func,
        required_stages: tuple[str, ...],
    ) -> None:
        """
        Register a backend-specific analyzer to the analyzer registry.

        This allows adapters to register custom analyzers for backend-specific
        analysis passes that are not part of the common analyzer registry.

        Args:
            analyzer_id: The analyzer identifier (e.g., "amd_buffer_ops")
            analyzer_func: The analyzer function with signature
                          (entry, procedure_checks) -> dict | None
            required_stages: Required stage names (e.g., ("ttgir", "amdgcn"))
        """
        from tritonparse.parse.ir_analysis import AnalysisRegistry

        # Use this adapter's name as affinity
        adapter_affinity = self.adapter_name
        AnalysisRegistry.register(
            analyzer_id, analyzer_func, required_stages, adapter_affinity
        )

    def normalize_device_string(self, device: str) -> str:
        return device

    def get_parser(self, parser_id: str):
        """
        Get parser function by parser_id from the parser registry.

        This is a generic implementation that works for most backends.
        Subclasses can override this method if they need custom parser resolution.

        Args:
            parser_id: The parser identifier (e.g., "generic_loc", "ptx_loc")

        Returns:
            The parser function for the given parser_id

        Raises:
            ValueError: If the parser_id is not found in the registry
        """
        from tritonparse.parse.ir_parser import ParserRegistry

        parser = ParserRegistry.get_parser(parser_id)
        if parser is None:
            available_parsers = ParserRegistry.list_parsers()
            raise ValueError(
                f"Parser '{parser_id}' not found. "
                f"Available parsers: {available_parsers}"
            )
        return parser

    def register_backend_parser(self, parser_id: str, parser_func) -> None:
        """
        Register a backend-specific parser to the parser registry.

        This allows adapters to register custom parsers for backend-specific
        IR formats that are not part of the common parser registry.

        Args:
            parser_id: The parser identifier (e.g., "ascend_ir")
            parser_func: The parser function
        """
        from tritonparse.parse.ir_parser import ParserRegistry

        ParserRegistry.register(parser_id, parser_func)


class NvidiaTritonAdapter(CompilationPipelineAdapter):
    def __init__(self):
        """Initialize and register backend-specific parsers and stage descriptors."""
        from tritonparse.parse.ir_parser import _parse_ptx_loc, _parse_sass_loc

        # Register NVIDIA-specific parsers
        self.register_backend_parser("ptx_loc", _parse_ptx_loc)
        self.register_backend_parser("sass_loc", _parse_sass_loc)

        # Register NVIDIA-specific derived artifacts
        from tritonparse.tools.disasm import extract as derive_sass

        self.register_backend_derived_artifact("cubin", "sass", "nvdisasm", derive_sass)

        # Pre-initialize stage descriptors (immutable objects, can be reused)
        self._stages = [
            IRStageDescriptor(
                "ttir", ".ttir", "TTIR", 10, True, True, "generic_loc", "mlir"
            ),
            IRStageDescriptor(
                "ttgir", ".ttgir", "TTGIR", 20, True, True, "generic_loc", "mlir"
            ),
            IRStageDescriptor(
                "llir", ".llir", "LLIR", 30, True, True, "generic_loc", "llvm"
            ),
            IRStageDescriptor("ptx", ".ptx", "PTX", 40, True, True, "ptx_loc", "ptx"),
            IRStageDescriptor(
                "cubin", ".cubin", "CUBIN", 50, False, False, "none", "plaintext"
            ),
            IRStageDescriptor(
                "sass", ".sass", "SASS", 60, True, True, "sass_loc", "asm"
            ),
            IRStageDescriptor(
                "json", ".json", "JSON", 100, True, False, "none", "json"
            ),
        ]

    @property
    def adapter_name(self) -> str:
        return "cuda_triton"

    @property
    def runtime_backend(self) -> str:
        return "cuda"

    @property
    def pytorch_module(self) -> str:
        return "torch.cuda"

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return self._stages


class AmdTritonAdapter(CompilationPipelineAdapter):
    def __init__(self):
        """Initialize and register backend-specific parsers and analyzers."""
        from tritonparse.parse.ir_analysis import _analyze_amd_buffer_ops
        from tritonparse.parse.ir_parser import _parse_amdgcn_loc

        # Register AMD-specific parsers
        self.register_backend_parser("amdgcn_loc", _parse_amdgcn_loc)

        # Register AMD-specific analyzers
        self.register_backend_analyzer(
            "amd_buffer_ops",
            _analyze_amd_buffer_ops,
            required_stages=("ttgir", "amdgcn"),
        )

        # Pre-initialize stage descriptors (immutable objects, can be reused)
        self._stages = [
            IRStageDescriptor(
                "ttir", ".ttir", "TTIR", 10, True, True, "generic_loc", "mlir"
            ),
            IRStageDescriptor(
                "ttgir", ".ttgir", "TTGIR", 20, True, True, "generic_loc", "mlir"
            ),
            IRStageDescriptor(
                "llir", ".llir", "LLIR", 30, True, True, "generic_loc", "llvm"
            ),
            IRStageDescriptor(
                "amdgcn", ".amdgcn", "AMDGCN", 40, True, True, "amdgcn_loc", "asm"
            ),
            IRStageDescriptor(
                "json", ".json", "JSON", 100, True, False, "none", "json"
            ),
        ]

    @property
    def adapter_name(self) -> str:
        return "hip_triton"

    @property
    def runtime_backend(self) -> str:
        return "hip"

    @property
    def pytorch_module(self) -> str:
        return "torch.cuda"

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return self._stages


class PipelineAdapterRegistry:
    """Registry for managing and resolving compilation pipeline adapters.

    Provides registration and resolution methods for different backend adapters.
    Adapters can be looked up by name or inferred from trace metadata.

    Note: Adapters are instantiated once during registration and reused for all
    subsequent resolutions. This ensures parsers are registered only once and
    stage descriptors are initialized only once.
    """

    def __init__(self) -> None:
        self._adapters: dict[str, CompilationPipelineAdapter] = {}

    def register(self, adapter_cls: type[CompilationPipelineAdapter]) -> None:
        adapter = adapter_cls()
        self._adapters[adapter.adapter_name.lower()] = adapter

    def create_all(self) -> list[CompilationPipelineAdapter]:
        return list(self._adapters.values())

    def resolve(
        self,
        *,
        adapter_name: str,
    ) -> CompilationPipelineAdapter:
        adapter = self._adapters.get(adapter_name.lower())
        if adapter is None:
            raise ValueError(
                "Unable to resolve adapter from adapter_name: "
                f"adapter_name={adapter_name!r}"
            )
        return adapter

    def resolve_from_trace(
        self,
        metadata: dict[str, Any],
    ) -> CompilationPipelineAdapter:
        backend_name = metadata.get("backend_name")
        if isinstance(backend_name, str):
            inferred_adapter_name = f"{backend_name}_triton"
            return self.resolve(adapter_name=inferred_adapter_name)

        raise ValueError(
            "Unable to resolve adapter from trace metadata: "
            f"backend_name={backend_name!r}"
        )


_REGISTRY = PipelineAdapterRegistry()
for _adapter_cls in (NvidiaTritonAdapter, AmdTritonAdapter):
    _REGISTRY.register(_adapter_cls)


def get_backend_registry() -> PipelineAdapterRegistry:
    return _REGISTRY
