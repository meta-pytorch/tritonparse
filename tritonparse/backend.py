#  Copyright (c) Meta Platforms, Inc. and affiliates.
from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable

from tritonparse.tp_logger import logger


def normalize_accelerator_device_string(device: str) -> str:
    """Normalize accelerator device strings to index 0, preserving CPU."""
    if not isinstance(device, str):
        return "cpu"

    normalized = device.strip()
    if not normalized or normalized == "cpu":
        return "cpu"

    prefix = normalized.split(":")[0]
    return f"{prefix}:0"


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
        derive_func: Callable that takes a source file path and returns derived content,
            or None if the tool is unavailable or derivation fails.
    """

    source_stage_name: str
    target_stage_name: str
    tool_name: str
    derive_func: Callable[[str], str | None]


# =============================================================================
# INSTANCE-LEVEL REGISTRIES
# =============================================================================
class DerivedArtifactRegistry:
    """Per-adapter registry for managing derived artifact info objects."""

    def __init__(self) -> None:
        self._registry: dict[str, DerivedArtifactInfo] = {}

    def register(self, info: DerivedArtifactInfo) -> None:
        key = info.target_stage_name
        if key in self._registry:
            logger.debug(
                f"Overwriting existing derived artifact registration for '{key}'"
            )
        self._registry[key] = info

    def get_by_target(self, target_stage_name: str) -> DerivedArtifactInfo | None:
        return self._registry.get(target_stage_name)

    def list_all(self) -> list[DerivedArtifactInfo]:
        return list(self._registry.values())

    def list_target_stage_names(self) -> list[str]:
        return list(self._registry.keys())


class ParserRegistry:
    """Per-adapter registry for managing IR parser functions."""

    def __init__(self) -> None:
        self._parsers: dict[str, Callable] = {}

    def register(self, parser_id: str, parser_func: Callable) -> None:
        """Register a parser function with the given parser_id.

        If parser_id is already registered, the existing parser is overwritten
        and a warning is logged.
        """
        if parser_id in self._parsers:
            logger.warning(
                f"Parser '{parser_id}' is already registered. "
                f"Overwriting with new parser function."
            )
        self._parsers[parser_id] = parser_func
        logger.debug(f"Registered parser: {parser_id}")

    def get_parser(self, parser_id: str) -> Callable | None:
        """Get a parser function by parser_id."""
        return self._parsers.get(parser_id)

    def list_parsers(self) -> list[str]:
        """List all registered parser IDs."""
        return list(self._parsers.keys())


@dataclass
class AnalyzerContext:
    """Per-call context passed to analyzers. Extensible without changing signatures."""

    procedure_checks: list[dict[str, Any]] | None = None


@dataclass
class AnalyzerInfo:
    """Information about a registered analyzer.

    Attributes:
        name: Analyzer name (e.g., "amd_buffer_ops")
        func: Analyzer function with signature (entry, ctx) -> dict | None
        required_stages: Tuple of stage names required (e.g., ("ttgir", "amdgcn"))
    """

    name: str
    func: Callable[[dict[str, Any], AnalyzerContext], dict[str, Any] | None]
    required_stages: tuple[str, ...]


class AnalysisRegistry:
    """Per-adapter registry for managing IR analysis functions and their metadata."""

    def __init__(self) -> None:
        self._analyzer_infos: dict[str, AnalyzerInfo] = {}

    def register(
        self,
        analyzer_id: str,
        analyzer_func: Callable,
        required_stages: tuple[str, ...],
    ) -> None:
        """Register an analyzer with its metadata."""
        if analyzer_id in self._analyzer_infos:
            logger.debug(
                f"Analyzer '{analyzer_id}' is already registered. Overwriting."
            )
        info = AnalyzerInfo(
            name=analyzer_id,
            func=analyzer_func,
            required_stages=required_stages,
        )
        self._analyzer_infos[analyzer_id] = info

    def get_analyzer_info(self, analyzer_id: str) -> AnalyzerInfo | None:
        """Get analyzer info by name."""
        return self._analyzer_infos.get(analyzer_id)

    def get_analyzer(self, analyzer_id: str) -> Callable | None:
        """Get the analyzer function by name."""
        info = self._analyzer_infos.get(analyzer_id)
        return info.func if info else None

    def list_analyzers(self) -> list[str]:
        """List all registered analyzer IDs."""
        return list(self._analyzer_infos.keys())

    def list_analyzer_infos(self) -> list[tuple[str, AnalyzerInfo]]:
        """List all registered (analyzer_id, AnalyzerInfo) pairs."""
        return list(self._analyzer_infos.items())


# =============================================================================
# ADAPTER BASE CLASS
# =============================================================================
class CompilationPipelineAdapter(ABC):
    """Abstract base class for compilation pipeline adapters.

    Adapters provide backend-specific configuration for different
    compilation pipelines (e.g., CUDA vs HIP). Each adapter defines the
    IR stages, derived artifacts, and analysis passes for its pipeline.

    Each adapter instance holds its own isolated registries (parsers,
    analyzers, derived artifacts), ensuring complete backend isolation.
    """

    adapter_name: str
    runtime_backend: str
    pytorch_module: str

    def __init__(self):
        self._stages: list[IRStageDescriptor] = []
        self._parser_registry = ParserRegistry()
        self._analysis_registry = AnalysisRegistry()
        self._derived_artifact_registry = DerivedArtifactRegistry()

        # Register common parsers (shared across all backends)
        from tritonparse.parse.ir_parser import _parse_generic_loc, _parse_none

        self._parser_registry.register("generic_loc", _parse_generic_loc)
        self._parser_registry.register("none", _parse_none)

        # Register common analyzers (shared across all backends)
        from tritonparse.parse.ir_analysis import (
            _analyze_loop_schedules_generic,
            _analyze_procedures_generic,
        )

        self._analysis_registry.register(
            "loop_schedules",
            _analyze_loop_schedules_generic,
            required_stages=("ttir", "ttgir"),
        )
        self._analysis_registry.register(
            "procedure_checks",
            _analyze_procedures_generic,
            required_stages=("ttgir",),
        )

    def get_ir_stages(self) -> list[IRStageDescriptor]:
        return self._stages

    def get_stage_by_name(self, stage_name: str) -> IRStageDescriptor | None:
        for stage in self.get_ir_stages():
            if stage.name == stage_name:
                return stage
        return None

    def get_applicable_derived_artifacts(
        self,
        enabled_derived_artifacts: set[str] | None = None,
    ) -> list[DerivedArtifactInfo]:
        """
        Get derived artifacts applicable to this adapter, filtered by user-enabled list.

        Validates user-provided names and warns about unknowns.

        Args:
            enabled_derived_artifacts: User-enabled target stage names (None = all)

        Returns:
            List of applicable DerivedArtifactInfo
        """
        all_artifacts = self._derived_artifact_registry.list_all()

        if enabled_derived_artifacts is not None:
            enabled_normalized = {n.lower() for n in enabled_derived_artifacts}
            known = {info.target_stage_name.lower() for info in all_artifacts}
            unknown = enabled_normalized - known
            if unknown:
                logger.warning(
                    f"TRITONPARSE_DERIVED_ARTIFACTS contains unknown target stage names: {unknown}. "
                    f"Available for {self.adapter_name}: {sorted(known)}"
                )
            return [
                info
                for info in all_artifacts
                if info.target_stage_name.lower() in enabled_normalized
            ]

        return all_artifacts

    def register_backend_derived_artifact(
        self,
        source_stage_name: str,
        target_stage_name: str,
        tool_name: str,
        derive_func: Callable[[str], str | None],
    ) -> None:
        """Register a backend-specific derived artifact to the adapter's registry."""
        self._derived_artifact_registry.register(
            DerivedArtifactInfo(
                source_stage_name=source_stage_name,
                target_stage_name=target_stage_name,
                tool_name=tool_name,
                derive_func=derive_func,
            )
        )

    def list_parsers(self) -> list[str]:
        """List all registered parser IDs (common + backend-specific)."""
        return self._parser_registry.list_parsers()

    def get_analysis_passes(self) -> list[str]:
        """
        Return list of analysis pass names for this adapter.

        All analyzers in the adapter's instance registry are included
        (common + backend-specific).
        """
        return self._analysis_registry.list_analyzers()

    def get_analyzer_required_stages(
        self, analyzer_name: str
    ) -> tuple[str, ...] | None:
        """Return required stages for the given analyzer, or None if not registered."""
        info = self._analysis_registry.get_analyzer_info(analyzer_name)
        return info.required_stages if info else None

    def get_executable_analyzers(
        self,
        file_content: dict[str, str],
        enabled_analyses: set[str] | None = None,
    ) -> list[str]:
        """
        Get list of analyzers that can be executed based on:
        1. User-enabled analyses (from environment variable)
        2. Available intermediate products (file_content)

        Also validates user-provided analyzer names and warns about unknowns.

        Args:
            file_content: Dictionary mapping file keys to file content
            enabled_analyses: User-enabled analysis names (None = all)

        Returns:
            List of executable analyzer names
        """
        # Validate user-provided names against known analyzers
        enabled_normalized = (
            {n.lower() for n in enabled_analyses}
            if enabled_analyses is not None
            else None
        )
        if enabled_normalized is not None:
            known = {name.lower() for name in self._analysis_registry.list_analyzers()}
            unknown = enabled_normalized - known
            if unknown:
                logger.warning(
                    f"TRITONPARSE_ANALYSIS contains unknown analyzer names: {unknown}. "
                    f"Available for {self.adapter_name}: {sorted(known)}"
                )

        declared_analyzers = self.get_analysis_passes()
        executable = []

        for analyzer_name in declared_analyzers:
            # Check 1: Is it enabled by user?
            if (
                enabled_normalized is not None
                and analyzer_name.lower() not in enabled_normalized
            ):
                continue

            # Check 2: Are required stages available?
            info = self._analysis_registry.get_analyzer_info(analyzer_name)
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
        ctx: AnalyzerContext,
    ) -> dict[str, Any]:
        """
        Execute the specified analysis pass.

        Args:
            pass_name: Analysis name (e.g., "amd_buffer_ops", "loop_schedules")
            entry: Trace entry (contains payload)
            ctx: Per-call analyzer context

        Returns:
            Analysis result dictionary

        Raises:
            ValueError: If the pass_name is not found in the registry
        """

        analyzer = self._analysis_registry.get_analyzer(pass_name)
        if analyzer is None:
            available = self._analysis_registry.list_analyzers()
            raise ValueError(
                f"Analyzer '{pass_name}' not found. Available analyzers: {available}"
            )

        return analyzer(entry, ctx)

    def register_backend_analyzer(
        self,
        analyzer_id: str,
        analyzer_func,
        required_stages: tuple[str, ...],
    ) -> None:
        """
        Register a backend-specific analyzer to the adapter's registry.

        Args:
            analyzer_id: The analyzer identifier (e.g., "amd_buffer_ops")
            analyzer_func: The analyzer function with signature
                          (entry, ctx) -> dict | None
            required_stages: Required stage names (e.g., ("ttgir", "amdgcn"))
        """
        self._analysis_registry.register(analyzer_id, analyzer_func, required_stages)

    def get_canonical_device_string(self) -> str:
        """Return the adapter's canonical accelerator device string."""
        return normalize_accelerator_device_string(self.pytorch_module)

    def get_parser(self, parser_id: str):
        """
        Get parser function by parser_id from the adapter's parser registry.

        Args:
            parser_id: The parser identifier (e.g., "generic_loc", "ptx_loc")

        Returns:
            The parser function for the given parser_id

        Raises:
            ValueError: If the parser_id is not found in the registry
        """
        parser = self._parser_registry.get_parser(parser_id)
        if parser is None:
            available_parsers = self._parser_registry.list_parsers()
            raise ValueError(
                f"Parser '{parser_id}' not found. "
                f"Available parsers: {available_parsers}"
            )
        return parser

    def register_backend_parser(self, parser_id: str, parser_func) -> None:
        """
        Register a backend-specific parser to the adapter's parser registry.

        Args:
            parser_id: The parser identifier (e.g., "ascend_ir")
            parser_func: The parser function
        """
        self._parser_registry.register(parser_id, parser_func)


class NvidiaTritonAdapter(CompilationPipelineAdapter):
    adapter_name: str = "cuda_triton"
    runtime_backend: str = "cuda"
    pytorch_module: str = "cuda"

    def __init__(self):
        super().__init__()

        # Register NVIDIA-specific parsers
        from tritonparse.parse.ir_parser import _parse_ptx_loc, _parse_sass_loc

        self._parser_registry.register("ptx_loc", _parse_ptx_loc)
        self._parser_registry.register("sass_loc", _parse_sass_loc)

        # Register NVIDIA-specific derived artifacts
        from tritonparse.tools.disasm import extract as derive_sass

        self._derived_artifact_registry.register(
            DerivedArtifactInfo(
                source_stage_name="cubin",
                target_stage_name="sass",
                tool_name="nvdisasm",
                derive_func=derive_sass,
            )
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


class AmdTritonAdapter(CompilationPipelineAdapter):
    adapter_name: str = "hip_triton"
    runtime_backend: str = "hip"
    pytorch_module: str = "cuda"

    def __init__(self):
        super().__init__()

        # Register AMD-specific parsers
        from tritonparse.parse.ir_parser import _parse_amdgcn_loc

        self._parser_registry.register("amdgcn_loc", _parse_amdgcn_loc)

        # Register AMD-specific analyzers
        from tritonparse.parse.ir_analysis import _analyze_amd_buffer_ops

        self._analysis_registry.register(
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


# =============================================================================
# ADAPTER REGISTRY (LAZY INITIALIZATION)
# =============================================================================
class PipelineAdapterRegistry:
    """Registry for managing and resolving compilation pipeline adapters.

    Adapters are registered as classes and lazily instantiated on first use.
    This avoids importing backend-specific modules until they are needed.
    """

    def __init__(self) -> None:
        self._adapter_classes: dict[str, type[CompilationPipelineAdapter]] = {}
        self._adapter_instances: dict[str, CompilationPipelineAdapter] = {}

    def register(self, adapter_cls: type[CompilationPipelineAdapter]) -> None:
        key = adapter_cls.adapter_name.lower()
        self._adapter_classes[key] = adapter_cls

    def _ensure_initialized(self, adapter_name: str) -> None:
        key = adapter_name.lower()
        if key not in self._adapter_instances and key in self._adapter_classes:
            self._adapter_instances[key] = self._adapter_classes[key]()

    def resolve(
        self,
        *,
        adapter_name: str,
    ) -> CompilationPipelineAdapter:
        self._ensure_initialized(adapter_name)
        adapter = self._adapter_instances.get(adapter_name.lower())
        if adapter is None:
            available = list(self._adapter_classes.keys())
            raise ValueError(
                "Unable to resolve adapter from adapter_name: "
                f"adapter_name={adapter_name!r}. Available: {available}"
            )
        return adapter

    def resolve_from_backend_name(
        self,
        backend_name: str,
    ) -> CompilationPipelineAdapter:
        inferred_adapter_name = f"{backend_name}_triton"
        return self.resolve(adapter_name=inferred_adapter_name)

    def resolve_from_trace(
        self,
        metadata: dict[str, Any],
    ) -> CompilationPipelineAdapter:
        backend_name = metadata.get("backend_name")
        if isinstance(backend_name, str):
            return self.resolve_from_backend_name(backend_name)

        raise ValueError(
            "Unable to resolve adapter from trace metadata: "
            f"backend_name={backend_name!r}"
        )


# Module-level registration (no instantiation at import time)
_REGISTRY = PipelineAdapterRegistry()
_REGISTRY.register(NvidiaTritonAdapter)
_REGISTRY.register(AmdTritonAdapter)


def get_backend_registry() -> PipelineAdapterRegistry:
    return _REGISTRY
