from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IRStageDescriptor:
    name: str
    extension: str
    display_name: str
    display_order: int
    is_text: bool
    supports_source_mapping: bool
    parser_id: str
    syntax_id: str


@dataclass(frozen=True)
class DerivedArtifactDescriptor:
    source_extension: str
    output_stage_name: str
    output_extension: str
    tool_name: str


@dataclass(frozen=True)
class AnalysisPassDescriptor:
    name: str
    required_stages: tuple[str, ...]


class CompilationPipelineAdapter(ABC):
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

    def classify_artifact(self, artifact_name: str) -> IRStageDescriptor | None:
        artifact_suffix = Path(artifact_name).suffix
        for stage in self.get_ir_stages():
            if stage.extension == artifact_suffix:
                return stage
        return None

    @property
    def known_stage_extensions(self) -> set[str]:
        return {stage.extension for stage in self.get_ir_stages()}

    def get_derived_artifacts(self) -> list[DerivedArtifactDescriptor]:
        return []

    def get_analysis_passes(self) -> list[AnalysisPassDescriptor]:
        return []

    def normalize_device_string(self, device: str) -> str:
        return device


class NvidiaTritonAdapter(CompilationPipelineAdapter):
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
        return [
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
        return [
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


class PipelineAdapterRegistry:
    def __init__(self) -> None:
        self._adapter_types: dict[str, type[CompilationPipelineAdapter]] = {}

    def register(self, adapter_cls: type[CompilationPipelineAdapter]) -> None:
        adapter = adapter_cls()
        self._adapter_types[adapter.adapter_name.lower()] = adapter_cls

    def create_all(self) -> list[CompilationPipelineAdapter]:
        return [adapter_cls() for adapter_cls in self._adapter_types.values()]

    def resolve(
        self,
        *,
        adapter_name: str,
    ) -> CompilationPipelineAdapter:
        adapter_cls = self._adapter_types.get(adapter_name.lower())
        if adapter_cls is None:
            raise ValueError(
                "Unable to resolve adapter from adapter_name: "
                f"adapter_name={adapter_name!r}"
            )
        return adapter_cls()

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


def _deserialize_stage_descriptor(raw_stage: dict[str, Any]) -> IRStageDescriptor:
    _validate_stage_dict(raw_stage)
    return IRStageDescriptor(
        name=str(raw_stage["name"]),
        extension=str(raw_stage["extension"]),
        display_name=str(raw_stage["display_name"]),
        display_order=int(raw_stage["display_order"]),
        is_text=bool(raw_stage["is_text"]),
        supports_source_mapping=bool(raw_stage["supports_source_mapping"]),
        parser_id=str(raw_stage["parser_id"]),
        syntax_id=str(raw_stage["syntax_id"]),
    )


def _validate_stage_dict(raw_stage: dict[str, Any]) -> None:
    """Strictly validate a raw stage dict from metadata.

    Raises ValueError if required fields are missing or have invalid types.
    """
    if not isinstance(raw_stage, dict):
        raise ValueError("stage descriptor must be a dict")

    required_fields = [
        "name",
        "extension",
        "display_name",
        "display_order",
        "is_text",
        "supports_source_mapping",
        "parser_id",
        "syntax_id",
    ]

    missing = [f for f in required_fields if f not in raw_stage]
    if missing:
        raise ValueError(f"Missing required stage descriptor fields: {missing}")

    if not isinstance(raw_stage["name"], str):
        raise ValueError("stage 'name' must be a string")
    if not isinstance(raw_stage["extension"], str):
        raise ValueError("stage 'extension' must be a string")
    if not isinstance(raw_stage["display_name"], str):
        raise ValueError("stage 'display_name' must be a string")
    if "display_order" in raw_stage:
        try:
            int(raw_stage["display_order"])
        except Exception:
            raise ValueError("stage 'display_order' must be an integer or integer-like")
    if not isinstance(raw_stage["parser_id"], str):
        raise ValueError("stage 'parser_id' must be a string")
    if not isinstance(raw_stage["syntax_id"], str):
        raise ValueError("stage 'syntax_id' must be a string")


def deserialize_stage_descriptors_from_event(
    event: dict[str, Any],
) -> list[IRStageDescriptor]:
    payload = event.get("payload", {})
    metadata = payload.get("metadata", {})
    serialized_stage_descriptors = metadata.get("stage_descriptors")
    if not (
        isinstance(serialized_stage_descriptors, list) and serialized_stage_descriptors
    ):
        return []
    stages = [
        _deserialize_stage_descriptor(raw_stage)
        for raw_stage in serialized_stage_descriptors
        if isinstance(raw_stage, dict)
    ]
    return sorted(stages, key=lambda stage: stage.display_order)


_REGISTRY = PipelineAdapterRegistry()
for _adapter_cls in (NvidiaTritonAdapter, AmdTritonAdapter):
    _REGISTRY.register(_adapter_cls)


def get_backend_registry() -> PipelineAdapterRegistry:
    return _REGISTRY
