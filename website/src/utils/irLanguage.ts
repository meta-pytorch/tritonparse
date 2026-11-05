import type { IRStageDescriptor } from "./dataLoader";

/**
 * Utility functions for IR language display
 */

/**
 * Get a user-friendly display name for the IR language.
 * When irStages is available, looks up the display_name from stage descriptors.
 * Otherwise falls back to hardcoded legacy logic for backward compatibility.
 */
export const getDisplayLanguage = (irType: string, irStages?: IRStageDescriptor[]): string => {
  // Try dynamic lookup from ir_stages. This mirrors
  // dataLoader's getIRStageDescriptor (match by artifact extension, so a
  // stage whose name differs from its extension still resolves), except
  // the longest extension wins: "kernel.clean_mlir" also ends with ".mlir".
  // Kept local so this module stays importable by the node unit tests,
  // which cannot load dataLoader's runtime dependencies.
  if (irStages && irStages.length > 0) {
    const lower = irType.toLowerCase();
    const stage = [...irStages]
      .sort((a, b) => b.extension.length - a.extension.length)
      .find((s) => s.extension && lower.endsWith(s.extension.toLowerCase()));
    if (stage) return stage.display_name;
  }

  // Fallback: legacy hardcoded logic
  const lower = irType.toLowerCase();
  if (lower.endsWith("ttgir")) {
    return "TTGIR (TritonGPU MLIR)";
  } else if (lower.endsWith("ttir")) {
    return "TTIR (Triton MLIR)";
  } else if (lower.endsWith("llir")) {
    return "LLIR (LLVM IR)";
  } else if (lower.endsWith("ptx")) {
    return "PTX (NVIDIA Parallel Thread Execution)";
  } else if (lower.endsWith("cubin")) {
    return "CUBIN (NVIDIA CUDA Binary)";
  } else if (lower.endsWith("python")) {
    return "Python";
  } else if (lower.endsWith("json")) {
    return "JSON";
  } else if (lower.endsWith("amdgcn")) {
    return "AMDGCN (AMD GCN Assembly)";
  } else if (lower.endsWith("sass")) {
    return "SASS (NVIDIA Shader Assembly)";
  } else if (lower.endsWith("clean_mlir")) {
    return "Clean MLIR (CuTe DSL)";
  } else if (lower.endsWith("mlir")) {
    return "MLIR (CuTe DSL)";
  } else {
    return irType;
  }
};
