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
  // Try dynamic lookup from ir_stages
  if (irStages && irStages.length > 0) {
    const type = irType.split('.').pop()?.toLowerCase() || irType.toLowerCase();
    const stage = irStages.find(s => s.name === type);
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
  } else {
    return irType;
  }
};
