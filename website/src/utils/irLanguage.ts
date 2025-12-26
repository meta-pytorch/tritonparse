/**
 * Utility functions for IR language display
 */

/**
 * Get a user-friendly display name for the IR language
 * @param irType - The type/name of IR file
 * @returns A human-readable language name
 */
export const getDisplayLanguage = (irType: string): string => {
  if (irType.toLowerCase().endsWith("ttgir")) {
    return "TTGIR (TritonGPU MLIR)";
  } else if (irType.toLowerCase().endsWith("ttir")) {
    return "TTIR (Triton MLIR)";
  } else if (irType.toLowerCase().endsWith("llir")) {
    return "LLIR (LLVM IR)";
  } else if (irType.toLowerCase().endsWith("ptx")) {
    return "PTX (NVIDIA Parallel Thread Execution)";
  } else if (irType.toLowerCase().endsWith("cubin")) {
    return "CUBIN (NVIDIA CUDA Binary)";
  } else if (irType.toLowerCase().endsWith("python")) {
    return "Python";
  } else if (irType.toLowerCase().endsWith("json")) {
    return "JSON";
  } else if (irType.toLowerCase().endsWith("amdgcn")) {
    return "AMDGCN (AMD GCN Assembly)";
  } else {
    return irType;
  }
};
