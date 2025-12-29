import React from "react";
import { IRFile } from "../utils/dataLoader";

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

interface TritonIRsProps {
  irFiles: Record<string, IRFile>;
  onViewIR: (irType: string) => void;
}

const TritonIRs: React.FC<TritonIRsProps> = ({ irFiles, onViewIR }) => {
  return (
    <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">Triton IRs</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {Object.keys(irFiles).map((irType) => (
          <div
            key={irType}
            className="bg-gray-50 rounded p-4 border border-gray-200 hover:bg-blue-50 hover:border-blue-200 cursor-pointer transition-colors"
            onClick={() => onViewIR(irType)}
          >
            <div className="flex items-start">
              <div className="flex-shrink-0">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6 text-blue-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <div className="ml-4">
                <h3 className="text-lg font-medium text-gray-800">
                  {getDisplayLanguage(irType)}
                </h3>
                <p className="text-sm text-gray-600 mt-1">
                  View full {irType.toUpperCase()} code
                </p>
              </div>
              <div className="ml-auto">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-gray-400"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TritonIRs;
