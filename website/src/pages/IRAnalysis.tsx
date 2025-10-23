import React from "react";
import { ProcessedKernel } from "../utils/dataLoader";

interface IRAnalysisProps {
  kernels: ProcessedKernel[];
  selectedKernel: number;
}

const formatMetadataValue = (value: any): string => {
  if (value === null) {
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
};

interface MetadataItemProps {
  label: string;
  value: React.ReactNode;
}

const MetadataItem: React.FC<MetadataItemProps> = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-sm font-medium text-gray-500">{label}</span>
    <span className="font-mono text-sm break-words">{value}</span>
  </div>
);

const IRAnalysis: React.FC<IRAnalysisProps> = ({ kernels, selectedKernel }) => {
  if (kernels.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">No kernel data available</div>
      </div>
    );
  }

  const kernel = kernels[selectedKernel];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Triton Kernel IR Analysis</h1>

      <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Kernel: {kernel.name}
        </h2>
      </div>

      {kernel.irAnalysis.amdBufferOps && (
        <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          AMD BufferOps Information
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <>
              <MetadataItem
                label="TTGIR Global Load Count"
                value={kernel.irAnalysis.amdBufferOps.total || "N/A"}
              />
              <MetadataItem
                label="TTGIR Global Store Count"
                value={kernel.irAnalysis.amdBufferOps.loads || "N/A"}
              />
              <MetadataItem
                label="TTGIR Buffer Load Count"
                value={kernel.irAnalysis.amdBufferOps.total || "N/A"}
              />
              <MetadataItem
                label="TTGIR Buffer Store Count"
                value={kernel.irAnalysis.amdBufferOps.loads || "N/A"}
              />
              <MetadataItem
                label="AMDGCN Global Load Count"
                value={kernel.irAnalysis.amdBufferOps.total || "N/A"}
              />
              <MetadataItem
                label="AMDGCN Global Store Count"
                value={kernel.irAnalysis.amdBufferOps.loads || "N/A"}
              />
              <MetadataItem
                label="AMDGCN Buffer Load Count"
                value={kernel.irAnalysis.amdBufferOps.total || "N/A"}
              />
              <MetadataItem
                label="AMDGCN Buffer Store Count"
                value={kernel.irAnalysis.amdBufferOps.loads || "N/A"}
              />
            </>
            </div>
          </div>
          )}
    </div>
  );
};

export default IRAnalysis;
