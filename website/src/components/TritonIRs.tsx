import React from "react";
import { IRFile } from "../utils/dataLoader";
import { getDisplayLanguage } from "../utils/irLanguage";
import { DocumentTextIcon, ChevronRightIcon } from "./icons";

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
                <DocumentTextIcon className="h-6 w-6 text-blue-600" />
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
                <ChevronRightIcon className="h-5 w-5 text-gray-400" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TritonIRs;
