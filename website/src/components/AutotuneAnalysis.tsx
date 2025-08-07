import React from "react";
import { AutotuneAnalysisEvent } from "../utils/dataLoader";

interface AutotuneAnalysisProps {
  analysis: AutotuneAnalysisEvent;
  currentKernelHash: string;
  onSelectKernel: (hash: string) => void;
}

const AutotuneAnalysis: React.FC<AutotuneAnalysisProps> = ({
  analysis,
  currentKernelHash,
  onSelectKernel,
}) => {
  return (
    <div className="bg-white rounded-lg p-4 mb-4 shadow border border-gray-200">
      <h3 className="text-lg font-medium mb-3 text-gray-800">
        Autotune Analysis for <code>{analysis.name}</code>
      </h3>
      <p className="text-sm text-gray-600 mb-4">
        This kernel was part of an autotuning session. The table below shows all
        configurations tested and the final selection.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th
                scope="col"
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                Config Parameters
              </th>
              <th
                scope="col"
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                Status
              </th>
              <th
                scope="col"
                className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
              >
                Action
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {analysis.configs.map((config, index) => {
              const isCurrent = config.compilation_hash === currentKernelHash;
              const isSelected = config.compilation_hash === analysis.selected_hash;
              return (
                <tr
                  key={index}
                  className={isCurrent ? "bg-blue-50" : "hover:bg-gray-50"}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-800">
                    {JSON.stringify(config.config_params)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {isCurrent && (
                      <span className="font-semibold text-blue-700">
                        ⬅️ Current
                      </span>
                    )}
                    {isSelected && (
                      <span className="font-semibold text-green-700 ml-2">
                        ✅ Selected
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    {!isCurrent && (
                      <button
                        onClick={() => onSelectKernel(config.compilation_hash)}
                        className="text-indigo-600 hover:text-indigo-900"
                      >
                        View Details
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default AutotuneAnalysis; 