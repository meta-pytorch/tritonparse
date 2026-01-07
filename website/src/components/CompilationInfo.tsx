import React from "react";

interface CompilationInfoProps {
  metadata: {
    compiler: string;
    version: string;
    timestamp: string;
  };
  targets: string[];
  options: Record<string, unknown>;
  statistics: Record<string, number>;
}

const CompilationInfo: React.FC<CompilationInfoProps> = ({
  metadata,
  targets,
  options,
  statistics,
}) => {
  return (
    <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">
        Compilation Information
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <h3 className="text-lg font-medium mb-2 text-blue-600">Metadata</h3>
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Compiler:</span>
              <span className="text-gray-800">{metadata.compiler}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Version:</span>
              <span className="text-gray-800">{metadata.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Timestamp:</span>
              <span className="text-gray-800">
                {new Date(metadata.timestamp).toLocaleString()}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <h3 className="text-lg font-medium mb-2 text-blue-600">Targets</h3>
          <div className="flex flex-wrap gap-2">
            {targets.map((target, index) => (
              <span
                key={index}
                className="px-2 py-1 bg-blue-50 rounded text-green-600 text-sm border border-green-200"
              >
                {target}
              </span>
            ))}
          </div>
        </div>

        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <h3 className="text-lg font-medium mb-2 text-blue-600">Options</h3>
          <div className="space-y-1 text-sm">
            {Object.entries(options).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-gray-600">{key}:</span>
                <span className="text-gray-800">{JSON.stringify(value)}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-50 rounded p-3 border border-gray-200">
          <h3 className="text-lg font-medium mb-2 text-blue-600">Statistics</h3>
          <div className="space-y-1 text-sm">
            {Object.entries(statistics).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-gray-600">
                  {key
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase())}
                  :
                </span>
                <span className="text-gray-800">
                  {key.includes("time") ? `${value} ms` : value}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompilationInfo;
