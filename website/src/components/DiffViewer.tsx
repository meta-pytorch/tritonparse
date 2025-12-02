import ArgumentViewer from "./ArgumentViewer";
import React from "react";
import StackDiffViewer from "./StackDiffViewer";

interface DiffViewerProps {
  diffs: any;
}

const DiffViewer: React.FC<DiffViewerProps> = ({ diffs }) => {
  if (!diffs || Object.keys(diffs).length === 0) {
    return (
      <p className="text-sm text-gray-500">No differing fields detected.</p>
    );
  }

  // Separate different kinds of diffs
  const extractedArgs = diffs.extracted_args;
  const stackDiff = diffs.stack;
  const otherDiffs = Object.fromEntries(
    Object.entries(diffs).filter(
      ([key]) => key !== "extracted_args" && key !== "stack"
    )
  );

  const renderSimpleDiff = (_key: string, data: any) => {
    if (data.diff_type === "summary") {
      return <p className="font-mono text-sm text-gray-800">{data.summary_text}</p>;
    }
    if (data.diff_type === "distribution") {
      return (
        <ul className="list-disc list-inside pl-2 text-sm">
          {data.values.map((item: any, index: number) => {
            const launchRanges = item.launches
              .map((r: any) =>
                r.start === r.end
                  ? `${r.start}`
                  : `${r.start}-${r.end}`
              )
              .join(", ");
            return (
              <li key={index} className="font-mono text-gray-800 break-all">
                <span className="font-mono bg-gray-100 px-1 rounded">
                  {JSON.stringify(item.value)}
                </span>
                <span className="text-gray-500 text-xs ml-2">
                  ({item.count} times, in launches: {launchRanges})
                </span>
              </li>
            );
          })}
        </ul>
      );
    }
    // Fallback for unexpected structures
    return <pre>{JSON.stringify(data, null, 2)}</pre>;
  };

  return (
    <div className="space-y-4">
      {extractedArgs && Object.keys(extractedArgs).length > 0 && (
        <div>
          <h5 className="text-md font-semibold mb-2 text-gray-700">
            Extracted Arguments
          </h5>
          <ArgumentViewer args={extractedArgs} isDiffViewer={true} />
        </div>
      )}

      {Object.keys(otherDiffs).length > 0 && (
        <div>
          <h5 className="text-md font-semibold mb-2 text-gray-700">
            Other Differing Fields
          </h5>
          <div className="space-y-3">
            {Object.entries(otherDiffs).map(([key, value]) => (
              <div
                key={key}
                className="w-full p-2 bg-white rounded border border-gray-200"
              >
                <span className="text-sm font-medium text-gray-600 block mb-1 break-all">
                  {key}
                </span>
                <div className="pl-4">{renderSimpleDiff(key, value)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {stackDiff && <StackDiffViewer stackDiff={stackDiff} />}

    </div>
  );
};

export default DiffViewer; 