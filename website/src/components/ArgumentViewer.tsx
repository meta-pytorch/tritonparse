import React, { useState } from 'react';
import { isTensorArg, extractTensorMetadata, formatTensorSummary } from '../utils/tensor';
import { ChevronRightIcon } from './icons';

interface LaunchRange {
    start: number;
    end: number;
}

interface DistributionValue {
    value: unknown;
    count: number;
    launches: LaunchRange[];
}

interface DistributionData {
    diff_type: 'summary' | 'distribution';
    summary_text?: string;
    values?: DistributionValue[];
}

// Renders the value distribution (e.g., "16 (2 times, in launches: 1-2)")
const DistributionCell: React.FC<{ data: DistributionData | null }> = ({ data }) => {
    if (!data) return null;
    if (data.diff_type === 'summary') {
        return <span className="text-gray-500 italic">{data.summary_text}</span>;
    }
    if (data.diff_type === 'distribution' && data.values) {
        return (
            <ul className="list-none m-0 p-0 space-y-1">
                {data.values.map((item: DistributionValue, index: number) => {
                    const launchRanges = item.launches
                        .map((r: LaunchRange) => (r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`))
                        .join(', ');
                    return (
                        <li key={index}>
                            <span className="font-mono bg-gray-100 px-1 rounded">{JSON.stringify(item.value)}</span>
                            <span className="text-gray-500 text-xs ml-2">({item.count} times, in launches: {launchRanges})</span>
                        </li>
                    );
                })}
            </ul>
        );
    }
    return <span className="font-mono">{JSON.stringify(data)}</span>;
};

// Renders a single row in the ArgumentViewer table
const ArgumentRow: React.FC<{
  argName: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- argData contains dynamic data from trace
  argData: any;
  isDiffViewer?: boolean;
}> = ({ argName, argData, isDiffViewer = false }) => {
    // Keep hooks unconditional to satisfy React rules of hooks
    const [showDetails, setShowDetails] = useState(false);
    const [isCollapsed, setIsCollapsed] = useState(false);
    // Case 1: This is a complex argument with internal differences
    if (isDiffViewer && argData.diff_type === "argument_diff") {
        const { sames, diffs } = argData;
        const hasSames = Object.keys(sames).length > 0;
        const hasDiffs = Object.keys(diffs).length > 0;

        return (
            <div className="bg-gray-50 border-b border-gray-200 last:border-b-0">
                <div
                    className="flex items-center space-x-4 p-2 cursor-pointer"
                    onClick={() => setIsCollapsed(!isCollapsed)}
                >
                    <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">
                        {argName}
                    </div>
                    <div className="flex-1 text-gray-500 italic text-sm flex items-center">
                        Complex argument with internal differences
                        <ChevronRightIcon
                            className={`w-4 h-4 ml-2 transform transition-transform ${
                                isCollapsed ? "" : "rotate-90"
                            }`}
                            strokeWidth={2}
                        />
                    </div>
                </div>
                {!isCollapsed && (
                    <div className="pb-2 px-4 space-y-3">
                        {hasSames && (
                            <div>
                                <h6 className="text-sm font-semibold text-gray-600 mb-1">
                                    Unchanged Properties
                                </h6>
                                <div className="space-y-1 pl-4">
                                    {Object.entries(sames).map(
                                        ([key, value]) => (
                                            <div
                                                key={key}
                                                className="flex items-start text-sm"
                                            >
                                                <span className="w-28 font-mono text-gray-500 flex-shrink-0">
                                                    {key}:
                                                </span>
                                                <span className="font-mono">
                                                    {JSON.stringify(value)}
                                                </span>
                                            </div>
                                        )
                                    )}
                                </div>
                            </div>
                        )}
                        {hasDiffs && (
                            <div>
                                <h6 className="text-sm font-semibold text-gray-600 mb-1">
                                    Differing Properties
                                </h6>
                                <div className="space-y-2 pl-4">
                                    {Object.entries(diffs).map(
                                        ([key, value]) => (
                                            <div
                                                key={key}
                                                className="flex items-start text-sm"
                                            >
                                                <span className="w-28 font-mono text-gray-500 flex-shrink-0">
                                                    {key}:
                                                </span>
                                                <div className="flex-1">
                                                    <DistributionCell
                                                        data={value as DistributionData | null}
                                                    />
                                                </div>
                                            </div>
                                        )
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    }
    // Case 2: This is a tensor argument (in the "Sames" table)
    if (isTensorArg(argData)) {
        const meta = extractTensorMetadata(argData);
        const summary = formatTensorSummary(meta);

        return (
            <div className="border-b border-gray-200 last:border-b-0">
                <div className="flex items-start space-x-4 p-2">
                    <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">
                        {argName}
                    </div>
                    <div className="w-48 flex-shrink-0 text-gray-600">
                        {argData.type}
                    </div>
                    <div className="flex-1 font-mono text-sm flex items-center flex-wrap">
                        <span className="mr-2 break-all">{summary}</span>
                        <button
                            className="text-xs text-blue-600 hover:text-blue-700 flex items-center"
                            onClick={() => setShowDetails(!showDetails)}
                            aria-expanded={showDetails}
                            aria-controls={`tensor-details-${argName}`}
                        >
                            Details
                            <ChevronRightIcon
                                className={`w-3 h-3 ml-1 transform transition-transform ${
                                    showDetails ? "" : "rotate-90"
                                }`}
                                strokeWidth={2}
                            />
                        </button>
                    </div>
                </div>
                {showDetails && (
                    <div id={`tensor-details-${argName}`} className="px-2 pb-3">
                        <div className="ml-[12rem] mr-[12rem] bg-gray-50 border border-gray-200 rounded p-2">
                            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                                {[
                                    ["dtype", meta.dtype],
                                    ["shape", meta.shape],
                                    ["strides", meta.strides],
                                    ["device", meta.device],
                                    ["length", meta.length],
                                    ["layout", meta.layout],
                                    ["contiguous", meta.contiguous],
                                    ["nbytes", meta.nbytes],
                                    ["requires_grad", meta.requires_grad],
                                    ["storage_offset", meta.storage_offset],
                                ].map(([label, value]) =>
                                    value !== undefined ? (
                                        <div
                                            key={String(label)}
                                            className="flex"
                                        >
                                            <span className="w-28 text-gray-600">
                                                {String(label)}
                                            </span>
                                            <span className="font-mono break-all">
                                                {Array.isArray(value)
                                                    ? `[${value.join(", ")}]`
                                                    : String(value)}
                                            </span>
                                        </div>
                                    ) : null
                                )}
                            </div>
                            <div className="mt-2">
                                <details>
                                    <summary className="cursor-pointer text-xs text-gray-600">
                                        Raw
                                    </summary>
                                    <pre className="text-xs whitespace-pre-wrap break-all mt-1">
                                        {JSON.stringify(argData, null, 2)}
                                    </pre>
                                </details>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        );
    }
    // Case 3: This is a simple argument (in the "Sames" table)
    return (
        <div className="flex items-start space-x-4 p-2 border-b border-gray-200 last:border-b-0">
            <div className="w-48 flex-shrink-0 font-semibold text-gray-800 break-all">
                {argName}
            </div>
            <div className="w-48 flex-shrink-0 text-gray-600">
                {argData.type}
            </div>
            <div className="flex-1 font-mono text-sm">
                {typeof argData.value !== "object" || argData.value === null ? (
                    <span>{String(argData.value)}</span>
                ) : (
                    <pre className="text-xs whitespace-pre-wrap break-all">
                        {JSON.stringify(argData, null, 2)}
                    </pre>
                )}
            </div>
        </div>
    );
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any -- args contain dynamic data from trace
const ArgumentViewer: React.FC<{ args: Record<string, any>; isDiffViewer?: boolean; }> = ({ args, isDiffViewer = false }) => {
    if (!args || Object.keys(args).length === 0) {
        return <div className="text-sm text-gray-500 p-2">No arguments to display.</div>;
    }

    // A "complex view" is needed if we are showing diffs and at least one of them is a complex argument_diff
    const isComplexView = isDiffViewer && Object.values(args).some(arg => arg.diff_type === 'argument_diff');

    return (
        <div className="border border-gray-200 rounded-md bg-white">
            {/* Render header only for the simple, non-complex table view */}
            {!isComplexView && (
                <div className="flex items-center space-x-4 p-2 bg-gray-100 font-bold text-gray-800 border-b border-gray-200">
                    <div className="w-48 flex-shrink-0">Argument Name</div>
                    <div className="w-48 flex-shrink-0">Type</div>
                    <div className="flex-1">Value</div>
                </div>
            )}

            {/* Rows */}
            <div>
                {Object.entries(args).map(([argName, argData]) => (
                    <ArgumentRow key={argName} argName={argName} argData={argData} isDiffViewer={isDiffViewer} />
                ))}
            </div>
        </div>
    );
};

export default ArgumentViewer;
