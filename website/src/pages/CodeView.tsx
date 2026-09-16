import React, { useEffect, useState, useMemo } from "react";
import { ProcessedKernel, getIRType, getDefaultPanels, IRStageDescriptor } from "../utils/dataLoader";
import CodeComparisonViewV2 from "../components/CodeComparisonViewV2";
import { getDisplayLanguage } from "../utils/irLanguage";
import { ArrowsRightLeftIcon } from "../components/icons";

/**
 * Props for the CodeView component
 */
interface CodeViewProps {
  kernels: ProcessedKernel[]; // Array of processed kernel data
  selectedKernel?: number; // Index of the currently selected kernel
  /** Actual load-source identity; main path passes App state. */
  sourceId?: string;
}

/**
 * Helper function to find default IR files for left and right panels.
 * Uses getDefaultPanels() to determine which stages to select, then
 * matches stage names to actual filenames in irFiles.
 * Falls back to legacy logic for old traces without ir_stages.
 */
function findDefaultIRFiles(irFiles: string[], irStages?: IRStageDescriptor[]): { left: string; right: string } {
  const defaultPanels = getDefaultPanels(irStages);

  const leftFile = irFiles.find(key => key.toLowerCase().includes(defaultPanels.left));
  const rightFile = irFiles.find(key => key.toLowerCase().includes(defaultPanels.right));

  const left = leftFile || irFiles[0] || "";
  const right = rightFile || irFiles[1] || irFiles[0] || "";

  return { left, right };
}

/**
 * Inner component that manages IR selection state
 * This is keyed by selectedKernel in the parent, so it remounts when kernel changes
 */
const CodeViewInner: React.FC<{
  kernel: ProcessedKernel;
  irFiles: string[];
  defaultIRFiles: { left: string; right: string };
  selectedKernel: number;
  sourceId?: string;
}> = ({ kernel, irFiles, defaultIRFiles, selectedKernel, sourceId }) => {
  // States to track selected IR files for left and right panels
  // Initialize with defaults - component remounts when kernel changes
  const [leftIR, setLeftIR] = useState<string>(defaultIRFiles.left);
  const [rightIR, setRightIR] = useState<string>(defaultIRFiles.right);

  // State to track if Python source code should be shown
  const [showPythonSource, setShowPythonSource] = useState<boolean>(true);

  // Word wrap for all three panels (same `wrap` URL param as File Diff).
  const [wordWrap, setWordWrap] = useState<"off" | "on">(() => {
    const w = new URLSearchParams(window.location.search).get("wrap");
    return w === "on" ? "on" : "off";
  });
  useEffect(() => {
    const url = new URL(window.location.href);
    // Preserve existing history.state (a router may own it) and skip the
    // write when the URL already reflects the value (absent == "off"), so
    // mounting never rewrites a URL the user didn't touch.
    const current = url.searchParams.get("wrap") ?? "off";
    if (current === wordWrap) return;
    url.searchParams.set("wrap", wordWrap);
    window.history.replaceState(window.history.state, "", url.toString());
  }, [wordWrap]);

  const hasPythonSource = !!kernel?.pythonSourceInfo?.code;

  // Memoized panel descriptors: CodeComparisonViewV2 is memo'd, so these
  // must keep referential identity across unrelated parent renders (e.g.
  // tab switches). Fresh object literals here re-rendered all viewers and
  // re-tokenized every row (~6s measured) on each switch.
  const leftPanel = useMemo(() => ({
    code: {
      content: kernel.irFiles[leftIR],
      source_mapping: kernel.sourceMappings?.[getIRType(leftIR)] || {}
    },
    title: leftIR
  }), [kernel, leftIR]);
  const rightPanel = useMemo(() => ({
    code: {
      content: kernel.irFiles[rightIR],
      source_mapping: kernel.sourceMappings?.[getIRType(rightIR)] || {}
    },
    title: rightIR
  }), [kernel, rightIR]);

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold text-gray-800">
          Code Comparison: [{selectedKernel}] {kernel.name}
        </h1>
        <label className="flex items-center gap-2 text-sm font-medium text-gray-700">
          Wrap:
          <select
            data-testid="comparison-wrap-select"
            value={wordWrap}
            onChange={(e) => setWordWrap(e.target.value as "off" | "on")}
            className="border border-gray-300 rounded px-2 py-1 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
            aria-label="Word wrap"
          >
            <option value="off">Off</option>
            <option value="on">On</option>
          </select>
        </label>
      </div>

      {/* IR file selector controls */}
      <div className="flex justify-between items-center mb-6 relative">
        <div className="w-[calc(50%-24px)] bg-gray-50 p-3 rounded-tl-lg rounded-tr-lg border border-gray-200">
          <label
            htmlFor="leftIRSelect"
            className="mb-1 font-medium text-gray-700 block"
          >
            Left Panel:
          </label>
          <select
            id="leftIRSelect"
            className="border border-gray-300 rounded px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-full"
            value={leftIR}
            onChange={(e) => setLeftIR(e.target.value)}
          >
            {irFiles.length === 0 && (
              <option value="">No IR files available</option>
            )}
            {irFiles.map((file) => (
              <option key={`left-${file}`} value={file}>
                {file}
              </option>
            ))}
          </select>
          {leftIR && (
            <div className="text-sm text-gray-600 mt-1">
              Language: {getDisplayLanguage(leftIR, kernel?.ir_stages)}
            </div>
          )}
        </div>

        {/* Swap button in the middle */}
        <button
          className="absolute left-1/2 top-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-400 hover:bg-gray-500 text-white font-medium rounded-full p-1.5 flex items-center justify-center shadow-sm opacity-80 z-10"
          onClick={() => {
            // Swap the left and right IR selections
            const temp = leftIR;
            setLeftIR(rightIR);
            setRightIR(temp);
          }}
          title="Swap panels"
        >
          <ArrowsRightLeftIcon className="h-4 w-4" />
        </button>

        <div className="w-[calc(50%-24px)] bg-gray-50 p-3 rounded-tl-lg rounded-tr-lg border border-gray-200">
          <label
            htmlFor="rightIRSelect"
            className="mb-1 font-medium text-gray-700 block"
          >
            Right Panel:
          </label>
          <select
            id="rightIRSelect"
            className="border border-gray-300 rounded px-3 py-2 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 w-full"
            value={rightIR}
            onChange={(e) => setRightIR(e.target.value)}
          >
            {irFiles.length === 0 && (
              <option value="">No IR files available</option>
            )}
            {irFiles.map((file) => (
              <option key={`right-${file}`} value={file}>
                {file}
              </option>
            ))}
          </select>
          {rightIR && (
            <div className="text-sm text-gray-600 mt-1">
              Language: {getDisplayLanguage(rightIR, kernel?.ir_stages)}
            </div>
          )}
        </div>
      </div>

      {/* Python Source Toggle (only shown if Python source is available) */}
      {hasPythonSource && (
        <div className="mb-4 bg-gray-50 p-3 rounded-lg border border-gray-200 flex items-center">
          <label className="flex items-center cursor-pointer">
            <div className="relative">
              <input
                type="checkbox"
                className="sr-only"
                checked={showPythonSource}
                onChange={(e) => setShowPythonSource(e.target.checked)}
              />
              <div className={`block w-10 h-6 rounded-full ${showPythonSource ? 'bg-blue-500' : 'bg-gray-400'}`}></div>
              <div className={`dot absolute left-1 top-1 bg-white w-4 h-4 rounded-full transition transform ${showPythonSource ? 'translate-x-4' : ''}`}></div>
            </div>
            <div className="ml-3 font-medium text-gray-700">
              Show Python Source Code
            </div>
          </label>
          {showPythonSource && kernel.pythonSourceInfo?.code && (
            <div className="ml-6 text-sm text-gray-600">
              Source: {kernel.pythonSourceInfo.file_path}
            </div>
          )}
        </div>
      )}

      {/* Side-by-side comparison of selected IR files */}
      {leftIR && rightIR ? (
        <div className="h-[calc(100vh-20rem)] bg-white rounded-lg overflow-auto resize-y min-h-48 shadow-sm border border-gray-200">
          <CodeComparisonViewV2
            leftPanel={leftPanel}
            rightPanel={rightPanel}
            py_code_info={kernel.pythonSourceInfo}
            showPythonSource={showPythonSource && hasPythonSource}
            pythonMapping={kernel.sourceMappings?.["python"]}
            irStages={kernel.ir_stages}
            // Main path passes the actual load-source identity; the File
            // Diff preview call site passes nothing and falls back to the
            // page-param derivation (preview data comes from the File Diff
            // session, not the main load).
            sourceId={
              sourceId ??
              new URLSearchParams(window.location.search).get("json_url") ??
              "local-data"
            }
            kernelId={kernel.metadata?.hash ?? selectedKernel}
            wordWrap={wordWrap}
          />
        </div>
      ) : (
        <div className="p-8 text-center text-gray-600">
          Select IR files to compare
        </div>
      )}
    </div>
  );
};

/**
 * CodeView component that shows a side-by-side comparison of different IR files
 * from the same kernel (typically TTGIR and PTX)
 */
const CodeView: React.FC<CodeViewProps> = ({ kernels, selectedKernel = 0, sourceId }) => {
  // Compute derived values (may be undefined if no valid kernel)
  const kernel = kernels && kernels.length > 0 && selectedKernel >= 0
    ? kernels[selectedKernel]
    : undefined;

  // Memoize irFiles to ensure stable reference for dependency arrays
  const irFiles = useMemo(
    () => (kernel ? Object.keys(kernel.irFiles) : []),
    [kernel]
  );

  // Compute default IR files
  const defaultIRFiles = useMemo(() => {
    if (irFiles.length === 0) return { left: "", right: "" };
    return findDefaultIRFiles(irFiles, kernel?.ir_stages);
  }, [irFiles, kernel?.ir_stages]);

  // Return a message if no kernel data is available
  if (!kernel) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">
          No data available for code comparison
        </div>
      </div>
    );
  }

  // Show message if no IR files are available
  if (irFiles.length === 0) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200">
          <h2 className="text-xl font-semibold text-yellow-800 mb-3">
            No IR Files Available
          </h2>
          <p className="text-yellow-700">
            No IR files found for this kernel. Please select a different kernel.
          </p>
        </div>
      </div>
    );
  }

  // Use key prop to force remount when kernel changes
  // This avoids calling setState in useEffect
  return (
    <CodeViewInner
      key={`kernel-${selectedKernel}`}
      kernel={kernel}
      irFiles={irFiles}
      defaultIRFiles={defaultIRFiles}
      selectedKernel={selectedKernel}
      sourceId={sourceId}
    />
  );
};

export default CodeView;
