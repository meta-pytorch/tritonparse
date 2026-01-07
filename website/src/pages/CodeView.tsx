import React, { useState, useMemo } from "react";
import { ProcessedKernel, getIRType } from "../utils/dataLoader";
import CodeComparisonView from "../components/CodeComparisonView";
import { getDisplayLanguage } from "../utils/irLanguage";
import { mapLanguageToHighlighter } from "../components/CodeViewer";
import { ArrowsRightLeftIcon } from "../components/icons";

/**
 * Props for the CodeView component
 */
interface CodeViewProps {
  kernels: ProcessedKernel[]; // Array of processed kernel data
  selectedKernel?: number; // Index of the currently selected kernel
}

/**
 * Helper function to find default IR files for left and right panels
 */
function findDefaultIRFiles(irFiles: string[]): { left: string; right: string } {
  let left = "";
  let right = "";

  const ttgirFile = irFiles.find(key => key.toLowerCase().includes("ttgir"));
  if (ttgirFile) {
    left = ttgirFile;
  } else if (irFiles.length > 0) {
    left = irFiles[0];
  }

  const ptxFile = irFiles.find(key => key.toLowerCase().includes("ptx"));
  if (ptxFile) {
    right = ptxFile;
  } else if (irFiles.length > 1) {
    right = irFiles[1];
  } else if (irFiles.length === 1) {
    right = irFiles[0];
  }

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
}> = ({ kernel, irFiles, defaultIRFiles }) => {
  // States to track selected IR files for left and right panels
  // Initialize with defaults - component remounts when kernel changes
  const [leftIR, setLeftIR] = useState<string>(defaultIRFiles.left);
  const [rightIR, setRightIR] = useState<string>(defaultIRFiles.right);

  // State to track if Python source code should be shown
  const [showPythonSource, setShowPythonSource] = useState<boolean>(true);

  const hasPythonSource = !!kernel?.pythonSourceInfo?.code;

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-4">
        Code Comparison: {kernel.name}
      </h1>

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
              Language: {getDisplayLanguage(leftIR)}
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
              Language: {getDisplayLanguage(rightIR)}
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
        <div className="h-[calc(100vh-20rem)] bg-white rounded-lg overflow-hidden shadow-sm border border-gray-200">
          <CodeComparisonView
            leftPanel={{
              code: {
                content: kernel.irFiles[leftIR],
                source_mapping: kernel.sourceMappings?.[getIRType(leftIR)] || {}
              },
              language: mapLanguageToHighlighter(leftIR),
              title: leftIR
            }}
            rightPanel={{
              code: {
                content: kernel.irFiles[rightIR],
                source_mapping: kernel.sourceMappings?.[getIRType(rightIR)] || {}
              },
              language: mapLanguageToHighlighter(rightIR),
              title: rightIR
            }}
            py_code_info={kernel.pythonSourceInfo}
            showPythonSource={showPythonSource && hasPythonSource}
            pythonMapping={kernel.sourceMappings?.["python"] || {}}
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
const CodeView: React.FC<CodeViewProps> = ({ kernels, selectedKernel = 0 }) => {
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
    return findDefaultIRFiles(irFiles);
  }, [irFiles]);

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
    />
  );
};

export default CodeView;
