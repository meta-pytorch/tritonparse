import React, { useState, useEffect } from "react";
import { ProcessedKernel, getIRType } from "../utils/dataLoader";
import CodeComparisonView from "../components/CodeComparisonView";
import { getDisplayLanguage } from "../components/TritonIRs";
import { mapLanguageToHighlighter } from "../components/CodeViewer";

/**
 * Props for the CodeView component
 */
interface CodeViewProps {
  kernels: ProcessedKernel[]; // Array of processed kernel data
  selectedKernel?: number; // Index of the currently selected kernel
}


/**
 * CodeView component that shows a side-by-side comparison of different IR files
 * from the same kernel (typically TTGIR and PTX)
 */
const CodeView: React.FC<CodeViewProps> = ({ kernels, selectedKernel = 0 }) => {
  // States to track selected IR files for left and right panels
  const [leftIR, setLeftIR] = useState<string>("");
  const [rightIR, setRightIR] = useState<string>("");

  // State to track if Python source code should be shown
  const [showPythonSource, setShowPythonSource] = useState<boolean>(true);

  // State to track the last selected kernel to detect changes
  const [lastSelectedKernel, setLastSelectedKernel] = useState<number>(selectedKernel);

  // Return a message if no kernel data is available
  if (!kernels || kernels.length === 0 || selectedKernel < 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">
          No data available for code comparison
        </div>
      </div>
    );
  }

  const kernel = kernels[selectedKernel];
  const irFiles = Object.keys(kernel.irFiles);

  // Check if Python source is available
  const hasPythonSource = !!kernel.pythonSourceInfo?.code;

  // Reset selections when kernel changes
  useEffect(() => {
    if (selectedKernel !== lastSelectedKernel) {
      // Only reset when switching between different kernels
      setLeftIR("");
      setRightIR("");
      setLastSelectedKernel(selectedKernel);
    }
  }, [selectedKernel, lastSelectedKernel]);

  // Set default IR files on initial render or when kernel changes
  useEffect(() => {
    // Skip setting defaults if both IR files are already selected
    if (leftIR && rightIR) {
      return;
    }

    // Try to find TTGIR and PTX files as defaults
    let defaultLeftIR = leftIR;
    let defaultRightIR = rightIR;

    // Only find defaults for empty selections
    if (!defaultLeftIR) {
      // Look for TTGIR first
      const ttgirFile = irFiles.find(key => key.toLowerCase().includes("ttgir"));
      if (ttgirFile) {
        defaultLeftIR = ttgirFile;
      } else if (irFiles.length > 0) {
        defaultLeftIR = irFiles[0];
      }
    }

    if (!defaultRightIR) {
      // Look for PTX first
      const ptxFile = irFiles.find(key => key.toLowerCase().includes("ptx"));
      if (ptxFile) {
        defaultRightIR = ptxFile;
      } else if (irFiles.length > 1) {
        defaultRightIR = irFiles[1];
      } else if (irFiles.length === 1) {
        defaultRightIR = irFiles[0]; // Use the same file if only one exists
      }
    }

    // Only set state if needed
    if (!leftIR && defaultLeftIR) {
      setLeftIR(defaultLeftIR);
    }
    if (!rightIR && defaultRightIR) {
      setRightIR(defaultRightIR);
    }
  }, [irFiles, leftIR, rightIR]);

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
          {/* This SVG represents a horizontal swap icon with two arrows pointing in opposite directions.
             It's used for the swap button that exchanges the content between left and right panels. */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path d="M8 5a1 1 0 100 2h5.586l-1.293 1.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L13.586 5H8zM12 15a1 1 0 100-2H6.414l1.293-1.293a1 1 0 10-1.414-1.414l-3 3a1 1 0 000 1.414l3 3a1 1 0 001.414-1.414L6.414 15H12z" />
          </svg>
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

export default CodeView;
