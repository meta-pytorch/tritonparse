import React, { useEffect, useState } from "react";
import SingleMonacoViewer from "./SingleMonacoViewer";
import { IRFile, IRStageDescriptor } from "../utils/dataLoader";
import { getDisplayLanguage } from "../utils/irLanguage";
import CopyCodeButton from "./CopyCodeButton";
import { ArrowLeftIcon } from "./icons";

/**
 * Props for the SingleCodeViewer component
 */
interface SingleCodeViewerProps {
  irFile?: IRFile; // IR file object containing content and source mappings
  irContent?: string; // Direct code content as string (alternative to irFile)
  title: string; // Title to display for the code view
  onBack: () => void; // Callback function when back button is clicked
  irStages?: IRStageDescriptor[];
  sourceId?: string; // Stable data-source identity for the doc token
  kernelId?: string | number; // Kernel hash or index within the source
}

/**
 * SingleCodeViewer component that displays a single IR file with syntax highlighting
 * Used for detailed viewing of a specific IR file
 */
const SingleCodeViewer: React.FC<SingleCodeViewerProps> = ({
  irFile,
  irContent,
  title,
  onBack,
  irStages,
  sourceId,
  kernelId,
}) => {
  // Determine content to display (either from direct content or from IRFile)
  const codeContent = irContent || (irFile ? irFile.content : "");
  const displayLanguage = getDisplayLanguage(title, irStages);

  // Word wrap (same `wrap` URL param as File Diff; default off for IR).
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

  return (
    <div className="p-6">
      {/* Header with back button and title */}
      <div className="flex items-center mb-4">
        <button
          onClick={onBack}
          className="text-blue-600 hover:text-blue-800 flex items-center mr-4"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-1" />
          Back
        </button>
        <div>
          <h1 className="text-2xl font-bold text-gray-800">{title}</h1>
          <p className="text-gray-600">Language: {displayLanguage}</p>
        </div>
      </div>

      {/* Code viewer container */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        {/* Panel title bar */}
        <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
          <span>{title}</span>
          <div className="flex items-center gap-2">
            <span className="text-sm bg-blue-700 px-2 py-1 rounded">
              {displayLanguage}
            </span>
            <label className="text-sm bg-blue-700 px-2 py-1 rounded flex items-center gap-1">
              Wrap:
              <select
                data-testid="single-wrap-select"
                value={wordWrap}
                onChange={(e) => setWordWrap(e.target.value as "off" | "on")}
                className="bg-blue-700 text-white text-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-white cursor-pointer"
                aria-label="Word wrap"
              >
                <option value="off">Off</option>
                <option value="on">On</option>
              </select>
            </label>
            <CopyCodeButton
              code={codeContent}
              className="text-sm bg-blue-700 px-2 py-1 rounded"
            />
          </div>
        </div>
        {/* Code content area with fixed height */}
        <div className="h-[calc(100vh-12rem)]">
          <SingleMonacoViewer
            irFile={irFile}
            irContent={irContent}
            title={title}
            irStages={irStages}
            sourceId={sourceId}
            kernelId={kernelId}
            wordWrap={wordWrap}
          />
        </div>
      </div>
    </div>
  );
};

export default SingleCodeViewer;
