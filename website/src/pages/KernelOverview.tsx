import React, { useState, useRef, useLayoutEffect, useCallback } from "react";
import ArgumentViewer from "../components/ArgumentViewer";
import DiffViewer from "../components/DiffViewer";
import { ProcessedKernel } from "../utils/dataLoader";
import ToggleSwitch from "../components/ToggleSwitch";
import { DocumentTextIcon, ChevronRightIcon } from "../components/icons";

interface KernelOverviewProps {
  /** A list of all processed kernels available for viewing. */
  kernels: ProcessedKernel[];
  /** The index of the currently selected kernel. */
  selectedKernel: number;
  /** Callback function to handle kernel selection. */
  onSelectKernel: (index: number) => void;
  /** Callback function to handle viewing an IR file. */
  onViewIR: (irType: string) => void;
}

/**
 * Determines if a metadata value is considered "long" and should be displayed at the end
 */
const isLongValue = (value: unknown): boolean => {
  const formattedString = formatMetadataValue(value);
  return formattedString.length > 50;
};

/**
 * Formats a value for display in the metadata section
 * @param value - The value to format
 * @returns Formatted string representation
 */
const formatMetadataValue = (value: unknown): string => {
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

/**
 * Component for displaying a single metadata item with consistent styling
 */
interface MetadataItemProps {
  label: string;
  value: React.ReactNode;
  span?: number; // Number of columns to span (default: 1)
}

const MetadataItem: React.FC<MetadataItemProps> = ({
  label,
  value,
  span = 1,
}) => (
  <div
    className={`flex flex-col ${span > 1 ? `col-span-${span}` : ""} ${
      span === 0 ? "col-span-full" : ""
    }`}
  >
    <span className="text-sm font-medium text-gray-500">{label}</span>
    <span className="font-mono text-sm break-words">{value}</span>
  </div>
);

/**
 * Stack entry interface for stack traces
 */
interface StackEntry {
  filename: string | number | [string, number];
  line: number;
  name: string;
  loc?: string;
}

/**
 * Gets the actual file path from a stack entry's filename
 * @param entry - The stack entry
 */
const getSourceFilePath = (entry: StackEntry): string => {
  if (typeof entry.filename === "string") {
    return entry.filename;
  }
  return "Invalid filename format";
};

/**
 * The main component for displaying an overview of Triton kernels.
 * It includes a kernel selector, metadata display, launch analysis, and IR file links.
 */
const KernelOverview: React.FC<KernelOverviewProps> = ({
  kernels,
  selectedKernel,
  onSelectKernel,
  onViewIR,
}) => {
  // State for controlling the sticky and collapsed behavior of the kernel selector
  const [isTiledKernelView, setIsTiledKernelView] = useState(false);
  const [isSticky, setIsSticky] = useState(true);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const buttonsContainerRef = useRef<HTMLDivElement>(null);

  /**
   * Adjusts the scroll position of the kernel buttons container to ensure
   * the selected kernel's row is visible when the header is sticky and collapsed.
   */
  const adjustScroll = useCallback(() => {
    if (isTiledKernelView && isSticky && isCollapsed && buttonsContainerRef.current) {
      const container = buttonsContainerRef.current;
      const selectedButton = container.children[selectedKernel] as
        | HTMLElement
        | undefined;

      if (selectedButton) {
        // Scroll the container to bring the selected button's row into view
        container.scrollTop = selectedButton.offsetTop;
      }
    }
  }, [isTiledKernelView, isSticky, isCollapsed, selectedKernel]);

  // Effect to adjust scroll on state changes and listen for window resizing
  useLayoutEffect(() => {
    adjustScroll();

    window.addEventListener("resize", adjustScroll);
    return () => {
      window.removeEventListener("resize", adjustScroll);
    };
  }, [adjustScroll, kernels]);

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
      <h1 className="text-2xl font-bold text-gray-800 mb-6">
        Triton Kernel Overview
      </h1>

      {/* Kernel Selection */}
      <div
        className={`bg-white rounded-lg shadow-sm border border-gray-200 transition-all duration-300 mb-4 ${
          isSticky ? "sticky top-4 z-10 p-2" : "p-4"
        }`}
        onMouseEnter={() => isSticky && setIsCollapsed(false)}
        onMouseLeave={() => isSticky && setIsCollapsed(true)}
      >
        <div className={`flex items-center gap-4 ${isSticky ? "mb-2" : "mb-4"}`}>
          <h2
            className={`${
              isSticky ? "text-lg" : "text-xl"
            } font-semibold text-gray-800`}
          >
            Available Kernels
          </h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span
                className={`${
                  isSticky ? "text-xs" : "text-sm"
                } text-gray-600`}
              >
                Tiled View
              </span>
              <ToggleSwitch isChecked={isTiledKernelView} onChange={setIsTiledKernelView} />
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`${
                  isSticky ? "text-xs" : "text-sm"
                } text-gray-600`}
              >
                Sticky Header
              </span>
              <ToggleSwitch isChecked={isSticky} onChange={setIsSticky} />
            </div>
            {!isTiledKernelView && (
              <div className="flex items-center gap-2">
                <span className={`${isSticky ? "text-xs" : "text-sm"} text-gray-600`}>
                  Select Kernel
                </span>
                <select
                  className="border border-gray-300 rounded-md p-1.5 text-sm bg-white"
                  value={selectedKernel}
                  onChange={(e) => onSelectKernel(Number(e.target.value))}
                >
                  {kernels.map((k, index) => (
                    <option key={index} value={index}>
                      {k.name}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>
        {isTiledKernelView && (
          <div
            ref={buttonsContainerRef}
            className={`flex flex-wrap transition-all duration-300 ${
              isSticky ? "gap-1" : "gap-2"
            } ${
              isSticky && isCollapsed
                ? "max-h-9 overflow-hidden"
                : "max-h-96"
            }`}
          >
            {kernels.map((k, index) => (
              <button
                key={index}
                className={`rounded-md transition-colors whitespace-nowrap ${
                  isSticky
                    ? "px-3 py-1 text-xs"
                    : "px-4 py-2 text-sm"
                } ${
                  index === selectedKernel
                    ? "bg-blue-100 border border-blue-300 text-blue-800"
                    : "bg-gray-50 border border-gray-200 hover:bg-blue-50 text-gray-800"
                }`}
                onClick={() => onSelectKernel(index)}
              >
                <div className="font-medium">{k.name}</div>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Kernel Details */}
      <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Kernel Details: {kernel.name}
        </h2>

        {/* Metadata Section */}
        {kernel.metadata && (
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Compilation Metadata
            </h3>
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
              {/* Short fields in responsive grid */}
              <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3 mb-4">
                {/* All short metadata fields */}
                {Object.entries(kernel.metadata)
                  .filter(([, value]) => !isLongValue(value))
                  .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                  .map(([key, value]) => {
                    return (
                      <MetadataItem
                        key={key}
                        label={key
                          .split("_")
                          .map(
                            (word) =>
                              word.charAt(0).toUpperCase() + word.slice(1)
                          )
                          .join(" ")}
                        value={formatMetadataValue(value)}
                      />
                    );
                  })}
              </div>

              {/* Long fields in separate section within same container */}
              {Object.entries(kernel.metadata).filter(([, value]) =>
                isLongValue(value)
              ).length > 0 && (
                <div className="space-y-3 border-t border-gray-200 pt-4">
                  {Object.entries(kernel.metadata)
                    .filter(([, value]) => isLongValue(value))
                    .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                    .map(([key, value]) => (
                      <div key={key} className="w-full">
                        <span className="text-sm font-medium text-gray-500 block mb-1">
                          {key
                            .split("_")
                            .map(
                              (word) =>
                                word.charAt(0).toUpperCase() + word.slice(1)
                            )
                            .join(" ")}
                        </span>
                        <span className="font-mono text-sm block break-all">
                          {formatMetadataValue(value)}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Launch Analysis Section */}
        {kernel.launchDiff && (
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Launch Analysis
            </h3>
            <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
              <p className="text-sm text-gray-700 mb-4">
                <span className="font-semibold">Total Launches:</span>{" "}
                {kernel.launchDiff.total_launches}
              </p>

              {/* Launch Index Map */}
              {kernel.launchDiff.launch_index_map && (
                <div className="mb-4">
                  <h4 className="text-md font-semibold mb-2 text-gray-800">
                    Launch Locations in Original Trace{" "}
                    <span className="text-sm font-normal text-gray-500">
                      (0-based line numbers)
                    </span>
                  </h4>
                  <div className="font-mono text-sm bg-gray-100 p-2 rounded">
                    {kernel.launchDiff.launch_index_map
                      .map((r: { start: number; end: number }) =>
                        r.start === r.end
                          ? `${r.start}`
                          : `${r.start}-${r.end}`
                      )
                      .join(", ")}
                  </div>
                </div>
              )}

              {/* Unchanged Fields */}
              {kernel.launchDiff.sames && Object.keys(kernel.launchDiff.sames).length > 0 && (
              <div className="mb-4">
                <h4 className="text-md font-semibold mb-2 text-gray-800">
                  Unchanged Launch Arguments
                </h4>
                <ArgumentViewer args={kernel.launchDiff.sames.extracted_args || {}} />
              </div>
              )}

              {(() => {
                if (!kernel.launchDiff.sames) return null;

                const otherSames = Object.fromEntries(
                  Object.entries(kernel.launchDiff.sames).filter(
                    ([key]) =>
                      key !== "compilation_metadata" &&
                      key !== "extracted_args" &&
                      key !== "event_type" &&
                      key !== "stack"
                  )
                );

                if (Object.keys(otherSames).length > 0) {
                  return (
                    <div className="mb-4">
                      <h4 className="text-md font-semibold mb-2 text-gray-800">
                        Other Unchanged Fields
                      </h4>
                      <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3 p-2 bg-white rounded border border-gray-200">
                        {Object.entries(otherSames).map(([key, value]) => (
                          <MetadataItem
                            key={key}
                            label={key
                              .split("_")
                              .map(
                                (word) =>
                                  word.charAt(0).toUpperCase() + word.slice(1)
                              )
                              .join(" ")}
                            value={formatMetadataValue(value)}
                          />
                        ))}
                      </div>
                    </div>
                  );
                }
                return null;
              })()}

              {/* Unchanged Stack Trace */}
              {kernel.launchDiff.sames && kernel.launchDiff.sames.stack && (
                <div className="mb-4">
                  <h4 className="text-md font-semibold mb-2 text-gray-800">
                    Unchanged Stack Trace
                  </h4>
                  <div className="bg-white p-3 rounded-md border border-gray-200 overflow-auto resize-y h-80 min-h-24">
                    {Array.isArray(kernel.launchDiff.sames.stack) ? (
                      kernel.launchDiff.sames.stack.map((entry: StackEntry, index: number) => (
                        <div key={index} className="mb-1 font-mono text-sm">
                          <span className="text-blue-600">
                            {getSourceFilePath(entry)}
                          </span>
                          :<span className="text-red-600">{entry.line}</span> -
                          <span className="text-green-600">{entry.name}</span> -
                          <span className="text-gray-700">{entry.loc}</span>
                        </div>
                      ))
                    ) : (
                      <pre className="font-mono text-sm text-gray-700 whitespace-pre-wrap break-all">
                        {typeof kernel.launchDiff.sames.stack === 'string'
                          ? kernel.launchDiff.sames.stack
                          : JSON.stringify(kernel.launchDiff.sames.stack, null, 2)}
                      </pre>
                    )}
                  </div>
                </div>
              )}

              {/* Differing Fields */}
              <div className="mb-4">
                <h4 className="text-md font-semibold mb-2 text-gray-800">
                  Differing Fields
                </h4>
                <DiffViewer diffs={(kernel.launchDiff.diffs ?? {}) as Record<string, unknown>} />
              </div>
            </div>
          </div>
        )}

        {/* Stack Trace */}
        <div className="mb-4">
          <h3 className="text-lg font-medium mb-2 text-gray-800">
            Compilation Stack Trace
          </h3>
          <div className="bg-gray-50 p-3 rounded-md border border-gray-200 overflow-auto resize-y h-80 min-h-24">
            {kernel.stack.map((entry, index) => (
              <div key={index} className="mb-1 font-mono text-sm">
                <span className="text-blue-600">
                  {getSourceFilePath(entry)}
                </span>
                :<span className="text-red-600">{entry.line}</span> -
                <span className="text-green-600">{entry.name}</span> -
                <span className="text-gray-700">{entry.loc}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-medium mb-2 text-gray-800">IR Files</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.keys(kernel.irFiles).map((irType) => (
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
                      {irType}
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">
                      View full IR code
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
      </div>
    </div>
  );
};

export default KernelOverview;
