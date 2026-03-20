import React, { useState } from "react";
import { ProcessedKernel, ProcedureCheckResult, DisplayAttribute } from "../utils/dataLoader";

interface IRAnalysisProps {
  kernels: ProcessedKernel[];
  selectedKernel: number;
}

/**
 * Loop schedule entry with prologue, loop body, and epilogue
 */
interface LoopSchedule {
  prologue?: string[];
  loop_body?: string[];
  epilogue?: string[];
}

const IRAnalysis: React.FC<IRAnalysisProps> = ({ kernels, selectedKernel }) => {
  const [expandedPatterns, setExpandedPatterns] = useState<Record<string, boolean>>({});
  const [expandedMessages, setExpandedMessages] = useState<Record<string, boolean>>({});
  const [expandedModuleAttrs, setExpandedModuleAttrs] = useState<Record<string, boolean>>({});
  const [expandedProcedures, setExpandedProcedures] = useState<Record<string, boolean>>({});
  const [expandedTileSize, setExpandedTileSize] = useState<Record<string, boolean>>({});

  if (kernels.length === 0) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">No kernel data available</div>
      </div>
    );
  }

  const kernel = kernels[selectedKernel];
  if (kernel.ir_analysis === null) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-gray-800">No IR Analysis available</div>
      </div>
    );
  }

  const io_counts = kernel.ir_analysis?.io_counts;
  const ttgir_info = io_counts?.["amd_ttgir_bufferops_count"];
  const amdgcn_info = io_counts?.["amd_gcn_bufferops_count"];
  const loop_schedule = kernel.ir_analysis?.loop_schedules;
  const procedure_checks = kernel.ir_analysis?.procedure_checks;
  const getCount = (info: Record<string, number> | undefined, key: string): string => { return info?.[key]?.toString() ?? "N/A"; };

  // Helper function to get procedure check status display
  const getProcedureCheckDisplay = (result: ProcedureCheckResult): { color: string; icon: string } => {
    if (result.detected) {
      return {
        color: "bg-green-100 text-green-800 border-green-200",
        icon: "✓"
      };
    } else {
      return {
        color: "bg-red-100 text-red-800 border-red-200",
        icon: "✗"
      };
    }
  };

  // Helper to get the value of an attribute from the dynamic attributes dict
  const getAttributeValue = (checkResult: ProcedureCheckResult, key: string): unknown => {
    return checkResult.attributes?.[key] ?? undefined;
  };

  // Helper to format an attribute value for display
  const formatAttributeValue = (value: unknown, type?: string): string => {
    if (value === null || value === undefined) return "N/A";
    if (type === "number" && typeof value === "number") {
      return value.toLocaleString();
    }
    return String(value);
  };

  // Get display attributes grouped by their group field
  const getGroupedAttributes = (displayAttrs: DisplayAttribute[]): Record<string, DisplayAttribute[]> => {
    const groups: Record<string, DisplayAttribute[]> = {};
    for (const attr of displayAttrs) {
      const group = attr.group || "other";
      if (!groups[group]) groups[group] = [];
      groups[group].push(attr);
    }
    return groups;
  };

  // Get the not-detected message: use the message from JSON config if available
  const getNotDetectedMessage = (name: string, checkResult: ProcedureCheckResult): string => {
    if (checkResult.message) {
      return checkResult.message.trim().replace("Detected", "NOT DETECTED");
    }
    return `Procedure "${name}" - NOT DETECTED

Check the specific conditions required for this procedure.`;
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Triton Kernel IR Analysis</h1>

      <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Kernel: [{selectedKernel}] {kernel.name}
        </h2>

        {io_counts && (ttgir_info || amdgcn_info) && (
          <>
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              AMD BufferOps Information
            </h3>

            <div className="bg-gray-50 p-4 rounded-md border border-gray-200 mb-6">
              <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3">
                {ttgir_info && (
                  <>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">Tiled Global Load Count</span>
                      <span className="font-mono text-sm break-words">{getCount(ttgir_info, "tt.load_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">Tiled Global Store Count</span>
                      <span className="font-mono text-sm break-words">{getCount(ttgir_info, "tt.store_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">Tiled Buffer Load Count</span>
                      <span className="font-mono text-sm break-words">{getCount(ttgir_info, "amdgpu.buffer_load_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">Tiled Buffer Store Count</span>
                      <span className="font-mono text-sm break-words">{getCount(ttgir_info, "amdgpu.buffer_store_count")}</span>
                    </div>
                  </>
                )}
                {amdgcn_info && (
                  <>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">AMDGCN Global Load Instruction Count</span>
                      <span className="font-mono text-sm break-words">{getCount(amdgcn_info, "global_load_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">AMDGCN Global Store Instruction Count</span>
                      <span className="font-mono text-sm break-words">{getCount(amdgcn_info, "global_store_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">AMDGCN Buffer Load Instruction Count</span>
                      <span className="font-mono text-sm break-words">{getCount(amdgcn_info, "buffer_load_count")}</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-gray-500">AMDGCN Buffer Store Instruction Count</span>
                      <span className="font-mono text-sm break-words">{getCount(amdgcn_info, "buffer_store_count")}</span>
                    </div>
                  </>
                )}
              </div>
            </div>
          </>
        )}

        {loop_schedule && loop_schedule.length > 0 && (
          <>
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Software Pipelining Schedule
            </h3>

            {loop_schedule.map((schedule: LoopSchedule, loopIndex: number) => {
              const prologue = schedule?.prologue || [];
              const loopBody = schedule?.loop_body || [];
              const epilogue = schedule?.epilogue || [];

              return (
                <div key={loopIndex} className="bg-gray-50 p-4 rounded-md border border-gray-200 mb-4">
                  <h4 className="text-md font-semibold mb-2 text-gray-700">
                    Software Pipelining for Loop {loopIndex + 1}
                  </h4>

                  {/* Prologue */}
                  {prologue.length > 0 && (
                    <div className="mb-3">
                      <div className="text-sm font-medium text-gray-600 mb-1">Prologue:</div>
                      <div className="bg-white p-2 rounded border border-gray-200 font-mono text-xs">
                        {prologue.map((line: string, idx: number) => (
                          <div key={idx} className="text-gray-700">
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Loop Body */}
                  <div className="mb-3">
                    <div className="text-sm font-medium text-gray-600 mb-1">Loop Body:</div>
                    <div className="bg-white p-2 rounded border border-gray-200">
                      <div className="font-mono text-xs text-gray-500 mb-1">for (...) {'{'}</div>
                      <div className="pl-4 font-mono text-xs">
                        {loopBody.length > 0 ? (
                          loopBody.map((line: string, idx: number) => (
                            <div key={idx} className="text-gray-700">
                              {line}
                            </div>
                          ))
                        ) : (
                          <div className="text-gray-400 italic">No operations in loop body</div>
                        )}
                      </div>
                      <div className="font-mono text-xs text-gray-500 mt-1">{'}'}</div>
                    </div>
                  </div>

                  {/* Epilogue */}
                  {epilogue.length > 0 && (
                    <div>
                      <div className="text-sm font-medium text-gray-600 mb-1">Epilogue:</div>
                      <div className="bg-white p-2 rounded border border-gray-200 font-mono text-xs">
                        {epilogue.map((line: string, idx: number) => (
                          <div key={idx} className="text-gray-700">
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </>
        )}

        {/* FileCheck-based Procedure Detection Section */}
        {procedure_checks && Object.keys(procedure_checks).length > 0 && (
          <>
            <h3 className="text-lg font-medium mb-3 text-gray-800">
              Procedure Detection (FileCheck)
            </h3>

            <div className="bg-gray-50 p-4 rounded-md border border-gray-200 mb-6">
              <div className="text-sm text-gray-600 mb-4">
                FileCheck-based pattern matching for detecting specific procedures in IR content.
                Uses <a href="https://pypi.org/project/filecheck/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Python filecheck</a> (LLVM FileCheck port).
              </div>

              <div className="space-y-3">
                {Object.entries(procedure_checks).map(([name, result]) => {
                  const display = getProcedureCheckDisplay(result as ProcedureCheckResult);
                  const checkResult = result as ProcedureCheckResult;
                  const isPatternExpanded = expandedPatterns[name] || false;
                  const isMessageExpanded = expandedMessages[name] || false;
                  const isModuleAttrsExpanded = expandedModuleAttrs[name] || false;
                  const isProcedureExpanded = expandedProcedures[name] || false;
                  return (
                    <div key={name} className="bg-white rounded border border-gray-200">
                      {/* Foldable Header */}
                      <button
                        onClick={() => setExpandedProcedures(prev => ({ ...prev, [name]: !isProcedureExpanded }))}
                        className="w-full p-3 flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
                      >
                        <div className="flex items-center">
                          <span className="mr-2 text-gray-500">{isProcedureExpanded ? "▼" : "▶"}</span>
                          <span className="font-medium text-gray-800">{name}</span>
                        </div>
                        <span className={`inline-flex items-center px-2 py-1 rounded text-sm font-medium border ${display.color}`}>
                          {display.icon} {checkResult.detected ? "Detected" : "Not Detected"}
                        </span>
                      </button>

                      {/* Foldable Content */}
                      {isProcedureExpanded && (
                        <div className="p-3 pt-0 border-t border-gray-100">
                          {/* Dynamic Attributes - Rendered from display_attributes config */}
                          {(() => {
                            const displayAttrs = checkResult.display_attributes || [];
                            if (displayAttrs.length === 0) return null;

                            const grouped = getGroupedAttributes(displayAttrs);
                            const parameterAttrs = [
                              ...(grouped["parameters"] || []),
                              ...(grouped["counters"] || []),
                            ];
                            const tileAttrs = grouped["tile_info"] || [];
                            const otherAttrs = Object.entries(grouped)
                              .filter(([g]) => !["parameters", "counters", "tile_info"].includes(g))
                              .flatMap(([, attrs]) => attrs);

                            // Check if any parameter/counter attrs have values
                            const hasParamValues = parameterAttrs.some(
                              attr => getAttributeValue(checkResult, attr.key) !== undefined
                            );
                            // Check if any tile attrs have values
                            const hasTileValues = tileAttrs.some(
                              attr => getAttributeValue(checkResult, attr.key) !== undefined
                            );

                            return (
                              <>
                                {/* Parameters & Counters Grid */}
                                {hasParamValues && (
                                  <div className="grid grid-cols-[repeat(auto-fit,_minmax(120px,_1fr))] gap-3 mb-3 bg-gray-50 p-3 rounded border border-gray-100">
                                    {parameterAttrs.map(attr => {
                                      const value = getAttributeValue(checkResult, attr.key);
                                      if (value === undefined) return null;
                                      return (
                                        <div key={attr.key} className="flex flex-col">
                                          <span className="text-xs font-medium text-gray-500">{attr.label}</span>
                                          <span className="font-mono text-sm">{formatAttributeValue(value, attr.type)}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}

                                {/* Tile Size Information - Foldable */}
                                {hasTileValues && (
                                  <div className="mb-3">
                                    <button
                                      onClick={() => setExpandedTileSize(prev => ({ ...prev, [name]: !expandedTileSize[name] }))}
                                      className="flex items-center text-xs font-medium text-blue-600 hover:text-blue-800 cursor-pointer mb-2"
                                    >
                                      <span className="mr-1">{expandedTileSize[name] ? "▼" : "▶"}</span>
                                      Tile Size Information
                                    </button>
                                    {expandedTileSize[name] && (
                                      <div className="grid grid-cols-[repeat(auto-fit,_minmax(120px,_1fr))] gap-3 bg-blue-50 p-3 rounded border border-blue-100">
                                        {tileAttrs.map(attr => {
                                          const value = getAttributeValue(checkResult, attr.key);
                                          if (value === undefined) return null;
                                          return (
                                            <div key={attr.key} className="flex flex-col">
                                              <span className="text-xs font-medium text-blue-600">{attr.label}</span>
                                              <span className="font-mono text-sm">{formatAttributeValue(value, attr.type)}</span>
                                            </div>
                                          );
                                        })}
                                      </div>
                                    )}
                                  </div>
                                )}

                                {/* Other attribute groups */}
                                {otherAttrs.length > 0 && otherAttrs.some(attr => getAttributeValue(checkResult, attr.key) !== undefined) && (
                                  <div className="grid grid-cols-[repeat(auto-fit,_minmax(120px,_1fr))] gap-3 mb-3 bg-gray-50 p-3 rounded border border-gray-100">
                                    {otherAttrs.map(attr => {
                                      const value = getAttributeValue(checkResult, attr.key);
                                      if (value === undefined) return null;
                                      return (
                                        <div key={attr.key} className="flex flex-col">
                                          <span className="text-xs font-medium text-gray-500">{attr.label}</span>
                                          <span className="font-mono text-sm">{formatAttributeValue(value, attr.type)}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                )}
                              </>
                            );
                          })()}

                      {/* Collapsible Module Attributes */}
                          {checkResult.module_attributes && (
                            <div className="mt-2">
                              <button
                                onClick={(e) => { e.stopPropagation(); setExpandedModuleAttrs(prev => ({ ...prev, [name]: !isModuleAttrsExpanded })); }}
                                className="flex items-center text-xs font-medium text-gray-500 hover:text-gray-700 cursor-pointer"
                              >
                                <span className="mr-1">{isModuleAttrsExpanded ? "▼" : "▶"}</span>
                                Module Attributes (TTGIR)
                              </button>
                              {isModuleAttrsExpanded && (
                                <div className="mt-1 bg-gray-50 p-2 rounded border border-gray-100 font-mono text-xs overflow-x-auto">
                                  <pre className="text-gray-600 whitespace-pre-wrap break-all">
                                    {checkResult.module_attributes}
                                  </pre>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Collapsible Check Patterns */}
                          {checkResult.match_details && checkResult.match_details.length > 0 && (
                            <div className="mt-2">
                              <button
                                onClick={(e) => { e.stopPropagation(); setExpandedPatterns(prev => ({ ...prev, [name]: !isPatternExpanded })); }}
                                className="flex items-center text-xs font-medium text-gray-500 hover:text-gray-700 cursor-pointer"
                              >
                                <span className="mr-1">{isPatternExpanded ? "▼" : "▶"}</span>
                                Check Patterns ({checkResult.match_details.length})
                              </button>
                              {isPatternExpanded && (
                                <div className="mt-1 bg-gray-50 p-2 rounded border border-gray-100 font-mono text-xs">
                                  {checkResult.match_details.map((detail: string, idx: number) => (
                                    <div key={idx} className="text-gray-600">
                                      {detail}
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}

                          {/* Collapsible Detailed Message (Criteria & Performance Implications) - when DETECTED */}
                          {checkResult.detected && checkResult.message && (
                            <div className="mt-3">
                              <button
                                onClick={(e) => { e.stopPropagation(); setExpandedMessages(prev => ({ ...prev, [name]: !isMessageExpanded })); }}
                                className="flex items-center text-sm font-medium text-blue-600 hover:text-blue-800 cursor-pointer"
                              >
                                <span className="mr-1">{isMessageExpanded ? "▼" : "▶"}</span>
                                Show Details (Criteria & Performance Implications)
                              </button>
                              {isMessageExpanded && (
                                <div className="mt-2 bg-blue-50 p-4 rounded-md border border-blue-200">
                                  <pre className="text-xs text-gray-700 whitespace-pre-wrap font-sans leading-relaxed">
                                    {checkResult.message.trim()}
                                  </pre>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Collapsible Conditions for Activation - when NOT DETECTED */}
                          {!checkResult.detected && (
                            <div className="mt-3">
                              <button
                                onClick={(e) => { e.stopPropagation(); setExpandedMessages(prev => ({ ...prev, [name]: !isMessageExpanded })); }}
                                className="flex items-center text-sm font-medium text-gray-500 hover:text-gray-700 cursor-pointer"
                              >
                                <span className="mr-1">{isMessageExpanded ? "▼" : "▶"}</span>
                                Show Details (Conditions for Activation)
                              </button>
                              {isMessageExpanded && (
                                <div className="mt-2 bg-gray-50 p-4 rounded-md border border-gray-200">
                                  <pre className="text-xs text-gray-600 whitespace-pre-wrap font-sans leading-relaxed">
                                    {getNotDetectedMessage(name, checkResult)}
                                  </pre>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          </>
          )}
      </div>
    </div>
  );
};

export default IRAnalysis;
