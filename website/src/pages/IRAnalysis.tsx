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

// Default display attributes configuration for backward compatibility
const DEFAULT_PARAMETER_ATTRS: DisplayAttribute[] = [
  { key: "num_warps", label: "Warps", type: "number", group: "parameters" },
  { key: "num_stages", label: "Stages", type: "number", group: "parameters" },
  { key: "num_pp_clusters", label: "PP Clusters", type: "number", group: "parameters" },
  { key: "cond_barrier_count", label: "Cond Barriers", type: "number", group: "counters" },
  { key: "setprio_count", label: "SetPrio Ops", type: "number", group: "counters" },
  { key: "dot_count", label: "Dot Operations", type: "number", group: "counters" },
];

const DEFAULT_TILE_ATTRS: DisplayAttribute[] = [
  { key: "tile_m", label: "Tile M", type: "number", group: "tile_info" },
  { key: "tile_n", label: "Tile N", type: "number", group: "tile_info" },
  { key: "tile_k", label: "Tile K", type: "number", group: "tile_info" },
  { key: "tile_size_bits", label: "Tile Size (bits)", type: "number", group: "tile_info" },
  { key: "input_dtype", label: "Input Type", type: "string", group: "tile_info" },
  { key: "output_dtype", label: "Output Type", type: "string", group: "tile_info" },
  { key: "mfma_m", label: "MFMA M", type: "number", group: "tile_info" },
  { key: "mfma_n", label: "MFMA N", type: "number", group: "tile_info" },
  { key: "mfma_k", label: "MFMA K", type: "number", group: "tile_info" },
];

// Export default attributes for use in dynamic rendering
export { DEFAULT_PARAMETER_ATTRS, DEFAULT_TILE_ATTRS };

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

  // Helper function to get not-detected message for each procedure type
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const getNotDetectedMessage = (name: string, _checkResult: ProcedureCheckResult): string => {
    // If the result has a custom message for not-detected state, use it
    // Otherwise, fall back to hardcoded messages for backward compatibility
    if (name === "BlockPingpong_Small") {
      return `BlockPingpong Small (1 PP Cluster) - NOT DETECTED

CONDITIONS FOR ACTIVATION:
  • numWarps = 4 (1 pingpong cluster)
  • numStages > 1 (software pipelining required)
  • 262,144 ≤ tileSize ≤ 16,777,216 bits
  • Exactly 2 global loads and 2 local loads in dot computation
  • Single tt.dot operation per loop iteration

PERFORMANCE IMPLICATIONS:
  • Uses setprio interleaving (no cond_barrier)
  • Warps from different blocks pingpong on same SIMD
  • Ideal for small, compute-bound tiles`;
    } else if (name === "BlockPingpong_Medium") {
      return `BlockPingpong Medium (2 PP Clusters) - NOT DETECTED

CONDITIONS FOR ACTIVATION:
  • numWarps = 8 (2 pingpong clusters)
  • numStages = 2 (exactly)
  • tileSize == 33,554,432 bits (medium tile exactly)
  • Exactly 2 local stores in dot computation
  • Exactly 2 global loads and 2 local loads

PERFORMANCE IMPLICATIONS:
  • Uses amdgpu.cond_barrier for inter-cluster sync
  • Warps from same block pingpong
  • Two tt.dot operations per loop iteration
  • Suitable for medium-sized compute-bound tiles`;
    } else if (name === "BlockPingpong_Large") {
      return `BlockPingpong Large (4 PP Clusters) - NOT DETECTED

CONDITIONS FOR ACTIVATION:
  • numWarps = 8 (4 pingpong clusters)
  • numStages = 2 (exactly)
  • tileSize ≥ 67,108,864 bits (large tile)
  • Exactly 2 local stores in dot computation
  • Exactly 2 global loads and 2 local loads
  • NOT mfma16×16×16 with kWidth=8 (register spilling)

PERFORMANCE IMPLICATIONS:
  • Uses amdgpu.cond_barrier for inter-cluster sync
  • Warps from same block pingpong
  • Four tt.dot operations per loop iteration
  • Dots split into 4 pieces to avoid register pressure`;
    } else {
      return `Procedure "${name}" - NOT DETECTED

Check the specific conditions required for this procedure.`;
    }
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
                          {/* Key Parameters Grid - Always displayed when data available */}
                          {((checkResult.num_warps !== null && checkResult.num_warps !== undefined) ||
                        (checkResult.num_stages !== null && checkResult.num_stages !== undefined) ||
                        (checkResult.num_pp_clusters !== null && checkResult.num_pp_clusters !== undefined) ||
                        (checkResult.cond_barrier_count !== null && checkResult.cond_barrier_count !== undefined) ||
                        (checkResult.setprio_count !== null && checkResult.setprio_count !== undefined) ||
                        (checkResult.dot_count !== null && checkResult.dot_count !== undefined)) && (
                            <div className="grid grid-cols-[repeat(auto-fit,_minmax(120px,_1fr))] gap-3 mb-3 bg-gray-50 p-3 rounded border border-gray-100">
                              {checkResult.num_warps !== null && checkResult.num_warps !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">Warps</span>
                                  <span className="font-mono text-sm">{checkResult.num_warps}</span>
                                </div>
                              )}
                              {checkResult.num_stages !== null && checkResult.num_stages !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">Stages</span>
                                  <span className="font-mono text-sm">{checkResult.num_stages}</span>
                                </div>
                              )}
                              {checkResult.num_pp_clusters !== null && checkResult.num_pp_clusters !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">PP Clusters</span>
                                  <span className="font-mono text-sm">{checkResult.num_pp_clusters}</span>
                                </div>
                              )}
                              {checkResult.cond_barrier_count !== null && checkResult.cond_barrier_count !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">Cond Barriers</span>
                                  <span className="font-mono text-sm">{checkResult.cond_barrier_count}</span>
                                </div>
                              )}
                              {checkResult.setprio_count !== null && checkResult.setprio_count !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">SetPrio Ops</span>
                                  <span className="font-mono text-sm">{checkResult.setprio_count}</span>
                                </div>
                              )}
                              {checkResult.dot_count !== null && checkResult.dot_count !== undefined && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-gray-500">Dot Operations</span>
                                  <span className="font-mono text-sm">{checkResult.dot_count}</span>
                                </div>
                              )}
                            </div>
                          )}

                          {/* Tile Size Information - Always visible when tile info available, foldable */}
                      {(checkResult.tile_m !== null && checkResult.tile_m !== undefined) ||
                       (checkResult.tile_n !== null && checkResult.tile_n !== undefined) ||
                       (checkResult.tile_k !== null && checkResult.tile_k !== undefined) ||
                       (checkResult.tile_size_bits !== null && checkResult.tile_size_bits !== undefined) ||
                       (checkResult.input_dtype !== null && checkResult.input_dtype !== undefined) ||
                       (checkResult.output_dtype !== null && checkResult.output_dtype !== undefined) ||
                       (checkResult.mfma_m !== null && checkResult.mfma_m !== undefined) ? (
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
                              {(checkResult.tile_m !== null && checkResult.tile_m !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Tile M</span>
                                  <span className="font-mono text-sm">{checkResult.tile_m}</span>
                                </div>
                              )}
                              {(checkResult.tile_n !== null && checkResult.tile_n !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Tile N</span>
                                  <span className="font-mono text-sm">{checkResult.tile_n}</span>
                                </div>
                              )}
                              {(checkResult.tile_k !== null && checkResult.tile_k !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Tile K</span>
                                  <span className="font-mono text-sm">{checkResult.tile_k}</span>
                                </div>
                              )}
                              {(checkResult.tile_size_bits !== null && checkResult.tile_size_bits !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Tile Size (bits)</span>
                                  <span className="font-mono text-sm">{checkResult.tile_size_bits.toLocaleString()}</span>
                                </div>
                              )}
                              {(checkResult.input_dtype !== null && checkResult.input_dtype !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Input Type</span>
                                  <span className="font-mono text-sm">{checkResult.input_dtype}</span>
                                </div>
                              )}
                              {(checkResult.output_dtype !== null && checkResult.output_dtype !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">Output Type</span>
                                  <span className="font-mono text-sm">{checkResult.output_dtype}</span>
                                </div>
                              )}
                              {(checkResult.mfma_m !== null && checkResult.mfma_m !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">MFMA M</span>
                                  <span className="font-mono text-sm">{checkResult.mfma_m}</span>
                                </div>
                              )}
                              {(checkResult.mfma_n !== null && checkResult.mfma_n !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">MFMA N</span>
                                  <span className="font-mono text-sm">{checkResult.mfma_n}</span>
                                </div>
                              )}
                              {(checkResult.mfma_k !== null && checkResult.mfma_k !== undefined) && (
                                <div className="flex flex-col">
                                  <span className="text-xs font-medium text-blue-600">MFMA K</span>
                                  <span className="font-mono text-sm">{checkResult.mfma_k}</span>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      ) : null}

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
