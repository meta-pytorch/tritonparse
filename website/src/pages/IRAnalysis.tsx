import React from "react";
import { ProcessedKernel } from "../utils/dataLoader";

interface IRAnalysisProps {
  kernels: ProcessedKernel[];
  selectedKernel: number;
}

const IRAnalysis: React.FC<IRAnalysisProps> = ({ kernels, selectedKernel }) => {
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
  const getCount = (info: Record<string, number> | undefined, key: string): string => { return info?.[key]?.toString() ?? "N/A"; };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">Triton Kernel IR Analysis</h1>

      <div className="bg-white rounded-lg p-4 mb-4 shadow-sm border border-gray-200">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Kernel: {kernel.name}
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

            {loop_schedule.map((schedule: any, loopIndex: number) => {
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
      </div>
    </div>
  );
};

export default IRAnalysis;
