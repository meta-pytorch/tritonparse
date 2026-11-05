/**
 * Unit tests for kernel extraction without on-disk IR paths.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  processKernelData,
  type LogEntry,
} from "../../../src/utils/dataLoader.ts";

const CUTEDSL_COMPILATION: LogEntry = {
  event_type: "compilation",
  payload: {
    metadata: { hash: "cutlass_flow_copy", name: "flow_copy" },
    file_content: {
      "kernel.mlir": "module {}",
      "kernel.ptx": "// ptx",
    },
    ir_stages: [
      {
        name: "mlir",
        extension: ".mlir",
        display_name: "MLIR (CuTe DSL)",
        display_order: 10,
        is_text: true,
        supports_source_mapping: true,
        syntax_id: "mlir",
      },
      {
        name: "ptx",
        extension: ".ptx",
        display_name: "PTX",
        display_order: 20,
        is_text: true,
        supports_source_mapping: true,
        syntax_id: "ptx",
      },
    ],
  },
};

test("compilation without file_path is still extracted as a kernel", () => {
  const kernels = processKernelData([CUTEDSL_COMPILATION]);
  assert.equal(kernels.length, 1);
  assert.equal(kernels[0].name, "kernel");
  assert.deepEqual(kernels[0].filePaths, {});
  assert.deepEqual(Object.keys(kernels[0].irFiles), [
    "kernel.mlir",
    "kernel.ptx",
  ]);
});

test("compilation without file_content is still skipped", () => {
  const entry: LogEntry = {
    event_type: "compilation",
    payload: {
      metadata: { hash: "abc", name: "x" },
      file_path: { "kernel.ptx": "/tmp/kernel.ptx" },
    },
  };
  assert.equal(processKernelData([entry]).length, 0);
});
