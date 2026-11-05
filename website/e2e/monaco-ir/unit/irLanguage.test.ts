/**
 * Unit tests for IR display-name resolution.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import type { IRStageDescriptor } from "../../../src/utils/dataLoader.ts";
import { getDisplayLanguage } from "../../../src/utils/irLanguage.ts";

const CUTEDSL_STAGES: IRStageDescriptor[] = [
  { name: "mlir", extension: ".mlir", display_name: "MLIR (CuTe DSL)", display_order: 10, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "clean_mlir", extension: ".clean_mlir", display_name: "Clean MLIR (CuTe DSL)", display_order: 15, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "ptx", extension: ".ptx", display_name: "PTX", display_order: 20, is_text: true, supports_source_mapping: true, syntax_id: "ptx" },
  { name: "sass", extension: ".sass", display_name: "SASS", display_order: 40, is_text: true, supports_source_mapping: true, syntax_id: "asm" },
];

test("dynamic lookup resolves cutedsl stages", () => {
  assert.equal(getDisplayLanguage("kernel.mlir", CUTEDSL_STAGES), "MLIR (CuTe DSL)");
  assert.equal(getDisplayLanguage("kernel.clean_mlir", CUTEDSL_STAGES), "Clean MLIR (CuTe DSL)");
  assert.equal(getDisplayLanguage("kernel.ptx", CUTEDSL_STAGES), "PTX");
  assert.equal(getDisplayLanguage("kernel.sass", CUTEDSL_STAGES), "SASS");
});

test("dynamic lookup matches by extension when the stage name differs", () => {
  const stages: IRStageDescriptor[] = [
    { name: "pre_opt_ir", extension: ".source", display_name: "Pre-optimization IR", display_order: 0, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
    ...CUTEDSL_STAGES,
  ];
  assert.equal(getDisplayLanguage("kernel.source", stages), "Pre-optimization IR");
});

test("dynamic lookup ignores the case of stage names", () => {
  const stages = CUTEDSL_STAGES.map((s) => ({ ...s, name: s.name.toUpperCase() }));
  assert.equal(getDisplayLanguage("kernel.ptx", stages), "PTX");
  assert.equal(getDisplayLanguage("kernel.clean_mlir", stages), "Clean MLIR (CuTe DSL)");
});

test("legacy fallback names mlir and clean_mlir without stages", () => {
  assert.equal(getDisplayLanguage("kernel.mlir"), "MLIR (CuTe DSL)");
  assert.equal(getDisplayLanguage("kernel.clean_mlir"), "Clean MLIR (CuTe DSL)");
});

test("legacy fallback keeps existing names", () => {
  assert.equal(getDisplayLanguage("kernel.ttgir"), "TTGIR (TritonGPU MLIR)");
  assert.equal(getDisplayLanguage("kernel.ptx"), "PTX (NVIDIA Parallel Thread Execution)");
  assert.equal(getDisplayLanguage("kernel.xyz"), "kernel.xyz");
});
