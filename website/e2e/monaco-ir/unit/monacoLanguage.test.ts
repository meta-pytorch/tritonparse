/**
 * Unit layer (§6.3.1) for Monaco language resolution (R6 closure fixtures:
 * old trace without stages, custom display_name, unknown syntax_id).
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  MONACO_LANGUAGE_IDS,
  mapFileToMonacoLanguage,
} from "../../../src/utils/monacoLanguage.ts";
import type { IRStageDescriptor } from "../../../src/utils/dataLoader.ts";

function stage(name: string, syntaxId: string, displayName?: string): IRStageDescriptor {
  return {
    name,
    extension: `.${name}`,
    display_name: displayName ?? name.toUpperCase(),
    display_order: 10,
    is_text: true,
    supports_source_mapping: true,
    syntax_id: syntaxId,
  };
}

test("stage syntax_id lookup covers every known id", () => {
  const cases: Array<[string, string]> = [
    ["mlir", MONACO_LANGUAGE_IDS.mlir],
    ["llvm", MONACO_LANGUAGE_IDS.llvm],
    ["ptx", MONACO_LANGUAGE_IDS.ptx],
    ["amdgcn", MONACO_LANGUAGE_IDS.asm],
    ["asm", MONACO_LANGUAGE_IDS.asm],
    ["python", MONACO_LANGUAGE_IDS.python],
    // Resolved plaintext: JSON/python-feature imports stay out (§4.6).
    ["json", MONACO_LANGUAGE_IDS.plaintext],
    ["c", MONACO_LANGUAGE_IDS.plaintext],
  ];
  for (const [syntaxId, expected] of cases) {
    const stages = [stage("ttgir", syntaxId)];
    assert.equal(mapFileToMonacoLanguage("kernel.ttgir", stages), expected);
  }
});

test("unknown syntax_id falls back to plaintext, never an unregistered id", () => {
  const stages = [stage("ttgir", "fancy-new-lang")];
  assert.equal(
    mapFileToMonacoLanguage("kernel.ttgir", stages),
    MONACO_LANGUAGE_IDS.plaintext
  );
});

test("prototype-named syntax_ids fall back to plaintext with warning", () => {
  // Exact counterexamples from rounds/002-codex.md (candidate 0513e65f).
  const registered = new Set(Object.values(MONACO_LANGUAGE_IDS));
  const warnings: string[] = [];
  const originalWarn = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map(String).join(" "));
  };
  try {
    for (const id of ["constructor", "toString", "__proto__", "hasOwnProperty", "valueOf"]) {
      const resolved = mapFileToMonacoLanguage("kernel.ttgir", [stage("ttgir", id)]);
      assert.equal(typeof resolved, "string");
      assert.ok(registered.has(resolved));
      assert.equal(resolved, MONACO_LANGUAGE_IDS.plaintext);
    }
  } finally {
    console.warn = originalWarn;
  }
  assert.equal(warnings.length, 5);
  assert.ok(warnings.every((w) => w.includes("unknown syntax_id")));
});

test("custom display_name never changes the resolved language (R6)", () => {
  const stages = [stage("ttgir", "mlir", "My Custom TTGIR!!!")];
  assert.equal(
    mapFileToMonacoLanguage("kernel.ttgir", stages),
    MONACO_LANGUAGE_IDS.mlir
  );
});

test("legacy extension fallback without stages (old traces)", () => {
  const cases: Array<[string, string]> = [
    ["k.ttgir", MONACO_LANGUAGE_IDS.mlir],
    ["k.ttir", MONACO_LANGUAGE_IDS.mlir],
    ["K.TTGIR", MONACO_LANGUAGE_IDS.mlir],
    ["k.llir", MONACO_LANGUAGE_IDS.llvm],
    ["k.ptx", MONACO_LANGUAGE_IDS.ptx],
    ["k.amdgcn", MONACO_LANGUAGE_IDS.asm],
    ["k.sass", MONACO_LANGUAGE_IDS.asm],
    ["python", MONACO_LANGUAGE_IDS.python],
    ["k.json", MONACO_LANGUAGE_IDS.plaintext],
    ["k.cubin", MONACO_LANGUAGE_IDS.plaintext],
    ["k.unknown", MONACO_LANGUAGE_IDS.plaintext],
  ];
  for (const [filename, expected] of cases) {
    assert.equal(mapFileToMonacoLanguage(filename, undefined), expected);
    assert.equal(mapFileToMonacoLanguage(filename, []), expected);
  }
});

test("unmatched stage name falls through to the extension fallback", () => {
  const stages = [stage("other", "mlir")];
  assert.equal(
    mapFileToMonacoLanguage("kernel.ptx", stages),
    MONACO_LANGUAGE_IDS.ptx
  );
});
