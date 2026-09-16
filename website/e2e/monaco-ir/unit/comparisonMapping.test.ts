/**
 * Unit layer (§6.3.1) for the comparison mapping math.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 *
 * Pins the legacy semantics ported from CodeComparisonView (stage discovery,
 * property naming, python file-match direction) plus the two deliberate
 * deviations: no parseInt forging (F19) and range filtering deferred to
 * normalizeHighlightLines (§4.4.1 single normalization point).
 */
import test from "node:test";
import assert from "node:assert/strict";
import type {
  IRStageDescriptor,
  SourceMapping,
} from "../../../src/utils/dataLoader.ts";
import {
  buildKernelKey,
  calculateMappedLines,
  calculatePythonLines,
  intersectRangeWithDoc,
  lineCountOfContent,
} from "../../../src/components/monaco/comparisonMapping.ts";
import { normalizeHighlightLines } from "../../../src/components/monaco/highlightMath.ts";

const STAGES: IRStageDescriptor[] = [
  { name: "ttir", extension: ".ttir", display_name: "TTIR", display_order: 10, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "ttgir", extension: ".ttgir", display_name: "TTGIR", display_order: 20, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "llir", extension: ".llir", display_name: "LLIR", display_order: 30, is_text: true, supports_source_mapping: true, syntax_id: "llvm" },
];

function entry(patch: Record<string, unknown>): Record<string, SourceMapping> {
  return { "5": { line: 5, ...patch } as SourceMapping };
}

test("mapped lines resolve through stage-discovered properties", () => {
  const out = calculateMappedLines(
    entry({ ttgir_lines: [10, 20], ttir_lines: [7] }),
    5,
    "kernel.ttgir",
    STAGES
  );
  assert.deepEqual(out, [10, 20]);
});

test("mapped lines fall back to the legacy stage list without ir_stages", () => {
  const out = calculateMappedLines(
    entry({ ptx_lines: [3] }),
    5,
    "kernel.ptx",
    undefined
  );
  assert.deepEqual(out, [3]);
});

test("mapped lines ignore stages without source-mapping support", () => {
  const stages: IRStageDescriptor[] = [
    { ...STAGES[1], supports_source_mapping: false },
  ];
  const out = calculateMappedLines(
    entry({ ttgir_lines: [10] }),
    5,
    "kernel.ttgir",
    stages
  );
  assert.deepEqual(out, []);
});

test("mapped lines return [] for missing entries, titles and properties", () => {
  assert.deepEqual(calculateMappedLines(entry({}), 5, "kernel.ttgir", STAGES), []);
  assert.deepEqual(calculateMappedLines(entry({ ttgir_lines: [1] }), 6, "kernel.ttgir", STAGES), []);
  assert.deepEqual(calculateMappedLines(undefined, 5, "kernel.ttgir", STAGES), []);
  // Target title with an unknown IR type matches no stage.
  assert.deepEqual(
    calculateMappedLines(entry({ ttgir_lines: [1] }), 5, "kernel.wat", STAGES),
    []
  );
});

test("pure-integer strings convert, forging strings pass through for normalize (F19)", () => {
  const raw = calculateMappedLines(
    entry({ ttgir_lines: ["42", "3.5", "4oops", 7, NaN, 2.5] }),
    5,
    "kernel.ttgir",
    STAGES
  );
  assert.deepEqual(raw, [42, "3.5", "4oops", 7, NaN, 2.5]);
  // The single normalization point drops the forgeries with diagnostics —
  // 3.5/4oops must never appear as lines 3/4.
  const normalized = normalizeHighlightLines(raw, { offset: 1, lineCount: 100 });
  assert.deepEqual(normalized.lines, [7, 42]);
  assert.equal(normalized.droppedInvalid, 4);
});

test("non-array mapping fields yield [] instead of throwing (F19)", () => {
  for (const bad of [7, "10", { 0: 1 }, null]) {
    const out = calculateMappedLines(
      entry({ ttgir_lines: bad as unknown as number[] }),
      5,
      "kernel.ttgir",
      STAGES
    );
    assert.deepEqual(out, [], `field ${JSON.stringify(bad)}`);
  }
});

test("python lines resolve absolute lines on file match", () => {
  const info = { code: "x = 1\n", file_path: "model.py", start_line: 451 };
  const out = calculatePythonLines(
    { "5": { line: 452, file: "/src/model.py" } },
    5,
    info
  );
  assert.deepEqual(out, [452]);
});

test("python lines keep the legacy file-match direction and guards", () => {
  const info = { code: "x = 1\n", file_path: "model.py", start_line: 1 };
  // Mapping-side file must contain the panel-side path (legacy direction).
  assert.deepEqual(
    calculatePythonLines({ "5": { line: 1, file: "/src/other.py" } }, 5, info),
    []
  );
  assert.deepEqual(
    calculatePythonLines({ "5": { line: 1 } }, 5, info),
    []
  );
  assert.deepEqual(
    calculatePythonLines({ "5": { line: 1, file: "/src/model.py" } }, 5, {
      ...info,
      code: "",
    }),
    []
  );
  assert.deepEqual(
    calculatePythonLines(undefined, 5, info),
    []
  );
  // Empty panel-side path must not match every mapping file (includes("")).
  assert.deepEqual(
    calculatePythonLines({ "5": { line: 1, file: "/src/model.py" } }, 5, {
      ...info,
      file_path: "",
    }),
    []
  );
});

test("python lines prefer the call site for inlined code", () => {
  const info = { code: "x = 1\n", file_path: "kernel.py", start_line: 288 };
  // tl.cdiv inlines language/standard.py:43 into the kernel at line 288. The
  // entry's own file/line describe the callee, which fails the file match and
  // would leave the user's line unhighlighted.
  const inlined: Record<string, SourceMapping> = {
    "21": {
      line: 43,
      file: "/triton/language/standard.py",
      is_callsite: true,
      inlined_at_file: "/src/kernel.py",
      inlined_at_line: 288,
    },
  };
  assert.deepEqual(calculatePythonLines(inlined, 21, info), [288]);

  // Plain entries carry neither field and are unaffected.
  assert.deepEqual(
    calculatePythonLines(
      { "21": { line: 288, file: "/src/kernel.py" } },
      21,
      info
    ),
    [288]
  );

  // Both fields are required: one without the other is malformed, and mixing a
  // caller file with a callee line would point at the wrong place silently.
  assert.deepEqual(
    calculatePythonLines(
      {
        "21": {
          line: 43,
          file: "/triton/language/standard.py",
          inlined_at_line: 288,
        },
      },
      21,
      info
    ),
    []
  );
});

test("python lines pass out-of-range candidates to the normalizer (§4.4.1)", () => {
  const info = { code: "a\nb\n", file_path: "m.py", start_line: 1 };
  // Legacy filtered here with console.error; V2 returns the raw candidate so
  // the single normalization point drops it with badge diagnostics.
  const raw = calculatePythonLines(
    { "5": { line: 999, file: "/x/m.py" } },
    5,
    info
  );
  assert.deepEqual(raw, [999]);
  const normalized = normalizeHighlightLines(raw, { offset: 1, lineCount: 2 });
  assert.deepEqual(normalized.lines, []);
  assert.equal(normalized.droppedOutOfRange, 1);
});

test("python lines never coerce illegal types; strict-int strings only", () => {
  const info = { code: "a\nb\n", file_path: "m.py", start_line: 1 };
  const doc = { offset: 1, lineCount: 2 };
  const solve = (line: unknown) => {
    const raw = calculatePythonLines(
      { "5": { line: line as number, file: "/x/m.py" } },
      5,
      info
    );
    return { raw, normalized: normalizeHighlightLines(raw, doc) };
  };
  // Boolean/array/hex/float-string values pass through verbatim so the
  // normalizer drops them with diagnostics instead of forging highlights.
  for (const bad of [true, [459], "0x1cb", "455.0", 3.5]) {
    const { raw, normalized } = solve(bad);
    assert.deepEqual(raw, [bad]);
    assert.deepEqual(normalized.lines, []);
    assert.equal(normalized.droppedInvalid, 1);
  }
  // Integers and pure-integer strings (same strict rule as the IR side).
  assert.deepEqual(solve(2).normalized.lines, [2]);
  assert.deepEqual(solve("2").raw, [2]);
  assert.deepEqual(solve("2").normalized.lines, [2]);
});

test("kernel key always qualifies the kernel with its source (§4.2.1)", () => {
  assert.equal(buildKernelKey("left", 0), '["left",0]');
  assert.equal(buildKernelKey("http://x/t.json", "abc"), '["http://x/t.json","abc"]');
  // Same index in different sources must differ; same hash in different
  // sources must differ (bare index/hash are rejected shapes).
  assert.notEqual(buildKernelKey("left", 0), buildKernelKey("right", 0));
  assert.notEqual(buildKernelKey("a", "h"), buildKernelKey("b", "h"));
});

test("function range intersects the document window, empty yields null", () => {
  const doc = { offset: 1, lineCount: 100 };
  assert.deepEqual(
    intersectRangeWithDoc({ start: 10, end: 20 }, doc),
    { start: 10, end: 20 }
  );
  assert.deepEqual(
    intersectRangeWithDoc({ start: 90, end: 200 }, doc),
    { start: 90, end: 100 }
  );
  assert.equal(intersectRangeWithDoc({ start: 101, end: 200 }, doc), null);
  assert.equal(intersectRangeWithDoc({ start: 50, end: 40 }, doc), null);
  assert.equal(intersectRangeWithDoc(undefined, doc), null);
  assert.equal(
    intersectRangeWithDoc({ start: 10.5, end: 20 }, doc),
    null
  );
});

test("function range respects offset windows (python snippet semantics)", () => {
  const doc = { offset: 451, lineCount: 100 };
  assert.deepEqual(
    intersectRangeWithDoc({ start: 400, end: 460 }, doc),
    { start: 451, end: 460 }
  );
  assert.equal(intersectRangeWithDoc({ start: 1, end: 100 }, doc), null);
});

test("line count matches Monaco model line semantics", () => {
  assert.equal(lineCountOfContent(""), 1);
  assert.equal(lineCountOfContent("a"), 1);
  assert.equal(lineCountOfContent("a\n"), 2);
  assert.equal(lineCountOfContent("a\nb\nc"), 3);
  assert.equal(lineCountOfContent("a\r\nb\r\n"), 3);
});
