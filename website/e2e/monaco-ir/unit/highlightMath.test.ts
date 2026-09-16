/**
 * Unit layer (§6.3.1) for highlight-set math.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  getAnchorGroupedLines,
  mergeContiguousLines,
  normalizeHighlightLines,
  toAbsolute,
  toPhysical,
} from "../../../src/components/monaco/highlightMath.ts";

test("offset conversions round-trip at offset 1 and 451", () => {
  for (const offset of [1, 451]) {
    for (const phys of [1, 2, 100, 7328]) {
      assert.equal(toPhysical(toAbsolute(phys, offset), offset), phys);
    }
  }
  assert.equal(toPhysical(451, 451), 1);
  assert.equal(toAbsolute(1, 451), 451);
});

test("normalize sorts, dedups and keeps the legal set", () => {
  const out = normalizeHighlightLines([30, 10, 20, 20, 10], { offset: 1, lineCount: 100 });
  assert.deepEqual(out.lines, [10, 20, 30]);
  assert.equal(out.droppedInvalid, 0);
  assert.equal(out.droppedOutOfRange, 0);
});

test("normalize drops invalid entries with diagnostics, never throws", () => {
  const out = normalizeHighlightLines(
    [5, "7", NaN, 2.5, null, undefined, {}, [3], Infinity],
    { offset: 1, lineCount: 100 }
  );
  assert.deepEqual(out.lines, [5]);
  assert.equal(out.droppedInvalid, 8);
  assert.equal(out.droppedOutOfRange, 0);
});

test("normalize drops out-of-range lines without clamping to the edge", () => {
  const out = normalizeHighlightLines([1, 99, 100, 101, 999, 0, -3], {
    offset: 1,
    lineCount: 100,
  });
  // 999 must NOT appear as forged line 100; 0/-3 must NOT appear as line 1.
  assert.deepEqual(out.lines, [1, 99, 100]);
  assert.equal(out.droppedInvalid, 0);
  assert.equal(out.droppedOutOfRange, 4);
});

test("normalize respects offset windows (python snippet semantics)", () => {
  const out = normalizeHighlightLines([450, 451, 500, 550, 551], {
    offset: 451,
    lineCount: 100,
  });
  assert.deepEqual(out.lines, [451, 500, 550]);
  assert.equal(out.droppedOutOfRange, 2);
});

test("normalize treats non-array input as empty", () => {
  for (const raw of [undefined, null, 42, "10", {}]) {
    const out = normalizeHighlightLines(raw, { offset: 1, lineCount: 10 });
    assert.deepEqual(out.lines, []);
  }
});

test("merge produces minimal contiguous spans", () => {
  assert.deepEqual(mergeContiguousLines([]), []);
  assert.deepEqual(mergeContiguousLines([7]), [[7, 7]]);
  assert.deepEqual(mergeContiguousLines([1, 2, 3, 5, 6, 10]), [
    [1, 3],
    [5, 6],
    [10, 10],
  ]);
  // Defensive: exact duplicates collapse instead of producing empty spans.
  assert.deepEqual(mergeContiguousLines([4, 4, 5]), [[4, 5]]);
});

test("anchor grouping collects the clicked line plus same-anchor lines", () => {
  const mapping = {
    "2": { line: 2, ttgir_line: 2 },
    "4": { line: 4, ttgir_line: 2 },
    "6": { line: 6, ttgir_line: 6 },
  };
  assert.deepEqual(getAnchorGroupedLines(mapping, "ttgir_line", 2), [2, 4]);
  assert.deepEqual(getAnchorGroupedLines(mapping, "ttgir_line", 4), [4, 2]);
  assert.deepEqual(getAnchorGroupedLines(mapping, "ttgir_line", 6), [6]);
});

test("anchor grouping without mapping yields only the clicked line", () => {
  const mapping = { "2": { line: 2, ttgir_line: 2 } };
  assert.deepEqual(getAnchorGroupedLines(mapping, "ttgir_line", 1), [1]);
  assert.deepEqual(getAnchorGroupedLines(undefined, "ttgir_line", 9), [9]);
  assert.deepEqual(getAnchorGroupedLines({ "3": { line: 3 } }, "ttgir_line", 3), [3]);
});

test("non-numeric mapping keys flow into normalize diagnostics, not decorations", () => {
  const mapping = {
    "2": { line: 2, ttgir_line: 2 },
    loc42: { line: 99, ttgir_line: 2 },
  };
  const grouped = getAnchorGroupedLines(mapping, "ttgir_line", 2);
  assert.equal(grouped.length, 2);
  const out = normalizeHighlightLines(grouped, { offset: 1, lineCount: 100 });
  assert.deepEqual(out.lines, [2]);
  assert.equal(out.droppedInvalid, 1);
});

test("I001: fractional/trailing-char keys never forge line numbers", () => {
  // Exact counterexamples from rounds/002-codex.md (candidate 0513e65f).
  const mapping = {
    "2": { ttgir_line: 7 },
    "3.5": { ttgir_line: 7 },
    "4oops": { ttgir_line: 7 },
  };
  const grouped = getAnchorGroupedLines(mapping, "ttgir_line", 2);
  const out = normalizeHighlightLines(grouped, { offset: 1, lineCount: 20 });
  assert.deepEqual(out.lines, [2]);
  assert.equal(out.droppedInvalid, 2);
  assert.equal(out.droppedOutOfRange, 0);
});
