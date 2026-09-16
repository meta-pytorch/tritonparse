/**
 * Unit layer for the standalone overview ruler (§4.8): marker positioning
 * parity with the legacy formula, and even sampling for large sets.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  RULER_OVERFLOW_THRESHOLD,
  RULER_SAMPLE_SIZE,
  markerTopPercent,
  sampleMarkers,
} from "../../../src/components/rulerMath.ts";

test("marker positions match the legacy formula incl. clamps", () => {
  // Legacy: ((line - start) / (count - 1)) * 100 clamped to 2..98.
  assert.equal(markerTopPercent(1, 1, 101), 2);
  assert.equal(markerTopPercent(101, 1, 101), 98);
  assert.equal(markerTopPercent(51, 1, 101), 50);
  assert.equal(markerTopPercent(26, 1, 101), 25);
  // Out-of-range lines pin to the clamps, never off-strip.
  assert.equal(markerTopPercent(0, 1, 101), 2);
  assert.equal(markerTopPercent(500, 1, 101), 98);
  // Offset documents (python snippet semantics).
  assert.equal(markerTopPercent(451, 451, 101), 2);
  assert.equal(markerTopPercent(551, 451, 101), 98);
  // Single-line documents pin to the top clamp.
  assert.equal(markerTopPercent(1, 1, 1), 2);
});

test("sampling keeps small sets intact and samples large sets evenly", () => {
  assert.deepEqual(sampleMarkers([], 200), []);
  assert.deepEqual(sampleMarkers([3, 7], 200), [3, 7]);
  const lines = Array.from({ length: 7328 }, (_, i) => i + 1);
  const sampled = sampleMarkers(lines, RULER_SAMPLE_SIZE);
  assert.equal(sampled.length, RULER_SAMPLE_SIZE);
  assert.equal(sampled[0], 1);
  assert.equal(sampled[sampled.length - 1], 7328);
  assert.deepEqual([...new Set(sampled)], sampled);
  for (let i = 1; i < sampled.length; i++) {
    assert.ok(sampled[i] > sampled[i - 1]);
  }
});

test("sampling guards degenerate max (never divides by zero)", () => {
  assert.deepEqual(sampleMarkers([3, 7, 9], 1), [3]);
  assert.deepEqual(sampleMarkers([], 1), []);
  assert.deepEqual(sampleMarkers([3, 7, 9], 0), []);
  assert.deepEqual(sampleMarkers([3, 7, 9], -5), []);
});

test("overflow threshold and sample size match the F20 contract", () => {
  assert.equal(RULER_OVERFLOW_THRESHOLD, 5000);
  assert.equal(RULER_SAMPLE_SIZE, 200);
});
