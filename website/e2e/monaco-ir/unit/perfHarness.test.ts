/**
 * Unit layer (§6.3.1) for the perf harness's pure helpers.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 *
 * Importing ../perf.mjs must not launch the runner: the import below doubles
 * as the main-guard regression test (these tests could never run if main()
 * fired on import).
 */
import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseArgs, summarize, normalizeExpectTriple, killProcAndCleanTmp } from "../perf.mjs";

const BASE = [
  "--scenario", "p2",
  "--artifact-dir", "/tmp/perf-test",
  "--trace-url", "http://127.0.0.1:1/t",
];

test("--lines parses a clean CSV", () => {
  assert.deepEqual(parseArgs([...BASE, "--lines", "3,2,3"]).lines, [3, 2, 3]);
});

test("--lines rejects non-numeric entries at parse time", () => {
  assert.throws(
    () => parseArgs([...BASE, "--lines", "3,x,5"]),
    /--lines must be a comma-separated list of positive integers/
  );
});

test("--lines rejects non-positive entries", () => {
  assert.throws(() => parseArgs([...BASE, "--lines", "3,0,5"]), /positive integers/);
  assert.throws(() => parseArgs([...BASE, "--lines", ""]), /positive integers/);
});

test("--lines defaults to null when absent", () => {
  assert.equal(parseArgs([...BASE]).lines, null);
});

test("summarize interpolates p50 for n=2 (not max)", () => {
  const s = summarize([20, 10]);
  assert.equal(s.n, 2);
  assert.equal(s.p50, 15);
  assert.equal(s.min, 10);
  assert.equal(s.max, 20);
});

test("summarize collapses n=1 and sorts input", () => {
  const one = summarize([7]);
  assert.deepEqual([one.min, one.p50, one.p95, one.max], [7, 7, 7, 7]);
  assert.equal(summarize([5, 1, 3, 2, 4]).p50, 3);
});

test("summarize p95 interpolates within the top rank", () => {
  // rank = 0.95 * 4 = 3.8 -> xs[3] + 0.8 * (xs[4] - xs[3]) = 4.8
  assert.ok(Math.abs(summarize([1, 2, 3, 4, 5]).p95 - 4.8) < 1e-9);
});

test("summarize throws on empty input", () => {
  assert.throws(() => summarize([]), /summarize: no samples/);
});

test("normalizeExpectTriple sorts every leg ascending", () => {
  const out = normalizeExpectTriple({ left: [9, 3], right: [5], python: [451, 120] });
  assert.deepEqual(out, { left: [3, 9], right: [5], python: [120, 451] });
});

test("normalizeExpectTriple does not mutate the input", () => {
  const triple = { left: [9, 3], right: [5], python: [451, 120] };
  normalizeExpectTriple(triple);
  assert.deepEqual(triple, { left: [9, 3], right: [5], python: [451, 120] });
});

test("killProcAndCleanTmp kills the proc and removes the profile dir", () => {
  const dir = mkdtempSync(join(tmpdir(), "perf-test-profile-"));
  writeFileSync(join(dir, "f"), "x");
  let killed = false;
  killProcAndCleanTmp({ kill: () => { killed = true; } }, dir);
  assert.equal(killed, true);
  assert.equal(existsSync(dir), false);
});

test("killProcAndCleanTmp tolerates a missing dir and a throwing kill", () => {
  const dir = join(tmpdir(), `perf-test-missing-${Date.now()}`);
  assert.doesNotThrow(() =>
    killProcAndCleanTmp({ kill: () => { throw new Error("gone"); } }, dir)
  );
});
