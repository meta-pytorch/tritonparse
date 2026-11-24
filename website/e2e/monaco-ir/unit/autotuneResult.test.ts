/**
 * Unit tests for the autotune benchmark table helpers.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  formatBenchmarkDuration,
  getAutotuneBenchmarkRows,
  type AutotuneResult,
} from "../../../src/utils/dataLoader.ts";

const BEST = "BLOCK_SIZE_M: 32, num_warps: 1";
const OTHER = "BLOCK_SIZE_M: 16, num_warps: 1";

// Each configs_timings value is the do_bench quantiles [p50, p20, p80] in ms.
const FRESH: AutotuneResult = {
  best_config: BEST,
  configs_timings: {
    [OTHER]: [0.258336, 0.2476096, 0.2698944],
    [BEST]: [0.2488, 0.2343936, 0.4318528],
  },
  benchmark_duration: 1.7164421081542969,
  cache_hit: false,
};

test("rows show the quantiles as median/p20/p80 ms, best config first", () => {
  assert.deepEqual(getAutotuneBenchmarkRows(FRESH), [
    { config: BEST, isBest: true, median: "0.249", p20: "0.234", p80: "0.432" },
    { config: OTHER, isBest: false, median: "0.258", p20: "0.248", p80: "0.270" },
  ]);
  assert.equal(formatBenchmarkDuration(FRESH.benchmark_duration), "1.716s");
});

test("a cache hit has no benchmark duration but keeps its timings", () => {
  const cacheHit: AutotuneResult = {
    ...FRESH,
    benchmark_duration: null,
    cache_hit: true,
  };
  assert.equal(formatBenchmarkDuration(cacheHit.benchmark_duration), null);
  assert.deepEqual(
    getAutotuneBenchmarkRows(cacheHit),
    getAutotuneBenchmarkRows(FRESH)
  );
});

test("missing timings render as N/A and empty entries are skipped", () => {
  const partial: AutotuneResult = {
    ...FRESH,
    configs_timings: { [BEST]: [0.25, null], [OTHER]: [] },
  };
  assert.deepEqual(getAutotuneBenchmarkRows(partial), [
    { config: BEST, isBest: true, median: "0.250", p20: "N/A", p80: "N/A" },
  ]);
  assert.deepEqual(
    getAutotuneBenchmarkRows({ ...FRESH, configs_timings: null }),
    []
  );
});
