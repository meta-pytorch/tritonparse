/**
 * Unit tests for the kernel metadata comparison helper.
 * Run: npm run test:unit (plain node --test with type stripping, no build).
 */
import test from "node:test";
import assert from "node:assert/strict";
import {
  changedMetadataKeys,
  formatMetadataValue,
} from "../../../src/utils/metadataDiff.ts";

const base = {
  name: "add_kernel",
  metadata: { hash: "aaa", num_warps: 4, num_stages: 3, shared: 0, flag: true },
};

test("same-name kernels report only the fields whose value differs", () => {
  const next = {
    name: "add_kernel",
    metadata: { hash: "bbb", num_warps: 8, num_stages: 3, shared: 0, flag: true },
  };
  const changed = changedMetadataKeys(base, next);
  assert.deepEqual([...changed.keys()].sort(), ["hash", "num_warps"]);
  assert.equal(changed.get("num_warps"), "4");
});

test("a key that exists on only one side counts as changed", () => {
  const next = { name: "add_kernel", metadata: { ...base.metadata, extra: [1, 2] } };
  assert.deepEqual([...changedMetadataKeys(base, next).keys()], ["extra"]);
  assert.equal(changedMetadataKeys(base, next).get("extra"), "");
});

test("different kernel names are not compared", () => {
  const other = { name: "mul_kernel", metadata: { ...base.metadata, hash: "zzz" } };
  assert.equal(changedMetadataKeys(base, other).size, 0);
});

test("missing metadata or missing previous kernel gives no highlights", () => {
  assert.equal(changedMetadataKeys(undefined, base).size, 0);
  assert.equal(changedMetadataKeys(base, { name: "add_kernel" }).size, 0);
});

test("object values compare by their displayed JSON", () => {
  const a = { name: "k", metadata: { cfg: { x: 1 } } };
  const b = { name: "k", metadata: { cfg: { x: 1 } } };
  const c = { name: "k", metadata: { cfg: { x: 2 } } };
  assert.equal(changedMetadataKeys(a, b).size, 0);
  assert.equal(changedMetadataKeys(a, c).size, 1);
  assert.equal(formatMetadataValue(null), "null");
});
