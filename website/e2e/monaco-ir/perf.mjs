#!/usr/bin/env node
/**
 * Phase 2 performance matrix runner (design §6.2, 005 tasks 4-5).
 *
 * Committed measurement entry: every scenario drives the real UI through
 * real input (CDP mouse/keyboard, no debug-driven calls) and writes raw
 * samples plus a summary to --artifact-dir as JSON. Inputs stay outside the
 * repo (user traces, generated inputs) and are referenced by URL/path +
 * recorded hashes, so any run is re-verifiable.
 *
 * Usage:
 *   node e2e/monaco-ir/perf.mjs --scenario p1-cold --trace-url URL
 *     --renderer monaco --base-url URL --artifact-dir DIR [--iterations N]
 *   node e2e/monaco-ir/perf.mjs --scenario p2 --trace-url URL_100K
 *     --renderer monaco --base-url URL --artifact-dir DIR
 *     --lines 3,2,3 --expect expects.json
 *   node e2e/monaco-ir/perf.mjs --scenario p4 --input-file LOCAL_3X20M
 *     --base-url URL --artifact-dir DIR
 * Scenarios: p1-cold, p1-tabs, p1-hot, p2, p3, p4, p5, p1-first-frame.
 * p1-tabs alternates real tab switches (comparison tabs or single/open
 * overview) and times each settle; --iterations sets the switch count.
 * p1-first-frame injects a passive rAF sampler before page load and records
 * first-placeholder / first-panels-registered / first-content frames with
 * the full raw frame array (I010).
 * p4 takes --input-file <local .ndjson/.ndjson.gz> instead of --trace-url.
 * p2 takes --lines CSV + --expect JSON ({ line: [exact set] }) for inputs
 * where re-clicking one line would pass trivially on the retained set.
 * p1-hot takes --expect JSON ({ pyOffset, clicks: { line: {left,right,python} } })
 * to confirm the exact triple plus all three panels' decorations per click
 * (I009); without it only the python set is confirmed (legacy fallback).
 * p1-cold and p3 take --view single|comparison (default comparison); the
 * 100k-line gate input runs as --view single.
 *
 * Measurement boundaries (§6.2):
 * - p1-cold: navigationStart -> editors/viewers mounted AND first content
 *   rows painted. Cold = fresh chrome profile per iteration.
 * - p1-hot: mousedown dispatch -> highlight sets applied + next rAF paint.
 * - p2: same hot boundary on the Single view + mapping counts.
 * - p3: 5s of real wheel scrolling with a passive in-page sampler only (no
 *   screenshots, no CDP traffic during the window except the wheel events).
 * - p4: §6.2 memory gate on the 3-pane × 20MB two-kernel input: same-session
 *   File Diff baseline (drift check, warn-only at >=5%), 1s heap polling
 *   peak, 30s stable, kernel
 *   cycle 0->1->0, 30s final settle. Heap = usedJSHeapSize; no forced GC.
 * - p5: longest rendered IR line + comparison mount time on the given trace;
 *   conclusive (>5000 chars) inputs also run the wrap-off horizontal scroll
 *   gate on the long-line panel.
 */
import { mkdirSync, writeFileSync, readFileSync, mkdtempSync, rmSync } from "node:fs";
import { dirname, join, normalize, relative, resolve, sep } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createServer } from "node:http";
import {
  findChrome,
  launchChrome,
  listTargets,
  connectPageTarget,
  evaluate,
  waitForFunction,
  mouseClick,
  keyPress,
  captureScreenshot,
} from "./cdp.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "fixtures");

function parseArgs(argv) {
  const out = {
    scenario: null,
    traceUrl: null,
    inputFile: null,
    renderer: "monaco",
    baseUrl: "http://localhost:5173",
    artifactDir: null,
    iterations: null,
    chrome: null,
    lines: null,
    expect: null,
    expectPath: null,
    view: "comparison",
    hot: false,
  };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--scenario") out.scenario = argv[++i];
    else if (argv[i] === "--trace-url") out.traceUrl = argv[++i];
    else if (argv[i] === "--input-file") out.inputFile = argv[++i];
    else if (argv[i] === "--renderer") out.renderer = argv[++i];
    else if (argv[i] === "--base-url") out.baseUrl = argv[++i];
    else if (argv[i] === "--artifact-dir") out.artifactDir = argv[++i];
    else if (argv[i] === "--iterations") out.iterations = Number(argv[++i]);
    else if (argv[i] === "--chrome") out.chrome = argv[++i];
    else if (argv[i] === "--lines") out.lines = argv[++i].split(",").map(Number);
    else if (argv[i] === "--expect") { out.expectPath = argv[++i]; out.expect = JSON.parse(readFileSync(out.expectPath, "utf8")); }
    else if (argv[i] === "--view") out.view = argv[++i];
    else if (argv[i] === "--hot") out.hot = true;
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  if (!["p1-cold", "p1-tabs", "p1-first-frame", "p1-hot", "p2", "p3", "p4", "p5"].includes(out.scenario)) {
    throw new Error("--scenario must be one of p1-cold p1-tabs p1-first-frame p1-hot p2 p3 p4 p5");
  }
  if (!out.artifactDir) throw new Error("--artifact-dir is required");
  // NaN is not nullish: without this, a non-numeric --iterations silently
  // yields zero-iteration scenarios that still write "passing" artifacts.
  if (out.iterations !== null && !(Number.isInteger(out.iterations) && out.iterations > 0)) {
    throw new Error("--iterations must be a positive integer");
  }
  // Same NaN-trap as --iterations: "--lines 3,x,5" parses to [3, NaN, 5]
  // and NaN later interpolates into evaluate code as a literal, failing
  // obscurely ("single line NaN not clickable") far from the typo.
  if (out.lines !== null && !(out.lines.length > 0 && out.lines.every((n) => Number.isInteger(n) && n > 0))) {
    throw new Error("--lines must be a comma-separated list of positive integers");
  }
  if (!["monaco", "legacy"].includes(out.renderer)) throw new Error("--renderer must be monaco|legacy");
  if (!["comparison", "single"].includes(out.view)) throw new Error("--view must be comparison|single");
  // Fail loudly on silently-ignored combinations (parseArgs style): --hot
  // is measured only by p1-cold on Single, and p3 Single is monaco-only
  // (legacy never enters Single, so the scroll would target the overview
  // page while the artifact claims view=single).
  if (out.hot && (out.scenario !== "p1-cold" || out.view !== "single")) {
    throw new Error("--hot requires --scenario p1-cold --view single");
  }
  if (out.scenario === "p3" && out.view === "single" && out.renderer !== "monaco") {
    throw new Error("p3 --view single requires --renderer monaco");
  }
  if (out.scenario === "p4" && !out.inputFile) throw new Error("p4 requires --input-file");
  if (out.scenario !== "p4" && !out.traceUrl) throw new Error(`${out.scenario} requires --trace-url`);
  out.baseUrl = out.baseUrl.replace(/\/$/, "").replace("://localhost", "://127.0.0.1");
  return out;
}

/** Mirror of run.mjs startFixtureServer (kept in sync deliberately). */
function startFixtureServer(extraRoutes = {}) {
  const server = createServer((req, res) => {
    try {
      const urlPath = decodeURIComponent(req.url.split("?")[0]).replace(/^\/+/, "");
      if (extraRoutes["/" + urlPath]) {
        const { body, type } = extraRoutes["/" + urlPath];
        res.writeHead(200, { "Content-Type": type, "Access-Control-Allow-Origin": "*" });
        res.end(body);
        return;
      }
      // Containment: never serve outside FIXTURES (`..` segments escape
      // the join). Localhost-only, but a 404 is cheaper than a hole.
      const filePath = normalize(join(FIXTURES, urlPath));
      const rel = relative(FIXTURES, filePath);
      if (rel === ".." || rel.startsWith(`..${sep}`)) {
        res.writeHead(404, { "Access-Control-Allow-Origin": "*" });
        res.end("no such fixture");
        return;
      }
      const body = readFileSync(filePath);
      res.writeHead(200, { "Content-Type": "application/octet-stream", "Access-Control-Allow-Origin": "*" });
      res.end(body);
    } catch {
      res.writeHead(404, { "Access-Control-Allow-Origin": "*" });
      res.end("no such fixture");
    }
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

async function freshPage(chromeOpt, extraArgs = []) {
  const userDataDir = mkdtempSync(join(tmpdir(), "monaco-perf-"));
  const { proc, port: debugPort } = await launchChrome({
    chromePath: chromeOpt ?? findChrome(),
    userDataDir,
    extraArgs: ["--window-size=1920,1080", ...extraArgs],
  });
  try {
    let pageTarget = null;
    for (let i = 0; i < 100; i++) {
      await new Promise((r) => setTimeout(r, 200));
      const targets = await listTargets(debugPort);
      pageTarget = targets.find((t) => t.type === "page");
      if (pageTarget) break;
    }
    if (!pageTarget) throw new Error("no page target in fresh chrome");
    const s = await connectPageTarget(pageTarget);
    await s.send("Page.enable");
    await s.send("Runtime.enable");
    return { s, proc, userDataDir };
  } catch (e) {
    // Setup failed after launch: kill the orphaned chrome, remove the
    // profile dir it can no longer need, then rethrow.
    killProcAndCleanTmp(proc, userDataDir);
    throw e;
  }
}

/** Kill a scenario chrome and remove its temp profile (multi-hundred-MB
 * dirs would otherwise accumulate under os.tmpdir() across perf runs). */
function killProcAndCleanTmp(proc, userDataDir) {
  try { proc.kill(); } catch { /* ignore */ }
  try {
    rmSync(userDataDir, { recursive: true, force: true });
  } catch (e) {
    console.warn(`warning: tmp profile cleanup failed: ${e?.message ?? e}`);
  }
}

function rendererParam(renderer) {
  return renderer === "monaco" ? "&renderer=monaco" : "";
}

function summarize(samples) {
  const xs = [...samples].sort((a, b) => a - b);
  if (xs.length === 0) throw new Error("summarize: no samples");
  // Linear-interpolation quantiles (numpy 'linear' default): with n=2,
  // p50 is the mean of the two samples instead of the max, so
  // low-iteration runs don't report a misleading p50 == max.
  const q = (p) => {
    const rank = p * (xs.length - 1);
    const lo = Math.floor(rank);
    const hi = Math.ceil(rank);
    return xs[lo] + (xs[hi] - xs[lo]) * (rank - lo);
  };
  return { n: xs.length, min: xs[0], p50: q(0.5), p95: q(0.95), max: xs[xs.length - 1] };
}

/** Wheel until a python absolute line's test point hits CONTENT_TEXT. */
async function wheelToPyLine(s, abs) {
  const c = await evaluate(s, `() => {
    const r = window.__TRITONPARSE_DEBUG.panels.python.editor.getDomNode().getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }`);
  for (let i = 0; i < 200; i++) {
    const xy = await evaluate(s, `() => {
      const D = window.__TRITONPARSE_DEBUG;
      const ed = D.panels.python.editor;
      const pos = ed.getScrolledVisiblePosition({ lineNumber: ${abs}, column: 1 });
      if (!pos) return null;
      const r = ed.getDomNode().getBoundingClientRect();
      const x = r.x + pos.left + 10, y = r.y + pos.top + pos.height / 2;
      if (y < r.y + 2 || y > r.y + r.height - 2) return null;
      const t = ed.getTargetAtClientPoint(x, y);
      if (!t || t.type !== D.monaco.editor.MouseTargetType.CONTENT_TEXT) return null;
      return { x, y };
    }`);
    if (xy) return xy;
    const dir = await evaluate(s, `() => {
      const ed = window.__TRITONPARSE_DEBUG.panels.python.editor;
      const vs = ed.getVisibleRanges()[0];
      if (!vs) return 1;
      const mid = (vs.startLineNumber + vs.endLineNumber) / 2;
      return ${abs} > mid ? 1 : -1;
    }`);
    await s.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: c.x, y: c.y, deltaX: 0, deltaY: dir * 300 });
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error(`py ${abs} never clickable`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Wait until the renderer main thread answers promptly (huge-trace parsing blocks it). */
async function waitResponsive(s, { tries = 12, settleMs = 5000 } = {}) {
  for (let i = 0; i < tries; i++) {
    const t0 = Date.now();
    try {
      await evaluate(s, `() => 1`);
      if (Date.now() - t0 < 10000) {
        await sleep(1500);
        return;
      }
    } catch {
      // CDP timed out: still blocked; keep waiting.
    }
    await sleep(settleMs);
  }
  throw new Error("renderer stayed unresponsive");
}

/** Single-shot evaluate retried across main-thread stalls (heavy phases).
 * Attempt-capped: a deterministically failing fnSource must loud-fail
 * instead of hanging the whole perf run. */
async function evalRetry(s, fnSource, what, maxAttempts = 10) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await evaluate(s, fnSource);
    } catch (e) {
      if (attempt >= maxAttempts) throw new Error(`eval ${what} failed after ${attempt} attempts: ${String(e).slice(0, 200)}`);
      console.log(`  eval ${what} attempt ${attempt} stalled, retrying: ${String(e).slice(0, 90)}`);
      await waitResponsive(s, { tries: 40 });
    }
  }
}

/** Real-mouse click at coordinates, retried across main-thread stalls.
 * Attempt-capped for the same loud-fail reason as evalRetry. */
async function resilientClick(s, resolve, what, maxAttempts = 10) {
  for (let attempt = 1; ; attempt++) {
    try {
      await waitResponsive(s, { tries: 40 });
      const pt = await resolve();
      if (!pt) throw new Error(`no target for ${what}`);
      await mouseClick(s, pt.x, pt.y);
      return;
    } catch (e) {
      if (String(e.message ?? e).includes("no target for")) throw e;
      if (attempt >= maxAttempts) throw new Error(`click ${what} failed after ${attempt} attempts: ${String(e).slice(0, 200)}`);
      console.log(`  click ${what} attempt ${attempt} stalled, retrying: ${String(e).slice(0, 90)}`);
    }
  }
}

/** Real-mouse click on an element found by selector+exact text. */
async function clickText(s, selector, text) {
  const resolve = async () => {
    const rect = await evaluate(
      s,
      `() => {
        const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
        const el = els.find((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
        if (!el) return null;
        el.scrollIntoView({ block: 'center' });
        const r = el.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      }`
    );
    if (!rect) return null;
    await sleep(400);
    return evaluate(
      s,
      `() => {
        const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
        const el = els.find((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
        const r = el.getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
      }`
    );
  };
  await resilientClick(s, resolve, `${selector} "${text}"`);
}

/** Change a native select with real keyboard input (arrows switch options). */
async function selectByKeyboard(s, targetValue) {
  const info = await evalRetry(
    s,
    `() => {
      // Generic values ("0"/"1") must match exactly one dropdown; fail
      // loudly on ambiguity instead of keystroking the wrong element.
      const els = [...document.querySelectorAll('select')].filter((e) =>
        [...e.options].some((op) => op.value === ${JSON.stringify(targetValue)}));
      if (els.length === 0) return null;
      if (els.length > 1) return { ambiguous: els.length };
      const el = els[0];
      el.scrollIntoView({ block: 'center' });
      const r = el.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2,
        options: [...el.options].map((o) => o.value), current: el.value };
    }`,
    `find select ${targetValue}`
  );
  if (!info) throw new Error(`no select offering ${targetValue}`);
  if (info.ambiguous !== undefined) throw new Error(`select ${targetValue} matches ${info.ambiguous} dropdowns (expected exactly 1)`);
  await sleep(300);
  await resilientClick(s, async () => ({ x: info.x, y: info.y }), `select ${targetValue}`);
  const cur = info.options.indexOf(info.current);
  const tgt = info.options.indexOf(targetValue);
  const key = tgt >= cur ? "ArrowDown" : "ArrowUp";
  const code = tgt >= cur ? 40 : 38;
  for (let i = 0; i < Math.abs(tgt - cur); i++) {
    await evalRetry(s, `() => 1`, "pre-key");
    for (;;) {
      try {
        await keyPress(s, key, { code: key, windowsVirtualKeyCode: code });
        break;
      } catch (e) {
        console.log(`  select key ${key} stalled, retrying: ${String(e).slice(0, 80)}`);
        await waitResponsive(s, { tries: 40 });
      }
    }
    await sleep(150);
  }
  for (;;) {
    try {
      await keyPress(s, "Escape", { code: "Escape", windowsVirtualKeyCode: 27 });
      break;
    } catch (e) {
      console.log(`  select Escape stalled, retrying: ${String(e).slice(0, 80)}`);
      await waitResponsive(s, { tries: 40 });
    }
  }
  await waitForFunction(
    s,
    `() => {
      const el = [...document.querySelectorAll('select')].find((e) =>
        [...e.options].some((op) => op.value === ${JSON.stringify(targetValue)}));
      return el && el.value === ${JSON.stringify(targetValue)} ? true : false;
    }`,
    { timeoutMs: 60000 }
  );
}

/** Resolve a text button's coordinates (untimed settle), click it, return tClick. */
async function clickTextTimed(s, selector, text) {
  const pt = await evaluate(s, `() => {
    const el = [...document.querySelectorAll(${JSON.stringify(selector)})]
      .find((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
    if (!el) return null;
    el.scrollIntoView({ block: 'center' });
    const r = el.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }`);
  if (!pt) throw new Error(`no ${selector} with text ${text}`);
  await new Promise((r) => setTimeout(r, 400));
  const tClick = Date.now();
  await mouseClick(s, pt.x, pt.y);
  return { tClick };
}

/** Real overview click entering Single on the first .tt card. */
async function openFirstSingle(s) {
  await waitForFunction(s, `() => [...document.querySelectorAll('h3')].length > 0 ? true : false`, { timeoutMs: 300000 });
  const h3 = await evaluate(s, `() => {
    const h = [...document.querySelectorAll('h3')].find((x) => x.textContent.includes(".tt"));
    if (!h) return null;
    h.scrollIntoView({ block: "center" });
    const r = h.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2, text: h.textContent.trim() };
  }`);
  if (!h3) throw new Error("no ttir/ttgir card found");
  await new Promise((r) => setTimeout(r, 400));
  const tClick = Date.now();
  await mouseClick(s, h3.x, h3.y);
  return { file: h3.text, tClick };
}

async function scenarioP1Cold(args) {
  const iterations = args.iterations ?? 3;
  const results = [];
  for (let i = 0; i < iterations; i++) {
    const { s, proc, userDataDir } = await freshPage(args.chrome);
    try {
      // I010: --hot on the comparison view measures a real hot open: load
      // the overview first (cold; the loader initializes), then open the IR
      // view through the tab click and time click->content.
      const hotComparison = args.hot && args.view === "comparison";
      const url = args.view === "single" || hotComparison
        ? `${args.baseUrl}/?json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`
        : `${args.baseUrl}/?view=ir_code_comparison&json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
      const tNav = Date.now();
      await s.send("Page.navigate", { url });
      let tClick = null;
      let file = null;
      if (args.view === "single") {
        ({ file, tClick } = await openFirstSingle(s));
      } else if (hotComparison) {
        await waitForFunction(s, `() => [...document.querySelectorAll('h3')].length > 0 ? true : false`, { timeoutMs: 300000 });
        await waitResponsive(s, { tries: 6, settleMs: 2000 });
        const tab = await evaluate(s, `() => {
          const el = [...document.querySelectorAll('button')].find((e) => e.textContent.trim() === 'IR Code');
          if (!el) return null;
          el.scrollIntoView({ block: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }`);
        if (!tab) throw new Error("no IR Code tab");
        await new Promise((r) => setTimeout(r, 400));
        tClick = Date.now();
        await mouseClick(s, tab.x, tab.y);
        file = "IR Code tab";
      }
      const readyFn =
        args.renderer === "monaco"
          ? (args.view === "single"
            ? `() => {
                const ed = window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor;
                if (!ed) return false;
                const rows = ed.getDomNode().querySelectorAll(".view-lines .view-line").length;
                return rows > 0 ? true : false;
              }`
            : `() => {
                const P = window.__TRITONPARSE_DEBUG?.panels;
                if (!(P?.left?.editor && P?.right?.editor && P?.python?.editor)) return false;
                const rows = P.left.editor.getDomNode().querySelectorAll(".view-lines .view-line").length;
                return rows > 0 ? true : false;
              }`)
          : `() => document.querySelectorAll('[data-line-number]').length > 100 ? true : false`;
      await waitForFunction(s, readyFn, { timeoutMs: 300000 });
      const tReady = Date.now();
      const info = await evaluate(s, `() => {
        const nav = performance.getEntriesByType("navigation")[0];
        const paint = performance.getEntriesByType("paint").map((p) => ({ name: p.name, t: Math.round(p.startTime) }));
        return {
          navMs: nav ? { domContentLoaded: Math.round(nav.domContentLoadedEventEnd), load: Math.round(nav.loadEventEnd) } : null,
          paint,
        };
      }`);
      const row = { iteration: i, coldNavToReadyMs: tReady - tNav, ...info };
      if (tClick !== null) {
        row.file = file;
        row.clickToContentMs = tReady - tClick;
      }
      if (args.hot && args.view === "single") {
        // Hot reopen in the same profile (loader already initialized): real
        // Back click, then re-enter Single and time click->content again.
        await clickText(s, "button", "Back");
        await waitForFunction(s, `() => [...document.querySelectorAll('h3')].length > 0 ? true : false`, { timeoutMs: 60000 });
        const second = await openFirstSingle(s);
        await waitForFunction(s, readyFn, { timeoutMs: 300000 });
        row.hotReopenMs = Date.now() - second.tClick;
      }
      results.push(row);
      console.log(`  iter ${i}: ready in ${tReady - tNav}ms${tClick !== null ? ` (click->content ${tReady - tClick}ms)` : ""}${row.hotReopenMs !== undefined ? ` (hot reopen ${row.hotReopenMs}ms)` : ""}`);
    } finally {
      try { s.close(); } catch { /* ignore */ }
      killProcAndCleanTmp(proc, userDataDir);
    }
  }
  return { scenario: "p1-cold", renderer: args.renderer, view: args.view, traceUrl: args.traceUrl, results };
}

async function scenarioP1Tabs(args) {
  // I010: repeated real tab switches with per-switch settle timing.
  // comparison: "IR Code" <-> "Kernel Overview"; single: h3 card <-> "Back".
  // Settle = endpoint mounted + first content painted (editors + rows for
  // monaco IR endpoints, [data-line-number] rows for legacy, h3 for overview).
  const switches = args.iterations ?? 6;
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    const url = `${args.baseUrl}/?json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    await s.send("Page.navigate", { url });
    const h3Fn = `() => [...document.querySelectorAll('h3')].length > 0 ? true : false`;
    await waitForFunction(s, h3Fn, { timeoutMs: 300000 });
    await waitResponsive(s, { tries: 6, settleMs: 2000 });
    const cmpReadyFn = args.renderer === "monaco"
      ? `() => {
          const P = window.__TRITONPARSE_DEBUG?.panels;
          if (!(P?.left?.editor && P?.right?.editor && P?.python?.editor)) return false;
          return P.left.editor.getDomNode().querySelectorAll(".view-lines .view-line").length > 0 ? true : false;
        }`
      : `() => document.querySelectorAll('[data-line-number]').length > 100 ? true : false`;
    const singleReadyFn = args.renderer === "monaco"
      ? `() => {
          const ed = window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor;
          if (!ed) return false;
          return ed.getDomNode().querySelectorAll(".view-lines .view-line").length > 0 ? true : false;
        }`
      : cmpReadyFn;
    const results = [];
    for (let i = 0; i < switches; i++) {
      if (args.view === "single") {
        if (i % 2 === 0) {
          const { file, tClick } = await openFirstSingle(s);
          await waitForFunction(s, singleReadyFn, { timeoutMs: 300000 });
          results.push({ switch: i, to: "single", file, settleMs: Date.now() - tClick });
        } else {
          const { tClick } = await clickTextTimed(s, "button", "Back");
          await waitForFunction(s, h3Fn, { timeoutMs: 120000 });
          results.push({ switch: i, to: "overview", settleMs: Date.now() - tClick });
        }
      } else {
        const tab = i % 2 === 0 ? "IR Code" : "Kernel Overview";
        const { tClick } = await clickTextTimed(s, "button", tab);
        await waitForFunction(s, tab === "IR Code" ? cmpReadyFn : h3Fn, { timeoutMs: 300000 });
        results.push({ switch: i, to: tab, settleMs: Date.now() - tClick });
      }
      console.log(`  switch ${i} -> ${results[i].to}: settle ${results[i].settleMs}ms`);
    }
    return {
      scenario: "p1-tabs", renderer: args.renderer, view: args.view,
      traceUrl: args.traceUrl, results,
      summary: { settleMs: summarize(results.map((r) => r.settleMs)) },
    };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

async function scenarioP1FirstFrame(args) {
  // I010: committed first-frame verification. A passive rAF sampler is
  // injected before any page script (addScriptToEvaluateOnNewDocument) and
  // records per-frame {placeholder visible, debug panels registered, content
  // rows painted}. Panels register after the synchronous model work inside
  // editor mount, so firstPlaceholderT < firstPanelsT <= firstContentT proves
  // the placeholder painted before the sync model work. Raw frames ship in
  // the result JSON.
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    await s.send("Page.addScriptToEvaluateOnNewDocument", {
      source: `window.__P1FF = { t0: -1, mark: -1, frames: [], done: false };
        (function rec(t) {
          if (window.__P1FF.done) return;
          if (window.__P1FF.t0 < 0) window.__P1FF.t0 = t;
          const ph = [...document.querySelectorAll('.mp-panel-loading')].some((n) => {
            const r = n.getBoundingClientRect(); return r.width > 0 && r.height > 0;
          });
          const panels = Object.keys(window.__TRITONPARSE_DEBUG?.panels ?? {}).length;
          const rows = document.querySelectorAll('.view-lines .view-line').length;
          window.__P1FF.frames.push({
            t: Math.round((t - window.__P1FF.t0) * 10) / 10,
            ph: ph ? 1 : 0, panels, rows,
          });
          requestAnimationFrame(rec);
        })(performance.now());`,
    });
    const url = args.view === "single"
      ? `${args.baseUrl}/?json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`
      : `${args.baseUrl}/?view=ir_code_comparison&json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    await s.send("Page.navigate", { url });
    if (args.view === "single") {
      await openFirstSingle(s);
      await evaluate(s, `() => { window.__P1FF.mark = performance.now(); return true; }`);
    }
    const readyFn = args.view === "single"
      ? `() => {
          const ed = window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor;
          if (!ed) return false;
          return ed.getDomNode().querySelectorAll(".view-lines .view-line").length > 0 ? true : false;
        }`
      : `() => {
          const P = window.__TRITONPARSE_DEBUG?.panels;
          if (!(P?.left?.editor && P?.right?.editor && P?.python?.editor)) return false;
          return P.left.editor.getDomNode().querySelectorAll(".view-lines .view-line").length > 0 ? true : false;
        }`;
    await waitForFunction(s, readyFn, { timeoutMs: 300000 });
    // Panels register in a React effect after mount/paint; extend sampling
    // until registration lands so the panels boundary is always captured.
    await waitForFunction(
      s,
      `() => Object.keys(window.__TRITONPARSE_DEBUG?.panels ?? {}).length > 0 ? true : false`,
      { timeoutMs: 60000 }
    );
    const data = await evaluate(s, `() => {
      window.__P1FF.done = true;
      return { t0: window.__P1FF.t0, mark: window.__P1FF.mark, frames: window.__P1FF.frames };
    }`);
    // No pre-click editor exists in either view, so global firsts are the
    // post-navigation (comparison) / post-click (single) firsts; the mark is
    // informational only (it trails the click by CDP roundtrips).
    const markRel = data.mark >= 0 ? data.mark - data.t0 : 0;
    const first = (pred) => {
      const f = data.frames.find(pred);
      return f ? f.t : null;
    };
    return {
      scenario: "p1-first-frame", renderer: args.renderer, view: args.view,
      traceUrl: args.traceUrl,
      markRelMs: Math.round(markRel * 10) / 10,
      firstPlaceholderFrameMs: first((f) => f.ph === 1),
      firstPanelsFrameMs: first((f) => f.panels > 0),
      firstContentFrameMs: first((f) => f.rows > 0),
      frameCount: data.frames.length,
      frames: data.frames,
    };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

/**
 * Normalize a p1-hot --expect click triple for comparison. The page-side
 * `got` sets are sorted ascending before the JSON.stringify equality check,
 * so the expected sets must be sorted too — otherwise a --expect file that
 * lists lines out of order never matches despite correct highlights (p2
 * already normalizes via expectedSorted for the same reason).
 */
function normalizeExpectTriple(triple) {
  const sorted = (xs) => [...xs].sort((a, b) => a - b);
  return { left: sorted(triple.left), right: sorted(triple.right), python: sorted(triple.python) };
}

async function scenarioP1Hot(args) {
  const iterations = args.iterations ?? 5;
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    const url = `${args.baseUrl}/?view=ir_code_comparison&json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    await s.send("Page.navigate", { url });
    const results = [];
    if (args.renderer === "monaco") {
      await waitForFunction(s, `() => {
        const P = window.__TRITONPARSE_DEBUG?.panels;
        return P?.left?.editor && P?.right?.editor && P?.python?.editor ? true : false;
      }`, { timeoutMs: 300000 });
      for (let i = 0; i < iterations; i++) {
        const abs = i % 2 === 0 ? 451 : 601;
        const xy = await wheelToPyLine(s, abs);
        const t0 = Date.now();
        await mouseClick(s, xy.x, xy.y);
        const triple = args.expect?.clicks?.[String(abs)] ?? null;
        if (triple) {
          // I009: confirm this input's exact highlight triple AND the three
          // panels' line decorations (python physical = absolute - offset + 1).
          const pyOffset = args.expect.pyOffset ?? 1;
          await waitForFunction(s, `() => {
            const P = window.__TRITONPARSE_DEBUG.panels;
            const exp = ${JSON.stringify(normalizeExpectTriple(triple))};
            for (const id of ["left", "right"]) {
              const got = P[id].editor.getModel().getAllDecorations()
                .filter((d) => d.options.className === 'mp-highlighted-line')
                .map((d) => d.range.startLineNumber).sort((a, b) => a - b);
              if (JSON.stringify(got) !== JSON.stringify(exp[id])) return false;
            }
            const pyExp = exp.python.map((a) => a - ${pyOffset} + 1);
            const pyGot = P.python.editor.getModel().getAllDecorations()
              .filter((d) => d.options.className === 'mp-highlighted-line')
              .map((d) => d.range.startLineNumber).sort((a, b) => a - b);
            if (JSON.stringify(pyGot) !== JSON.stringify(pyExp)) return false;
            return true;
          }`, { timeoutMs: 15000, pollingMs: 25 });
        } else {
          await waitForFunction(s, `() => {
            const P = window.__TRITONPARSE_DEBUG.panels;
            return JSON.stringify(P.python.getHighlights()) === ${JSON.stringify(JSON.stringify([abs]))} ? true : false;
          }`, { timeoutMs: 15000, pollingMs: 25 });
        }
        const tSets = Date.now();
        // Next frame after the sets landed.
        await evaluate(s, `() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`, { awaitPromise: true });
        const tFrame = Date.now();
        const sets = await evaluate(s, `() => {
          const P = window.__TRITONPARSE_DEBUG.panels;
          return { left: P.left.getHighlights().length, right: P.right.getHighlights().length };
        }`);
        results.push({ iteration: i, line: abs, clickToSetsMs: tSets - t0, clickToNextFrameMs: tFrame - t0, ...sets });
        console.log(`  iter ${i} (py${abs}): sets ${tSets - t0}ms, frame ${tFrame - t0}ms`);
      }
    } else {
      await waitForFunction(s, `() => document.querySelectorAll('[data-line-number]').length > 100 ? true : false`, { timeoutMs: 300000 });
      for (let i = 0; i < iterations; i++) {
        const abs = i % 2 === 0 ? 451 : 601;
        const xy = await evaluate(s, `() => {
          const py = document.querySelector('code.language-python');
          const row = py.querySelector('[data-line-number="${abs}"]');
          if (!row) return null;
          row.scrollIntoView({ block: "center" });
          const r = row.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }`);
        if (!xy) throw new Error(`legacy py ${abs} row missing`);
        await new Promise((r) => setTimeout(r, 400));
        const t0 = Date.now();
        await mouseClick(s, xy.x, xy.y);
        const triple = args.expect?.clicks?.[String(abs)] ?? null;
        if (triple) {
          // I009 legacy leg: exact highlighted rows on all three panels
          // (document order: left block, right block, then python).
          await waitForFunction(s, `() => {
            const exp = ${JSON.stringify(normalizeExpectTriple(triple))};
            const rows = (root) => [...root.querySelectorAll('.highlighted-line')]
              .map((r) => Number(r.getAttribute("data-line-number"))).sort((a, b) => a - b);
            const mlir = document.querySelectorAll('code.language-mlir');
            const py = document.querySelector('code.language-python');
            if (mlir.length < 2 || !py) return false;
            return JSON.stringify(rows(mlir[0])) === JSON.stringify(exp.left)
              && JSON.stringify(rows(mlir[1])) === JSON.stringify(exp.right)
              && JSON.stringify(rows(py)) === JSON.stringify(exp.python) ? true : false;
          }`, { timeoutMs: 15000, pollingMs: 25 });
        } else {
          await waitForFunction(s, `() => {
            const py = document.querySelector('code.language-python');
            const hit = [...py.querySelectorAll('.highlighted-line')].map((r) => Number(r.getAttribute("data-line-number")));
            return JSON.stringify(hit) === ${JSON.stringify(JSON.stringify([abs]))} ? true : false;
          }`, { timeoutMs: 15000, pollingMs: 25 });
        }
        const tSets = Date.now();
        await evaluate(s, `() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`, { awaitPromise: true });
        const tFrame = Date.now();
        results.push({ iteration: i, line: abs, clickToSetsMs: tSets - t0, clickToNextFrameMs: tFrame - t0 });
        console.log(`  iter ${i} (py${abs}): sets ${tSets - t0}ms, frame ${tFrame - t0}ms`);
      }
    }
    const sets = results.map((r) => r.clickToSetsMs);
    const frames = results.map((r) => r.clickToNextFrameMs);
    return { scenario: "p1-hot", renderer: args.renderer, traceUrl: args.traceUrl, results, summary: { clickToSetsMs: summarize(sets), clickToNextFrameMs: summarize(frames) } };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

async function scenarioP2(args) {
  // Single view hot click with full-table anchor grouping: measures the
  // click boundary and counts real mappings (group size + scanned entries).
  // --lines "3,2,3" rotates clicked lines (re-clicking one line would pass
  // trivially on the retained set); --expect FILE asserts exact highlight
  // sets per line ({ "3": [3, 10003, ...], ... }).
  const iterations = args.iterations ?? 3;
  const lines = args.lines ?? [3];
  if (!args.expect && iterations > 1 && new Set(lines).size < 2) {
    throw new Error("p2 without --expect needs --lines rotation (at least 2 distinct lines): re-clicking one line passes trivially on retained highlights");
  }
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    const url = `${args.baseUrl}/?json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    await s.send("Page.navigate", { url });
    await waitForFunction(s, `() => [...document.querySelectorAll('h3')].length > 0 ? true : false`, { timeoutMs: 300000 });
    // Enter Single on the first kernel's first IR (real overview click).
    const h3 = await evaluate(s, `() => {
      const h = [...document.querySelectorAll('h3')].find((x) => x.textContent.includes(".tt"));
      if (!h) return null;
      h.scrollIntoView({ block: "center" });
      const r = h.getBoundingClientRect();
      return { x: r.x + r.width / 2, y: r.y + r.height / 2, text: h.textContent.trim() };
    }`);
    if (!h3) throw new Error("no ttir/ttgir card found");
    await new Promise((r) => setTimeout(r, 400));
    await mouseClick(s, h3.x, h3.y);
    await waitForFunction(s, `() => !!window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor`, { timeoutMs: 120000 });
    const results = [];
    for (let i = 0; i < iterations; i++) {
      const line = lines[i % lines.length];
      const expected = args.expect?.[String(line)] ?? null;
      // Sorted symmetrically with the in-page `got`: user-supplied --expect
      // JSON may list lines in any order, and an unsorted literal would
      // never equal the sorted observation (misleading timeout).
      const expectedSorted = expected === null ? null : [...expected].sort((a, b) => a - b);
      // Click a line near the top (real mouse, CONTENT_TEXT pre-checked).
      const xy = await evaluate(s, `() => {
        const D = window.__TRITONPARSE_DEBUG;
        const ed = D.panels["single-viewer"].editor;
        const pos = ed.getScrolledVisiblePosition({ lineNumber: ${line}, column: 1 });
        if (!pos) return null;
        const r = ed.getDomNode().getBoundingClientRect();
        const x = r.x + pos.left + 10, y = r.y + pos.top + pos.height / 2;
        const t = ed.getTargetAtClientPoint(x, y);
        if (!t || t.type !== D.monaco.editor.MouseTargetType.CONTENT_TEXT) return null;
        return { x, y };
      }`);
      if (!xy) throw new Error(`single line ${line} not clickable`);
      // No-expect mode waits for the highlight SET TO CHANGE, not merely to
      // be non-empty: retained highlights from the previous iteration would
      // satisfy `length > 0` immediately and record fake near-zero latency.
      const preSig = expected === null
        ? await evaluate(s, `() => JSON.stringify(window.__TRITONPARSE_DEBUG.panels["single-viewer"].getHighlights())`)
        : null;
      const t0 = Date.now();
      await mouseClick(s, xy.x, xy.y);
      if (expected) {
        await waitForFunction(s, `() => {
          const ed = window.__TRITONPARSE_DEBUG.panels["single-viewer"].editor;
          const got = ed.getModel().getAllDecorations()
            .filter((d) => d.options.className === 'mp-highlighted-line')
            .map((d) => d.range.startLineNumber).sort((a, b) => a - b);
          return JSON.stringify(got) === ${JSON.stringify(JSON.stringify(expectedSorted))} ? true : false;
        }`, { timeoutMs: 60000, pollingMs: 25 });
      } else {
        try {
          await waitForFunction(s, `() => JSON.stringify(window.__TRITONPARSE_DEBUG.panels["single-viewer"].getHighlights()) !== ${JSON.stringify(preSig)} ? true : false`, { timeoutMs: 15000, pollingMs: 25 });
        } catch {
          throw new Error(
            `p2 line ${line}: highlight set unchanged 15s after click — same-line re-click without --expect, an unmapped line, or a missed click; rotate --lines or pass --expect with the exact set`
          );
        }
      }
      const tSets = Date.now();
      await evaluate(s, `() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))`, { awaitPromise: true });
      const tFrame = Date.now();
      const counts = await evaluate(s, `() => {
        const p = window.__TRITONPARSE_DEBUG.panels["single-viewer"];
        return { groupSize: p.getHighlights().length };
      }`);
      results.push({ iteration: i, file: h3.text, line, clickToSetsMs: tSets - t0, clickToNextFrameMs: tFrame - t0, ...counts });
      console.log(`  iter ${i}: line ${line} sets ${tSets - t0}ms, frame ${tFrame - t0}ms, group ${counts.groupSize}`);
    }
    return { scenario: "p2", renderer: args.renderer, traceUrl: args.traceUrl, results };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

async function scenarioP3(args) {
  // 5s passive scroll sampling: the ONLY CDP traffic during the window is
  // the wheel events themselves (no screenshots, no evaluates, no recording).
  // --view single scrolls the Single viewer (the 100k-line gate input).
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    const url = args.view === "single"
      ? `${args.baseUrl}/?json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`
      : `${args.baseUrl}/?view=ir_code_comparison&json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    await s.send("Page.navigate", { url });
    if (args.view === "single" && args.renderer === "monaco") {
      await openFirstSingle(s);
      await waitForFunction(s, `() => {
        const ed = window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor;
        if (!ed) return false;
        return ed.getDomNode().querySelectorAll(".view-lines .view-line").length > 0 ? true : false;
      }`, { timeoutMs: 300000 });
    } else if (args.renderer === "monaco") {
      await waitForFunction(s, `() => {
        const P = window.__TRITONPARSE_DEBUG?.panels;
        return P?.left?.editor ? true : false;
      }`, { timeoutMs: 300000 });
    } else {
      await waitForFunction(s, `() => document.querySelectorAll('[data-line-number]').length > 100 ? true : false`, { timeoutMs: 300000 });
    }
    // Install the passive sampler, then stay silent for 5s of wheeling.
    await evaluate(s, `() => {
      window.__p3 = { frames: [], longtasks: [], raf0: 0, count: 0 };
      const rec = (t) => {
        const a = window.__p3;
        if (a.raf0) a.frames.push(t - a.raf0);
        a.raf0 = t;
        a.count++;
        requestAnimationFrame(rec);
      };
      requestAnimationFrame(rec);
      try {
        new PerformanceObserver((list) => {
          for (const e of list.getEntries()) window.__p3.longtasks.push(Math.round(e.duration * 10) / 10);
        }).observe({ entryTypes: ["longtask"] });
      } catch { /* longtask unsupported */ }
    }`);
    const wheelTarget = args.view === "single" && args.renderer === "monaco"
      ? "single-viewer"
      : "left";
    const c = await evaluate(s, args.renderer === "monaco"
      ? `() => {
          const r = window.__TRITONPARSE_DEBUG.panels[${JSON.stringify(wheelTarget)}].editor.getDomNode().getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }`
      : `() => {
          const v = document.querySelector('code.language-mlir');
          const r = v.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
        }`);
    const tStart = Date.now();
    // 5s window: wheel down, then back up if we hit the bottom. The flip
    // fires once per 2.5s slot (slot counter, not a time-window predicate:
    // the ~100ms loop period can land twice inside a 120ms window and flip
    // twice, cancelling itself).
    let dir = 1;
    let flippedSlot = 0;
    while (Date.now() - tStart < 5000) {
      await s.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: c.x, y: c.y, deltaX: 0, deltaY: dir * 240 });
      await new Promise((r) => setTimeout(r, 100));
      const slot = Math.floor((Date.now() - tStart) / 2500);
      if (slot > flippedSlot) {
        flippedSlot = slot;
        dir = -dir;
      }
    }
    const samples = await evaluate(s, `() => ({ frames: window.__p3.frames, longtasks: window.__p3.longtasks, count: window.__p3.count })`);
    const gaps = samples.frames;
    return {
      scenario: "p3", renderer: args.renderer, view: args.view, traceUrl: args.traceUrl,
      windowMs: Date.now() - tStart,
      frameGaps: summarize(gaps.map((g) => Math.round(g * 100) / 100)),
      longtasks: { n: samples.longtasks.length, totalMs: Math.round(samples.longtasks.reduce((a, b) => a + b, 0) * 10) / 10 },
      rawFrames: gaps.length,
    };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

async function scenarioP4(args, fixturePort) {
  // Design §6.2 memory gate on the three-pane × 20MB two-kernel input:
  // same-session trace load -> File Diff opened+stable (baseline: two heap
  // samples 5s apart, drift check (warn-only); IR panel models absent) -> IR comparison
  // open with 1s heap polling (peak) -> 30s stable -> kernel cycle 0->1->0
  // through the real overview select -> 30s final settle. Heap unit is
  // performance.memory.usedJSHeapSize (precise-memory-info flag); model
  // inventory (uri/chars/lines) is recorded per sample. No forced GC.
  const { s, proc, userDataDir } = await freshPage(args.chrome, ["--enable-precise-memory-info"]);
  try {
    const big = `http://127.0.0.1:${fixturePort}/p4-input.ndjson.gz`;
    const url =
      `${args.baseUrl}/?json_url=${encodeURIComponent(big)}` +
      `&json_b_url=${encodeURIComponent(big)}&ir=ttgir&renderer=monaco&debug=1`;
    await s.send("Page.navigate", { url });
    await waitForFunction(
      s,
      `() => document.body.textContent.includes('reviewer_memory_0') ? true : false`,
      { timeoutMs: 300000 }
    );
    await waitResponsive(s, { tries: 40 });
    // Passive in-page sampler (reads heap + model inventory only).
    await evalRetry(s, `() => {
      window.__PERF_P4 = {
        samples: [], annotations: [], timer: null,
        inventory() {
          return (window.__TRITONPARSE_DEBUG?.monaco?.editor?.getModels() ?? []).map((m) => ({
            uri: m.uri.toString(), characters: m.getValueLength(), lines: m.getLineCount(),
          }));
        },
        sample(label) {
          const point = { label, at: performance.now(), utc: new Date().toISOString(),
            usedJSHeapSize: performance.memory?.usedJSHeapSize ?? null, models: this.inventory() };
          this.samples.push(point);
          return point;
        },
        start(label) {
          if (this.timer !== null) throw new Error("p4 polling already running");
          this.sample(label + ":start");
          this.timer = setInterval(() => this.sample(label), 1000);
        },
        stop(label) {
          if (this.timer !== null) clearInterval(this.timer);
          this.timer = null;
          return this.sample(label + ":stop");
        },
        annotate(label) { this.annotations.push({ label, at: performance.now() }); },
      };
      return !!window.__PERF_P4;
    }`, "install p4 sampler");
    const sample = (label) =>
      evalRetry(s, `() => window.__PERF_P4.sample(${JSON.stringify(label)})`, `sample ${label}`);

    console.log("  phase: open File Diff");
    await clickText(s, "button", "File Diff");
    await waitForFunction(
      s,
      `() => {
        const D = window.__TRITONPARSE_DEBUG;
        return !!D?.panels?.filediff?.diffEditor && D.getModels().length === 2 ? true : false;
      }`,
      { timeoutMs: 300000 }
    );
    await sleep(3000);
    const preBaseline = await evalRetry(s, `() => ({
      panels: Object.keys(window.__TRITONPARSE_DEBUG.panels ?? {}),
    })`, "preBaseline");
    for (const id of ["left", "right", "python", "single-viewer"]) {
      if (preBaseline.panels.includes(id)) {
        throw new Error(`IR panel ${id} exists before baseline: ${JSON.stringify(preBaseline.panels)}`);
      }
    }
    const b1 = await sample("baseline");
    await sleep(5000);
    const b2 = await sample("baseline");
    // Fail fast with a targeted error: without precise-memory-info the heap
    // readings are null and the run would otherwise die minutes later with
    // the generic "summarize: no samples" after collecting unusable data.
    if (typeof b1.usedJSHeapSize !== "number" || typeof b2.usedJSHeapSize !== "number") {
      throw new Error("p4: performance.memory.usedJSHeapSize unavailable (Chrome needs --enable-precise-memory-info); heap assertions cannot run");
    }
    const drift = Math.abs(b2.usedJSHeapSize - b1.usedJSHeapSize) / b1.usedJSHeapSize;
    console.log(`  baseline ${(b1.usedJSHeapSize / 1048576).toFixed(1)}MB drift ${(drift * 100).toFixed(3)}%`);
    // Warn-only, not a hard gate: GC between the two samples legitimately
    // moves the heap, so a throw would be flaky. Investigate a leak only
    // if the warning reproduces. (NaN from a missing heap reading warns.)
    if (!(drift < 0.05)) {
      console.warn(`  WARNING: baseline drift ${(drift * 100).toFixed(3)}% >= 5%`);
    }

    console.log("  phase: open IR comparison (kernel 0)");
    await evalRetry(s, `() => window.__PERF_P4.start('p4')`, "startMemory");
    await clickText(s, "button", "IR Code");
    await waitForFunction(
      s,
      `() => !!window.__TRITONPARSE_DEBUG?.panels?.left?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.right?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.python?.editor ? true : false`,
      { timeoutMs: 300000 }
    );
    await evalRetry(s, `() => window.__PERF_P4.annotate('ir-open-kernel0')`, "annotate open");
    await sleep(30000);
    const stable = await sample("stable-30s");
    console.log(`  stable-30s ${(stable.usedJSHeapSize / 1048576).toFixed(1)}MB models=${stable.models.length}`);

    console.log("  phase: kernel 0 -> 1");
    await clickText(s, "button", "Kernel Overview");
    await selectByKeyboard(s, "1");
    await clickText(s, "button", "IR Code");
    await waitForFunction(
      s,
      `() => !!window.__TRITONPARSE_DEBUG?.panels?.left?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.right?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.python?.editor ? true : false`,
      { timeoutMs: 300000 }
    );
    await sleep(5000);
    const k1 = await sample("kernel1");
    console.log(`  kernel1 ${(k1.usedJSHeapSize / 1048576).toFixed(1)}MB models=${k1.models.length}`);

    console.log("  phase: kernel 1 -> 0");
    await clickText(s, "button", "Kernel Overview");
    await selectByKeyboard(s, "0");
    await clickText(s, "button", "IR Code");
    await waitForFunction(
      s,
      `() => !!window.__TRITONPARSE_DEBUG?.panels?.left?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.right?.editor &&
        !!window.__TRITONPARSE_DEBUG?.panels?.python?.editor ? true : false`,
      { timeoutMs: 300000 }
    );
    await sleep(5000);
    const k0 = await sample("kernel0-again");
    console.log(`  kernel0-again ${(k0.usedJSHeapSize / 1048576).toFixed(1)}MB models=${k0.models.length}`);
    await sleep(25000);
    const fin = await sample("final-settle");
    console.log(`  final-settle ${(fin.usedJSHeapSize / 1048576).toFixed(1)}MB models=${fin.models.length}`);

    await evalRetry(s, `() => window.__PERF_P4.stop('end')`, "stopMemory");
    const records = await evalRetry(
      s,
      `() => ({ samples: window.__PERF_P4.samples, annotations: window.__PERF_P4.annotations })`,
      "read p4"
    );
    try {
      const png = await captureScreenshot(s);
      writeFileSync(join(args.artifactDir, "p4-final.png"), png);
    } catch (e) {
      console.log(`  p4 screenshot skipped: ${String(e).slice(0, 120)}`);
    }
    const heap = records.samples.map((m) => m.usedJSHeapSize).filter((v) => typeof v === "number");
    const denominatorTextBytes = stable.models
      .filter((m) => m.uri.startsWith("file:"))
      .reduce((n, m) => n + m.characters * 2, 0);
    return {
      scenario: "p4", inputFile: args.inputFile, renderer: "monaco",
      chromeFlags: ["--enable-precise-memory-info"],
      denominatorTextBytes,
      baseline: { b1, b2, drift },
      stable30s: stable, kernel1: k1, kernel0Again: k0, finalSettle: fin,
      heapSummary: summarize(heap),
      annotations: records.annotations, samples: records.samples,
    };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

/**
 * Move a comparison panel until a physical line is CONTENT_TEXT-clickable,
 * using real keyboard input. CDP-injected wheel events move Monaco by a
 * fixed ~50px each, so reaching a deep line (e.g. 3908) needs keyboard
 * paging: one Ctrl+End jump, then a directional PageUp/PageDown loop.
 */
async function keyToPanelLine(s, panelId, line) {
  const c = await evaluate(s, `() => {
    const r = window.__TRITONPARSE_DEBUG.panels[${JSON.stringify(panelId)}].editor.getDomNode().getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  }`);
  // Focus the editor with a real click (empty-mapping inputs: no highlights).
  await mouseClick(s, c.x, c.y);
  await new Promise((r) => setTimeout(r, 300));
  // Jump to the document end; a failed jump still converges via paging.
  await keyPress(s, "End", { code: "End", windowsVirtualKeyCode: 35, modifiers: 2 });
  await new Promise((r) => setTimeout(r, 500));
  const clickable = () => evaluate(s, `() => {
    const D = window.__TRITONPARSE_DEBUG;
    const ed = D.panels[${JSON.stringify(panelId)}].editor;
    const pos = ed.getScrolledVisiblePosition({ lineNumber: ${line}, column: 1 });
    if (!pos) return null;
    const r = ed.getDomNode().getBoundingClientRect();
    const x = r.x + pos.left + 10, y = r.y + pos.top + pos.height / 2;
    if (y < r.y + 2 || y > r.y + r.height - 2) return null;
    const t = ed.getTargetAtClientPoint(x, y);
    if (!t || t.type !== D.monaco.editor.MouseTargetType.CONTENT_TEXT) return null;
    return { x, y };
  }`);
  for (let i = 0; i < 300; i++) {
    const xy = await clickable();
    if (xy) return { ...c, lineX: xy.x, lineY: xy.y };
    const dir = await evaluate(s, `() => {
      const ed = window.__TRITONPARSE_DEBUG.panels[${JSON.stringify(panelId)}].editor;
      const vs = ed.getVisibleRanges()[0];
      if (!vs) return 1;
      const mid = (vs.startLineNumber + vs.endLineNumber) / 2;
      return ${line} > mid ? 1 : -1;
    }`);
    if (dir > 0) await keyPress(s, "PageDown", { code: "PageDown", windowsVirtualKeyCode: 34 });
    else await keyPress(s, "PageUp", { code: "PageUp", windowsVirtualKeyCode: 33 });
    await new Promise((r) => setTimeout(r, 60));
  }
  throw new Error(`${panelId} line ${line} never clickable`);
}

async function scenarioP5(args) {
  // Long-line input characterization + comparison mount on the given trace.
  // Requires a sourced real >5000-char IR line (005 task 5); without one the
  // scenario reports the measured maximum and stays inconclusive. On a
  // conclusive input it also drives the §6.2 wrap gate: wrap must be off,
  // and real horizontal wheel input must move the long-line panel with
  // content still rendered (equivalent to the legacy <pre> behavior).
  const { s, proc, userDataDir } = await freshPage(args.chrome);
  try {
    const url = `${args.baseUrl}/?view=ir_code_comparison&json_url=${encodeURIComponent(args.traceUrl)}${rendererParam(args.renderer)}&debug=1`;
    const t0 = Date.now();
    await s.send("Page.navigate", { url });
    await waitForFunction(s, `() => {
      const P = window.__TRITONPARSE_DEBUG?.panels;
      return P?.left?.editor && P?.right?.editor && P?.python?.editor ? true : false;
    }`, { timeoutMs: 300000 });
    const tReady = Date.now();
    const longest = await evaluate(s, `() => {
      const P = window.__TRITONPARSE_DEBUG.panels;
      let best = { len: 0, panel: null, line: 0 };
      for (const id of ["left", "right"]) {
        const v = P[id].editor.getModel().getValue().split("\\n");
        for (let i = 0; i < v.length; i++) {
          if (v[i].length > best.len) best = { len: v[i].length, panel: id, line: i + 1 };
        }
      }
      return best;
    }`);
    const result = { scenario: "p5", renderer: args.renderer, traceUrl: args.traceUrl, mountMs: tReady - t0, longestRenderedIrLine: longest, conclusive: longest.len > 5000 };
    if (!result.conclusive) return result;
    // Wrap gate on the panel holding the longest real line. Scroll width is
    // measured after the long line is laid out (Monaco sizes lazily).
    const wrapOpt = await evaluate(s, `() => {
      const D = window.__TRITONPARSE_DEBUG;
      const ed = D.panels[${JSON.stringify(longest.panel)}].editor;
      return { wrapOff: ed.getOption(D.monaco.editor.EditorOption.wrappingInfo).isViewportWrapping === false };
    }`);
    if (!wrapOpt.wrapOff) throw new Error(`wrap is not off on ${longest.panel}`);
    const c = await keyToPanelLine(s, longest.panel, longest.line);
    const wrap = await evaluate(s, `() => {
      const ed = window.__TRITONPARSE_DEBUG.panels[${JSON.stringify(longest.panel)}].editor;
      return {
        clientWidth: Math.round(ed.getLayoutInfo().width),
        scrollWidth: Math.round(ed.getScrollWidth()),
        scrollLeft: Math.round(ed.getScrollLeft()),
      };
    }`);
    if (!(wrap.scrollWidth > wrap.clientWidth)) {
      throw new Error(`no horizontal overflow on ${longest.panel}: ${JSON.stringify(wrap)}`);
    }
    for (let i = 0; i < 10; i++) {
      await s.send("Input.dispatchMouseEvent", { type: "mouseWheel", x: c.x, y: c.y, deltaX: 300, deltaY: 0 });
      await new Promise((r) => setTimeout(r, 120));
    }
    const after = await evaluate(s, `() => {
      const ed = window.__TRITONPARSE_DEBUG.panels[${JSON.stringify(longest.panel)}].editor;
      return {
        scrollLeft: Math.round(ed.getScrollLeft()),
        renderedLines: ed.getDomNode().querySelectorAll(".view-lines .view-line").length,
      };
    }`);
    if (!(after.scrollLeft > wrap.scrollLeft)) {
      throw new Error(`horizontal scroll did not move ${longest.panel}: ${wrap.scrollLeft} -> ${after.scrollLeft}`);
    }
    if (!(after.renderedLines > 0)) throw new Error(`${longest.panel} blank after horizontal scroll`);
    console.log(`  wrap-off horizontal scroll ${longest.panel}: ${wrap.scrollLeft} -> ${after.scrollLeft} (scrollWidth ${wrap.scrollWidth})`);
    try {
      const png = await captureScreenshot(s);
      writeFileSync(join(args.artifactDir, "p5-long-line.png"), png);
    } catch (e) {
      console.log(`  p5 screenshot skipped: ${String(e).slice(0, 120)}`);
    }
    return { ...result, wrapGate: { wrapOff: wrapOpt.wrapOff, ...wrap, ...after } };
  } finally {
    try { s.close(); } catch { /* ignore */ }
    killProcAndCleanTmp(proc, userDataDir);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.artifactDir, { recursive: true });
  const extraRoutes = {};
  if (args.scenario === "p4") {
    extraRoutes["/p4-input.ndjson.gz"] = { body: readFileSync(args.inputFile), type: "application/octet-stream" };
  }
  const { server, port } = await startFixtureServer(extraRoutes);
  try {
    let result;
    if (args.scenario === "p1-cold") result = await scenarioP1Cold(args);
    else if (args.scenario === "p1-tabs") result = await scenarioP1Tabs(args);
    else if (args.scenario === "p1-first-frame") result = await scenarioP1FirstFrame(args);
    else if (args.scenario === "p1-hot") result = await scenarioP1Hot(args);
    else if (args.scenario === "p2") result = await scenarioP2(args);
    else if (args.scenario === "p3") result = await scenarioP3(args);
    else if (args.scenario === "p4") result = await scenarioP4(args, port);
    else result = await scenarioP5(args);
    result.recordedAt = new Date().toISOString();
    result.baseUrl = args.baseUrl;
    result.params = {
      iterations: args.iterations,
      lines: args.lines,
      expectPath: args.expectPath,
      view: args.view,
      hot: args.hot,
    };
    const stamp = `${args.scenario}-${args.renderer}`;
    writeFileSync(join(args.artifactDir, `${stamp}.json`), JSON.stringify(result, null, 1));
    console.log(`PERF ${args.scenario}/${args.renderer} DONE -> ${args.artifactDir}/${stamp}.json`);
  } finally {
    // Awaited close: fire-and-forget lets the process exit mid-teardown;
    // closeAllConnections() drops the dead-Chrome keep-alives first so the
    // close callback resolves instead of hanging on lingering sockets.
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }
}

// Dual-use module: unit tests import the pure helpers; the runner only
// starts when this file is invoked as a script (not when imported).
export { parseArgs, summarize, normalizeExpectTriple, killProcAndCleanTmp };
const invokedAsScript =
  process.argv[1] != null &&
  pathToFileURL(resolve(process.argv[1])).href === import.meta.url;
if (invokedAsScript) {
  main().catch((err) => {
    console.error(`PERF FAILED: ${err && err.message ? err.message : err}`);
    process.exit(1);
  });
}

