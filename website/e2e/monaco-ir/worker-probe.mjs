#!/usr/bin/env node
/**
 * Worker + delivery probe for the Monaco IR migration (Phase 1).
 *
 * Usage:
 *   node e2e/monaco-ir/worker-probe.mjs --base-url URL --artifact-dir DIR [--chrome PATH]
 *   node e2e/monaco-ir/worker-probe.mjs --static-dir DIR --artifact-dir DIR [--chrome PATH]
 *
 * --base-url targets a dev/preview server; --static-dir serves a directory
 * (e.g. a standalone-delivery folder holding only standalone.html) over HTTP
 * so the single-file build is verified with no adjacent assets.
 *
 * What it proves (beyond "a worker was created" / "a diff rendered"):
 * - the $computeDiff request/response protocol on the wire: every worker
 *   reply matches an observed request by (worker, vsWorker, req/seq), the
 *   reply carries the exact expected change, and no reply carries err;
 * - exactly one blob: worker, zero worker errors, zero main-thread fallback
 *   warnings;
 * - zero failed/4xx resource requests with the CDN blocked and cache
 *   disabled, and no AMD/CDN/vs/worker-file requests (fonts included).
 *
 * All page input is real CDP mouse/keyboard; the Worker subclass installed
 * via addScriptToEvaluateOnNewDocument only observes postMessage traffic.
 */
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
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
  const out = { baseUrl: null, staticDir: null, artifactDir: null, chrome: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--base-url") out.baseUrl = argv[++i];
    else if (argv[i] === "--static-dir") out.staticDir = argv[++i];
    else if (argv[i] === "--artifact-dir") out.artifactDir = argv[++i];
    else if (argv[i] === "--chrome") out.chrome = argv[++i];
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  if (!out.artifactDir) throw new Error("--artifact-dir is required");
  if ((out.baseUrl && out.staticDir) || (!out.baseUrl && !out.staticDir)) {
    throw new Error("exactly one of --base-url / --static-dir is required");
  }
  if (out.baseUrl) {
    out.baseUrl = out.baseUrl.replace(/\/$/, "").replace("://localhost", "://127.0.0.1");
  }
  return out;
}

/** Mirror of run.mjs buildLongFileDiffPair (kept in sync deliberately). */
function buildLongFileDiffPair() {
  const base = JSON.parse(readFileSync(join(FIXTURES, "single-basic.ndjson"), "utf8").trim());
  const lines = [];
  for (let i = 1; i <= 240; i++) {
    lines.push(`// worker-probe filler line ${i} with enough unchanged text to align both sides`);
  }
  lines[2] = `// worker-probe long line ${"x".repeat(6000)}`;
  const left = structuredClone(base);
  const right = structuredClone(base);
  const l = [...lines];
  l[119] = "%changed = arith.constant 11 : i32 // worker-probe exact expected difference";
  const r = [...lines];
  r[119] = "%changed = arith.constant 22 : i32 // worker-probe exact expected difference";
  left.payload.file_content["e2e_kernel.ttgir"] = `${l.join("\n")}\n`;
  right.payload.file_content["e2e_kernel.ttgir"] = `${r.join("\n")}\n`;
  return {
    "/probe-left.ndjson": `${JSON.stringify(left)}\n`,
    "/probe-right.ndjson": `${JSON.stringify(right)}\n`,
  };
}

function startFixtureServer(extra) {
  const server = createServer((req, res) => {
    const name = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const send = (body, type) => {
      res.writeHead(200, { "Content-Type": type, "Access-Control-Allow-Origin": "*" });
      res.end(body);
    };
    if (extra[name]) {
      send(extra[name], "application/x-ndjson");
      return;
    }
    const file = join(FIXTURES, name.replace(/^\//, ""));
    if (name.endsWith(".ndjson") && existsSync(file)) {
      send(readFileSync(file), "application/x-ndjson");
      return;
    }
    res.writeHead(404).end("no such fixture");
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

const MIME = { ".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".map": "application/json", ".wasm": "application/wasm" };

function startStaticServer(dir) {
  const server = createServer((req, res) => {
    const name = decodeURIComponent(new URL(req.url, "http://x").pathname);
    const file = name === "/" ? join(dir, "standalone.html") : join(dir, name.replace(/^\//, ""));
    if (!existsSync(file)) {
      res.writeHead(404).end("missing delivery file");
      return;
    }
    res.writeHead(200, { "Content-Type": MIME[extname(file)] ?? "application/octet-stream" });
    res.end(readFileSync(file));
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({ server, port: server.address().port }));
  });
}

/** Observation-only Worker subclass (page-side source, own implementation). */
const OBSERVER_SOURCE = `(() => {
  const observations = { workers: [], requests: [], responses: [], errors: [] };
  Object.defineProperty(window, "__TP_WORKER_OBSERVATIONS", { value: observations });
  const NativeWorker = window.Worker;
  const ids = new WeakMap();
  const pending = new Set();
  const key = (id, worker, n) => id + ":" + worker + ":" + n;
  window.Worker = class ObservedWorker extends NativeWorker {
    constructor(...args) {
      super(...args);
      const id = observations.workers.length + 1;
      ids.set(this, id);
      observations.workers.push({ id, url: String(args[0]), createdAt: performance.now(), terminatedAt: null });
      this.addEventListener("message", (event) => {
        const v = event.data;
        // Record EVERY reply with a matched flag: dropping unmatched replies
        // here would make the driver's no-unmatched-reply check vacuous.
        if (v && v.type === 1) {
          observations.responses.push({
            id, vsWorker: v.vsWorker, seq: v.seq,
            matched: pending.has(key(id, v.vsWorker, v.seq)),
            changes: v.res && v.res.changes ? v.res.changes.map((c) => (Array.isArray(c) ? c.slice(0, 4) : null)) : null,
            identical: v.res ? !!v.res.identical : null,
            error: v.err ?? null, at: performance.now(),
          });
        }
      });
      this.addEventListener("error", (event) => observations.errors.push({ id, message: event.message, at: performance.now() }));
      this.addEventListener("messageerror", () => observations.errors.push({ id, message: "messageerror", at: performance.now() }));
    }
    postMessage(value, ...rest) {
      const id = ids.get(this);
      // Track every request for reply matching (any method), but only log
      // $computeDiff in requests so the driver's request assertions keep
      // their meaning.
      if (value && value.type === 0) {
        pending.add(key(id, value.vsWorker, value.req));
        if (value.method === "$computeDiff") {
          observations.requests.push({ id, vsWorker: value.vsWorker, req: value.req, at: performance.now() });
        }
      }
      return super.postMessage(value, ...rest);
    }
    terminate(...args) {
      const entry = observations.workers.find((w) => w.id === ids.get(this));
      if (entry) entry.terminatedAt = performance.now();
      return super.terminate(...args);
    }
  };
})();`;

function assertEqual(actual, expected, label) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${label}: expected ${e}, got ${a}`);
  console.log(`  ok ${label} = ${a}`);
}

async function main() {
  for (const key of ["NO_PROXY", "no_proxy"]) {
    const cur = (process.env[key] ?? "").split(",").map((s) => s.trim()).filter(Boolean);
    for (const host of ["localhost", "127.0.0.1"]) {
      if (!cur.includes(host)) cur.push(host);
    }
    process.env[key] = cur.join(",");
  }
  const args = parseArgs(process.argv.slice(2));
  mkdirSync(args.artifactDir, { recursive: true });

  // Resources hoisted so one try/finally owns ALL cleanup: an early throw
  // (unreachable base, fixture/chrome failure) must not leak servers or
  // leave chrome running with the event loop alive.
  let staticServer = null;
  let fixtureServer = null;
  let proc = null;
  const requests = [];
  const responses = [];
  const failed = [];
  const errors = [];
  const warnings = [];
  let passed = false;
  let failure = null;
  let observations = null;
  let baseUrl = args.baseUrl;
  try {
    if (args.staticDir) {
      const files = readdirSync(args.staticDir);
      console.log(`delivery dir holds: ${files.join(", ")}`);
      staticServer = (await startStaticServer(args.staticDir)).server;
      const port = staticServer.address().port;
      baseUrl = `http://127.0.0.1:${port}`;
    }
    try {
      const res = await fetch(baseUrl, { signal: AbortSignal.timeout(10000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      throw new Error(`base URL unreachable (${baseUrl}): ${err.message}`);
    }

    const pair = buildLongFileDiffPair();
    const started = await startFixtureServer(pair);
    fixtureServer = started.server;
    const fixturePort = started.port;
    const leftUrl = `http://127.0.0.1:${fixturePort}/probe-left.ndjson`;
    const rightUrl = `http://127.0.0.1:${fixturePort}/probe-right.ndjson`;
    const singleUrl = `http://127.0.0.1:${fixturePort}/single-basic.ndjson`;

    const chromePath = args.chrome ?? findChrome();
    const launched = await launchChrome({
      chromePath,
      userDataDir: mkdtempSync(join(tmpdir(), "monaco-worker-probe-")),
      extraArgs: ["--window-size=1920,1080"],
    });
    proc = launched.proc;
    const debugPort = launched.port;
    console.log(`chrome ${chromePath} (debug :${debugPort}), fixtures :${fixturePort}, base ${baseUrl}`);

    const targets = await listTargets(debugPort);
    const pageTarget = targets.find((t) => t.type === "page");
    if (!pageTarget) throw new Error("no page target in fresh chrome");
    const s = await connectPageTarget(pageTarget);
    await s.send("Page.enable");
    await s.send("Runtime.enable");
    await s.send("Network.enable");
    await s.send("Network.setCacheDisabled", { cacheDisabled: true });
    await s.send("Network.setBlockedURLs", { urls: ["*://*.jsdelivr.net/*", "*://jsdelivr.net/*", "*://*.unpkg.com/*"] });
    s.on("Network.requestWillBeSent", (p) => requests.push({ url: p.request.url, type: p.type }));
    // Navigation-cancelled requests surface as net::ERR_ABORTED loadingFailed
    // events on the shared session: without this filter, an in-flight request
    // from the single-viewer document would fail the "failed resources"
    // assertion after the file_diff Page.navigate for reasons unrelated to
    // worker delivery. All other failures still fail loudly below.
    s.on("Network.loadingFailed", (p) => {
      if (p.errorText === "net::ERR_ABORTED") return;
      failed.push({ error: p.errorText, blocked: p.blockedReason });
    });
    s.on("Network.responseReceived", (p) => responses.push({ url: p.response.url, status: p.response.status }));
    s.on("Runtime.consoleAPICalled", (p) => {
      const text = (p.args ?? []).map((a) => a.value ?? a.description ?? "").join(" ");
      if (p.type === "error") errors.push(text.slice(0, 300));
      if (p.type === "warning") warnings.push(text.slice(0, 300));
    });
    s.on("Runtime.exceptionThrown", (p) => {
      errors.push(`uncaught: ${(p.exceptionDetails.exception?.description ?? p.exceptionDetails.text ?? "").slice(0, 300)}`);
    });
    const { identifier: observerId } = await s.send("Page.addScriptToEvaluateOnNewDocument", { source: OBSERVER_SOURCE });

    const readObs = () => evaluate(s, `() => window.__TP_WORKER_OBSERVATIONS`);

    async function clickText(selector, text) {
      const rect = await evaluate(
        s,
        `() => {
          const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
          const el = els.find((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
          if (!el) return null;
          el.scrollIntoView({ block: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2, visible: r.bottom > 0 && r.top < window.innerHeight };
        }`
      );
      if (!rect) throw new Error(`no ${selector} with text ${text}`);
      await new Promise((r) => setTimeout(r, 400));
      await mouseClick(s, rect.x, rect.y);
    }

    // ---- Single: no diff worker traffic expected ----
    console.log("STEP single without diff worker");
    await s.send("Page.navigate", {
      url: `${baseUrl}/?json_url=${encodeURIComponent(singleUrl)}&renderer=monaco&debug=1`,
    });
    await waitForFunction(
      s,
      `() => [...document.querySelectorAll('h3')].some((h) => h.textContent.trim() === 'e2e_kernel.ttgir')`,
      { timeoutMs: 60000 }
    );
    await clickText("h3", "e2e_kernel.ttgir");
    await waitForFunction(
      s,
      `() => !!window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor`,
      { timeoutMs: 60000 }
    );
    const xy = await evaluate(s, `() => {
      const ed = window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor;
      const pos = ed.getScrolledVisiblePosition({ lineNumber: 2, column: 1 });
      const r = ed.getDomNode().getBoundingClientRect();
      return { x: r.x + 250, y: r.y + pos.top + pos.height / 2 };
    }`);
    await mouseClick(s, xy.x, xy.y);
    const highlights = await waitForFunction(
      s,
      `() => { const h = window.__TRITONPARSE_DEBUG.panels['single-viewer'].getHighlights(); return h.length > 0 ? h : false; }`,
      { timeoutMs: 15000 }
    );
    assertEqual(highlights, [2, 4], "single highlights");
    observations = await readObs();
    assertEqual(observations.requests.length, 0, "single $computeDiff requests");
    assertEqual(observations.errors, [], "single worker errors");

    // ---- File Diff: full worker protocol proof ----
    console.log("STEP file diff worker protocol");
    await s.send("Page.navigate", {
      url: `${baseUrl}/?view=file_diff&json_url=${encodeURIComponent(leftUrl)}` +
        `&json_b_url=${encodeURIComponent(rightUrl)}&ir=ttgir&wrap=on&debug=1`,
    });
    await waitForFunction(
      s,
      `() => {
        const D = window.__TRITONPARSE_DEBUG;
        if (!D?.panels?.filediff?.diffEditor) return false;
        if (D.getModels().length !== 2) return false;
        const label = [...document.querySelectorAll('div')].find((d) => d.textContent.startsWith('IR Type:'));
        return label && label.textContent.includes('ttgir') ? true : false;
      }`,
      { timeoutMs: 60000 }
    );
    const diffResponse = await waitForFunction(
      s,
      `() => {
        const o = window.__TP_WORKER_OBSERVATIONS;
        const hit = o.responses.find((r) => !r.error && r.changes && r.changes.length === 1 && r.changes[0][0] === 120);
        return hit ? hit : false;
      }`,
      { timeoutMs: 60000 }
    );
    assertEqual(diffResponse.changes[0], [120, 121, 120, 121], "worker change range");
    assertEqual(diffResponse.error, null, "worker reply error");
    observations = await readObs();
    // One worker per mounted diff widget; unmounts terminate theirs (StrictMode
    // double-mounts and remounts create-then-terminate). Assert live count.
    const alive = observations.workers.filter((w) => w.terminatedAt === null);
    assertEqual(alive.length, 1, "live worker count");
    // Vite inlines ?worker&inline as blob: only at build time; dev serves the
    // same-origin worker file. Either form is accepted on dev/preview, but a
    // standalone delivery (no adjacent assets) MUST be blob:-inlined.
    const workerUrl = alive[0].url;
    const isBlob = workerUrl.startsWith("blob:");
    // Resolve relative worker specifiers (e.g. `assets/editor.worker.js`,
    // `./assets/...`) against the page origin before classifying: a raw
    // prefix test would false-fail them as non-local. The recorded URL is
    // kept untruncated so resolution sees the full specifier.
    let isLocal = false;
    try {
      isLocal = new URL(workerUrl, `${baseUrl}/`).origin === new URL(baseUrl).origin;
    } catch {
      isLocal = workerUrl.startsWith("/") || workerUrl.startsWith(baseUrl);
    }
    if (args.staticDir && !isBlob) {
      throw new Error(`standalone worker is not blob:-inlined: ${workerUrl}`);
    }
    if (!isBlob && !isLocal) {
      throw new Error(`worker is neither blob: nor same-origin: ${workerUrl}`);
    }
    console.log(`  ok worker ${workerUrl.slice(0, 80)}...`);
    // Every reply matches an observed request by (worker, vsWorker, req/seq).
    // The matched flag is recorded by the observer for EVERY reply, so this
    // check can actually fail (a driver-side re-derivation would need the
    // full request log to stay meaningful).
    for (const r of observations.responses) {
      if (!r.matched) {
        throw new Error(`unmatched worker reply: ${JSON.stringify(r)}`);
      }
      if (r.error) throw new Error(`worker reply error: ${JSON.stringify(r)}`);
    }
    console.log(`  ok ${observations.responses.length} replies matched to observed requests (${observations.requests.length} $computeDiff)`);
    assertEqual(observations.errors, [], "worker errors");

    // Operate the diff (real input) and confirm continued worker service.
    async function selectValue(targetValue) {
      const info = await evaluate(
        s,
        `() => {
          const el = [...document.querySelectorAll('select')].find((e) =>
            [...e.options].some((op) => op.value === ${JSON.stringify(targetValue)}));
          if (!el) return null;
          el.scrollIntoView({ block: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2,
            options: [...el.options].map((o) => o.value), current: el.value };
        }`
      );
      if (!info) throw new Error(`no select offering ${targetValue}`);
      await new Promise((r) => setTimeout(r, 300));
      await mouseClick(s, info.x, info.y);
      const cur = info.options.indexOf(info.current);
      const tgt = info.options.indexOf(targetValue);
      const down = tgt >= cur;
      for (let i = 0; i < Math.abs(tgt - cur); i++) {
        await keyPress(s, down ? "ArrowDown" : "ArrowUp", { code: down ? "ArrowDown" : "ArrowUp", windowsVirtualKeyCode: down ? 40 : 38 });
        await new Promise((r) => setTimeout(r, 150));
      }
      await keyPress(s, "Escape", { code: "Escape", windowsVirtualKeyCode: 27 });
      await waitForFunction(
        s,
        `() => {
          const el = [...document.querySelectorAll('select')].find((e) =>
            [...e.options].some((op) => op.value === ${JSON.stringify(targetValue)}));
          return el && el.value === ${JSON.stringify(targetValue)} ? true : false;
        }`,
        { timeoutMs: 10000 }
      );
    }
    const responseCount = () => evaluate(s, `() => window.__TP_WORKER_OBSERVATIONS.responses.length`);
    await selectValue("off");
    await new Promise((r) => setTimeout(r, 600));
    const wrapState = await evaluate(s, `() => {
      const D = window.__TRITONPARSE_DEBUG;
      const de = D.panels.filediff.diffEditor;
      const E = D.monaco.editor.EditorOption;
      return [de.getOriginalEditor().getOption(E.wrappingInfo).isViewportWrapping,
        de.getModifiedEditor().getOption(E.wrappingInfo).isViewportWrapping];
    }`);
    assertEqual(wrapState, [false, false], "wrap off both sides");
    await clickText("label", "Only changes");
    await waitForFunction(s, `() => document.querySelectorAll('.diff-hidden-lines').length > 0`, { timeoutMs: 15000 });
    console.log("  ok only-changes hides rows");
    // I005: the codicon font must load (data URI on standalone delivery).
    await waitForFunction(s, `() => document.fonts.check('16px codicon') ? true : false`, { timeoutMs: 30000 });
    console.log("  ok codicon font loaded");
    await clickText("label", "Only changes");
    await waitForFunction(s, `() => document.querySelectorAll('.diff-hidden-lines').length === 0`, { timeoutMs: 15000 });
    // The wrap/only-changes toggles above emit worker replies of their own;
    // snapshot the count only after traffic quiets, so the IR-switch check
    // below observes the switch's own reply rather than toggle leftovers.
    let seen = -1;
    {
      let lastCount = -1;
      let stableSince = Date.now();
      const quietDeadline = Date.now() + 15000;
      for (;;) {
        const n = await responseCount();
        const now = Date.now();
        if (n !== lastCount) {
          lastCount = n;
          stableSince = now;
        } else if (now - stableSince >= 1000) {
          break;
        }
        if (now > quietDeadline) {
          throw new Error(`worker traffic never quieted (responses=${n})`);
        }
        await new Promise((r) => setTimeout(r, 250));
      }
      seen = lastCount;
    }
    await selectValue("llir");
    await waitForFunction(s, `() => window.__TP_WORKER_OBSERVATIONS.responses.length > ${seen}`, { timeoutMs: 60000 });
    console.log("  ok worker served IR switch");
    // Re-baseline: the original ttgir reply (line-120 change) still sits in
    // the accumulated list, so an unscoped `.some()` would pass vacuously.
    const beforeSwitchBack = await evaluate(s, `() => window.__TP_WORKER_OBSERVATIONS.responses.length`);
    await selectValue("ttgir");
    await waitForFunction(
      s,
      `() => window.__TP_WORKER_OBSERVATIONS.responses.slice(${beforeSwitchBack}).some((r) => !r.error && r.changes && r.changes[0] && r.changes[0][0] === 120)`,
      { timeoutMs: 60000 }
    );
    observations = await readObs();
    const aliveAfter = observations.workers.filter((w) => w.terminatedAt === null);
    assertEqual(aliveAfter.length, 1, "still one live worker after switches");
    const models = await evaluate(s, `() => window.__TRITONPARSE_DEBUG.getModels().length`);
    assertEqual(models, 2, "model count after switches");
    const shot = await captureScreenshot(s);
    writeFileSync(join(args.artifactDir, "worker-probe-filediff.png"), shot);

    // ---- Resource hygiene (CDN blocked, cache disabled) ----
    assertEqual(errors, [], "console errors");
    const badStatus = responses.filter((r) => r.status >= 400);
    assertEqual(badStatus, [], "http error statuses");
    assertEqual(failed, [], "failed resources");
    if (warnings.some((t) => /Could not create web worker|Falling back to loading web worker|main thread.*fallback/i.test(t))) {
      throw new Error(`worker fallback warning: ${JSON.stringify(warnings)}`);
    }
    // AMD/CDN/worker-file requests are banned everywhere; font files may load
    // same-origin on dev/preview, but the standalone delivery (no adjacent
    // assets) must be fully self-contained via the inlined data URI (I005).
    const badRequests = requests.filter((r) =>
      /jsdelivr\.net|unpkg\.com|\/monaco-tmp\/|\/vs\/loader\.js|\/vs\/editor\/editor\.main|(?:json|ts|css|html)\.worker/i.test(r.url));
    assertEqual(badRequests, [], "amd/cdn/worker requests");
    if (args.staticDir) {
      const fontRequests = requests.filter((r) => /\.(ttf|woff2?|otf|eot)(\?|$)/i.test(r.url));
      assertEqual(fontRequests, [], "standalone font requests");
    }

    await s.send("Page.removeScriptToEvaluateOnNewDocument", { identifier: observerId });
    s.close();
    passed = true;
    console.log("WORKER PROBE PASSED");
  } catch (err) {
    failure = err.stack;
    console.error(`WORKER PROBE FAILED: ${err.message}`);
    process.exitCode = 1;
  } finally {
    // Finally must never throw (it would mask the real failure): guard the
    // artifact write and tolerate partially-initialized resources.
    try {
      const result = {
        baseUrl, passed, failure,
        observations,
        requests: requests.map((r) => r.url),
        responses: responses.map((r) => `${r.status} ${r.url}`),
        failed, errors, warnings,
      };
      writeFileSync(join(args.artifactDir, "worker-probe.json"), `${JSON.stringify(result, null, 2)}\n`);
    } catch (e) {
      console.error(`worker-probe artifact write failed: ${e?.message ?? e}`);
    }
    try { proc?.kill(); } catch { /* already dead */ }
    try { fixtureServer?.close(); } catch { /* never served */ }
    try { staticServer?.close(); } catch { /* never served */ }
  }
}

// Pre-try failures (bad args, artifact dir) have no servers to clean;
// report them with the same shape instead of an unhandled rejection.
main().catch((err) => {
  console.error(`WORKER PROBE FAILED: ${err?.message ?? err}`);
  process.exitCode = 1;
});
