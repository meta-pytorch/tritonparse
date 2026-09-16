#!/usr/bin/env node
/**
 * Committed e2e runner for the Monaco IR migration (§6.3 layers 2–3).
 *
 * Usage: npm run test:e2e-monaco [-- --base-url URL --artifact-dir DIR --chrome PATH]
 *
 * - Serves e2e/monaco-ir/fixtures over a local HTTP server (CORS *) and loads
 *   them through the real ?json_url pipeline.
 * - Launches a pristine headless Chrome, connects over CDP, and drives every
 *   input with trusted Input.* events (real mouse/keyboard).
 * - Reads state only through the ?debug=1 hook and Monaco model APIs.
 * - Prerequisite: a vite dev server (or preview) serving the site at --base-url.
 */
import { createServer } from "node:http";
import { mkdtempSync, readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname, normalize, relative, sep } from "node:path";
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
  typeText,
  captureScreenshot,
} from "./cdp.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const FIXTURES = join(HERE, "fixtures");

function parseArgs(argv) {
  const out = { baseUrl: "http://localhost:5173", artifactDir: null, chrome: null };
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--base-url") out.baseUrl = argv[++i];
    else if (argv[i] === "--artifact-dir") out.artifactDir = argv[++i];
    else if (argv[i] === "--chrome") out.chrome = argv[++i];
    else throw new Error(`unknown arg: ${argv[i]}`);
  }
  out.baseUrl = out.baseUrl.replace(/\/$/, "");
  // Node resolves localhost to ::1 first without curl-style fallback, while
  // the dev server typically binds 127.0.0.1; normalize so fetch and the
  // page URL both use the IPv4 loopback.
  out.baseUrl = out.baseUrl.replace("://localhost", "://127.0.0.1");
  return out;
}

/**
 * Deterministic long File Diff pair: 240-line ttgir with exactly one changed
 * line (120: 11 vs 22), one 6000-char overflow line (3), plus small llir and
 * identical json on both sides. Generated (not checked in) so the Context
 * hidden-row counts and the JSON IR leg stay in sync with this runner.
 */
function buildLongFileDiffPair() {
  const base = JSON.parse(readFileSync(join(FIXTURES, "single-basic.ndjson"), "utf8").trim());
  const lines = [];
  for (let i = 1; i <= 240; i++) {
    lines.push(`// e2e File Diff filler line ${i} with enough unchanged text to align both sides`);
  }
  lines[2] = `// e2e long line ${"x".repeat(6000)}`;
  const left = structuredClone(base);
  const right = structuredClone(base);
  const l = [...lines];
  l[119] = "%changed = arith.constant 11 : i32 // e2e exact expected difference";
  const r = [...lines];
  r[119] = "%changed = arith.constant 22 : i32 // e2e exact expected difference";
  left.payload.file_content["e2e_kernel.ttgir"] = `${l.join("\n")}\n`;
  right.payload.file_content["e2e_kernel.ttgir"] = `${r.join("\n")}\n`;
  const json = `${JSON.stringify({ e2e: "filediff-long", lines: 240 }, null, 2)}\n`;
  left.payload.file_content["e2e_kernel.json"] = json;
  right.payload.file_content["e2e_kernel.json"] = json;
  return {
    "filediff-long-left.ndjson": `${JSON.stringify(left)}\n`,
    "filediff-long-right.ndjson": `${JSON.stringify(right)}\n`,
  };
}

function startFixtureServer(generated) {
  const server = createServer((req, res) => {
    const name = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\//, "");
    const send = (body) => {
      res.writeHead(200, {
        "Content-Type": "application/x-ndjson",
        "Access-Control-Allow-Origin": "*",
      });
      res.end(body);
    };
    if (generated[name]) {
      send(generated[name]);
      return;
    }
    // Containment (mirrored in perf.mjs): never serve outside FIXTURES.
    const file = normalize(join(FIXTURES, name));
    const rel = relative(FIXTURES, file);
    if (!name.endsWith(".ndjson") || rel === ".." || rel.startsWith(`..${sep}`) || !existsSync(file)) {
      res.writeHead(404).end("no such fixture");
      return;
    }
    send(readFileSync(file));
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      resolve({ server, port: server.address().port });
    });
  });
}

function assertEqual(actual, expected, label) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) throw new Error(`${label}: expected ${e}, got ${a}`);
  console.log(`  ok ${label} = ${a}`);
}

const results = [];
async function step(name, fn) {
  const t0 = Date.now();
  try {
    await fn();
    results.push({ name, status: "pass", ms: Date.now() - t0 });
    console.log(`PASS ${name} (${Date.now() - t0}ms)`);
  } catch (err) {
    results.push({ name, status: "fail", ms: Date.now() - t0, error: String(err).slice(0, 400) });
    console.error(`FAIL ${name}: ${err.message}`);
    throw err;
  }
}

async function main() {
  // Everything this runner fetches is loopback; never send it through a proxy.
  for (const key of ["NO_PROXY", "no_proxy"]) {
    const cur = (process.env[key] ?? "").split(",").map((s) => s.trim()).filter(Boolean);
    for (const host of ["localhost", "127.0.0.1"]) {
      if (!cur.includes(host)) cur.push(host);
    }
    process.env[key] = cur.join(",");
  }
  const args = parseArgs(process.argv.slice(2));
  const artifactDir = args.artifactDir ?? join(HERE, "last-run");
  mkdirSync(artifactDir, { recursive: true });

  // Fail fast when no dev server is serving the base URL.
  try {
    const res = await fetch(args.baseUrl, { signal: AbortSignal.timeout(10000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (err) {
    throw new Error(`base URL unreachable (${args.baseUrl}); start vite dev/preview first: ${err.message}`);
  }

  const generated = buildLongFileDiffPair();
  for (const [name, body] of Object.entries(generated)) {
    writeFileSync(join(artifactDir, name), body);
  }
  const { server, port: fixturePort } = await startFixtureServer(generated);
  const fixtureUrl = `http://127.0.0.1:${fixturePort}/single-basic.ndjson`;
  const chromePath = args.chrome ?? findChrome();
  const { proc, port: debugPort } = await launchChrome({
    chromePath,
    userDataDir: mkdtempSync(join(tmpdir(), "monaco-e2e-")),
    extraArgs: ["--window-size=1920,1080"],
  });
  console.log(`chrome ${chromePath} (debug :${debugPort}), fixtures :${fixturePort}`);

  const failures = [];
  try {
    const targets = await listTargets(debugPort);
    const pageTarget = targets.find((t) => t.type === "page");
    if (!pageTarget) throw new Error("no page target in fresh chrome");
    const s = await connectPageTarget(pageTarget);
    await s.send("Page.enable");
    await s.send("Runtime.enable");

    const consoleErrors = [];
    const consoleWarnings = [];
    s.on("Runtime.consoleAPICalled", (p) => {
      const text = (p.args ?? []).map((a) => a.value ?? a.description ?? "").join(" ");
      if (p.type === "error") consoleErrors.push(text.slice(0, 300));
      if (p.type === "warning") consoleWarnings.push(text.slice(0, 300));
    });
    s.on("Runtime.exceptionThrown", (p) => {
      const d = p.exceptionDetails ?? {};
      const desc = d.exception?.description ?? d.text ?? "";
      const stack = (d.stackTrace?.callFrames ?? []).slice(0, 3)
        .map((f) => `${f.functionName}@${f.url.split("/").pop()}:${f.lineNumber}`).join(" <- ");
      consoleErrors.push(`uncaught: ${desc.slice(0, 300)} [${stack.slice(0, 200)}]`);
    });

    const shot = async (name) => {
      const png = await captureScreenshot(s);
      const file = join(artifactDir, name);
      writeFileSync(file, png);
      console.log(`  shot ${file}`);
    };

    const decorations = () =>
      evaluate(s, `() => {
        const ed = window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor;
        return ed.getModel().getAllDecorations()
          .filter((d) => d.options.className === 'mp-highlighted-line')
          .map((d) => d.range.startLineNumber).sort((a, b) => a - b);
      }`);

    /** Real-mouse click on a card/heading found by text (overview navigation). */
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
      // Let the programmatic scroll settle, then re-read the rect: the click
      // itself is always a real CDP mouse event at viewport coordinates.
      await new Promise((r) => setTimeout(r, 400));
      const settled = await evaluate(
        s,
        `() => {
          const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
          const el = els.find((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2, visible: r.bottom > 0 && r.top < window.innerHeight };
        }`
      );
      if (!settled.visible) throw new Error(`${selector} ${text} not visible after scroll`);
      await mouseClick(s, settled.x, settled.y);
    }

    /** Viewport center of a selector match (read-only query; caller clicks). */
    async function rectOf(selector, index = 0) {
      const rect = await evaluate(
        s,
        `() => {
          const el = document.querySelectorAll(${JSON.stringify(selector)})[${index}];
          if (!el) return null;
          el.scrollIntoView({ block: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2,
            visible: r.bottom > 0 && r.top < window.innerHeight };
        }`
      );
      if (!rect) throw new Error(`no ${selector}[${index}]`);
      await new Promise((r) => setTimeout(r, 400));
      const settled = await evaluate(
        s,
        `() => {
          const el = document.querySelectorAll(${JSON.stringify(selector)})[${index}];
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2,
            visible: r.bottom > 0 && r.top < window.innerHeight };
        }`
      );
      if (!settled.visible) throw new Error(`${selector}[${index}] not visible after scroll`);
      return settled;
    }

    /** clickText with an occurrence index (e.g. two "Load" buttons). */
    async function clickNthText(selector, text, index) {
      const rect = await evaluate(
        s,
        `() => {
          const els = [...document.querySelectorAll(${JSON.stringify(selector)})]
            .filter((e) => (e.textContent || '').trim() === ${JSON.stringify(text)});
          const el = els[${index}];
          if (!el) return null;
          el.scrollIntoView({ block: 'center' });
          const r = el.getBoundingClientRect();
          return { x: r.x + r.width / 2, y: r.y + r.height / 2, visible: r.bottom > 0 && r.top < window.innerHeight };
        }`
      );
      if (!rect) throw new Error(`no ${selector}[${index}] with text ${text}`);
      await new Promise((r) => setTimeout(r, 400));
      await mouseClick(s, rect.x, rect.y);
    }

    /** Real Ctrl+A (select-all) for replacing input contents. */
    async function ctrlA() {
      for (const type of ["rawKeyDown", "keyUp"]) {
        await s.send("Input.dispatchKeyEvent", {
          type, key: "a", code: "KeyA", windowsVirtualKeyCode: 65, modifiers: 2,
        });
      }
    }

    /** Real-mouse click on a Monaco line; polls decorations to expected. */
    async function clickLine(line, expected) {
      // I003: the debug/editor API is read-only (§6.3) — no reveal/scroll/layout
      // calls. The short fixture fits the initial viewport, so every target
      // line already has a visible position; a case that needs scrolling must
      // use real wheel/keyboard input, never editor API.
      const xy = await evaluate(s, `() => {
        const ed = window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor;
        const pos = ed.getScrolledVisiblePosition({ lineNumber: ${line}, column: 1 });
        if (!pos) return null;
        const r = ed.getDomNode().getBoundingClientRect();
        const x = r.x + 250, y = r.y + pos.top + pos.height / 2;
        if (y < 0 || y > window.innerHeight || x < 0 || x > window.innerWidth) return null;
        return { x, y };
      }`);
      if (!xy) throw new Error(`line ${line} not in initial viewport; add real-input scrolling instead of debug-driven reveal`);
      await mouseClick(s, xy.x, xy.y);
      const deadline = Date.now() + 5000;
      for (;;) {
        const got = await decorations();
        if (JSON.stringify(got) === JSON.stringify(expected)) {
          console.log(`  ok click line ${line} -> ${JSON.stringify(got)}`);
          return;
        }
        if (Date.now() > deadline) {
          throw new Error(`click line ${line}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(got)}`);
        }
        await new Promise((r) => setTimeout(r, 150));
      }
    }

    const appUrl =
      `${args.baseUrl}/?json_url=${encodeURIComponent(fixtureUrl)}&renderer=monaco&debug=1`;
    await step("load fixture trace", async () => {
      await s.send("Page.navigate", { url: appUrl });
      await waitForFunction(
        s,
        `() => [...document.querySelectorAll('h3')].some((h) => h.textContent.trim() === 'e2e_kernel.ttgir')`,
        { timeoutMs: 60000 }
      );
    });

    await step("open Single ttgir (real click)", async () => {
      await clickText("h3", "e2e_kernel.ttgir");
      await waitForFunction(
        s,
        `() => !!window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor`,
        { timeoutMs: 60000 }
      );
      const lang = await evaluate(
        s,
        `() => window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor.getModel().getLanguageId()`
      );
      assertEqual(lang, "triton-mlir", "ttgir language");
    });

    await step("anchor group click incl. invalid/out-of-range filtering (F9/F19)", async () => {
      // Fixture group for anchor value 2: lines 2,4 plus key 999 (out of range)
      // and keys locX/3.5/4oops (non-integer) — all must be filtered with a
      // visible diagnostic (I001: 3.5/4oops must not forge lines 3/4).
      await clickLine(2, [2, 4]);
      const badge = await waitForFunction(
        s,
        `() => { const el = document.querySelector('[data-testid="mp-diagnostics-badge"]'); return el ? el.textContent.trim() : false; }`,
        { timeoutMs: 10000 }
      );
      assertEqual(badge, "4 mappings ignored", "diagnostics badge");
      await shot("e2e-single-click2.png");
    });

    await step("overlapping group keeps exact set, then clears (F2)", async () => {
      await clickLine(4, [2, 4]);
      await clickLine(6, [6]);
      await clickLine(1, [1]);
    });

    await step("diagnostic warning for dropped lines is visible", async () => {
      const hit = consoleWarnings.find(
        (w) => w.includes("dropped 3 invalid / 1 out-of-range") && w.includes("e2e_kernel.ttgir")
      );
      if (!hit) throw new Error(`no dropped-lines warning; warnings: ${JSON.stringify(consoleWarnings)}`);
      console.log(`  ok warning: ${hit.slice(0, 160)}`);
    });

    await step("doc switch clears highlights, llir resolves llvm (F18 basic)", async () => {
      await clickText("button", "Back");
      await waitForFunction(
        s,
        `() => [...document.querySelectorAll('h3')].some((h) => h.textContent.trim() === 'e2e_kernel.llir')`,
        { timeoutMs: 30000 }
      );
      await clickText("h3", "e2e_kernel.llir");
      await waitForFunction(
        s,
        `() => !!window.__TRITONPARSE_DEBUG?.panels?.['single-viewer']?.editor`,
        { timeoutMs: 60000 }
      );
      const state = await evaluate(s, `() => {
        const p = window.__TRITONPARSE_DEBUG.panels['single-viewer'];
        return { lang: p.editor.getModel().getLanguageId(), highlights: p.getHighlights() };
      }`);
      assertEqual(state.lang, "triton-llvm", "llir language");
      assertEqual(state.highlights, [], "highlights after doc switch");
      assertEqual(await decorations(), [], "decorations after doc switch");
      await shot("e2e-single-llir.png");
    });

    await step("back removes debug hook (F10)", async () => {
      await clickText("button", "Back");
      await waitForFunction(
        s,
        `() => Object.keys(window.__TRITONPARSE_DEBUG?.panels ?? {}).length === 0`,
        { timeoutMs: 30000 }
      );
      // Console errors are asserted once at the very end (more suites follow).
    });

    // ---- Single ruler suite (F14/F20): 6001-line overflow set ----
    const rulerFixtureUrl = `http://127.0.0.1:${fixturePort}/ruler-6001.ndjson`;
    const rulerUrl =
      `${args.baseUrl}/?json_url=${encodeURIComponent(rulerFixtureUrl)}&renderer=monaco&debug=1`;

    const singleState = () =>
      evaluate(s, `() => {
        const p = window.__TRITONPARSE_DEBUG.panels['single-viewer'];
        const ed = p.editor;
        const spans = ed.getModel().getAllDecorations()
          .filter((d) => d.options.className === 'mp-highlighted-line')
          .map((d) => [d.range.startLineNumber, d.range.endLineNumber]);
        const ruler = document.querySelector('[data-testid="overview-ruler"]');
        const pane = document.querySelector('[data-testid="single-viewer-monaco-panel"]');
        const rr = ruler?.getBoundingClientRect();
        const pr = pane?.getBoundingClientRect();
        return {
          highlights: p.getHighlights(),
          spans,
          markers: ruler ? ruler.querySelectorAll('[data-testid^="overview-marker-"]').length : -1,
          badge: document.querySelector('[data-testid="ruler-overflow-badge"]')?.textContent.trim() ?? null,
          diag: document.querySelector('[data-testid="mp-diagnostics-badge"]')?.textContent.trim() ?? null,
          rulerRect: rr ? { x: rr.x, y: rr.y, w: rr.width, h: rr.height } : null,
          paneRect: pr ? { x: pr.x, y: pr.y, w: pr.width, h: pr.height } : null,
        };
      }`);

    /** Poll until a ruler click centers the line (smooth scroll settles). */
    async function waitCentered(line) {
      await waitForFunction(
        s,
        `() => {
          const D = window.__TRITONPARSE_DEBUG;
          const ed = D.panels['single-viewer'].editor;
          const lh = ed.getOption(D.monaco.editor.EditorOption.lineHeight);
          const h = ed.getLayoutInfo().height;
          const want = ed.getTopForLineNumber(${line}) - (h - lh) / 2;
          return Math.abs(ed.getScrollTop() - want) <= lh ? true : false;
        }`,
        { timeoutMs: 15000 }
      );
    }

    await step("ruler overflow: full set in state, sampled strip (F20)", async () => {
      await s.send("Page.navigate", { url: rulerUrl });
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
      // Real click on line 2; the full 6001-line group lands in state while
      // the strip samples to 200 markers (I001-style filtering still applies:
      // 3 invalid + 2 out-of-range keys are dropped with a badge).
      const xy = await evaluate(s, `() => {
        const ed = window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor;
        const pos = ed.getScrolledVisiblePosition({ lineNumber: 2, column: 1 });
        const r = ed.getDomNode().getBoundingClientRect();
        return { x: r.x + 250, y: r.y + pos.top + pos.height / 2 };
      }`);
      await mouseClick(s, xy.x, xy.y);
      await waitForFunction(
        s,
        `() => {
          const h = window.__TRITONPARSE_DEBUG.panels['single-viewer'].getHighlights();
          return h.length === 6001 ? true : false;
        }`,
        { timeoutMs: 15000 }
      );
      const full = await singleState();
      assertEqual(full.highlights.length, 6001, "full highlight set in state");
      assertEqual([full.highlights[0], full.highlights[6000]], [2, 6002], "set endpoints");
      assertEqual(full.spans, [[2, 6002]], "decorations merged full span");
      assertEqual(full.markers, 200, "sampled strip markers");
      assertEqual(full.badge, "200/6001", "overflow badge");
      assertEqual(full.diag, "5 mappings ignored", "diagnostics badge");
      // Geometry: own 14px strip beside the editor, never overlapping it.
      assertEqual(full.rulerRect.w, 14, "ruler strip width");
      if (!(full.rulerRect.x >= full.paneRect.x + full.paneRect.w - 1)) {
        throw new Error(`ruler overlaps editor: ${JSON.stringify({ rulerRect: full.rulerRect, paneRect: full.paneRect })}`);
      }
      const interception = await evaluate(s, `() => {
        const pane = document.querySelector('[data-testid="single-viewer-monaco-panel"]');
        const pr = pane.getBoundingClientRect();
        const el = document.elementFromPoint(pr.x + pr.width - 6, pr.y + pr.height / 2);
        return {
          inMonaco: !!el?.closest('.monaco-editor'),
          inRuler: !!el?.closest('[data-testid="overview-ruler"]'),
        };
      }`);
      assertEqual(interception, { inMonaco: true, inRuler: false }, "no ruler interception at editor edge");
      await shot("e2e-ruler-overflow.png");
    });

    await step("marker centers target, set unchanged, keyboard works (F14)", async () => {
      const last = await rectOf('[data-testid="overview-marker-6002"]');
      await mouseClick(s, last.x, last.y);
      await waitCentered(6002);
      let st = await singleState();
      assertEqual(st.highlights.length, 6001, "marker click keeps set");
      // Already-visible target is still centered; keyboard Enter activates.
      await waitCentered(6002);
      await keyPress(s, "Enter", { code: "Enter", windowsVirtualKeyCode: 13 });
      await waitCentered(6002);
      st = await singleState();
      assertEqual(st.highlights.length, 6001, "keyboard activation keeps set");
      await shot("e2e-ruler-centered.png");
    });

    await step("overflow popup lists all + copy-all resolves full list (F20)", async () => {
      // Observation-only spy on the platform clipboard API: the click below
      // is real input; this only records what the app resolved to write.
      await evaluate(s, `() => {
        window.__e2e_clipboard = [];
        const clip = navigator.clipboard;
        const orig = clip.writeText.bind(clip);
        clip.writeText = (t) => {
          window.__e2e_clipboard.push(t);
          return orig(t).catch(() => {});
        };
      }`);
      const badge = await rectOf('[data-testid="ruler-overflow-badge"]');
      await mouseClick(s, badge.x, badge.y);
      const popup = await waitForFunction(
        s,
        `() => {
          const items = [...document.querySelectorAll('.code-overview-popup-item')];
          return items.length === 6001
            ? { n: items.length, first: items[0].textContent, last: items[6000].textContent }
            : false;
        }`,
        { timeoutMs: 15000 }
      );
      assertEqual(popup, { n: 6001, first: "Line 2", last: "Line 6002" }, "popup full list");
      await shot("e2e-ruler-popup.png");
      await clickText("button", "Copy all");
      const copied = await waitForFunction(
        s,
        `() => window.__e2e_clipboard.length > 0 ? window.__e2e_clipboard[0] : false`,
        { timeoutMs: 10000 }
      );
      const expected = Array.from({ length: 6001 }, (_, i) => 2 + i).join(", ");
      if (copied !== expected) {
        throw new Error(
          `copy-all mismatch: len ${copied.length} vs ${expected.length}, ` +
          `head ${JSON.stringify(copied.slice(0, 40))}, tail ${JSON.stringify(copied.slice(-40))}`
        );
      }
      console.log(`  ok copy-all full sorted list (len ${copied.length})`);
      await keyPress(s, "Escape", { code: "Escape", windowsVirtualKeyCode: 27 });
      await waitForFunction(
        s,
        `() => !document.querySelector('[data-testid="ruler-overflow-popup"]')`,
        { timeoutMs: 10000 }
      );
    });

    await step("replacement click clears ruler overflow (F2)", async () => {
      // Real Ctrl+End jumps near the document end; then click line 6010.
      const focus = await evaluate(s, `() => {
        const ed = window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor;
        const pos = ed.getScrolledVisiblePosition({ lineNumber: 2, column: 1 });
        const r = ed.getDomNode().getBoundingClientRect();
        return { x: r.x + 250, y: r.y + pos.top + pos.height / 2 };
      }`);
      await mouseClick(s, focus.x, focus.y);
      await s.send("Input.dispatchKeyEvent", { type: "rawKeyDown", key: "End", code: "End", windowsVirtualKeyCode: 35, modifiers: 2 });
      await s.send("Input.dispatchKeyEvent", { type: "keyUp", key: "End", code: "End", windowsVirtualKeyCode: 35, modifiers: 2 });
      await waitForFunction(
        s,
        `() => !!window.__TRITONPARSE_DEBUG.panels['single-viewer'].editor.getScrolledVisiblePosition({ lineNumber: 6010, column: 1 })`,
        { timeoutMs: 15000 }
      );
      await clickLine(6010, [6010]);
      const st = await singleState();
      assertEqual(st.markers, 1, "single marker after replacement");
      assertEqual(st.badge, null, "overflow badge gone");
      assertEqual(st.diag, null, "diagnostics badge gone");
      await shot("e2e-ruler-replaced.png");
    });

    // ---- File Diff suite (F17): shared-loader regression + baseline fixes ----
    const fixtureBUrl = `http://127.0.0.1:${fixturePort}/filediff-right.ndjson`;
    const fileDiffUrl =
      `${args.baseUrl}/?view=file_diff&json_url=${encodeURIComponent(fixtureUrl)}` +
      `&json_b_url=${encodeURIComponent(fixtureBUrl)}&ir=ttgir&wrap=on&debug=1`;

    const filediffState = () =>
      evaluate(s, `() => {
        const D = window.__TRITONPARSE_DEBUG;
        const de = D.panels.filediff.diffEditor;
        const E = D.monaco.editor.EditorOption;
        const o = de.getOriginalEditor(), m = de.getModifiedEditor();
        return {
          models: D.getModels(),
          origH: o.getLayoutInfo().height, modH: m.getLayoutInfo().height,
          origW: o.getLayoutInfo().width,
          origWrap: o.getOption(E.wrappingInfo).isViewportWrapping,
          modWrap: m.getOption(E.wrappingInfo).isViewportWrapping,
          origRO: o.getOption(E.readOnly), modRO: m.getOption(E.readOnly),
          origScrollW: o.getScrollWidth(), modScrollW: m.getScrollWidth(),
        };
      }`);

    const waitFileDiff = (ir) =>
      waitForFunction(
        s,
        `() => {
          const D = window.__TRITONPARSE_DEBUG;
          if (!D?.panels?.filediff?.diffEditor) return false;
          if (D.getModels().length !== 2) return false;
          const label = [...document.querySelectorAll('div')].find((d) => d.textContent.startsWith('IR Type:'));
          return label && label.textContent.includes(${JSON.stringify(ir)}) ? true : false;
        }`,
        { timeoutMs: 60000 }
      );

    /** Change a native select with real keyboard input (arrows switch options). */
    async function selectByKeyboard(targetValue) {
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
      const key = tgt >= cur ? "ArrowDown" : "ArrowUp";
      const code = tgt >= cur ? 40 : 38;
      for (let i = 0; i < Math.abs(tgt - cur); i++) {
        await keyPress(s, key, { code: key, windowsVirtualKeyCode: code });
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

    await step("open File Diff direct, panes laid out (F17 layout)", async () => {
      await s.send("Page.navigate", { url: fileDiffUrl });
      await waitFileDiff("ttgir");
      const st = await filediffState();
      assertEqual(st.models.length, 2, "initial model count");
      // Fresh-mount wrap must already be split-free (the inline-fallback
      // transient used to stick the original pane at override2='off').
      assertEqual([st.origWrap, st.modWrap], [true, true], "initial wrap both sides");
      if (st.origH <= 200 || st.modH <= 200) {
        throw new Error(`panes collapsed without resize: origH=${st.origH} modH=${st.modH}`);
      }
      console.log(`  ok panes ${Math.round(st.origH)}px, no resize needed`);
      await shot("e2e-filediff-initial.png");
    });

    await step("IR switch keeps model count stable (F17 lifecycle)", async () => {
      for (const ir of ["llir", "ttgir"]) {
        await selectByKeyboard(ir);
        await waitFileDiff(ir);
        const st = await filediffState();
        assertEqual(st.models.length, 2, `model count after -> ${ir}`);
        if (st.origH <= 200 || st.modH <= 200) {
          throw new Error(`panes collapsed after -> ${ir}: ${st.origH}x${st.modH}`);
        }
      }
      // Rapid double switch with no settle between: the first widget remounts
      // while its init/diff may still be pending (pre-onMount unmount edge).
      await selectByKeyboard("llir");
      await selectByKeyboard("ttgir");
      await waitFileDiff("ttgir");
      const st = await filediffState();
      assertEqual(st.models.length, 2, "model count after rapid double switch");
      if (st.origH <= 200 || st.modH <= 200) {
        throw new Error(`panes collapsed after rapid switch: ${st.origH}x${st.modH}`);
      }
      await shot("e2e-filediff-ir-switched.png");
    });

    await step("wrap toggle syncs both panes (F17 wrap)", async () => {
      await selectByKeyboard("off");
      await new Promise((r) => setTimeout(r, 600));
      let st = await filediffState();
      assertEqual([st.origWrap, st.modWrap], [false, false], "wrap off both sides");
      if (!(st.origScrollW > st.origW)) {
        throw new Error(`expected horizontal overflow with wrap off: scrollW=${st.origScrollW} w=${st.origW}`);
      }
      await shot("e2e-filediff-wrap-off.png");
      await selectByKeyboard("on");
      await new Promise((r) => setTimeout(r, 600));
      st = await filediffState();
      assertEqual([st.origWrap, st.modWrap], [true, true], "wrap on both sides");
      if (!(st.origScrollW <= st.origW + 1 && st.modScrollW <= st.origW + 1)) {
        throw new Error(`overflow persists with wrap on: ${st.origScrollW}/${st.modScrollW} vs ${st.origW}`);
      }
      await shot("e2e-filediff-wrap-on.png");
      // Exact baseline repro: Wrap off -> switch IR -> Wrap on. A stale
      // mount-time capture used to leave the panes split (original off).
      await selectByKeyboard("off");
      await selectByKeyboard("llir");
      await waitFileDiff("llir");
      await selectByKeyboard("ttgir");
      await waitFileDiff("ttgir");
      await selectByKeyboard("on");
      await new Promise((r) => setTimeout(r, 600));
      st = await filediffState();
      assertEqual([st.origWrap, st.modWrap], [true, true], "wrap on both sides after IR switch");
      if (!(st.origScrollW <= st.origW + 1 && st.modScrollW <= st.origW + 1)) {
        throw new Error(`split panes after wrap->IR->wrap: ${st.origScrollW}/${st.modScrollW} vs ${st.origW}`);
      }
    });

    await step("only-changes + context via real input (F17 options)", async () => {
      await clickText("label", "Only changes");
      await new Promise((r) => setTimeout(r, 800));
      let st = await filediffState();
      assertEqual(st.models.length, 2, "model count with only-changes");
      await shot("e2e-filediff-only-changes.png");
      await clickText("label", "Only changes");
      await new Promise((r) => setTimeout(r, 800));
      st = await filediffState();
      assertEqual(st.models.length, 2, "model count after only-changes off");
    });

    await step("context changes visible rows + json leg (F17 context)", async () => {
      const longLeft = `http://127.0.0.1:${fixturePort}/filediff-long-left.ndjson`;
      const longRight = `http://127.0.0.1:${fixturePort}/filediff-long-right.ndjson`;
      const longUrl =
        `${args.baseUrl}/?view=file_diff&json_url=${encodeURIComponent(longLeft)}` +
        `&json_b_url=${encodeURIComponent(longRight)}&ir=ttgir&wrap=on&debug=1`;
      await s.send("Page.navigate", { url: longUrl });
      await waitFileDiff("ttgir");
      // Real diff content: exactly one line change at line 120.
      const changes = await waitForFunction(
        s,
        `() => {
          const c = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getLineChanges();
          return c && c.length === 1 ? c : false;
        }`,
        { timeoutMs: 30000 }
      );
      assertEqual(
        [changes[0].originalStartLineNumber, changes[0].modifiedStartLineNumber],
        [120, 120],
        "single real change at line 120"
      );
      const hiddenTotal = () =>
        evaluate(s, `() => [...document.querySelectorAll('.diff-hidden-lines')]
          .map((e) => Number(e.textContent.match(/(\\d+) hidden lines/)?.[1] ?? 0))
          .reduce((a, b) => a + b, 0)`);
      // Monaco renders spaces as nbsp in view lines; normalize before match.
      const renderedHas = (text) =>
        evaluate(s, `() => [...document.querySelectorAll('.monaco-diff-editor .modified .view-lines .view-line')]
          .some((e) => (e.textContent || '').replace(/\\u00a0/g, ' ').includes(${JSON.stringify(text)}))`);
      await clickText("label", "Only changes");
      await waitForFunction(s, `() => document.querySelectorAll('.diff-hidden-lines').length > 0`, { timeoutMs: 15000 });
      const before = await hiddenTotal();
      if (!(before > 0)) throw new Error("only-changes hid nothing");
      // Context 3 (default): line 119 visible, line 110 hidden. The viewport
      // relayout trails the hidden widgets, so poll for the settled render.
      await waitForFunction(
        s,
        `() => [...document.querySelectorAll('.monaco-diff-editor .modified .view-lines .view-line')]
          .some((e) => (e.textContent || '').replace(/\\u00a0/g, ' ').includes('filler line 119')) ? true : false`,
        { timeoutMs: 15000 }
      );
      console.log("  ok context line 119 rendered");
      assertEqual(await renderedHas("filler line 110"), false, "line 110 hidden");
      // Type 10 into the Context box with real input: more rows visible.
      const num = await rectOf('input[type="number"]');
      await mouseClick(s, num.x, num.y);
      await ctrlA();
      await typeText(s, "10");
      await keyPress(s, "Tab", { code: "Tab", windowsVirtualKeyCode: 9 });
      await waitForFunction(
        s,
        `() => {
          const total = [...document.querySelectorAll('.diff-hidden-lines')]
            .map((e) => Number(e.textContent.match(/(\\d+) hidden lines/)?.[1] ?? 0))
            .reduce((a, b) => a + b, 0);
          return total > 0 && total < ${before} ? total : false;
        }`,
        { timeoutMs: 15000 }
      );
      await waitForFunction(
        s,
        `() => [...document.querySelectorAll('.monaco-diff-editor .modified .view-lines .view-line')]
          .some((e) => (e.textContent || '').replace(/\\u00a0/g, ' ').includes('filler line 110')) ? true : false`,
        { timeoutMs: 15000 }
      );
      console.log("  ok line 110 revealed by context 10");
      await shot("e2e-filediff-context10.png");
      await clickText("label", "Only changes");
      await waitForFunction(s, `() => document.querySelectorAll('.diff-hidden-lines').length === 0`, { timeoutMs: 15000 });
      // JSON leg: identical json both sides, plaintext policy, models stable.
      await selectByKeyboard("json");
      await waitFileDiff("json");
      let st = await filediffState();
      assertEqual(st.models.length, 2, "model count on json");
      const jsonLang = await evaluate(s, `() => {
        const de = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor;
        return de.getModifiedEditor().getModel().getLanguageId();
      }`);
      assertEqual(jsonLang, "plaintext", "json stays plaintext");
      await selectByKeyboard("ttgir");
      await waitFileDiff("ttgir");
      st = await filediffState();
      assertEqual(st.models.length, 2, "model count back on ttgir");
    });

    await step("panes are readonly (F17 readonly)", async () => {
      const before = await evaluate(s, `() => {
        const de = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor;
        return [de.getOriginalEditor().getModel().getValue(), de.getModifiedEditor().getModel().getValue()];
      }`);
      const st = await filediffState();
      assertEqual([st.origRO, st.modRO], [true, true], "readonly flags");
      // Focus the original pane with a real click, then type real keys.
      const xy = await evaluate(s, `() => {
        const de = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor;
        const ed = de.getOriginalEditor();
        const pos = ed.getScrolledVisiblePosition({ lineNumber: 2, column: 10 });
        const r = ed.getDomNode().getBoundingClientRect();
        return { x: r.x + pos.left, y: r.y + pos.top + pos.height / 2 };
      }`);
      await mouseClick(s, xy.x, xy.y);
      await typeText(s, "XYZ");
      await new Promise((r) => setTimeout(r, 400));
      const after = await evaluate(s, `() => {
        const de = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor;
        return [de.getOriginalEditor().getModel().getValue(), de.getModifiedEditor().getModel().getValue()];
      }`);
      assertEqual(after, before, "content unchanged after typing");
    });

    await step("native handle drag resizes both panes (F17 resize)", async () => {
      // The container is intentionally fixed-pixel (mount-computed, user
      // resizable via the native resize-y handle); panes must track handle
      // drags through the ResizeObserver relay — driven here by a real mouse.
      const dims = () =>
        evaluate(s, `() => {
          const de = window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor;
          const inner = document.querySelector('[data-testid="file-diff-view"] > div');
          const r = inner.getBoundingClientRect();
          return {
            containerH: r.height, x: r.x + r.width - 8, y: r.y + r.height - 4,
            origH: de.getOriginalEditor().getLayoutInfo().height,
            modH: de.getModifiedEditor().getLayoutInfo().height,
          };
        }`);
      async function dragHandle(dy) {
        await evaluate(s, `() => { document.querySelector('[data-testid="file-diff-view"] > div').scrollIntoView({ block: 'end' }); }`);
        await new Promise((r) => setTimeout(r, 300));
        const d = await dims();
        const press = { type: "mousePressed", x: d.x, y: d.y, button: "left", buttons: 1, clickCount: 1 };
        await s.send("Input.dispatchMouseEvent", press);
        for (const f of [0.25, 0.5, 0.75, 1]) {
          await s.send("Input.dispatchMouseEvent", { type: "mouseMoved", x: d.x, y: d.y + dy * f, button: "left", buttons: 1 });
          await new Promise((r) => setTimeout(r, 60));
        }
        await s.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: d.x, y: d.y + dy, button: "left", buttons: 0, clickCount: 1 });
        await new Promise((r) => setTimeout(r, 500));
        return dims();
      }
      const before = await dims();
      const grown = await dragHandle(150);
      const dContainer = grown.containerH - before.containerH;
      if (!(dContainer > 100 && dContainer < 200)) {
        throw new Error(`handle drag did not resize container: ${Math.round(before.containerH)} -> ${Math.round(grown.containerH)}`);
      }
      if (Math.abs((grown.origH - before.origH) - dContainer) > 60 ||
          Math.abs((grown.modH - before.modH) - dContainer) > 60) {
        throw new Error(`panes did not track drag: orig ${Math.round(grown.origH - before.origH)} mod ${Math.round(grown.modH - before.modH)} vs container ${Math.round(dContainer)}`);
      }
      if (Math.abs(grown.origH - grown.modH) > 2) {
        throw new Error(`panes diverged after drag: ${grown.origH}x${grown.modH}`);
      }
      console.log(`  ok panes tracked drag (+${Math.round(dContainer)}px)`);
      await shot("e2e-filediff-resized.png");
      const restored = await dragHandle(-150);
      if (Math.abs(restored.origH - before.origH) > 60) {
        throw new Error(`panes did not recover: ${Math.round(restored.origH)} vs ${Math.round(before.origH)}`);
      }
    });

    const globalCounts = () =>
      evaluate(s, `() => {
        const D = window.__TRITONPARSE_DEBUG;
        return {
          models: D.getModels().length,
          diffWidgets: D.monaco.editor.getDiffEditors().length,
        };
      }`);
    const waitCounts = (models, widgets) =>
      waitForFunction(
        s,
        `() => {
          const D = window.__TRITONPARSE_DEBUG;
          return D.getModels().length === ${models} &&
            D.monaco.editor.getDiffEditors().length === ${widgets} ? true : false;
        }`,
        { timeoutMs: 60000 }
      );

    await step("all-ir multi-instance expand/collapse lifecycle (F17)", async () => {
      await clickText("button", "All IRs");
      await waitForFunction(
        s,
        `() => [...document.querySelectorAll('button')].some((b) => b.textContent.trim() === 'llir')`,
        { timeoutMs: 30000 }
      );
      await clickText("button", "ttgir");
      await waitCounts(2, 1);
      await clickText("button", "llir");
      await waitCounts(4, 2);
      await shot("e2e-filediff-all-two.png");
      // Collapse the last-opened widget while the other stays mounted.
      await clickText("button", "llir");
      await waitCounts(2, 1);
      if (consoleErrors.length > 0) {
        throw new Error(`console errors after collapse unmount: ${JSON.stringify(consoleErrors)}`);
      }
      console.log("  ok collapse unmount without errors");
      // All -> Single -> All round-trip, then collapse everything.
      await clickText("button", "Single IR");
      await waitFileDiff("ttgir");
      await clickText("button", "All IRs");
      await waitCounts(2, 1);
      await clickText("button", "ttgir");
      await waitCounts(0, 0);
      if (consoleErrors.length > 0) {
        throw new Error(`console errors after full collapse: ${JSON.stringify(consoleErrors)}`);
      }
      await clickText("button", "Single IR");
      await waitFileDiff("ttgir");
      const st = await filediffState();
      assertEqual(st.models.length, 2, "model count back in single");
    });

    await step("source-url replacement reloads mounted diff in place (F17)", async () => {
      // Swap both sources through the File Diff-local Load controls; kernel
      // indexes stay 0 and the selected IR exists in both traces, so the
      // mounted widget updates its content (keys omit source identity).
      const longLeft = `http://127.0.0.1:${fixturePort}/filediff-long-left.ndjson`;
      const longRight = `http://127.0.0.1:${fixturePort}/filediff-long-right.ndjson`;
      const origValue = () =>
        evaluate(s, `() => window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getOriginalEditor().getModel().getValue()`);
      const modValue = () =>
        evaluate(s, `() => window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getModifiedEditor().getModel().getValue()`);
      const beforeOrig = await origValue();
      const leftInput = await rectOf('input[type="url"]', 0);
      await mouseClick(s, leftInput.x, leftInput.y);
      await ctrlA();
      await typeText(s, longRight);
      await clickNthText("button", "Load", 0);
      await waitForFunction(
        s,
        `() => window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getOriginalEditor().getModel().getValue() !== ${JSON.stringify(beforeOrig)} ? true : false`,
        { timeoutMs: 60000 }
      );
      let counts = await globalCounts();
      assertEqual(counts, { models: 2, diffWidgets: 1 }, "counts after left replacement");
      const beforeMod = await modValue();
      const rightInput = await rectOf('input[type="url"]', 1);
      await mouseClick(s, rightInput.x, rightInput.y);
      await ctrlA();
      await typeText(s, longLeft);
      await clickNthText("button", "Load", 1);
      await waitForFunction(
        s,
        `() => window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getModifiedEditor().getModel().getValue() !== ${JSON.stringify(beforeMod)} ? true : false`,
        { timeoutMs: 60000 }
      );
      counts = await globalCounts();
      assertEqual(counts, { models: 2, diffWidgets: 1 }, "counts after right replacement");
      const st = await filediffState();
      assertEqual([st.origWrap, st.modWrap], [true, true], "wrap holds after replacement");
      const changes = await evaluate(s, `() => window.__TRITONPARSE_DEBUG.panels.filediff.diffEditor.getLineChanges()`);
      assertEqual(changes?.length, 1, "real diff after replacement");
      await shot("e2e-filediff-replaced.png");
    });

    await step("preview round-trip unmounts/remounts cleanly (F17)", async () => {
      // Preview uses exclusive rendering (not keep-alive): navigating away is
      // a real component unmount, returning is a fresh remount. The old
      // hideDiff flag used to leave a blank diff on return (never restored).
      const errorsBefore = consoleErrors.length;
      await clickText("button", "Left → IR Code");
      await waitForFunction(
        s,
        `() => new URLSearchParams(window.location.search).get('view') === 'ir_code_comparison'`,
        { timeoutMs: 30000 }
      );
      await waitCounts(0, 0);
      if (consoleErrors.length !== errorsBefore) {
        throw new Error(`console errors on preview unmount: ${JSON.stringify(consoleErrors.slice(errorsBefore))}`);
      }
      console.log("  ok preview unmount without errors");
      await clickText("button", "File Diff");
      await waitFileDiff("ttgir");
      const st = await filediffState();
      assertEqual(st.models.length, 2, "model count after preview remount");
      assertEqual([st.origWrap, st.modWrap], [true, true], "wrap after preview remount");
      if (st.origH <= 200 || st.modH <= 200) {
        throw new Error(`panes blank after preview round-trip: ${st.origH}x${st.modH}`);
      }
      console.log("  ok diff visible after preview round-trip");
    });

    await step("tab switch keep-alive preserves diff (F17 lifecycle)", async () => {
      await clickText("button", "Kernel Overview");
      await waitForFunction(s, `() => document.body.textContent.includes("Kernel Details")`, { timeoutMs: 30000 });
      await clickText("button", "File Diff");
      await waitFileDiff("ttgir");
      const st = await filediffState();
      assertEqual(st.models.length, 2, "model count after tab round-trip");
      if (st.origH <= 200 || st.modH <= 200) {
        throw new Error(`panes collapsed after tab round-trip: ${st.origH}x${st.modH}`);
      }
      await shot("e2e-filediff-final.png");
    });

    await step("no console errors across all suites (F10/F17)", async () => {
      if (consoleErrors.length > 0) {
        throw new Error(`console errors: ${JSON.stringify(consoleErrors)}`);
      }
      console.log("  ok no console errors across all suites");
    });

    s.close();
  } catch (err) {
    failures.push(err);
  } finally {
    proc.kill();
    server.close();
  }

  console.log(`\n${results.filter((r) => r.status === "pass").length}/${results.length} steps passed`);
  if (failures.length > 0) {
    console.error(`E2E FAILED: ${failures[0].message}`);
    process.exit(1);
  }
  console.log("E2E PASSED");
}

main().catch((err) => {
  console.error(`E2E FAILED: ${err.message}`);
  process.exit(1);
});
