/**
 * Minimal zero-dependency Chrome DevTools Protocol client (Node 24+).
 *
 * Shared by the committed e2e runner (run.mjs) and ad-hoc dev verification.
 * Uses only global fetch/WebSocket/child_process. Real mouse/keyboard input
 * always goes through CDP Input.* — never through app-internal callbacks.
 */
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";

/** Locate a Chrome binary: $CHROME_PATH, then well-known candidates. */
export function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "/tmp/fbpkg/one_world.browsers.chrome/latest/opt/google/chrome/chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ].filter(Boolean);
  for (const c of candidates) {
    if (c && existsSync(c)) return c;
  }
  throw new Error(
    `no Chrome binary found (tried ${candidates.join(", ")}); set CHROME_PATH`
  );
}

/**
 * Launch headless Chrome with a fresh profile on an ephemeral debug port.
 * Resolves once "DevTools listening on ..." appears on stderr.
 */
export function launchChrome({ chromePath, userDataDir, extraArgs = [] }) {
  return new Promise((resolve, reject) => {
    const args = [
      "--headless=new",
      "--remote-debugging-port=0",
      `--user-data-dir=${userDataDir}`,
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      "--no-sandbox",
      "--disable-gpu",
      "--disable-dev-shm-usage",
      "about:blank",
      ...extraArgs,
    ];
    const proc = spawn(chromePath, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderr = "";
    const timer = setTimeout(() => {
      proc.kill();
      reject(new Error(`timed out waiting for DevTools endpoint; stderr: ${stderr.slice(-500)}`));
    }, 30000);
    proc.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      const m = stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);
      if (m) {
        clearTimeout(timer);
        const port = Number(new URL(m[1]).port);
        resolve({ proc, wsEndpoint: m[1], port });
      }
    });
    proc.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    proc.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`chrome exited early with code ${code}; stderr: ${stderr.slice(-500)}`));
    });
  });
}

/** GET /json/list on a debug port. */
export async function listTargets(port) {
  const res = await fetch(`http://127.0.0.1:${port}/json/list`);
  if (!res.ok) throw new Error(`/json/list -> ${res.status}`);
  return res.json();
}

/** Attach to one page target's webSocketDebuggerUrl. */
export function connectPageTarget(pageTarget) {
  return CdpSession.connect(pageTarget.webSocketDebuggerUrl);
}

export class CdpSession {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id !== undefined && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`CDP ${msg.error.message} (code ${msg.error.code})`));
        else resolve(msg.result);
      } else if (msg.method) {
        const cbs = this.listeners.get(msg.method) ?? [];
        for (const cb of cbs) cb(msg.params);
      }
    };
  }

  static connect(wsUrl) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => resolve(new CdpSession(ws));
      ws.onerror = (err) => reject(err);
    });
  }

  send(method, params = {}, { timeoutMs = 30000 } = {}) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CDP ${method} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  on(method, cb) {
    if (!this.listeners.has(method)) this.listeners.set(method, []);
    this.listeners.get(method).push(cb);
  }

  close() {
    this.ws.close();
  }
}

/**
 * Evaluate a JS function in the page and return its JSON value.
 * fnSource is invoked with the given args; it must return serializable data.
 */
export async function evaluate(session, fnSource, { awaitPromise = false, args = [] } = {}) {
  const res = await session.send("Runtime.evaluate", {
    expression: `(${fnSource})(${args.map((a) => JSON.stringify(a)).join(",")})`,
    returnByValue: true,
    awaitPromise,
  });
  if (res.exceptionDetails) {
    throw new Error(`page eval threw: ${JSON.stringify(res.exceptionDetails).slice(0, 500)}`);
  }
  return res.result?.value;
}

/** Poll a JS function until it returns truthy (serializable). */
export async function waitForFunction(session, fnSource, { timeoutMs = 30000, pollingMs = 250 } = {}) {
  const deadline = Date.now() + timeoutMs;
  let last;
  for (;;) {
    last = await evaluate(session, fnSource).catch((e) => `__ERR__${e.message}`);
    if (last && !String(last).startsWith("__ERR__")) return last;
    if (Date.now() > deadline) {
      throw new Error(`waitForFunction timed out: ${fnSource.slice(0, 200)} (last: ${String(last).slice(0, 200)})`);
    }
    await new Promise((r) => setTimeout(r, pollingMs));
  }
}

/** Real trusted mouse click at CSS-viewport coordinates. */
export async function mouseClick(session, x, y) {
  for (const type of ["mousePressed", "mouseReleased"]) {
    await session.send("Input.dispatchMouseEvent", {
      type,
      x,
      y,
      button: "left",
      clickCount: 1,
    });
  }
}

/**
 * Real trusted key press (e.g. "Enter", "Tab"). Special keys need CDP key
 * metadata: keyPress(s, "ArrowDown", { code: "ArrowDown", windowsVirtualKeyCode: 40 }).
 */
export async function keyPress(session, key, extra = {}) {
  // Non-printable keys (arrows/Tab/Enter) travel as rawKeyDown.
  const down = extra.code ? "rawKeyDown" : "keyDown";
  await session.send("Input.dispatchKeyEvent", { type: down, key, ...extra });
  await session.send("Input.dispatchKeyEvent", { type: "keyUp", key, ...extra });
}

/** Type text with one trusted key event pair per character. */
export async function typeText(session, text) {
  for (const ch of text) {
    await session.send("Input.dispatchKeyEvent", { type: "keyDown", text: ch });
    await session.send("Input.dispatchKeyEvent", { type: "keyUp" });
  }
}

export async function captureScreenshot(session) {
  const res = await session.send("Page.captureScreenshot", { format: "png" });
  return Buffer.from(res.data, "base64");
}
