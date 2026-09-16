/**
 * Deterministic Monaco loading (design §4.13).
 *
 * ESM instance injection + inline editor worker. No CDN/AMD requests, no
 * runtime `vs/` resource tree — the single-file build stays self-contained.
 *
 * This module MUST be imported before any @monaco-editor/react consumer
 * (first import in main.tsx): loader.config() has to precede loader.init(),
 * and MonacoEnvironment.getWorker has to precede the first editor/worker use.
 */
import { loader } from "@monaco-editor/react";
// NOTE: specifiers use the package exports map ("./*": "./esm/vs/*.js").
// `monaco-editor/esm/vs/...` double-prefix forms are wrong; these resolve to
// the same files the loader's own Monaco type is declared from.
import * as monaco from "monaco-editor/editor/editor.api";
// editor.api does not self-register languages or these contributions; each is
// required explicitly (§4.13 registration list). JSON/C are intentionally NOT
// imported and stay plaintext (no diagnostics, no dedicated worker).
import "monaco-editor/languages/definitions/python/register";
import "monaco-editor/editor/contrib/find/browser/findController";
import "monaco-editor/editor/standalone/browser/quickAccess/standaloneGotoLineQuickAccess";
// Codicon font + base styles: without this registration the icon
// private-use characters render as tofu (find widget arrows/close/toggles,
// diff hidden-line expanders). This pulls @font-face + codicon.ttf into the
// bundle; the standalone inliner embeds the font as a data URI.
import "monaco-editor/features/codicon/register";
import EditorWorker from "monaco-editor/editor/editor.worker?worker&inline";

declare global {
  interface Window {
    MonacoEnvironment?: {
      getWorker?: (moduleId: string, label: string) => Worker;
    };
    __TRITONPARSE_DEBUG?: {
      panels?: Record<string, unknown>;
      /** Read-only model inventory for lifecycle tests (F17/R1). */
      getModels?: () => string[];
      /** ESM monaco namespace for test-side reads (option enums, URIs). */
      monaco?: typeof monaco;
    };
  }
}

// Single worker route: the inlined editor worker runs from a blob URL, so no
// separate .js request exists in either build. The getWorker constructor route
// never passes through the string-typed getWorkerUrl path.
self.MonacoEnvironment = {
  getWorker: () => new EditorWorker(),
};

// Instance injection: the loader resolves init() immediately and skips every
// AMD script request (@monaco-editor/loader `if (state.monaco)` branch).
loader.config({ monaco });

if (new URLSearchParams(window.location.search).get("debug") === "1") {
  if (!window.__TRITONPARSE_DEBUG) {
    window.__TRITONPARSE_DEBUG = {};
  }
  window.__TRITONPARSE_DEBUG.getModels = () =>
    monaco.editor.getModels().map((model) => model.uri.toString());
  // Read-only namespace access for tests (e.g. EditorOption enums). Tests must
  // only read through it, never drive editors (§6.3).
  window.__TRITONPARSE_DEBUG.monaco = monaco;
}
