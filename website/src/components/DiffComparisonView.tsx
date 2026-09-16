import React, { useEffect, useMemo, useRef, useState } from "react";
import { DiffEditor } from "@monaco-editor/react";

interface DiffOptions {
  context?: number; // lines of context when hiding unchanged regions
  wordWrap?: "off" | "on";
  onlyChanged?: boolean;
}

interface DiffComparisonViewProps {
  leftContent: string;
  rightContent: string;
  language?: string;
  height?: string;
  options?: DiffOptions;
}

// Monaco editor types vary by version, so we need to use a loose type for the editor options
interface MonacoEditorOptions {
  readOnly: boolean;
  renderSideBySide: boolean;
  renderOverviewRuler: boolean;
  renderIndicators: boolean;
  diffWordWrap: "off" | "on";
  wordWrap: "off" | "on" | "wordWrapColumn" | "bounded";
  wordWrapOverride1: "off" | "on" | "inherit";
  wordWrapOverride2: "off" | "on" | "inherit";
  wordWrapMinified: boolean;
  wrappingStrategy: "simple" | "advanced";
  originalEditable: boolean;
  ignoreTrimWhitespace: boolean;
  useInlineViewWhenSpaceIsLimited: boolean;
  hideUnchangedRegions?: { enabled: boolean; contextLineCount: number; revealLineCount: number };
  diffAlgorithm: "legacy" | "advanced";
  scrollbar: { vertical: "auto" | "hidden" | "visible"; horizontal: "auto" | "hidden" | "visible"; horizontalScrollbarSize: number };
  minimap: { enabled: boolean };
  scrollBeyondLastLine: boolean;
  automaticLayout: boolean;
}

// Monaco diff editor interface (minimal types for our usage).
interface MonacoDiffModel {
  original?: { dispose?: () => void } | null;
  modified?: { dispose?: () => void } | null;
}

interface MonacoDiffEditor {
  getOriginalEditor?: () => MonacoSubEditor | undefined;
  getModifiedEditor?: () => MonacoSubEditor | undefined;
  getDomNode?: () => HTMLElement | undefined;
  layout?: (dimension?: { width: number; height: number }) => void;
  // Typed ONLY for the unmount detach step (see below). editor.dispose()
  // stays untyped and uncalled: the shell belongs to the wrapper.
  getModel?: () => MonacoDiffModel | null | undefined;
  setModel?: (model: null) => void;
}

interface MonacoSubEditor {
  updateOptions?: (options: Record<string, unknown>) => void;
}

const DiffComparisonView: React.FC<DiffComparisonViewProps> = ({
  leftContent,
  rightContent,
  language = "plaintext",
  height = "calc(100vh - 12rem)",
  options,
}) => {
  const monacoOptions = useMemo(() => {
    // Always pass a full object: updateOptions({hideUnchangedRegions: undefined})
    // does not reliably reset a previously enabled value.
    // The "Unchanged lines shown around each change" control maps to
    // contextLineCount (surrounding rows); revealLineCount is only the manual
    // per-click expansion step, kept equal so expansions match the context —
    // but floored at 1, since context=0 would otherwise make every expansion
    // reveal 0 rows and the expand control a no-op.
    const context = Math.max(0, options?.context ?? 3);
    const hideUnchanged = {
      enabled: options?.onlyChanged ?? false,
      contextLineCount: context,
      revealLineCount: Math.max(1, context),
    };
    const wrap = options?.wordWrap ?? "on";
    const wrapping = wrap === "on";

    const opts: MonacoEditorOptions = {
      readOnly: true,
      renderSideBySide: true,
      renderOverviewRuler: true,
      renderIndicators: true,
      // Diff-editor level wrap plus both per-side overrides must all follow
      // the user option, otherwise Wrap=off has no visible effect.
      diffWordWrap: wrap,
      wordWrap: wrap,
      // Force both sides to honor wrap regardless of per-side defaults
      wordWrapOverride1: wrap,
      wordWrapOverride2: wrap,
      wordWrapMinified: true,
      wrappingStrategy: "advanced",
      // Ensure even original (left) honors wrapping consistently
      originalEditable: false,
      // Always ignore leading/trailing whitespace (the Monaco default):
      // it is never a meaningful IR difference, so no toggle is offered.
      ignoreTrimWhitespace: true,
      // Never fall back to the inline view below 900px: the inline branch
      // writes wordWrapOverride1/2='off' onto the ORIGINAL pane only, and the
      // side-by-side restore writes back override1 but never override2 — the
      // effective wrap (override2 wins when set) sticks at 'off' forever.
      // This view is always side-by-side; disabling the fallback removes the
      // transient at its root (the single writer below also owns overrides).
      useInlineViewWhenSpaceIsLimited: false,
      // Monaco types may vary by version; these options are valid at runtime
      hideUnchangedRegions: hideUnchanged,
      // Prefer advanced algorithm if available
      diffAlgorithm: "advanced",
      // Only hide the horizontal scrollbar when wrapping; with Wrap=off the
      // user needs it to reach content past the viewport edge.
      scrollbar: {
        vertical: 'auto',
        horizontal: wrapping ? 'hidden' : 'auto',
        horizontalScrollbarSize: wrapping ? 0 : 10,
      },
      // keep view lean
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      // No automaticLayout: its ResizeObserver layouts on every size change
      // including the transition to/from 0x0 when the tab is hidden, which
      // burned ~5s per tab switch on large models (measured via LoAF). The
      // effect below lays out only when the container has a real size.
      automaticLayout: false,
    };
    return opts;
  // Depend on individual fields: callers pass a fresh object literal each
  // render, and [options] would rebuild (and re-apply) options every time.
  }, [options?.onlyChanged, options?.context, options?.wordWrap]);

  const editorRef = useRef<MonacoDiffEditor | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Manual layout on container resize (replaces automaticLayout): the tab
  // keep-alive hides this view with display:none, and laying out a 0-size
  // editor is both useless and expensive, so zero-size notifications are
  // skipped. Dimensions are passed explicitly from the observed container:
  // a bare layout() measures Monaco's own root element, which can never
  // re-grow once it has collapsed (Monaco clamps to a 5px minimum), so
  // self-measurement leaves a blank 5px-tall editor after un-hiding.
  // Layout runs synchronously in the observer (which the browser already
  // throttles to frame boundaries) — never behind rAF, which does not fire
  // reliably in background/headless pages.
  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      try { editorRef.current?.layout?.({ width: rect.width, height: rect.height }); } catch { /* editor may be disposed */ }
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Single wrap-application path (F17 wrap fix): the CURRENT option is applied
  // to both panes whenever it or the contents change. The previous design kept
  // mount-time timers and model/diff listeners that re-applied a stale closure
  // capture of `options`, racing fresh updates and leaving the panes split
  // (original off / modified on). There is exactly one writer now, always fresh.
  // It also owns the pane-level wordWrapOverride1/2: the effective wrap reads
  // override2 first, so writing wordWrap alone cannot heal a stuck override
  // left behind by the diff widget's inline-fallback transient (see above).
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    try {
      const wrap = options?.wordWrap ?? "on";
      const wrapping = wrap === "on";
      const original = editor.getOriginalEditor?.();
      const modified = editor.getModifiedEditor?.();
      const shared = { wordWrap: wrap, wordWrapOverride1: wrap, wordWrapOverride2: wrap, wordWrapMinified: true, wrappingStrategy: 'advanced', scrollbar: { horizontal: wrapping ? 'hidden' : 'auto', horizontalScrollbarSize: wrapping ? 0 : 10 } };
      original?.updateOptions?.(shared);
      modified?.updateOptions?.(shared);
    } catch { /* Monaco may throw if editor is disposed */ }
  }, [options?.wordWrap, leftContent, rightContent]);

  // Unmount cleanup: detach-then-dispose the MODELS here, dispose NOTHING else.
  // File Diff-specific ownership adjustment to the shared rule (§4.1): this
  // component owns model disposal (detach, then modified, then original);
  // the @monaco-editor/react wrapper owns the diff-editor shell and still
  // runs its own teardown afterwards (its model-disposal lines no-op on the
  // detached null via optional chaining). editor.dispose() is deliberately
  // never called here.
  // Actual mechanism (reviewer-reproduced, artifacts/wrapper-compatibility):
  // diffEditorWidget registers onWillDispose on each attached model; disposing
  // an attached model reports "TextModel got disposed before DiffEditorWidget
  // model got reset" and then resets to null. The report is thrown
  // asynchronously (errors.js rethrows via setTimeout), so it does NOT abort
  // the wrapper's remaining cleanup — registries return to zero even with the
  // default wrapper. Detaching first avoids the uncaught error entirely, which
  // is what a normal unmount requires; the e2e asserts both halves: model
  // count stable AND zero console errors.
  // Pre-onMount unmount needs no branch: the wrapper creates the widget,
  // models, setModel and onMount synchronously in one post-init effect, so
  // editorRef is set whenever models exist and there is nothing to detach
  // when it is still null.
  useEffect(() => {
    return () => {
      try {
        const editor = editorRef.current;
        const model = editor?.getModel?.();
        try {
          editor?.setModel?.(null);
        } catch {
          /* ignore */
        }
        try {
          model?.modified?.dispose?.();
        } catch {
          /* ignore */
        }
        try {
          model?.original?.dispose?.();
        } catch {
          /* ignore */
        }
      } catch {
        /* ignore */
      }
      try {
        // Guarded delete: in All-IRs mode several diff views share the key.
        const w = window as unknown as {
          __TRITONPARSE_DEBUG?: { panels?: Record<string, { diffEditor?: unknown }> };
        };
        const panels = w.__TRITONPARSE_DEBUG?.panels;
        if (panels?.filediff?.diffEditor === editorRef.current) {
          delete panels.filediff;
        }
      } catch {
        /* ignore */
      }
      editorRef.current = null;
    };
  }, []);

  // Vertical resizable container: keep width 100%, allow drag to change height
  const initialPxHeight = useMemo(() => {
    // If a pixel value is provided, use it directly
    if (typeof height === 'string') {
      const pxMatch = height.match(/(\d+)px$/);
      if (pxMatch) {
        try { return parseInt(pxMatch[1], 10); } catch { /* fallthrough */ }
      }

      // Support calc(100vh - Xrem)
      const calcRemMatch = height.match(/calc\(100vh\s*-\s*(\d+(?:\.\d+)?)rem\)/i);
      if (calcRemMatch && typeof window !== 'undefined') {
        const rem = parseFloat(calcRemMatch[1]);
        const remPx = rem * 16; // assume 1rem = 16px baseline
        return Math.max(240, Math.round(window.innerHeight - remPx));
      }

      // Support plain vh values (e.g., 80vh)
      const vhMatch = height.match(/(\d+(?:\.\d+)?)vh/i);
      if (vhMatch && typeof window !== 'undefined') {
        const vh = parseFloat(vhMatch[1]);
        return Math.max(240, Math.round(window.innerHeight * (vh / 100)));
      }
    }

    // Fallback: viewport height minus 16rem (~256px) if available; otherwise 600px
    if (typeof window !== 'undefined') {
      return Math.max(240, window.innerHeight - 256);
    }
    return 600;
  }, [height]);

  const [containerHeight, setContainerHeight] = useState<number>(initialPxHeight);

  return (
    <div className="w-full border border-gray-200 rounded bg-white" data-testid="file-diff-view">
      <div
        ref={containerRef}
        className="w-full resize-y overflow-auto"
        style={{ height: `${containerHeight}px`, minHeight: 240 }}
        // Browser native resize-y changes element height; the ResizeObserver
        // effect above relays non-zero sizes to the editor (no autoLayout).
        onMouseUp={() => {
          // Capture final height after drag (optional state sync)
          try {
            const node = editorRef.current?.getDomNode?.();
            if (node?.parentElement) {
              const h = node.parentElement.clientHeight;
              if (h > 0) setContainerHeight(h);
            }
          } catch { /* ignore resize errors */ }
        }}
      >
      <DiffEditor
        height="100%"
        language={language === "python" ? "python" : "plaintext"}
        original={leftContent ?? ""}
        modified={rightContent ?? ""}
        options={monacoOptions}
        theme="light"
        onMount={(editor) => {
          try {
            // Cast to our interface for type-safe access
            const diffEditor = editor as unknown as MonacoDiffEditor;
            editorRef.current = diffEditor;
            // Read-only debug hook (?debug=1): tests read wrapping/readonly
            // state through it, never drive input (§6.3).
            if (new URLSearchParams(window.location.search).get("debug") === "1") {
              const w = window as unknown as {
                __TRITONPARSE_DEBUG?: { panels?: Record<string, unknown> };
              };
              if (!w.__TRITONPARSE_DEBUG) {
                w.__TRITONPARSE_DEBUG = {};
              }
              if (!w.__TRITONPARSE_DEBUG.panels) {
                w.__TRITONPARSE_DEBUG.panels = {};
              }
              w.__TRITONPARSE_DEBUG.panels.filediff = { diffEditor };
            }
            // First layout with the measured container size (F17 collapse fix):
            // the RO above may already have fired while no editor existed, and
            // a bare layout() self-measures the collapsed 5px root. Never wait
            // for a user resize to show content.
            const node = containerRef.current;
            if (node) {
              const rect = node.getBoundingClientRect();
              if (rect.width > 0 && rect.height > 0) {
                try {
                  diffEditor.layout?.({ width: rect.width, height: rect.height });
                } catch {
                  /* ignore */
                }
              }
            }
          } catch {
            // swallow errors
          }
        }}
        loading={<div className="p-4 text-gray-600">Loading diff viewer...</div>}
      />
      </div>
    </div>
  );
};

export default DiffComparisonView;
