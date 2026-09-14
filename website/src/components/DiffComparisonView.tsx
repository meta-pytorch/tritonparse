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
  hideUnchangedRegions?: { enabled: boolean; revealLineCount: number };
  diffAlgorithm: "legacy" | "advanced";
  scrollbar: { vertical: "auto" | "hidden" | "visible"; horizontal: "auto" | "hidden" | "visible"; horizontalScrollbarSize: number };
  minimap: { enabled: boolean };
  scrollBeyondLastLine: boolean;
  automaticLayout: boolean;
}

// Monaco diff editor interface (minimal types for our usage)
interface MonacoDiffEditor {
  getOriginalEditor?: () => MonacoSubEditor | undefined;
  getModifiedEditor?: () => MonacoSubEditor | undefined;
  setModel?: (model: unknown) => void;
  dispose?: () => void;
  getDomNode?: () => HTMLElement | undefined;
  onDidUpdateDiff?: (callback: () => void) => void;
}

interface MonacoSubEditor {
  updateOptions?: (options: Record<string, unknown>) => void;
  setModel?: (model: unknown) => void;
  layout?: () => void;
  onDidLayoutChange?: (callback: () => void) => void;
  onDidChangeModel?: (callback: () => void) => void;
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
    const hideUnchanged = {
      enabled: options?.onlyChanged ?? false,
      revealLineCount: Math.max(0, options?.context ?? 3),
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
      automaticLayout: true,
    };
    return opts;
  // Depend on individual fields: callers pass a fresh object literal each
  // render, and [options] would rebuild (and re-apply) options every time.
  }, [options?.onlyChanged, options?.context, options?.wordWrap]);

  const editorRef = useRef<MonacoDiffEditor | null>(null);

  // Keep both panes in sync when options change
  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) return;
    try {
      const wrap = options?.wordWrap ?? "on";
      const wrapping = wrap === "on";
      const original = editor.getOriginalEditor?.();
      const modified = editor.getModifiedEditor?.();
      const shared = { wordWrap: wrap, wordWrapMinified: true, wrappingStrategy: 'advanced', scrollbar: { horizontal: wrapping ? 'hidden' : 'auto', horizontalScrollbarSize: wrapping ? 0 : 10 } };
      original?.updateOptions?.(shared);
      modified?.updateOptions?.(shared);
    } catch { /* Monaco may throw if editor is disposed */ }
  }, [options?.wordWrap]);

  // Ensure diff editor is fully disposed on unmount to avoid Monaco race conditions
  useEffect(() => {
    return () => {
      try {
        const editor = editorRef.current;
        if (editor) {
          try { editor.setModel?.(null); } catch { /* ignore */ }
          try { editor.getOriginalEditor?.()?.setModel?.(null); } catch { /* ignore */ }
          try { editor.getModifiedEditor?.()?.setModel?.(null); } catch { /* ignore */ }
          try { editor.dispose?.(); } catch { /* ignore */ }
        }
      } catch { /* ignore cleanup errors */ }
      editorRef.current = null;
      try {
        // Clean up any global state
        const win = window as { __DIFF?: unknown };
        win.__DIFF = undefined;
      } catch { /* ignore */ }
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
    <div className="w-full border border-gray-200 rounded bg-white">
      <div
        className="w-full resize-y overflow-auto"
        style={{ height: `${containerHeight}px`, minHeight: 240 }}
        // Browser native resize-y changes element height; Monaco autoLayout observes size
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
        // Ensure both panes use the same wrapping and scrollbar behavior
        onMount={(editor) => {
          try {
            // Cast to our interface for type-safe access
            const diffEditor = editor as unknown as MonacoDiffEditor;
            editorRef.current = diffEditor;

            const applyWrap = () => {
              try {
                const wrap = options?.wordWrap ?? "on";
                const wrapping = wrap === "on";
                const original = diffEditor.getOriginalEditor?.();
                const modified = diffEditor.getModifiedEditor?.();
                const shared = { wordWrap: wrap, wordWrapMinified: true, wrappingStrategy: 'advanced', wrappingIndent: 'same', scrollbar: { horizontal: wrapping ? 'hidden' : 'auto', horizontalScrollbarSize: wrapping ? 0 : 10 } };
                original?.updateOptions?.(shared);
                modified?.updateOptions?.(shared);
                // Force layout after changing wrap
                try { original?.layout?.(); } catch { /* ignore */ }
                try { modified?.layout?.(); } catch { /* ignore */ }
              } catch {
                // swallow errors
              }
            };

            // Apply at several timing points to avoid initialization overwrites
            applyWrap();
            requestAnimationFrame(() => applyWrap());
            setTimeout(() => applyWrap(), 0);
            setTimeout(() => applyWrap(), 100);
            setTimeout(() => applyWrap(), 300);

            // Re-apply on diff/layout/model changes
            try { diffEditor.onDidUpdateDiff?.(() => applyWrap()); } catch { /* ignore */ }
            try { diffEditor.getOriginalEditor?.()?.onDidLayoutChange?.(() => applyWrap()); } catch { /* ignore */ }
            try { diffEditor.getModifiedEditor?.()?.onDidLayoutChange?.(() => applyWrap()); } catch { /* ignore */ }
            try { diffEditor.getOriginalEditor?.()?.onDidChangeModel?.(() => applyWrap()); } catch { /* ignore */ }
            try { diffEditor.getModifiedEditor?.()?.onDidChangeModel?.(() => applyWrap()); } catch { /* ignore */ }
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
