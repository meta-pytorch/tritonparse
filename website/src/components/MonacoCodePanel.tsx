/**
 * Read-only single-file Monaco panel for the IR views (design §4.1, §4.10).
 *
 * Phase 0 spike subset: mount/lifecycle, decorations from a parent-guarded
 * highlight set, click-to-map input chain (§4.3), offset conversions (§4.4),
 * manual layout (§3.3) and the read-only debug hook (§6.3.3). functionRange
 * and initialLine arrive with the comparison migration (Phase 2).
 *
 * Ownership contract (§3.3/R1): the @monaco-editor/react wrapper owns the
 * editor and the model. This file never calls editor.dispose(),
 * model.dispose() or setModel(); cleanup only clears the decorations
 * collection, disposes event listeners and releases own refs.
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { Editor } from "@monaco-editor/react";
import type { editor as monacoEditorNamespace } from "monaco-editor";
import type * as monaco from "monaco-editor";
import { registerIrLanguages } from "./monaco/registerIrLanguages";
import {
  mergeContiguousLines,
  toAbsolute,
  toPhysical,
} from "./monaco/highlightMath";
import "./MonacoCodePanel.css";

type IStandaloneCodeEditor = monacoEditorNamespace.IStandaloneCodeEditor;
type IEditorDecorationsCollection = monacoEditorNamespace.IEditorDecorationsCollection;

export interface MonacoCodePanelProps {
  /** Stable panel identity; also scopes the model path and debug hook key. */
  viewerId: string;
  /** Full document text (stable reference => wrapper keeps the model). */
  content: string;
  /** Registered Monaco language id (see mapFileToMonacoLanguage). */
  monacoLanguage: string;
  /** Absolute number shown for physical line 1. Default 1. */
  lineOffset?: number;
  /**
   * Parent-guarded highlight set (absolute numbers, normalized): entries
   * whose doc mismatches already arrived as []. Applied verbatim.
   */
  effectiveLines: number[];
  /** Original entry token, assertion use only (parent already guarded). */
  entryDoc: object;
  /** Current document token (reveal-once judgement + assertion). */
  docToken: object;
  /**
   * Absolute [start, end] function range (inclusive), already intersected
   * with the document by the parent (§4.4.1). Undefined/empty => no range
   * decorations. Only the comparison python panel passes this (§4.5).
   */
  functionRange?: { start: number; end: number };
  /**
   * Absolute first-positioning target, consumed once per docToken (§4.10):
   * skipped when a same-token highlight is present (highlight reveal wins)
   * or the user already scrolled. Only the comparison python panel passes
   * this (full-file => function_start_line, snippet => start_line).
   */
  initialLine?: number;
  /** Editor font size. Default 14. */
  fontSize?: number;
  /** Fired with the absolute line number on text click (§4.3). */
  onLineClick?: (absoluteLine: number) => void;
  /** Fired once the editor instance is ready (rulers use it to scroll). */
  onMount?: (viewerId: string, editor: IStandaloneCodeEditor) => void;
}

interface DebugPanels {
  __TRITONPARSE_DEBUG?: {
    panels?: Record<string, { editor: IStandaloneCodeEditor; getHighlights: () => number[] }>;
  };
}

const MonacoCodePanel: React.FC<MonacoCodePanelProps> = ({
  viewerId,
  content,
  monacoLanguage,
  lineOffset = 1,
  effectiveLines,
  entryDoc,
  docToken,
  fontSize = 14,
  onLineClick,
  onMount,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  // Ready chain (R1): onMount must setEditor (state, not just a ref) so the
  // decorations/listeners/layout effects below re-run with the instance.
  const [editor, setEditor] = useState<IStandaloneCodeEditor | null>(null);
  const monacoRef = useRef<typeof monaco | null>(null);
  const collectionRef = useRef<IEditorDecorationsCollection | null>(null);
  const prevRevealRef = useRef<{ doc: object; lines: number[] } | null>(null);
  const linesRef = useRef<number[]>(effectiveLines);
  // Synced in an effect (ref writes during render are banned); the debug
  // hook reads through this mirror.
  useEffect(() => {
    linesRef.current = effectiveLines;
  }, [effectiveLines]);

  const options = useMemo<monacoEditorNamespace.IStandaloneEditorConstructionOptions>(
    () => ({
      readOnly: true,
      minimap: { enabled: false },
      folding: false,
      occurrencesHighlight: "off",
      selectionHighlight: false,
      renderLineHighlight: "none",
      scrollBeyondLastLine: false,
      // smoothScrolling:true is required for ScrollType.Smooth reveals (§4.1).
      smoothScrolling: true,
      wordWrap: "off",
      scrollbar: { vertical: "auto", horizontal: "auto" },
      lineNumbers: (n: number) => String(toAbsolute(n, lineOffset)),
      fontSize,
      fontFamily: "SFMono-Regular, Menlo, Consolas, monospace",
      automaticLayout: false,
      fixedOverflowWidgets: true,
    }),
    [fontSize, lineOffset]
  );

  // Manual layout (§3.3 template): non-zero resizes only; synchronous in the
  // observer (no rAF — it does not fire reliably in background/headless pages).
  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect || rect.width <= 0 || rect.height <= 0) return;
      try {
        editor?.layout?.({ width: rect.width, height: rect.height });
      } catch {
        /* editor may be disposed */
      }
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, [editor]);

  // Decorations collection lifecycle: created per editor instance, cleared on
  // release. The collection is editor-bound, not model-bound: content changes
  // only re-set() it (decorations effect below), never recreate it.
  useEffect(() => {
    if (!editor) return;
    const collection = editor.createDecorationsCollection([]);
    collectionRef.current = collection;
    return () => {
      try {
        collection.clear();
      } catch {
        /* editor may be disposed */
      }
      if (collectionRef.current === collection) {
        collectionRef.current = null;
      }
    };
  }, [editor]);

  // Click input chain (§4.3): same-line press+release with an empty selection.
  // Offset/handler changes re-register so no stale-offset closure survives.
  useEffect(() => {
    if (!editor || !monacoRef.current) return;
    const monacoApi = monacoRef.current;
    let downPos: monaco.IPosition | null = null;
    const down = editor.onMouseDown((e) => {
      downPos =
        e.target.type === monacoApi.editor.MouseTargetType.CONTENT_TEXT && e.target.position
          ? e.target.position
          : null;
    });
    const up = editor.onMouseUp((e) => {
      const pos = e.target.position;
      if (
        downPos &&
        pos &&
        pos.lineNumber === downPos.lineNumber &&
        editor.getSelection()?.isEmpty()
      ) {
        onLineClick?.(toAbsolute(pos.lineNumber, lineOffset));
      }
      downPos = null;
    });
    return () => {
      down.dispose();
      up.dispose();
    };
  }, [editor, lineOffset, onLineClick]);

  // Highlight application (§4.2): full-replacement set() + reveal-once.
  // A mismatched doc arrives as [] from the parent; set([]) clears.
  useEffect(() => {
    if (!editor || !monacoRef.current) return;
    const monacoApi = monacoRef.current;
    console.assert(
      entryDoc === docToken || effectiveLines.length === 0,
      "[monaco] parent must guard highlight doc"
    );
    const spans = mergeContiguousLines(effectiveLines);
    try {
      collectionRef.current?.set(
        spans.map(([s, e]) => ({
          range: new monacoApi.Range(toPhysical(s, lineOffset), 1, toPhysical(e, lineOffset), 1),
          options: {
            isWholeLine: true,
            className: "mp-highlighted-line",
            zIndex: 10,
            overviewRuler: {
              color: "rgba(245,158,11,.9)",
              position: monacoApi.editor.OverviewRulerLane.Right,
            },
          },
        }))
      );
    } catch {
      /* editor may be disposed */
    }
    const prev = prevRevealRef.current;
    const changed =
      !prev ||
      prev.doc !== docToken ||
      prev.lines.length !== effectiveLines.length ||
      prev.lines.some((line, i) => line !== effectiveLines[i]);
    if (effectiveLines.length > 0 && changed) {
      try {
        editor.revealLineInCenterIfOutsideViewport(
          toPhysical(effectiveLines[0], lineOffset),
          monacoApi.editor.ScrollType.Smooth
        );
      } catch {
        /* editor may be disposed */
      }
    }
    prevRevealRef.current = { doc: docToken, lines: effectiveLines };
  }, [effectiveLines, entryDoc, docToken, content, editor, lineOffset]);

  // Read-only debug hook (§6.3.3): tests read editor/decorations through it,
  // never drive input. The key is deleted on unmount (F10 assertion).
  useEffect(() => {
    if (!editor) return;
    if (new URLSearchParams(window.location.search).get("debug") !== "1") return;
    const w = window as unknown as DebugPanels;
    if (!w.__TRITONPARSE_DEBUG) {
      w.__TRITONPARSE_DEBUG = {};
    }
    if (!w.__TRITONPARSE_DEBUG.panels) {
      w.__TRITONPARSE_DEBUG.panels = {};
    }
    w.__TRITONPARSE_DEBUG.panels[viewerId] = {
      editor,
      getHighlights: () => linesRef.current,
    };
    return () => {
      const panels = (window as unknown as DebugPanels).__TRITONPARSE_DEBUG?.panels;
      if (panels) {
        delete panels[viewerId];
      }
    };
  }, [editor, viewerId]);

  return (
    <div
      ref={containerRef}
      className="mp-panel-container"
      data-testid={`${viewerId}-monaco-panel`}
    >
      <Editor
        path={`tritonparse/${viewerId}`}
        value={content}
        language={monacoLanguage}
        options={options}
        theme="light"
        height="100%"
        loading={<div className="mp-panel-loading">Loading editor…</div>}
        beforeMount={(m) => registerIrLanguages(m)}
        onMount={(ed, m) => {
          monacoRef.current = m;
          setEditor(ed);
          // First layout with the measured container size (R5): do not wait
          // for the RO, which may already have fired before the editor existed.
          const node = containerRef.current;
          if (node) {
            const rect = node.getBoundingClientRect();
            if (rect.width > 0 && rect.height > 0) {
              try {
                ed.layout({ width: rect.width, height: rect.height });
              } catch {
                /* ignore */
              }
            }
          }
          onMount?.(viewerId, ed);
        }}
      />
    </div>
  );
};

export default MonacoCodePanel;
