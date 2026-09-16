/**
 * Single IR view on the Monaco panel (design §4.9, Phase 0 spike).
 *
 * Owns the Single highlight truth source as { doc, lines } (§4.2.1): the doc
 * token covers file/content/mapping/stages identity, the parent guards every
 * render (mismatch => []), and a doc change clears the truth source so stale
 * highlights can neither render nor reveal on the new document (F18).
 * Click mapping reuses the anchor-grouping semantics; output always flows
 * through normalizeHighlightLines before setHighlights (§4.4.1).
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import MonacoCodePanel from "./MonacoCodePanel";
import OverviewRuler from "./OverviewRuler";
import type { editor as monacoEditorNamespace } from "monaco-editor";
import type { IRFile, IRStageDescriptor } from "../utils/dataLoader";
import { getGroupingAnchor } from "../utils/dataLoader";
import { mapFileToMonacoLanguage } from "../utils/monacoLanguage";
import {
  getAnchorGroupedLines,
  normalizeHighlightLines,
  toPhysical,
} from "./monaco/highlightMath";

/** Opaque document token: reference equality means same document. */
export type DocToken = object;

export interface PanelHighlight {
  doc: DocToken;
  lines: number[];
}

interface PanelDiagnostics {
  doc: DocToken;
  droppedInvalid: number;
  droppedOutOfRange: number;
}

interface SingleMonacoViewerProps {
  irFile?: IRFile;
  irContent?: string;
  /** Filename: doubles as doc identity and language source (no App change). */
  title: string;
  irStages?: IRStageDescriptor[];
}

/** Initial entry token: distinct from every real doc so the guard yields []. */
const EMPTY_DOC: DocToken = {};

const SingleMonacoViewer: React.FC<SingleMonacoViewerProps> = ({
  irFile,
  irContent,
  title,
  irStages,
}) => {
  const codeContent = irContent || (irFile ? irFile.content : "");
  const sourceMapping = irFile?.source_mapping;
  const monacoLanguage = mapFileToMonacoLanguage(title, irStages);

  // Single renders one kernel of one loaded trace: token deps are file /
  // mapping / stages / content references. The comparison view additionally
  // folds a source-qualified kernelKey in (Phase 2, §4.2.1).
  // Deps are identity inputs, intentionally unread: any reference change
  // rebuilds the token. The token stays opaque (no fields, no retention).
  const currentDoc: DocToken = useMemo(
    () => ({}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [title, sourceMapping, irStages, codeContent]
  );

  const [highlight, setHighlight] = useState<PanelHighlight>(() => ({
    doc: EMPTY_DOC,
    lines: [],
  }));
  const [diagnostics, setDiagnostics] = useState<PanelDiagnostics>(() => ({
    doc: EMPTY_DOC,
    droppedInvalid: 0,
    droppedOutOfRange: 0,
  }));

  // Doc-change clearing: any token change resets the truth source (switching
  // back does not restore, matching legacy ref-loss semantics). Synchronous
  // (no setTimeout): a deferred clear could wipe a click that lands for the
  // new doc before the timer fires. The click guard skips the clear when the
  // new doc already has a click in the same commit batch.
  const lastClickDocRef = useRef<DocToken | null>(null);
  useEffect(() => {
    if (lastClickDocRef.current === currentDoc) return;
    setHighlight({ doc: currentDoc, lines: [] });
    setDiagnostics({ doc: currentDoc, droppedInvalid: 0, droppedOutOfRange: 0 });
  }, [currentDoc]);

  // One memoized split per content: the click handler and the ruler prop
  // below share it instead of each splitting (up to 100k lines) per event
  // and per render.
  const lineCount = useMemo(
    () => (codeContent === "" ? 1 : codeContent.split("\n").length),
    [codeContent]
  );

  const handleLineClick = useCallback(
    (absoluteLine: number) => {
      const anchorProperty = `${getGroupingAnchor(irStages)}_line`;
      const grouped = getAnchorGroupedLines(sourceMapping, anchorProperty, absoluteLine);
      const normalized = normalizeHighlightLines(grouped, { offset: 1, lineCount });
      if (normalized.droppedInvalid > 0 || normalized.droppedOutOfRange > 0) {
        console.warn(
          `[monaco] dropped ${normalized.droppedInvalid} invalid / ${normalized.droppedOutOfRange} out-of-range mapped lines in ${title}`
        );
      }
      lastClickDocRef.current = currentDoc;
      setHighlight({ doc: currentDoc, lines: normalized.lines });
      setDiagnostics({
        doc: currentDoc,
        droppedInvalid: normalized.droppedInvalid,
        droppedOutOfRange: normalized.droppedOutOfRange,
      });
    },
    [sourceMapping, irStages, lineCount, title, currentDoc]
  );

  const effectiveLines = highlight.doc === currentDoc ? highlight.lines : [];
  const showDiagnostics =
    diagnostics.doc === currentDoc &&
    (diagnostics.droppedInvalid > 0 || diagnostics.droppedOutOfRange > 0);
  const droppedTotal = diagnostics.droppedInvalid + diagnostics.droppedOutOfRange;

  // Ruler reveal path (F14): the editor instance arrives via onMount; marker
  // clicks only scroll (never remap), matching legacy ruler semantics. The
  // marker contract is revealLineInCenter on the physical line — an
  // already-visible target is still centered — distinct from the
  // highlight-driven automatic reveal, which uses the if-outside rule (§4.8).
  // The omitted scrollType defaults to ScrollType.Smooth (verified in the
  // installed codeEditorWidget.js), so no editor-API import is needed here.
  const editorRef = useRef<monacoEditorNamespace.IStandaloneCodeEditor | null>(null);
  const handlePanelMount = useCallback(
    (_viewerId: string, editor: monacoEditorNamespace.IStandaloneCodeEditor) => {
      editorRef.current = editor;
    },
    []
  );
  const handleMarkerClick = useCallback((absoluteLine: number) => {
    try {
      editorRef.current?.revealLineInCenter(toPhysical(absoluteLine, 1));
    } catch (e) {
      // Usually a disposed editor (unmount race); warn so a real reveal
      // failure is diagnosable instead of silently swallowed.
      console.warn(`[monaco] marker reveal failed for line ${absoluteLine}`, e);
    }
  }, []);

  return (
    <div className="mp-single-wrap">
      <MonacoCodePanel
        viewerId="single-viewer"
        content={codeContent}
        monacoLanguage={monacoLanguage}
        lineOffset={1}
        effectiveLines={effectiveLines}
        entryDoc={highlight.doc}
        docToken={currentDoc}
        fontSize={16}
        onLineClick={handleLineClick}
        onMount={handlePanelMount}
      />
      <OverviewRuler
        lineCount={lineCount}
        startingLineNumber={1}
        highlightedLines={effectiveLines}
        onMarkerClick={handleMarkerClick}
        layout="side"
      />
      {showDiagnostics && (
        <div
          className="mp-diagnostics-badge"
          data-testid="mp-diagnostics-badge"
          role="status"
          title={`${diagnostics.droppedInvalid} invalid mappings, ${diagnostics.droppedOutOfRange} out-of-range mappings ignored`}
        >
          {droppedTotal} mappings ignored
        </div>
      )}
    </div>
  );
};

export default SingleMonacoViewer;
