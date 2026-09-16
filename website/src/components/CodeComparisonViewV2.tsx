/**
 * IR comparison view on Monaco panels (design §3.1, §4.2).
 *
 * Same panel props as the legacy CodeComparisonView plus the source-qualified
 * kernel identity (§4.2.1). Owns the three-panel highlight truth source as
 * { doc, lines } entries: every render guards each entry against its current
 * doc token (mismatch => []), and any token change clears the whole truth
 * source so stale highlights can neither render nor reveal on a new document
 * (F18). Click mapping reuses the ported legacy math (comparisonMapping.ts);
 * every output flows through normalizeHighlightLines before setHighlights
 * (§4.4.1), and the single normalized result drives decorations, ruler,
 * reveal and the diagnostics badges.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Panel, Group, Separator } from "react-resizable-panels";
import MonacoCodePanel from "./MonacoCodePanel";
import OverviewRuler from "./OverviewRuler";
import CopyCodeButton from "./CopyCodeButton";
import type { editor as monacoEditorNamespace } from "monaco-editor";
import {
  IRFile,
  IRStageDescriptor,
  PythonSourceCodeInfo,
  SourceMapping,
} from "../utils/dataLoader";
import { getDisplayLanguage } from "../utils/irLanguage";
import { mapFileToMonacoLanguage, MONACO_LANGUAGE_IDS } from "../utils/monacoLanguage";
import {
  EMPTY_DOC,
  normalizeHighlightLines,
  toPhysical,
  type DocToken,
  type HighlightDoc,
  type PanelDiagnostics,
  type PanelHighlight,
} from "./monaco/highlightMath";
import {
  buildKernelKey,
  calculateMappedLines,
  calculatePythonLines,
  intersectRangeWithDoc,
  lineCountOfContent,
} from "./monaco/comparisonMapping";

/**
 * Props for a single code panel.
 */
interface CodePanelProps {
  code?: IRFile;
  content?: string;
  title?: string;
}

/**
 * Props for the CodeComparisonViewV2 component.
 */
interface CodeComparisonViewV2Props {
  leftPanel: CodePanelProps;
  rightPanel: CodePanelProps;
  py_code_info?: PythonSourceCodeInfo;
  showPythonSource?: boolean;
  pythonMapping?: Record<string, SourceMapping>;
  irStages?: IRStageDescriptor[];
  /**
   * Stable data-source identity (load URL or slot name), independent of
   * kernel data. Combined with kernelId into the kernelKey (§4.2.1).
   */
  sourceId: string;
  /** Kernel identity within the source: metadata.hash ?? selected index. */
  kernelId: string | number;
}

/**
 * Unified highlight state interface (same panels as the legacy view).
 */
interface HighlightState {
  left: PanelHighlight;
  right: PanelHighlight;
  python: PanelHighlight;
}

/**
 * Per-panel diagnostics state, keyed by the same doc as the highlight entry.
 */
interface DiagnosticsState {
  left: PanelDiagnostics;
  right: PanelDiagnostics;
  python: PanelDiagnostics;
}

/**
 * Panel data interface for cached computations (same shape as legacy).
 */
interface PanelData {
  title: string;
  content: string;
  sourceMapping: Record<string, SourceMapping>;
  displayLanguage: string;
}

/**
 * Python info interface for cached computations (same shape as legacy).
 */
interface PythonInfo {
  code: string;
  file_path: string;
  start_line: number;
  isFullFileMode: boolean;
  function_start_line?: number;
  function_end_line?: number;
}

type PanelId = "left" | "right" | "python";

/**
 * Normalize one panel's raw mapping candidates and report diagnostics.
 * The single normalized output drives decorations, ruler, reveal and the
 * badge (§4.4.1); nothing consumes the raw set. Module scope (not inline
 * in the component): it closes over nothing, so the useCallback handlers
 * below keep a stable reference and exhaustive-deps stays meaningful.
 */
const applyPanelSet = (
  panel: PanelId,
  doc: DocToken,
  docInfo: HighlightDoc,
  raw: unknown[]
): { highlight: PanelHighlight; diagnostics: PanelDiagnostics } => {
  const normalized = normalizeHighlightLines(raw, docInfo);
  if (normalized.droppedInvalid > 0 || normalized.droppedOutOfRange > 0) {
    console.warn(
      `[monaco] dropped ${normalized.droppedInvalid} invalid / ${normalized.droppedOutOfRange} out-of-range mapped lines in comparison/${panel}`
    );
  }
  return {
    highlight: { doc, lines: normalized.lines },
    diagnostics: {
      doc,
      droppedInvalid: normalized.droppedInvalid,
      droppedOutOfRange: normalized.droppedOutOfRange,
    },
  };
};

const CodeComparisonViewV2: React.FC<CodeComparisonViewV2Props> = ({
  leftPanel,
  rightPanel,
  py_code_info,
  showPythonSource = false,
  pythonMapping,
  irStages,
  sourceId,
  kernelId,
}) => {
  // ==================== Memoized Computations ====================

  /**
   * Memoized left panel data (same derivation as the legacy view).
   */
  const leftPanel_data = useMemo<PanelData>(
    () => ({
      title: leftPanel.title || "TTGIR",
      content: leftPanel.content || leftPanel.code?.content || "",
      sourceMapping: leftPanel.code?.source_mapping || {},
      displayLanguage: getDisplayLanguage(leftPanel.title || "TTGIR", irStages),
    }),
    [leftPanel.title, leftPanel.content, leftPanel.code, irStages]
  );

  /**
   * Memoized right panel data (same derivation as the legacy view).
   */
  const rightPanel_data = useMemo<PanelData>(
    () => ({
      title: rightPanel.title || "PTX",
      content: rightPanel.content || rightPanel.code?.content || "",
      sourceMapping: rightPanel.code?.source_mapping || {},
      displayLanguage: getDisplayLanguage(rightPanel.title || "PTX", irStages),
    }),
    [rightPanel.title, rightPanel.content, rightPanel.code, irStages]
  );

  /**
   * Memoized Python source info (same derivation as the legacy view).
   */
  const pythonInfo = useMemo<PythonInfo>(
    () => ({
      code: py_code_info?.code || "",
      file_path: py_code_info?.file_path || "",
      start_line: py_code_info?.start_line || 1,
      isFullFileMode:
        py_code_info?.start_line === 1 &&
        py_code_info?.function_start_line !== undefined,
      function_start_line: py_code_info?.function_start_line,
      function_end_line: py_code_info?.function_end_line,
    }),
    [py_code_info]
  );

  const leftLineCount = useMemo(
    () => lineCountOfContent(leftPanel_data.content),
    [leftPanel_data.content]
  );
  const rightLineCount = useMemo(
    () => lineCountOfContent(rightPanel_data.content),
    [rightPanel_data.content]
  );
  const pythonLineCount = useMemo(
    () => lineCountOfContent(pythonInfo.code),
    [pythonInfo.code]
  );

  // ==================== Document Identity (§4.2.1) ====================

  // Source-qualified kernel key: the hash/index only identifies a kernel
  // within one data source. A plain string: React compares token deps by
  // value, so equal keys keep the token stable across renders.
  const kernelKey = useMemo(
    () => buildKernelKey(sourceId, kernelId),
    [sourceId, kernelId]
  );

  // Per-panel doc tokens. Deps are identity inputs, intentionally unread: any
  // reference change rebuilds the token. Tokens stay opaque (no fields, no
  // retention). The tokens live in this parent (not in the panels), so the
  // python toggle — which remounts only the python panel — keeps the panel
  // identity and the state re-applies on reopen (F15).
  const leftDoc: DocToken = useMemo(
    () => ({}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      kernelKey,
      leftPanel_data.title,
      leftPanel_data.content,
      leftPanel_data.sourceMapping,
      irStages,
    ]
  );
  const rightDoc: DocToken = useMemo(
    () => ({}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      kernelKey,
      rightPanel_data.title,
      rightPanel_data.content,
      rightPanel_data.sourceMapping,
      irStages,
    ]
  );
  const pythonDoc: DocToken = useMemo(
    () => ({}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [
      kernelKey,
      pythonInfo.file_path,
      pythonInfo.start_line,
      pythonInfo.code,
      pythonMapping,
      irStages,
    ]
  );

  const leftDocInfo: HighlightDoc = useMemo(
    () => ({ offset: 1, lineCount: leftLineCount }),
    [leftLineCount]
  );
  const rightDocInfo: HighlightDoc = useMemo(
    () => ({ offset: 1, lineCount: rightLineCount }),
    [rightLineCount]
  );
  const pythonDocInfo: HighlightDoc = useMemo(
    () => ({ offset: pythonInfo.start_line, lineCount: pythonLineCount }),
    [pythonInfo.start_line, pythonLineCount]
  );

  // ==================== Highlight Truth Source (§4.2) ====================

  const [highlights, setHighlights] = useState<HighlightState>(() => ({
    left: { doc: EMPTY_DOC, lines: [] },
    right: { doc: EMPTY_DOC, lines: [] },
    python: { doc: EMPTY_DOC, lines: [] },
  }));
  const [diagnostics, setDiagnostics] = useState<DiagnosticsState>(() => ({
    left: { doc: EMPTY_DOC, droppedInvalid: 0, droppedOutOfRange: 0 },
    right: { doc: EMPTY_DOC, droppedInvalid: 0, droppedOutOfRange: 0 },
    python: { doc: EMPTY_DOC, droppedInvalid: 0, droppedOutOfRange: 0 },
  }));

  // Doc-change clearing (legacy semantics: any panel's document change
  // clears all three panels). Switching back does not restore, matching the
  // legacy ref-loss behavior (F18). Applied SYNCHRONOUSLY in the effect:
  // React flushes pending passive effects before handling discrete input, so
  // a click after a doc change always observes (and extends) the cleared
  // truth. A setTimeout-deferred clear breaks that guarantee — the timeout
  // can fire after a fast click's setState and wipe a live highlight set
  // (observed as an e2e flake). No loop: the update never changes the tokens.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- prop-change reset, see above
    setHighlights({
      left: { doc: leftDoc, lines: [] },
      right: { doc: rightDoc, lines: [] },
      python: { doc: pythonDoc, lines: [] },
    });
    setDiagnostics({
      left: { doc: leftDoc, droppedInvalid: 0, droppedOutOfRange: 0 },
      right: { doc: rightDoc, droppedInvalid: 0, droppedOutOfRange: 0 },
      python: { doc: pythonDoc, droppedInvalid: 0, droppedOutOfRange: 0 },
    });
  }, [leftDoc, rightDoc, pythonDoc]);

  // Parent guard (§4.2.1): panels and rulers only ever receive sets whose
  // doc matches the current token; a mismatch renders as [] (and the empty
  // set() actively clears stale decorations in the panel).
  const leftEffective =
    highlights.left.doc === leftDoc ? highlights.left.lines : [];
  const rightEffective =
    highlights.right.doc === rightDoc ? highlights.right.lines : [];
  const pythonEffective =
    highlights.python.doc === pythonDoc ? highlights.python.lines : [];

  // ==================== Click Handlers ====================

  /**
   * Handle line click in left or right panel (legacy structure: same guards,
   * same mapping calls; the three updateHighlights calls become one
   * setHighlights with normalized sets).
   */
  const handlePanelLineClick = useCallback(
    (lineNumber: number, panelType: "left" | "right") => {
      const isLeftPanel = panelType === "left";
      const sourcePanel = isLeftPanel ? leftPanel_data : rightPanel_data;
      const targetPanel = isLeftPanel ? rightPanel_data : leftPanel_data;
      const sourceMapping = sourcePanel.sourceMapping;

      // No mapping entry: highlight the clicked line, clear the others.
      const rawTarget: unknown[] = [];
      const rawPython: unknown[] = [];
      if (sourceMapping && sourceMapping[lineNumber]) {
        const mapped = calculateMappedLines(
          sourceMapping,
          lineNumber,
          targetPanel.title,
          irStages
        );
        rawTarget.push(...mapped);
        if (showPythonSource && py_code_info?.code) {
          rawPython.push(
            ...calculatePythonLines(sourceMapping, lineNumber, pythonInfo)
          );
        }
      }

      const leftRaw = isLeftPanel ? [lineNumber] : rawTarget;
      const rightRaw = isLeftPanel ? rawTarget : [lineNumber];
      const left = applyPanelSet("left", leftDoc, leftDocInfo, leftRaw);
      const right = applyPanelSet("right", rightDoc, rightDocInfo, rightRaw);
      const python = applyPanelSet("python", pythonDoc, pythonDocInfo, rawPython);
      setHighlights({
        left: left.highlight,
        right: right.highlight,
        python: python.highlight,
      });
      setDiagnostics({
        left: left.diagnostics,
        right: right.diagnostics,
        python: python.diagnostics,
      });
    },
    [
      leftPanel_data,
      rightPanel_data,
      leftDoc,
      rightDoc,
      pythonDoc,
      leftDocInfo,
      rightDocInfo,
      pythonDocInfo,
      irStages,
      showPythonSource,
      py_code_info,
      pythonInfo,
    ]
  );

  /**
   * Handle line click in the Python panel (absolute line number; same
   * structure as the legacy handler).
   */
  const handlePythonLineClick = useCallback(
    (lineNumber: number) => {
      let rawLeft: unknown[] = [];
      let rawRight: unknown[] = [];
      const mapping = pythonMapping?.[lineNumber.toString()];
      if (mapping) {
        rawLeft = calculateMappedLines(
          { [lineNumber.toString()]: mapping },
          lineNumber,
          leftPanel_data.title,
          irStages
        );
        rawRight = calculateMappedLines(
          { [lineNumber.toString()]: mapping },
          lineNumber,
          rightPanel_data.title,
          irStages
        );
      }

      const left = applyPanelSet("left", leftDoc, leftDocInfo, rawLeft);
      const right = applyPanelSet("right", rightDoc, rightDocInfo, rawRight);
      const python = applyPanelSet(
        "python",
        pythonDoc,
        pythonDocInfo,
        [lineNumber]
      );
      setHighlights({
        left: left.highlight,
        right: right.highlight,
        python: python.highlight,
      });
      setDiagnostics({
        left: left.diagnostics,
        right: right.diagnostics,
        python: python.diagnostics,
      });
    },
    [
      pythonMapping,
      leftPanel_data.title,
      rightPanel_data.title,
      leftDoc,
      rightDoc,
      pythonDoc,
      leftDocInfo,
      rightDocInfo,
      pythonDocInfo,
      irStages,
    ]
  );

  const handleLeftLineClick = useCallback(
    (lineNumber: number) => handlePanelLineClick(lineNumber, "left"),
    [handlePanelLineClick]
  );

  const handleRightLineClick = useCallback(
    (lineNumber: number) => handlePanelLineClick(lineNumber, "right"),
    [handlePanelLineClick]
  );

  // ==================== Ruler Reveal Paths (§4.8) ====================

  // Editor instances arrive via onMount; marker clicks only scroll (never
  // remap). The marker contract is revealLineInCenter on the physical line,
  // distinct from the highlight-driven if-outside automatic reveal.
  const editorsRef = useRef<
    Record<PanelId, monacoEditorNamespace.IStandaloneCodeEditor | null>
  >({ left: null, right: null, python: null });
  const handlePanelMount = useCallback(
    (viewerId: string, editor: monacoEditorNamespace.IStandaloneCodeEditor) => {
      if (viewerId === "left" || viewerId === "right" || viewerId === "python") {
        editorsRef.current[viewerId] = editor;
      }
    },
    []
  );
  const handleLeftMarkerClick = useCallback((absoluteLine: number) => {
    try {
      editorsRef.current.left?.revealLineInCenter(toPhysical(absoluteLine, 1));
    } catch {
      /* editor may be disposed */
    }
  }, []);
  const handleRightMarkerClick = useCallback((absoluteLine: number) => {
    try {
      editorsRef.current.right?.revealLineInCenter(toPhysical(absoluteLine, 1));
    } catch {
      /* editor may be disposed */
    }
  }, []);
  const handlePythonMarkerClick = useCallback(
    (absoluteLine: number) => {
      try {
        editorsRef.current.python?.revealLineInCenter(
          toPhysical(absoluteLine, pythonInfo.start_line)
        );
      } catch {
        /* editor may be disposed */
      }
    },
    [pythonInfo.start_line]
  );

  // ==================== Python Range & Initial Line ====================

  // Function range intersected with the python window (§4.4.1); empty =>
  // undefined => no range decorations. Full-file mode only (legacy parity).
  const pythonFunctionRange = useMemo(() => {
    if (!pythonInfo.isFullFileMode) return undefined;
    if (
      pythonInfo.function_start_line == null ||
      pythonInfo.function_end_line == null
    ) {
      return undefined;
    }
    return (
      intersectRangeWithDoc(
        {
          start: pythonInfo.function_start_line,
          end: pythonInfo.function_end_line,
        },
        pythonDocInfo
      ) ?? undefined
    );
  }, [pythonInfo, pythonDocInfo]);

  // Initial positioning target (legacy parity): full-file => function start,
  // snippet => window start.
  const pythonInitialLine = pythonInfo.isFullFileMode
    ? pythonInfo.function_start_line
    : pythonInfo.start_line;

  const leftMonacoLanguage = mapFileToMonacoLanguage(
    leftPanel_data.title,
    irStages
  );
  const rightMonacoLanguage = mapFileToMonacoLanguage(
    rightPanel_data.title,
    irStages
  );

  // ==================== Scroll Tip State ====================

  const [showScrollTip, setShowScrollTip] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("tritonparse_hideScrollTip") !== "true";
    }
    return true;
  });

  const handleDismissScrollTip = useCallback(() => {
    setShowScrollTip(false);
    if (typeof window !== "undefined") {
      localStorage.setItem("tritonparse_hideScrollTip", "true");
    }
  }, []);

  // ==================== Render ====================

  const renderDiagnosticsBadge = (panel: PanelId, entry: PanelDiagnostics) => {
    const currentDoc =
      panel === "left" ? leftDoc : panel === "right" ? rightDoc : pythonDoc;
    if (entry.doc !== currentDoc) return null;
    const total = entry.droppedInvalid + entry.droppedOutOfRange;
    if (total === 0) return null;
    return (
      <div
        className="mp-diagnostics-badge"
        data-testid={`mp-diagnostics-badge-${panel}`}
        role="status"
        title={`${entry.droppedInvalid} invalid mappings, ${entry.droppedOutOfRange} out-of-range mappings ignored`}
      >
        {total} mappings ignored
      </div>
    );
  };

  return (
    <div style={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {showScrollTip && (
        <div
          style={{
            backgroundColor: "#e7f3ff",
            borderBottom: "1px solid #b3d7ff",
            padding: "6px 16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            fontSize: "13px",
            color: "#0066cc",
            flexShrink: 0,
          }}
        >
          <span>
            💡 Tip: Click any line to highlight corresponding code in other IR
            panels. Use{" "}
            <kbd
              style={{
                backgroundColor: "#f0f0f0",
                border: "1px solid #ccc",
                borderRadius: "3px",
                padding: "2px 6px",
                fontFamily: "monospace",
                fontSize: "12px",
                margin: "0 2px",
              }}
            >
              Shift
            </kbd>{" "}
            + Mouse Wheel to scroll horizontally.
          </span>
          <button
            onClick={handleDismissScrollTip}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              fontSize: "16px",
              color: "#666",
              padding: "0 4px",
            }}
            title="Dismiss tip"
            aria-label="Dismiss tip"
          >
            <span aria-hidden="true">✕</span>
          </button>
        </div>
      )}
      <Group orientation="horizontal" style={{ flex: 1, minHeight: 0 }}>
        {/* Left Panel */}
        <Panel defaultSize={33} minSize={20}>
          <div
            style={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              position: "relative",
            }}
          >
            <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center min-w-0">
              <span
                className="truncate flex-1 min-w-0 mr-2"
                title={leftPanel_data.title}
              >
                {leftPanel_data.title}
              </span>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                  {leftPanel_data.displayLanguage}
                </span>
                <CopyCodeButton
                  code={leftPanel_data.content}
                  className="text-sm bg-blue-700 px-2 py-1 rounded"
                />
              </div>
            </div>
            <div style={{ flex: 1, overflow: "hidden" }}>
              <div className="mp-panel-row">
                <MonacoCodePanel
                  viewerId="left"
                  content={leftPanel_data.content}
                  monacoLanguage={leftMonacoLanguage}
                  lineOffset={1}
                  effectiveLines={leftEffective}
                  entryDoc={highlights.left.doc}
                  docToken={leftDoc}
                  fontSize={14}
                  onLineClick={handleLeftLineClick}
                  onMount={handlePanelMount}
                />
                <OverviewRuler
                  lineCount={leftLineCount}
                  startingLineNumber={1}
                  highlightedLines={leftEffective}
                  onMarkerClick={handleLeftMarkerClick}
                          />
                {renderDiagnosticsBadge("left", diagnostics.left)}
              </div>
            </div>
          </div>
        </Panel>

        <Separator
          style={{
            width: "4px",
            backgroundColor: "#ddd",
            cursor: "col-resize",
          }}
        />

        {/* Right Panel */}
        <Panel defaultSize={33} minSize={20}>
          <div
            style={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              position: "relative",
            }}
          >
            <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center min-w-0">
              <span
                className="truncate flex-1 min-w-0 mr-2"
                title={rightPanel_data.title}
              >
                {rightPanel_data.title}
              </span>
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                  {rightPanel_data.displayLanguage}
                </span>
                <CopyCodeButton
                  code={rightPanel_data.content}
                  className="text-sm bg-blue-700 px-2 py-1 rounded"
                />
              </div>
            </div>
            <div style={{ flex: 1, overflow: "hidden" }}>
              <div className="mp-panel-row">
                <MonacoCodePanel
                  viewerId="right"
                  content={rightPanel_data.content}
                  monacoLanguage={rightMonacoLanguage}
                  lineOffset={1}
                  effectiveLines={rightEffective}
                  entryDoc={highlights.right.doc}
                  docToken={rightDoc}
                  fontSize={14}
                  onLineClick={handleRightLineClick}
                  onMount={handlePanelMount}
                />
                <OverviewRuler
                  lineCount={rightLineCount}
                  startingLineNumber={1}
                  highlightedLines={rightEffective}
                  onMarkerClick={handleRightMarkerClick}
                          />
                {renderDiagnosticsBadge("right", diagnostics.right)}
              </div>
            </div>
          </div>
        </Panel>

        {/* Python Source Panel (Optional) */}
        {showPythonSource && py_code_info && (
          <>
            <Separator
              style={{
                width: "4px",
                backgroundColor: "#ddd",
                cursor: "col-resize",
              }}
            />
            <Panel defaultSize={34} minSize={20}>
              <div
                style={{
                  height: "100%",
                  display: "flex",
                  flexDirection: "column",
                  position: "relative",
                }}
              >
                <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center min-w-0">
                  <span
                    className="truncate flex-1 min-w-0 mr-2"
                    title={
                      pythonInfo.isFullFileMode
                        ? "Python Source (Full File)"
                        : "Python Source"
                    }
                  >
                    {pythonInfo.isFullFileMode
                      ? "Python Source (Full File)"
                      : "Python Source"}
                  </span>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                      python
                    </span>
                    <CopyCodeButton
                      code={pythonInfo.code}
                      className="text-sm bg-blue-700 px-2 py-1 rounded"
                    />
                  </div>
                </div>
                <div style={{ flex: 1, overflow: "hidden" }}>
                  <div className="mp-panel-row">
                    <MonacoCodePanel
                      viewerId="python"
                      content={pythonInfo.code}
                      monacoLanguage={MONACO_LANGUAGE_IDS.python}
                      lineOffset={pythonInfo.start_line}
                      effectiveLines={pythonEffective}
                      entryDoc={highlights.python.doc}
                      docToken={pythonDoc}
                      functionRange={pythonFunctionRange}
                      initialLine={pythonInitialLine}
                      fontSize={14}
                      onLineClick={handlePythonLineClick}
                      onMount={handlePanelMount}
                    />
                    <OverviewRuler
                      lineCount={pythonLineCount}
                      startingLineNumber={pythonInfo.start_line}
                      highlightedLines={pythonEffective}
                      onMarkerClick={handlePythonMarkerClick}
                                  />
                    {renderDiagnosticsBadge("python", diagnostics.python)}
                  </div>
                </div>
              </div>
            </Panel>
          </>
        )}
      </Group>
    </div>
  );
};

export default React.memo(CodeComparisonViewV2);
