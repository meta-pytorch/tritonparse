/**
 * Legacy-only highlight-bus subscriber around the shared OverviewRuler.
 *
 * CodeViewer must not own this subscription: every mapping event would
 * re-render the whole Prism syntax subtree (React.memo cannot skip a
 * component's own state updates). Keeping the bus state inside this small
 * adapter confines event updates to the ruler; the syntax tree only renders
 * on real prop changes. Dies with the legacy renderer in Phase 4.
 */
import React, { useEffect, useMemo, useState } from "react";
import OverviewRuler from "./OverviewRuler";
import { createLegacyMarkerClickHandler } from "./legacyRulerScroll";
import {
  getCodeViewerHighlights,
  HIGHLIGHT_LINES_EVENT,
  type HighlightLinesEventDetail,
} from "./highlightEvents";

interface LegacyRulerAdapterProps {
  viewerId: string;
  lineCount: number;
  startingLineNumber: number;
  /** Static fallback before the first bus event arrives. */
  fallbackLines: number[];
}

const LegacyRulerAdapter: React.FC<LegacyRulerAdapterProps> = ({
  viewerId,
  lineCount,
  startingLineNumber,
  fallbackLines,
}) => {
  // The retained store seeds late mounts, then the highlight bus (filtered by
  // viewerId) takes over.
  const [eventHighlightedLines, setEventHighlightedLines] = useState<
    number[] | null
  >(() => getCodeViewerHighlights(viewerId) ?? null);
  const rulerLines = eventHighlightedLines ?? fallbackLines;

  useEffect(() => {
    const handleHighlightLines = (event: Event) => {
      const detail = (event as CustomEvent<HighlightLinesEventDetail>).detail;
      if (detail.viewerId === viewerId) {
        setEventHighlightedLines(detail.lineNumbers);
      }
    };

    window.addEventListener(HIGHLIGHT_LINES_EVENT, handleHighlightLines);
    return () => {
      window.removeEventListener(HIGHLIGHT_LINES_EVENT, handleHighlightLines);
    };
  }, [viewerId]);

  const lastRulerLine = startingLineNumber + lineCount - 1;
  const visibleRulerLines = useMemo(
    () =>
      Array.from(new Set(rulerLines))
        .filter((line) => line >= startingLineNumber && line <= lastRulerLine)
        .sort((a, b) => a - b),
    [rulerLines, startingLineNumber, lastRulerLine]
  );
  const handleRulerMarker = useMemo(
    () => createLegacyMarkerClickHandler(viewerId, startingLineNumber, lineCount),
    [viewerId, startingLineNumber, lineCount]
  );

  return (
    <OverviewRuler
      key={viewerId}
      lineCount={lineCount}
      startingLineNumber={startingLineNumber}
      highlightedLines={visibleRulerLines}
      onMarkerClick={handleRulerMarker}
    />
  );
};

export default React.memo(LegacyRulerAdapter);
