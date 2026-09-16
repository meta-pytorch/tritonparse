/**
 * Props-driven overview ruler for the Monaco views (§4.8).
 *
 * The strip owns its own 14px flex column beside the editor (§4.8/R9), so it
 * can neither cover the native overview ruler/scrollbar nor intercept their
 * pointer events. Scrolling is injected: panels pass an editor-API reveal.
 * Large sets are sampled with an overflow badge + full-list popup (F20).
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  RULER_OVERFLOW_THRESHOLD,
  RULER_SAMPLE_SIZE,
  markerTopPercent,
  sampleMarkers,
} from "./rulerMath";
import "./OverviewRuler.css";

export interface RulerOverflow {
  total: number;
  shown: number;
}

export interface OverviewRulerProps {
  lineCount: number;
  startingLineNumber: number;
  /** Absolute, normalized, ascending highlight lines (parent-guarded). */
  highlightedLines: number[];
  /** Fired with the absolute line when a marker (or popup entry) activates. */
  onMarkerClick: (line: number) => void;
}

const OverviewRuler: React.FC<OverviewRulerProps> = ({
  lineCount,
  startingLineNumber,
  highlightedLines,
  onMarkerClick,
}) => {
  const [popupOpen, setPopupOpen] = useState(false);
  const overflow = highlightedLines.length > RULER_OVERFLOW_THRESHOLD;
  const shownLines = overflow
    ? sampleMarkers(highlightedLines, RULER_SAMPLE_SIZE)
    : highlightedLines;

  // Dialog behavior: the badge keeps focus after opening, so a container
  // onKeyDown never sees Escape. Close from the document level instead,
  // move focus into the dialog on open, trap Tab inside, and return focus
  // to the badge on close (aria-modal + initial focus + focus return).
  const badgeRef = useRef<HTMLButtonElement | null>(null);
  const popupRef = useRef<HTMLDivElement | null>(null);
  // Cached trap ends: the popup holds 5000+ buttons, so the Tab handler
  // must not querySelectorAll on every keystroke. Recomputed when the
  // popup opens or its line set changes (content is static otherwise).
  const trapEndsRef = useRef<{ first: HTMLElement; last: HTMLElement } | null>(null);
  const closePopup = useCallback(() => {
    setPopupOpen(false);
    badgeRef.current?.focus();
  }, []);
  useEffect(() => {
    if (!popupOpen || !popupRef.current) {
      trapEndsRef.current = null;
      return;
    }
    const items = popupRef.current.querySelectorAll("button:not([disabled])");
    trapEndsRef.current =
      items.length > 0
        ? { first: items[0] as HTMLElement, last: items[items.length - 1] as HTMLElement }
        : null;
  }, [popupOpen, highlightedLines]);
  useEffect(() => {
    if (!popupOpen) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closePopup();
        return;
      }
      // Document-level Tab trap: focus starts on the dialog div itself
      // (tabIndex -1, outside the tab order), so a div-level onKeyDown can
      // neither see nor stop Shift+Tab from escaping to the page behind.
      if (event.key !== "Tab" || !popupRef.current) return;
      const ends = trapEndsRef.current;
      if (!ends) return;
      const active = document.activeElement;
      const inside = popupRef.current.contains(active);
      if (event.shiftKey) {
        if (active === ends.first || active === popupRef.current || !inside) {
          event.preventDefault();
          ends.last.focus();
        }
      } else if (active === ends.last || !inside) {
        event.preventDefault();
        ends.first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    popupRef.current?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [popupOpen, closePopup]);
  // Auto-close when the overflow condition clears under an open popup: the
  // badge unmounts with it, so focus return has no target; closing avoids
  // a stale popup whose trigger is gone.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- prop-change reset (one extra render, no cascade)
    if (popupOpen && !overflow) closePopup();
  }, [popupOpen, overflow, closePopup]);

  const copyAll = useCallback(() => {
    const text = highlightedLines.join(", ");
    try {
      const clipboard = navigator.clipboard;
      if (clipboard) {
        // The promise rejection is swallowed: a denied clipboard must never
        // surface as an unhandled rejection; the list stays readable.
        clipboard.writeText(text).catch(() => {});
      }
    } catch {
      /* clipboard unavailable; the list stays readable + clickable */
    }
  }, [highlightedLines]);

  // The strip always renders its 14px column — even empty — so showing or
  // clearing markers never changes the editor width. Mounting the strip on
  // first highlight used to fire the panel ResizeObserver mid-reveal and
  // freeze the smooth highlight animation part-way (measured: stuck at 117
  // instead of centering on 428).
  // An empty strip keeps its 14px column (layout stability) but drops the
  // grey track so it no longer reads as a dead scrollbar.
  const empty = highlightedLines.length === 0;
  return (
    <div
      className={`code-overview-ruler${empty ? " is-empty" : ""}`}
      data-testid="overview-ruler"
      aria-label="Highlighted lines overview"
    >
      {shownLines.map((line) => (
        <button
          key={line}
          type="button"
          className="code-overview-marker"
          style={{ top: `${markerTopPercent(line, startingLineNumber, lineCount)}%` }}
          title={`Line ${line}`}
          aria-label={`Scroll to highlighted line ${line}`}
          data-testid={`overview-marker-${line}`}
          onClick={() => onMarkerClick(line)}
        />
      ))}
      {overflow && (
        <button
          type="button"
          ref={badgeRef}
          className="code-overview-overflow"
          data-testid="ruler-overflow-badge"
          title={`Ruler shows ${shownLines.length}/${highlightedLines.length} markers; full highlights in editor`}
          aria-label={`Show all ${highlightedLines.length} highlighted lines`}
          onClick={() => setPopupOpen((open) => !open)}
        >
          {shownLines.length}/{highlightedLines.length}
        </button>
      )}
      {popupOpen && (
        <div
          className="code-overview-popup"
          data-testid="ruler-overflow-popup"
          role="dialog"
          aria-modal="true"
          aria-label={`All ${highlightedLines.length} highlighted lines`}
          tabIndex={-1}
          ref={popupRef}
        >
          <div className="code-overview-popup-header">
            <span>
              All {highlightedLines.length} lines (ruler shows {shownLines.length})
            </span>
            <button type="button" onClick={copyAll} title="Copy all line numbers">
              Copy all
            </button>
            <button type="button" onClick={closePopup} aria-label="Close">
              ×
            </button>
          </div>
          <div className="code-overview-popup-list">
            {highlightedLines.map((line) => (
              <button
                key={line}
                type="button"
                className="code-overview-popup-item"
                onClick={() => onMarkerClick(line)}
              >
                Line {line}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default React.memo(OverviewRuler);
