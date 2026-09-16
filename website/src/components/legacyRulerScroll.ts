/**
 * Legacy DOM scroll handler for the overview ruler, moved verbatim from the
 * previously embedded ruler (now scoped by viewerId instead of the click
 * event): centered row scroll, Monaco container fallback, row-height probe
 * jump, fractional guess. Dies with legacy in Phase 4.
 */
export function createLegacyMarkerClickHandler(
  viewerId: string | undefined,
  startingLineNumber: number,
  lineCount: number
): (line: number) => void {
  return (lineNumber: number) => {
    if (!viewerId) return;

    const container = document.querySelector(
      `[data-viewer-id="${viewerId}"]`
    ) as HTMLElement | null;
    if (!container) return;

    const target = container.querySelector(
      `[data-line-number="${lineNumber}"]`
    ) as HTMLElement | null;
    if (target) {
      const containerRect = container.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const centeredTop = container.scrollTop + targetRect.top -
        containerRect.top - container.clientHeight / 2;
      container.scrollTo({ top: Math.max(0, centeredTop), behavior: "smooth" });
      return;
    }

    const monacoContainer = container.querySelector(".monaco-editor");
    if (monacoContainer) {
      monacoContainer.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    // A virtualized viewer may not have the requested line in the DOM yet.
    // Virtualized rows have uniform height, so measure one rendered row
    // (exact under any font size) and jump precisely instead of guessing by
    // fraction: with virtualization active above 100KB this fallback is the
    // common path, not the exception.
    const probe = container.querySelector(
      '[data-line-number]'
    ) as HTMLElement | null;
    const rowHeight = probe ? probe.getBoundingClientRect().height : 0;
    if (rowHeight > 0) {
      const lineTop = (lineNumber - startingLineNumber) * rowHeight;
      container.scrollTo({
        top: Math.max(0, lineTop - container.clientHeight / 2),
        behavior: 'smooth',
      });
      return;
    }

    // No measurable row (empty viewer): fall back to a fractional guess.
    const fraction = lineCount <= 1
      ? 0
      : (lineNumber - startingLineNumber) / (lineCount - 1);
    const maxScrollTop = Math.max(0, container.scrollHeight - container.clientHeight);
    container.scrollTo({ top: fraction * maxScrollTop, behavior: "smooth" });
  };
}

