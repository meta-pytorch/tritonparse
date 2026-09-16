/**
 * Pure highlight-set math shared by the Monaco panels.
 *
 * Design references: §4.4 (offset conversions), §4.4.1 (normalization),
 * §4.2 (contiguous merge before building decorations), §4.9 (Single anchor
 * grouping, semantics replicated from SingleCodeViewer.handleLineClick).
 *
 * This module is intentionally dependency-free (no React, no Monaco) so the
 * same code runs in the browser and under plain `node --test` (unit layer).
 */

/** Document slice a highlight set was computed against (absolute line numbers). */
export interface HighlightDoc {
  /** Absolute line number shown for the first physical line. */
  offset: number;
  /** Total physical line count of the document. */
  lineCount: number;
}

export interface NormalizedLines {
  /** Legal set: finite integers, in range, deduplicated, ascending. */
  lines: number[];
  /** Count of dropped non-integer/NaN/non-number entries. */
  droppedInvalid: number;
  /** Count of dropped out-of-range entries (never clamped to the edge). */
  droppedOutOfRange: number;
}

/** Absolute (displayed) -> physical (Monaco, 1-based) line number. */
export function toPhysical(abs: number, lineOffset: number): number {
  return abs - lineOffset + 1;
}

/** Physical (Monaco, 1-based) -> absolute (displayed) line number. */
export function toAbsolute(phys: number, lineOffset: number): number {
  return phys + lineOffset - 1;
}

/**
 * Normalize a raw highlight candidate set against a document.
 *
 * - Non-numbers, NaN and non-integers are dropped and counted as invalid.
 *   (String coercion belongs to the mapping layer, not here.)
 * - Out-of-range lines are dropped and counted, never clamped: clamping
 *   would forge a false correspondence on the boundary line (§4.4.1).
 * - The surviving set is deduplicated and sorted ascending so decorations,
 *   ruler and reveal all consume one identical collection.
 */
export function normalizeHighlightLines(raw: unknown, doc: HighlightDoc): NormalizedLines {
  const items: unknown[] = Array.isArray(raw) ? raw : [];
  const seen = new Set<number>();
  let droppedInvalid = 0;
  let droppedOutOfRange = 0;
  const first = doc.offset;
  const last = doc.offset + doc.lineCount - 1;
  for (const item of items) {
    if (typeof item !== "number" || !Number.isInteger(item)) {
      droppedInvalid += 1;
      continue;
    }
    if (item < first || item > last) {
      droppedOutOfRange += 1;
      continue;
    }
    seen.add(item);
  }
  return {
    lines: [...seen].sort((a, b) => a - b),
    droppedInvalid,
    droppedOutOfRange,
  };
}

/**
 * Merge a sorted-unique line list into contiguous [start, end] spans
 * (inclusive, absolute numbers) so one decoration covers each run.
 * Exact duplicates are skipped defensively; unsorted input is not supported.
 */
export function mergeContiguousLines(sortedUniqueLines: number[]): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  for (const line of sortedUniqueLines) {
    const last = spans[spans.length - 1];
    if (last && line === last[1] + 1) {
      last[1] = line;
    } else if (!last || line > last[1]) {
      spans.push([line, line]);
    }
  }
  return spans;
}

/** Minimal structural view of one source-mapping entry. */
export interface AnchorMappingEntry {
  [key: string]: unknown;
}

/**
 * Strict whole-key line parsing (I001). parseInt would truncate "3.5" to 3
 * and "4oops" to 4, forging highlight lines that normalize can no longer
 * recognize as invalid. Only pure integer keys convert; anything else yields
 * NaN so the caller drops it through normalizeHighlightLines with diagnostics.
 */
function parseMappingKey(key: string): number {
  if (!/^-?\d+$/.test(key)) {
    return NaN;
  }
  return Number(key);
}

/**
 * Single-view anchor grouping: the clicked line plus every line sharing its
 * anchor value. Key comparison uses strict whole-key parsing (I001):
 * non-integer keys yield NaN, which the caller must pipe through
 * normalizeHighlightLines to drop with diagnostics.
 *
 * Lines without a mapping, or without the anchor property, yield only the
 * clicked line itself (§2.2 item 3 semantics for the single panel).
 */
export function getAnchorGroupedLines(
  sourceMapping: Record<string, AnchorMappingEntry> | undefined,
  anchorProperty: string,
  lineNumber: number
): number[] {
  const clicked = sourceMapping?.[String(lineNumber)];
  if (!clicked || clicked[anchorProperty] == null) {
    return [lineNumber];
  }
  const anchorValue = clicked[anchorProperty];
  const related = Object.entries(sourceMapping ?? {})
    .filter(
      ([key, mapping]) =>
        mapping[anchorProperty] === anchorValue && parseMappingKey(key) !== lineNumber
    )
    .map(([key]) => parseMappingKey(key));
  return [lineNumber, ...related];
}
