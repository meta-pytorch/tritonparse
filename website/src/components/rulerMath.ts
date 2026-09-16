/**
 * Pure overview-ruler math (§4.8), dependency-free for `node --test`.
 * Positioning replicates the legacy embedded ruler formula verbatim.
 */

/** Marker count above which the strip samples + shows the overflow badge. */
export const RULER_OVERFLOW_THRESHOLD = 5000;
/** Markers drawn when sampling. */
export const RULER_SAMPLE_SIZE = 200;

/**
 * Vertical marker position as a percentage of the strip (legacy formula,
 * preserved verbatim for visual parity, including the single-line pin).
 */
export function markerTopPercent(
  line: number,
  startingLineNumber: number,
  lineCount: number
): number {
  const rawPosition =
    lineCount <= 1 ? 0 : ((line - startingLineNumber) / (lineCount - 1)) * 100;
  return Math.min(98, Math.max(2, rawPosition));
}

/**
 * Evenly sample a sorted line list down to `max` entries (first+last kept).
 * Input at/under max returns as-is. Pure, unit-tested.
 */
export function sampleMarkers(lines: number[], max: number): number[] {
  // Degenerate max must not reach the (max - 1) divisor below: max<=0
  // keeps nothing, max==1 keeps the first entry (first+last contract).
  if (max <= 0) {
    return [];
  }
  if (max === 1) {
    return lines.length > 0 ? [lines[0]] : [];
  }
  if (lines.length <= max) {
    return lines;
  }
  const sampled: number[] = [];
  for (let i = 0; i < max; i++) {
    sampled.push(lines[Math.floor((i * (lines.length - 1)) / (max - 1))]);
  }
  return sampled;
}
