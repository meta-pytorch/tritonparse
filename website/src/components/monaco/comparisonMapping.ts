/**
 * Pure comparison mapping math (design §4.2–§4.4).
 *
 * Mapping semantics live outside React so unit tests and panels share one
 * implementation. Two deliberate deviations, both required by the approved
 * design:
 *
 * 1. No parseInt/Number forging (F19): `parseInt(line, 10)`
 *    truncated "3.5" to 3 and "4oops" to 4, and `Number(mapping.line)`
 *    converted true to 1 and [459] to 459, inventing highlight lines the
 *    normalizer could no longer recognize as invalid. Only pure-integer
 *    strings convert; anything else passes through verbatim so
 *    normalizeHighlightLines drops it with diagnostics. Non-array `*_lines`
 *    fields yield [] instead of throwing inside .map() (legacy crashed:
 *    TypeError).
 * 2. Single normalization point (§4.4.1): the mapping layer returns raw
 *    candidates (absolute line numbers); range/illegal filtering happens only
 *    in normalizeHighlightLines, whose single output drives decorations,
 *    ruler and reveal. In particular calculatePythonLines no longer filters
 *    out-of-range lines itself (legacy did, with a console.error); the
 *    normalizer drops them and the caller surfaces the diagnostics badge.
 *
 * Everything else — stage discovery, property naming, the python file
 * `includes` match direction, empty-input behavior — is legacy-identical and
 * pinned by unit tests.
 *
 * Dependency-light on purpose (dataLoader types + getIRType only) so the same
 * file runs in the browser and under plain `node --test` (unit layer).
 */
import {
  getIRType,
  type IRStageDescriptor,
  type SourceMapping,
} from "../../utils/dataLoader.ts";
import type { HighlightDoc } from "./highlightMath.ts";

/** Minimal python-panel identity needed by the IR→Python mapping. */
export interface PythonPanelInfo {
  code: string;
  file_path: string;
  start_line: number;
}

/**
 * Strict whole-string integer parsing (shared with the anchor
 * grouping): "42" -> 42, but "3.5"/"4oops"/"" stay verbatim so the
 * normalizer counts them as invalid instead of highlighting forged lines.
 */
function parseStrictIntString(value: string): number | string {
  if (/^-?\d+$/.test(value)) {
    return Number(value);
  }
  return value;
}

/**
 * Calculate mapped lines from a source IR mapping to a target panel.
 *
 * @param sourceMappings Source mapping record of the clicked panel.
 * @param lineNumber Clicked line (absolute == physical for IR panels).
 * @param targetTitle Target panel title (determines the IR type via getIRType).
 * @param irStages Trace stage descriptors for dynamic stage discovery.
 * @returns Raw candidate lines (unknown[]: integers plus whatever illegal
 *   entries the trace carried). The caller MUST pipe the result through
 *   normalizeHighlightLines before use.
 */
export function calculateMappedLines(
  sourceMappings: Record<string, SourceMapping> | undefined,
  lineNumber: number,
  targetTitle: string,
  irStages?: IRStageDescriptor[]
): unknown[] {
  const lineKey = lineNumber.toString();
  const sourceMapping = sourceMappings?.[lineKey];
  if (!sourceMapping) return [];

  const targetIRType = getIRType(targetTitle);

  const irTypesToCheck: Array<{ type: string; property: string }> =
    irStages && irStages.length > 0
      ? irStages
          .filter((s) => s.supports_source_mapping)
          .map((s) => ({ type: s.name, property: `${s.name}_lines` }))
      : [
          { type: "ttgir", property: "ttgir_lines" },
          { type: "ttir", property: "ttir_lines" },
          { type: "ptx", property: "ptx_lines" },
          { type: "llir", property: "llir_lines" },
          { type: "amdgcn", property: "amdgcn_lines" },
          { type: "sass", property: "sass_lines" },
        ];

  for (const { type, property } of irTypesToCheck) {
    if (
      targetIRType === type &&
      sourceMapping[property as keyof SourceMapping] !== undefined
    ) {
      const lines = sourceMapping[property as keyof SourceMapping] as unknown;
      // Non-array fields (number/string/object/null) yield no mapping instead
      // of throwing inside .map(); fixed expectations in unit tests.
      if (!Array.isArray(lines)) return [];
      return lines.map((line) =>
        typeof line === "string" ? parseStrictIntString(line) : line
      );
    }
  }

  return [];
}

/**
 * Calculate Python lines (absolute) from an IR source mapping.
 *
 * Keeps the legacy guards (missing mapping/file, empty python code) and the
 * legacy file-match direction (`mapping.file.includes(pythonInfo.file_path)`).
 * Range filtering is intentionally NOT done here: the raw absolute candidate
 * flows into normalizeHighlightLines, which drops out-of-range lines with
 * diagnostics (§4.4.1) instead of the legacy console.error.
 *
 * For inlined code the entry's own `file`/`line` describe the callee, so
 * `inlined_at_file` / `inlined_at_line` are preferred when both are present.
 */
export function calculatePythonLines(
  sourceMapping: Record<string, SourceMapping> | undefined,
  lineNumber: number,
  pythonInfo: PythonPanelInfo
): unknown[] {
  // file_path must be non-empty: includes("") matches every mapping file.
  if (!sourceMapping || !pythonInfo.code || !pythonInfo.file_path) return [];

  const lineKey = lineNumber.toString();
  const mapping = sourceMapping[lineKey];
  if (!mapping) return [];

  // Inlined code -- tl.cdiv, tl.sum, tl.dot -- reports `file`/`line` for the
  // *callee*, a line in a Triton library file rather than in the kernel on
  // screen, so the file check below rejects it and the user's line never
  // highlights. `inlined_at_*` carries the outermost frame of the inline
  // chain, which is the line the user wrote. Both fields are required: one
  // without the other is malformed, and mixing a caller file with a callee
  // line would silently point at the wrong place.
  const inlined =
    mapping.inlined_at_file != null && mapping.inlined_at_line != null;
  const file = inlined ? mapping.inlined_at_file : mapping.file;
  const line = inlined ? mapping.inlined_at_line : mapping.line;
  if (!file || line == null) return [];

  // Legacy match direction preserved: the mapping-side file path contains the
  // panel-side file path (absolute build paths vs. recorded prefixes).
  if (typeof file !== "string" || !file.includes(pythonInfo.file_path)) {
    return [];
  }

  // Same strict rule as calculateMappedLines: integers pass through
  // verbatim, pure-integer strings convert, and everything else (boolean,
  // array, hex/float strings, objects) passes through verbatim so
  // normalizeHighlightLines drops it with diagnostics. Never Number()-coerce:
  // Number(true) === 1 and Number([459]) === 459 forge highlight lines the
  // normalizer can no longer recognize as invalid.
  const raw: unknown = line;
  return [typeof raw === "string" ? parseStrictIntString(raw) : raw];
}

/**
 * Source-qualified kernel identity (§4.2.1/R4): the kernel hash or index only
 * identifies a kernel WITHIN one data source, so the source id is always the
 * first element. Bare index and bare hash are both rejected shapes — callers
 * pass the slot/URL source id explicitly.
 */
export function buildKernelKey(
  sourceId: string,
  kernelHashOrIndex: string | number
): string {
  return JSON.stringify([sourceId, kernelHashOrIndex]);
}

/** Absolute [start, end] function range (inclusive) from trace metadata. */
export interface FunctionRange {
  start: number;
  end: number;
}

/**
 * Intersect a function range with a document window (§4.4.1): ranges fully
 * outside the document yield null (no decorations), partial ranges clip to
 * the visible window. Non-integer bounds yield null — line decorations
 * cannot represent fractional lines.
 */
export function intersectRangeWithDoc(
  range: FunctionRange | undefined,
  doc: HighlightDoc
): FunctionRange | null {
  if (!range) return null;
  if (!Number.isInteger(range.start) || !Number.isInteger(range.end)) {
    return null;
  }
  const first = doc.offset;
  const last = doc.offset + doc.lineCount - 1;
  const start = Math.max(range.start, first);
  const end = Math.min(range.end, last);
  if (start > end) return null;
  return { start, end };
}

/**
 * Physical line count of a document string, matching Monaco's model line
 * count: "" is one (empty) line, "a\n" is two lines, "\n" splits CRLF too.
 */
export function lineCountOfContent(content: string): number {
  if (content === "") return 1;
  return content.split("\n").length;
}
