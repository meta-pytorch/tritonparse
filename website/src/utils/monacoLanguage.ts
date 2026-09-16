import type { IRStageDescriptor } from "./dataLoader";

/**
 * Monaco language ids provided by the IR migration (§4.6).
 * Every id returned by mapFileToMonacoLanguage must be registered
 * (custom Monarch grammars) or a Monaco builtin used as-is.
 */
export const MONACO_LANGUAGE_IDS = {
  mlir: "triton-mlir",
  llvm: "triton-llvm",
  ptx: "triton-ptx",
  asm: "triton-asm",
  python: "python",
  plaintext: "plaintext",
} as const;

/**
 * syntax_id (trace-declared, Prism-oriented) -> Monaco language id.
 * json/c intentionally map to plaintext: the JSON language contribution is
 * not imported (it would enable diagnostics plus a dedicated worker),
 * and C has no proven legacy coloring behavior (§4.6).
 *
 * A Map: indexing a plain object with a trace-controlled key would
 * read inherited properties ("constructor"/"toString" return functions,
 * "__proto__" returns an object), skipping the plaintext fallback.
 */
const SYNTAX_ID_TO_MONACO: ReadonlyMap<string, string> = new Map([
  ["mlir", MONACO_LANGUAGE_IDS.mlir],
  ["llvm", MONACO_LANGUAGE_IDS.llvm],
  ["ptx", MONACO_LANGUAGE_IDS.ptx],
  ["amdgcn", MONACO_LANGUAGE_IDS.asm],
  ["asm", MONACO_LANGUAGE_IDS.asm],
  ["python", MONACO_LANGUAGE_IDS.python],
  ["json", MONACO_LANGUAGE_IDS.plaintext],
  ["c", MONACO_LANGUAGE_IDS.plaintext],
]);

/**
 * Resolve a Monaco language id from the raw filename (R6: never reverse-map
 * from the display name, which is not invertible for custom display_names).
 *
 * Lookup order: ir_stages syntax_id first, then the extension fallback.
 * Unknown ids fall back to plaintext with a warning.
 */
export function mapFileToMonacoLanguage(
  filename: string,
  irStages?: IRStageDescriptor[]
): string {
  if (irStages && irStages.length > 0) {
    const type = filename.split(".").pop()?.toLowerCase() || filename.toLowerCase();
    const stage = irStages.find((s) => s.name === type);
    if (stage) {
      const mapped = SYNTAX_ID_TO_MONACO.get(stage.syntax_id);
      if (mapped) {
        return mapped;
      }
      console.warn(
        `[monaco] unknown syntax_id "${stage.syntax_id}" for "${filename}", falling back to plaintext`
      );
      return MONACO_LANGUAGE_IDS.plaintext;
    }
  }

  const lower = filename.toLowerCase();
  if (lower.endsWith("ttgir") || lower.endsWith("ttir")) {
    return MONACO_LANGUAGE_IDS.mlir;
  } else if (lower.endsWith("llir")) {
    return MONACO_LANGUAGE_IDS.llvm;
  } else if (lower.endsWith("ptx")) {
    return MONACO_LANGUAGE_IDS.ptx;
  } else if (lower.endsWith("amdgcn") || lower.endsWith("sass")) {
    return MONACO_LANGUAGE_IDS.asm;
  } else if (lower === "python") {
    return MONACO_LANGUAGE_IDS.python;
  }
  return MONACO_LANGUAGE_IDS.plaintext;
}
