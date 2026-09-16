/**
 * One-time registration of the Triton IR Monarch languages (§4.6).
 *
 * Called from the panel's `beforeMount` (runs before the first editor is
 * created); the module-level guard makes repeat calls no-ops. Only the four
 * custom ids are registered here — `python` is a Monaco builtin and
 * `plaintext` needs nothing.
 */
import type * as monaco from "monaco-editor";
import { MONACO_LANGUAGE_IDS } from "../../utils/monacoLanguage";
import {
  tritonAsmLanguage,
  tritonLlvmLanguage,
  tritonMlirLanguage,
  tritonPtxLanguage,
} from "./irGrammars";

const registeredIds = new Set<string>();

export function registerIrLanguages(m: typeof monaco): void {
  const defs: Array<[string, monaco.languages.IMonarchLanguage]> = [
    [MONACO_LANGUAGE_IDS.mlir, tritonMlirLanguage],
    [MONACO_LANGUAGE_IDS.llvm, tritonLlvmLanguage],
    [MONACO_LANGUAGE_IDS.ptx, tritonPtxLanguage],
    [MONACO_LANGUAGE_IDS.asm, tritonAsmLanguage],
  ];
  for (const [id, def] of defs) {
    if (registeredIds.has(id)) {
      continue;
    }
    registeredIds.add(id);
    m.languages.register({ id });
    m.languages.setMonarchTokensProvider(id, def);
  }
}
