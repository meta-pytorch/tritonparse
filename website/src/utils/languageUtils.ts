import type { IRStageDescriptor } from "./dataLoader";

/**
 * Maps our internal language names to syntax highlighter languages.
 * When irStages is available, looks up the syntax_id from stage descriptors.
 * Otherwise falls back to hardcoded legacy logic for backward compatibility.
 */
export const mapLanguageToHighlighter = (language: string, irStages?: IRStageDescriptor[]): string => {
  // Try dynamic lookup from ir_stages
  if (irStages && irStages.length > 0) {
    const type = language.split('.').pop()?.toLowerCase() || language.toLowerCase();
    const stage = irStages.find(s => s.name === type);
    if (stage) return stage.syntax_id;
  }

  // Fallback: legacy hardcoded logic
  const lowerCaseLanguage = language.toLowerCase();
  if (lowerCaseLanguage.endsWith("ttgir") || lowerCaseLanguage.endsWith("ttir")) {
    return 'mlir';
  } else if (lowerCaseLanguage.endsWith("llir")) {
    return 'llvm';
  } else if (lowerCaseLanguage.endsWith("ptx")) {
    return 'ptx';
  } else if (lowerCaseLanguage.endsWith("amdgcn")) {
    return 'amdgcn';
  } else if (lowerCaseLanguage.endsWith("sass")) {
    return 'asm';
  } else if (lowerCaseLanguage === "python") {
    return 'python';
  }

  return 'plaintext';
};
