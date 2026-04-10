/**
 * Maps our internal language names to syntax highlighter languages
 * @param language Internal language identifier
 * @returns Syntax highlighter language identifier
 */
export const mapLanguageToHighlighter = (language: string): string => {
  const lowerCaseLanguage = language.toLowerCase();

  // Handle language types with endsWith for better accuracy
  if (lowerCaseLanguage.endsWith("ttgir") || lowerCaseLanguage.endsWith("ttir")) {
    return 'mlir';
  } else if (lowerCaseLanguage.endsWith("llir")) {
    return 'llvm';
  } else if (lowerCaseLanguage.endsWith("ptx")) {
    return 'ptx';
  } else if (lowerCaseLanguage.endsWith("amdgcn")) {
    return 'amdgcn';
  } else if (lowerCaseLanguage.endsWith("sass")) {
    return 'asm';  // SASS is NVIDIA assembly, use generic asm highlighting
  } else if (lowerCaseLanguage === "python") {
    return 'python';
  }

  return 'plaintext';
};
