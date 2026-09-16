/**
 * First-version Monarch grammars for the Triton IR dialects (§4.6).
 *
 * Bar for the spike: not worse than plaintext, with keyword/value/type/
 * comment/string distinguishable. Per-language fallback to plaintext stays
 * available if a grammar misbehaves (C1 rule). These definitions are pure
 * data; registration happens in registerIrLanguages.ts.
 */
import type { languages } from "monaco-editor";

const commonBrackets: languages.IMonarchLanguage["brackets"] = [
  { open: "{", close: "}", token: "delimiter.curly" },
  { open: "[", close: "]", token: "delimiter.square" },
  { open: "(", close: ")", token: "delimiter.parenthesis" },
  { open: "<", close: ">", token: "delimiter.angle" },
];

const doubleQuotedString: languages.IMonarchLanguage["tokenizer"] = {
  string: [
    [/[^\\"]+/, "string"],
    [/\\./, "string.escape"],
    [/"/, "string", "@pop"],
  ],
};

/** MLIR dialects: ttgir/ttir (§4.6: %ssa, #alias, @sym, op.name, !type, loc strings). */
export const tritonMlirLanguage: languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".mlir",
  brackets: commonBrackets,
  keywords: ["module", "func", "return", "true", "false"],
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [/"/, "string", "@string"],
      [/%[A-Za-z0-9_$.]+/, "variable"],
      [/@[A-Za-z0-9_$.]+/, "type"],
      [/#[A-Za-z0-9_$.]+/, "type"],
      [/\^[A-Za-z0-9_.]+/, "type"],
      [/![A-Za-z0-9_.]+/, "type"],
      [/0[xX][0-9a-fA-F]+/, "number.hex"],
      [/\d+(\.\d+)?([eE][+-]?\d+)?/, "number"],
      // Dotted op names (tt.load, arith.constant) before plain identifiers.
      [/[A-Za-z_][\w$]*(\.[\w$]+)+/, "keyword"],
      [/[A-Za-z_][\w$]*/, { cases: { "@keywords": "keyword", "@default": "identifier" } }],
      [/[{}()[\]<>]/, "@brackets"],
      [/[=,;:]/, "delimiter"],
    ],
    ...doubleQuotedString,
  },
};

/** LLVM IR: llir (§4.6: comments, strings, numbers, %/@ values, labels). */
export const tritonLlvmLanguage: languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".llvm",
  brackets: commonBrackets,
  keywords: [
    "define", "declare", "ret", "br", "call", "load", "store", "alloca",
    "getelementptr", "phi", "select", "icmp", "fcmp", "add", "sub", "mul",
    "shl", "lshr", "ashr", "and", "or", "xor", "sext", "zext", "trunc",
    "bitcast", "unreachable", "resume", "invoke", "landingpad", "private",
    "internal", "unnamed_addr", "constant", "global", "align", "tail",
    "fast", "arcp", "contract", "afn", "reassoc", "ninf", "nnan", "nsz",
  ],
  typeKeywords: [
    "i1", "i8", "i16", "i32", "i64", "float", "double", "half", "bfloat",
    "x86_fp80", "ptr", "void", "label", "token", "metadata",
  ],
  tokenizer: {
    root: [
      [/;.*$/, "comment"],
      [/"/, "string", "@string"],
      [/%[A-Za-z0-9_$.]+/, "variable"],
      [/@[A-Za-z0-9_$.]+/, "type"],
      [/![A-Za-z0-9]+/, "type"],
      [/0[xX][0-9a-fA-F]+/, "number.hex"],
      [/-?\d+(\.\d+)?([eE][+-]?\d+)?/, "number"],
      [/[A-Za-z_][\w$.]*:/, "type"],
      [
        /[A-Za-z][\w$]*/,
        { cases: { "@keywords": "keyword", "@typeKeywords": "type", "@default": "identifier" } },
      ],
      [/[{}()[\]<>]/, "@brackets"],
      [/[=,*]/, "delimiter"],
    ],
    ...doubleQuotedString,
  },
};

const ptxMnemonics = [
  "abs", "activemask", "add", "and", "atom", "bar", "bfind", "bra", "brkpt",
  "brev", "brx", "call", "clz", "cos", "cp", "cvt", "cvta", "div", "elect",
  "ex2", "exit", "fma", "fns", "getctarank", "isspacep", "ld", "lg2", "lop3",
  "mad", "mad24", "mapa", "match", "max", "mbarrier", "min", "mma", "mov",
  "movmatrix", "mul", "mul24", "nanosleep", "neg", "not", "or", "pmevent",
  "popc", "prmt", "rcp", "redux", "rem", "ret", "rsqrt", "sad", "selp",
  "set", "setp", "shfl", "shl", "shr", "sin", "sqrt",
  "st", "sub", "suld", "suq", "sust", "tensormap", "testp", "tex", "trap",
  "txq", "vabsdiff", "vote", "wgmma", "xor",
];

/** PTX: line-oriented directives, predicates, addresses (§4.6). */
export const tritonPtxLanguage: languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".ptx",
  brackets: commonBrackets,
  keywords: ptxMnemonics,
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [/\/\*/, "comment", "@blockComment"],
      [/"/, "string", "@string"],
      // Directives (.target, .reg, .param, .f32, ...) before identifiers.
      [/\.\w[\w.]*/, "keyword"],
      [/%[\w$.]+/, "variable"],
      // Mnemonic prefixes (ld.global, st.shared, ...) before identifiers.
      [
        new RegExp(`\\b(?:${ptxMnemonics.join("|")})(?=\\.|\\s|;|$)`),
        "keyword",
      ],
      [/0[xX][0-9a-fA-F]+/, "number.hex"],
      [/-?\d+(\.\d+)?([eE][+-]?\d+)?/, "number"],
      [/[A-Za-z_][\w$.]*:/, "type"],
      [/[A-Za-z_][\w$.]*/, { cases: { "@keywords": "keyword", "@default": "identifier" } }],
      [/[{}()[\]<>]/, "@brackets"],
      [/[=,;]/, "delimiter"],
    ],
    blockComment: [
      [/[^/*]+/, "comment"],
      [/\*\//, "comment", "@pop"],
      [/[/*]/, "comment"],
    ],
    ...doubleQuotedString,
  },
};

/** Generic asm: amdgcn/sass/cubin disassembly text (§4.6). */
export const tritonAsmLanguage: languages.IMonarchLanguage = {
  defaultToken: "",
  tokenPostfix: ".asm",
  brackets: commonBrackets,
  keywords: [
    "mov", "add", "sub", "mul", "mad", "ld", "st", "nop", "ret", "bra",
    "call", "bar", "sync", "exit", "trap", "setp", "selp", "shl", "shr",
  ],
  tokenizer: {
    root: [
      [/\/\/.*$/, "comment"],
      [/;.*$/, "comment"],
      [/#.*$/, "comment"],
      [/\/\*/, "comment", "@blockComment"],
      [/"/, "string", "@string"],
      // AMDGCN scalar/vector regs, SASS regs/uniform/special/predicates.
      [/\b[sv]\d+\b/, "variable"],
      [/\bR\d+\b/, "variable"],
      [/\bUR\d+\b/, "variable"],
      [/\bSR_\w+\b/, "variable"],
      [/\bP\d+\b/, "variable"],
      [/0[xX][0-9a-fA-F]+/, "number.hex"],
      [/-?\d+(\.\d+)?([eE][+-]?\d+)?/, "number"],
      [/[A-Za-z_][\w$.]*:/, "type"],
      [/[A-Za-z_][\w$.]*/, { cases: { "@keywords": "keyword", "@default": "identifier" } }],
      [/[{}()[\]<>]/, "@brackets"],
      [/[=,]/, "delimiter"],
    ],
    blockComment: [
      [/[^/*]+/, "comment"],
      [/\*\//, "comment", "@pop"],
      [/[/*]/, "comment"],
    ],
    ...doubleQuotedString,
  },
};
