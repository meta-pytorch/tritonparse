/**
 * Committed fixture page for the comparison view (e2e only).
 *
 * Rendered when `?view=comparison_fixture` (see App.tsx); never used by the
 * normal product flows. Mounts the real CodeComparisonViewV2 once and swaps
 * exactly one identity input per button with real <button> elements, so the
 * F18 same-mount counterexamples (content-only, mapping-only, offset-only,
 * file_path-only, sourceId-only) run through genuine React prop updates —
 * never through debug-driven calls. The buttons also cover the python toggle
 * (F15) and the snippet/full-file mode switch (F6/F7/F13 legs).
 *
 * Fixture mapping contract (fixed expectations, also asserted by e2e).
 * Entry shapes mirror real traces: IR entries carry the python `line` +
 * `file` plus their own `<stage>_line` and cross-stage lines; python
 * entries carry cross-stage lines (strings for 451, as produced; numbers
 * for 452 to cover both spellings).
 * - py 451 -> left [10,20], right [30,40]; py 452 -> left [20,30],
 *   right [40,50] (F2 partial-intersection replacement pair).
 * - py 453 has no mapping (F3: only the clicked line lights).
 * - py 454 carries illegal entries: left [9999,"3.5","4oops",12] filters to
 *   [12] with a 3-dropped badge, right has a non-array field -> [] (F19).
 * - left 10 -> right [30] + python [455] (F1 IR leg).
 */
import React, { useMemo, useState } from "react";
import CodeComparisonViewV2 from "./CodeComparisonViewV2";
import {
  IRStageDescriptor,
  PythonSourceCodeInfo,
  SourceMapping,
} from "../utils/dataLoader";

const STAGES: IRStageDescriptor[] = [
  { name: "ttir", extension: ".ttir", display_name: "TTIR", display_order: 10, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "ttgir", extension: ".ttgir", display_name: "TTGIR", display_order: 20, is_text: true, supports_source_mapping: true, syntax_id: "mlir" },
  { name: "llir", extension: ".llir", display_name: "LLIR", display_order: 30, is_text: true, supports_source_mapping: true, syntax_id: "llvm" },
];

/** Padded line so real-mouse clicks land on CONTENT_TEXT across the panel. */
function pad(tag: string, i: number): string {
  return `// ${tag} line ${i} with enough trailing text to keep every line clickable wide ok`;
}

const LEFT_LINES = 200;
const RIGHT_LINES = 200;
const LEFT_CONTENT = Array.from({ length: LEFT_LINES }, (_, i) =>
  pad("fixture-ttgir", i + 1)
).join("\n");
const RIGHT_CONTENT = Array.from({ length: RIGHT_LINES }, (_, i) =>
  pad("fixture-ttir", i + 1)
).join("\n");

const LEFT_MAPPING: Record<string, SourceMapping> = {
  "10": {
    line: 455,
    file: "/src/fixture_model.py",
    ttgir_line: 10,
    ttir_lines: [30],
  },
  "20": {
    line: 456,
    file: "/src/fixture_model.py",
    ttgir_line: 20,
    ttir_lines: [50],
  },
};

const RIGHT_MAPPING: Record<string, SourceMapping> = {
  "30": {
    line: 455,
    file: "/src/fixture_model.py",
    ttir_line: 30,
    ttgir_lines: [10],
  },
  "40": {
    line: 456,
    file: "/src/fixture_model.py",
    ttir_line: 40,
    ttgir_lines: [20],
  },
  "50": {
    line: 456,
    file: "/src/fixture_model.py",
    ttir_line: 50,
    ttgir_lines: [20],
  },
};

const PYTHON_MAPPING: Record<string, SourceMapping> = {
  // Real python entries carry only cross-stage lines; the required `line`
  // field is present for the type but unused by the mapping math.
  "451": { line: 451, ttgir_lines: ["10", "20"], ttir_lines: ["30", "40"] },
  "452": { line: 452, ttgir_lines: [20, 30], ttir_lines: [40, 50] },
  "454": {
    ttgir_lines: [9999, "3.5", "4oops", 12],
    ttir_lines: "not-an-array",
  } as unknown as SourceMapping,
};

// 60 lines so the snippet-mode python panel scrolls: the F18 suite must
// provably scroll python down before asserting the return to 0.
const PY_SNIPPET_LINES = 60;
const PY_SNIPPET_CODE = Array.from({ length: PY_SNIPPET_LINES }, (_, i) =>
  `# fixture python snippet line ${451 + i} padded to keep clicks on text ok`
).join("\n");
const PY_SNIPPET: PythonSourceCodeInfo = {
  file_path: "/src/fixture_model.py",
  start_line: 451,
  code: PY_SNIPPET_CODE,
};

const PY_FULL_LINES = 100;
const PY_FULL_CODE = Array.from({ length: PY_FULL_LINES }, (_, i) =>
  `# fixture python full-file line ${i + 1} padded to keep clicks on text ok`
).join("\n");
const PY_FULL: PythonSourceCodeInfo = {
  file_path: "/src/fixture_model.py",
  start_line: 1,
  code: PY_FULL_CODE,
  function_start_line: 80,
  function_end_line: 85,
};

interface Overrides {
  leftContent?: string;
  leftMapping?: Record<string, SourceMapping>;
  pyInfo?: PythonSourceCodeInfo;
  sourceId?: string;
}

const BASE_SOURCE_ID = "fixture-a";

const ComparisonFixture: React.FC = () => {
  // Base objects are module constants reused by reference; each swap button
  // overrides exactly one slice, so every other identity input keeps its
  // reference (the F18 ref-discipline requirement).
  const [overrides, setOverrides] = useState<Overrides>({});
  const [showPython, setShowPython] = useState(true);
  const [fullFile, setFullFile] = useState(false);

  const pyInfo = useMemo(
    () => overrides.pyInfo ?? (fullFile ? PY_FULL : PY_SNIPPET),
    [overrides.pyInfo, fullFile]
  );
  const leftContent = overrides.leftContent ?? LEFT_CONTENT;
  const leftMapping = overrides.leftMapping ?? LEFT_MAPPING;
  const sourceId = overrides.sourceId ?? BASE_SOURCE_ID;

  const leftPanel = useMemo(
    () => ({
      code: { content: leftContent, source_mapping: leftMapping },
      title: "fixture_kernel.ttgir",
    }),
    [leftContent, leftMapping]
  );
  const rightPanel = useMemo(
    () => ({
      code: { content: RIGHT_CONTENT, source_mapping: RIGHT_MAPPING },
      title: "fixture_kernel.ttir",
    }),
    []
  );

  // Each swap replaces (never merges): every button overrides exactly one
  // slice from the base state, so consecutive swaps cannot accumulate stale
  // overrides from earlier swaps.
  const swap = (patch: Overrides) => setOverrides({ ...patch });

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          padding: "8px 12px",
          borderBottom: "1px solid #e5e7eb",
          display: "flex",
          gap: "8px",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <h1 style={{ fontSize: "14px", fontWeight: 600, marginRight: "8px" }}>
          Comparison fixture (e2e only)
        </h1>
        <button
          type="button"
          data-testid="fixture-content-only"
          onClick={() => swap({ leftContent: `${LEFT_CONTENT}\n// swapped` })}
        >
          content-only
        </button>
        <button
          type="button"
          data-testid="fixture-mapping-only"
          onClick={() =>
            swap({
              leftMapping: {
                ...LEFT_MAPPING,
                "11": { line: 457, file: "/src/fixture_model.py", ttgir_line: 11, ttir_lines: [31] },
              },
            })
          }
        >
          mapping-only
        </button>
        <button
          type="button"
          data-testid="fixture-offset-only"
          onClick={() =>
            swap({ pyInfo: { ...PY_SNIPPET, start_line: 452 } })
          }
        >
          offset-only
        </button>
        <button
          type="button"
          data-testid="fixture-file-path-only"
          onClick={() =>
            swap({
              pyInfo: {
                ...PY_SNIPPET,
                file_path: "/src/fixture_model_renamed.py",
              },
            })
          }
        >
          file-path-only
        </button>
        <button
          type="button"
          data-testid="fixture-source-id-only"
          onClick={() => swap({ sourceId: "fixture-b" })}
        >
          source-id-only
        </button>
        <button
          type="button"
          data-testid="fixture-python-toggle"
          onClick={() => setShowPython((v) => !v)}
        >
          python-toggle
        </button>
        <button
          type="button"
          data-testid="fixture-python-mode"
          onClick={() => setFullFile((v) => !v)}
        >
          python-mode
        </button>
        <button
          type="button"
          data-testid="fixture-reset"
          onClick={() => {
            setOverrides({});
            setShowPython(true);
            setFullFile(false);
          }}
        >
          reset
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <CodeComparisonViewV2
          leftPanel={leftPanel}
          rightPanel={rightPanel}
          py_code_info={pyInfo}
          showPythonSource={showPython}
          pythonMapping={PYTHON_MAPPING}
          irStages={STAGES}
          sourceId={sourceId}
          kernelId="k0"
        />
      </div>
    </div>
  );
};

export default ComparisonFixture;
