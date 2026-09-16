/**
 * Committed fixture page for the Single view (Phase 3 e2e, I013).
 *
 * Rendered when `?view=single_fixture` (see App.tsx); never used by the
 * normal product flows. Mounts the real SingleMonacoViewer once and swaps
 * exactly one identity input per button with real <button> elements, so the
 * Single F18 same-mount counterexamples (source-only, content-only,
 * mapping-only) run through genuine React prop updates — never through
 * debug-driven calls. Mirrors ComparisonFixture's ref discipline: base
 * objects are module constants reused by reference; each swap button
 * overrides exactly one slice.
 *
 * Fixture mapping contract (fixed expectations, also asserted by e2e).
 * Mirrors single-basic.ndjson anchor semantics (ttgir anchor):
 * - click 2 -> [2,4] with a 4-dropped badge (999 out-of-range; locX, 3.5,
 *   4oops invalid). Click 6 -> [6]; click 8 -> [8] (anchor 100 dropped).
 * - Lines without mapping light only themselves.
 */
import React, { useMemo, useState } from "react";
import SingleMonacoViewer from "./SingleMonacoViewer";
import {
  IRStageDescriptor,
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

const CONTENT_LINES = 60;
const BASE_CONTENT = Array.from({ length: CONTENT_LINES }, (_, i) =>
  pad("fixture-ttgir", i + 1)
).join("\n");

const BASE_MAPPING: Record<string, SourceMapping> = {
  "2": { line: 2, ttgir_line: 2 },
  "4": { line: 4, ttgir_line: 2 },
  "6": { line: 6, ttgir_line: 6 },
  "8": { line: 8, ttgir_line: 100 },
  "999": { line: 999, ttgir_line: 2 },
  "locX": { line: 99, ttgir_line: 2 },
  "3.5": { line: 3, ttgir_line: 2 },
  "4oops": { line: 4, ttgir_line: 2 },
};

interface Overrides {
  content?: string;
  mapping?: Record<string, SourceMapping>;
  sourceId?: string;
}

const BASE_SOURCE_ID = "fixture-a";
const TITLE = "fixture_kernel.ttgir";
const KERNEL_ID = "k0";

const SingleFixture: React.FC = () => {
  const [overrides, setOverrides] = useState<Overrides>({});
  // Unrelated state: flipping it forces an ordinary same-mount rerender
  // with all identity inputs untouched (retention control).
  const [touched, setTouched] = useState(false);

  const content = overrides.content ?? BASE_CONTENT;
  const mapping = overrides.mapping ?? BASE_MAPPING;
  const sourceId = overrides.sourceId ?? BASE_SOURCE_ID;

  // The irFile wrapper is rebuilt when its slices change, but the doc token
  // depends on the inner content/mapping references (not the wrapper), so
  // untouched slices keep their identity — same as App's wrapper.
  const irFile = useMemo(
    () => ({ content, source_mapping: mapping }),
    [content, mapping]
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
          Single fixture (e2e only){touched ? " (touched)" : ""}
        </h1>
        <button
          type="button"
          data-testid="fixture-single-source-only"
          onClick={() => swap({ sourceId: "fixture-b" })}
        >
          source-only
        </button>
        <button
          type="button"
          data-testid="fixture-single-content-only"
          onClick={() => swap({ content: `${BASE_CONTENT}\n// swapped` })}
        >
          content-only
        </button>
        <button
          type="button"
          data-testid="fixture-single-mapping-only"
          onClick={() =>
            swap({
              mapping: {
                ...BASE_MAPPING,
                "6": { line: 6, ttgir_line: 7 },
              },
            })
          }
        >
          mapping-only
        </button>
        <button
          type="button"
          data-testid="fixture-single-touch"
          onClick={() => setTouched((v) => !v)}
        >
          touch
        </button>
        <button
          type="button"
          data-testid="fixture-single-reset"
          onClick={() => {
            setOverrides({});
            setTouched(false);
          }}
        >
          reset
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <SingleMonacoViewer
          irFile={irFile}
          title={TITLE}
          irStages={STAGES}
          sourceId={sourceId}
          kernelId={KERNEL_ID}
        />
      </div>
    </div>
  );
};

export default SingleFixture;
