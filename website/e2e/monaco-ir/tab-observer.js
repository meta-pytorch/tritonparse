/**
 * Committed passive tab-settle observer for the p1-tabs perf scenario.
 *
 * Installed once per page via Runtime.evaluate; each timed switch arms it with
 * explicit expected panels, sends ONE real mouse click, then awaits
 * completion() with awaitPromise. No CDP traffic runs inside the timed window
 * except the click itself. All editor/model access is read-only (no reveal,
 * scroll, layout, or model-mutating calls).
 *
 * Boundary (design §6.2 tab settle):
 * - inputAt: performance.now() captured in-page on trusted mousedown/keydown.
 * - match per expected panel: editor present + VISIBLE (rect > 20px and inside
 *   the viewport; hidden keep-alive DOM with 0x0 rects never matches) +
 *   rendered text present + every explicitly expected field equal (lineCount,
 *   characters, editorId, modelId, scrollTop, scrollLeft, highlights,
 *   physicalLines). Unspecified fields are not compared.
 * - eligible: all panels matched AND geometry stable across two consecutive
 *   frames (id, width, height, scrollTop, scrollLeft, editorId, modelId).
 * - finish: the second consecutive eligible frame ("matched-and-next-frame").
 *   durationMs = frameAfterMatchAt - inputAt, both performance.now() values
 *   recorded inside the page callbacks (the rAF timestamp argument is stored
 *   for correlation only and never used as an observation time).
 *
 * Kinds: 'tab' (return to an already-mounted view; arming throws only when
 * every expected panel is already visible, and full document/retention
 * fields are compared) and
 * 'tab-mount' (first mount; no panel may exist when armed, and only
 * visibility + rendered text + stability are required since IDs are unknowable
 * before creation; the driver records the post-mount inventory separately).
 */
(() => {
  const key = "__PERF_TABS";
  if (window[key]) throw Error("A tab observation is already installed.");
  let active = null;
  let raf = null;
  let timer = null;
  const history = [];
  const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const panel = (id) => {
    const p = window.__TRITONPARSE_DEBUG?.panels?.[id];
    const e = p?.editor;
    const m = e?.getModel();
    if (!e || !m) return { id, present: false, visible: false };
    const node = e.getDomNode();
    const r = node?.getBoundingClientRect();
    const visible = !!r && r.width > 20 && r.height > 20 && r.right > 0 && r.bottom > 0 && r.left < innerWidth && r.top < innerHeight;
    const textRows = visible ? [...node.querySelectorAll(".view-lines .view-line")] : [];
    return {
      id, present: true, visible,
      editorId: e.getId(), modelId: m.id, modelUri: m.uri.toString(),
      lineCount: m.getLineCount(), characters: m.getValueLength(),
      width: r?.width ?? 0, height: r?.height ?? 0,
      scrollTop: e.getScrollTop(), scrollLeft: e.getScrollLeft(),
      hasRenderedText: textRows.some((row) => row.textContent.length > 0),
      highlights: p.getHighlights().slice(),
      decorations: m.getAllDecorations().filter((d) => d.options.className === "mp-highlighted-line").map((d) => [d.range.startLineNumber, d.range.endLineNumber]),
      renderedHighlights: visible ? node.querySelectorAll(".mp-highlighted-line").length : 0,
    };
  };
  // Sorted: getAllDecorations() order is not a contract, so the retention
  // comparison must not depend on decoration iteration order.
  const physicalLines = (spans) =>
    spans.flatMap(([start, end]) => Array.from({ length: end - start + 1 }, (_, i) => start + i)).sort((a, b) => a - b);
  const matches = (point, wanted) => {
    if (!point.present || !point.visible || !point.hasRenderedText) return false;
    for (const field of ["lineCount", "characters", "editorId", "modelId", "scrollTop", "scrollLeft"]) {
      if (wanted[field] !== undefined && wanted[field] !== point[field]) return false;
    }
    if (wanted.highlights !== undefined && !eq(wanted.highlights, point.highlights)) return false;
    if (wanted.physicalLines !== undefined && !eq([...wanted.physicalLines].sort((a, b) => a - b), physicalLines(point.decorations))) return false;
    return true;
  };
  function finish(reason) {
    if (!active) return;
    if (raf !== null) cancelAnimationFrame(raf);
    if (timer !== null) clearTimeout(timer);
    const run = active;
    const result = {
      label: run.settings.label, kind: run.settings.kind, inputAt: run.inputAt,
      input: run.input, finishedAt: performance.now(), reason,
      firstMatchAt: run.firstMatchAt, frameAfterMatchAt: run.frameAfterMatchAt,
      durationMs: run.frameAfterMatchAt === null || run.inputAt === null ? null : run.frameAfterMatchAt - run.inputAt,
      settings: run.settings, initial: run.initial, frames: run.frames,
    };
    history.push(result); active = null; raf = null; timer = null; run.resolve(result);
  }
  function frame(rafTimestamp) {
    if (!active) return;
    const run = active;
    if (run.inputAt === null) { raf = requestAnimationFrame(frame); return; }
    const points = run.settings.panels.map((wanted) => panel(wanted.id));
    const matched = points.every((p, i) => matches(p, run.settings.panels[i]));
    const geometry = points.map((p) => [p.id, p.width, p.height, p.scrollTop, p.scrollLeft, p.editorId, p.modelId]);
    const stable = run.previousGeometry !== null && eq(geometry, run.previousGeometry);
    const observedAt = performance.now();
    run.frames.push({ rafTimestamp, observedAt, matched, stable, panels: points });
    run.previousGeometry = geometry;
    const eligible = matched && stable;
    if (eligible) {
      if (run.firstMatchAt === null) run.firstMatchAt = observedAt;
      else {
        run.frameAfterMatchAt = observedAt;
        finish("matched-and-next-frame"); return;
      }
    } else run.firstMatchAt = null;
    if (run.frames.length >= (run.settings.maxFrames ?? 1800)) { finish("frame-limit"); return; }
    raf = requestAnimationFrame(frame);
  }
  const input = (event) => {
    if (!active || active.inputAt !== null) return;
    active.inputAt = performance.now();
    active.input = { type: event.type, button: event.button ?? null, key: event.key ?? null };
  };
  for (const type of ["mousedown", "keydown"]) window.addEventListener(type, input, { capture: true, passive: true });
  window[key] = {
    arm(settings) {
      if (active) throw Error("Finish the current observation first.");
      if (!["tab", "tab-mount"].includes(settings.kind)) throw Error("Unknown tab measurement kind.");
      if (!settings.panels?.length) throw Error("Explicit expected panels are required.");
      if (document.visibilityState !== "visible") throw Error("The target tab must be visible.");
      const initial = settings.panels.map((p) => panel(p.id));
      if (settings.kind === "tab" && initial.every((p) => p.visible)) {
        throw Error("The expected view is already visible; a return requires hidden panels.");
      }
      if (settings.kind === "tab-mount" && initial.some((p) => p.present)) {
        throw Error("A mount requires absent panels; this view is already mounted.");
      }
      let resolve;
      const promise = new Promise((r) => { resolve = r; });
      active = {
        settings, initial, resolve, promise, frames: [], previousGeometry: null,
        inputAt: null, input: null, firstMatchAt: null, frameAfterMatchAt: null,
      };
      timer = setTimeout(() => finish("timeout"), settings.timeoutMs ?? 90000);
      raf = requestAnimationFrame(frame);
      return { armed: true, initial };
    },
    completion() {
      if (active) return active.promise;
      if (history.length) return Promise.resolve(history.at(-1));
      throw Error("No observation has started.");
    },
    read() { return { active: active !== null, history }; },
    cancel() { finish("cancelled"); },
    dispose() {
      finish("disposed");
      for (const type of ["mousedown", "keydown"]) window.removeEventListener(type, input, { capture: true });
      delete window[key]; return history;
    },
  };
})();
