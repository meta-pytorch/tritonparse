/**
 * Committed first-frame observer for the p1-first-frame perf scenario (I010).
 *
 * Installed via Page.addScriptToEvaluateOnNewDocument, so it runs before any
 * page script. A passive rAF sampler: every frame records {rafTimestamp,
 * observedAt, editor-loading placeholder visible, content rows painted, debug
 * panels registered}. observedAt is performance.now() read inside the
 * callback; the rAF timestamp argument is stored for correlation only.
 *
 * Design branch (P1 gate line 533, O2 resolved DELETE): the CodeView
 * deferred-mount placeholder was deleted after the hot-open gates passed, so
 * first-frame evidence is the content-first-frame time. The per-panel
 * Editor.loading placeholder is kept as an init placeholder per R1, but per
 * R7 it cannot prove visible frames during the synchronous model work, so it
 * is recorded informatively only and never asserted as pre-work evidence.
 */
(() => {
  if (window.__P1FF2) throw Error("A first-frame observation is already installed.");
  const S = (window.__P1FF2 = {
    done: false, rafCount: 0, frames: [],
  });
  const rec = (rafTimestamp) => {
    if (S.done) return;
    S.rafCount++;
    const ph = [...document.querySelectorAll(".mp-panel-loading")].some((n) => {
      const r = n.getBoundingClientRect(); return r.width > 0 && r.height > 0;
    });
    S.frames.push({
      rafTimestamp,
      observedAt: performance.now(),
      ph: ph ? 1 : 0,
      rows: document.querySelectorAll(".view-lines .view-line").length,
      panels: Object.keys(window.__TRITONPARSE_DEBUG?.panels ?? {}).length,
    });
    if (S.rafCount < 20000) requestAnimationFrame(rec);
  };
  requestAnimationFrame(rec);
})();
