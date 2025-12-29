import React, { useCallback, useEffect, useMemo, useState } from "react";
import DiffComparisonView from "../components/DiffComparisonView";
import { useFileDiffSession } from "../context/FileDiffSession";
import { ProcessedKernel, loadLogData, loadLogDataFromFile, processKernelData, getIRType } from "../utils/dataLoader";
import "../types/global.d.ts";

type DiffMode = "single" | "all";

interface FileDiffViewProps {
  kernelsLeft: ProcessedKernel[];
  selectedLeftIndex: number;
  leftLoadedUrl: string | null;
}

const PARAM_VIEW = "view";
const PARAM_JSON_B_URL = "json_b_url";
const PARAM_KERNEL_HASH_A = "kernel_hash_a";
const PARAM_KERNEL_HASH_B = "kernel_hash_b";
const PARAM_MODE = "mode";
const PARAM_IR = "ir";
const PARAM_IGNORE_WS = "ignore_ws";
const PARAM_WORD_LEVEL = "word_level";
const PARAM_CONTEXT = "context";
const PARAM_WRAP = "wrap";
const PARAM_ONLY_CHANGED = "only_changed";

function findKernelIndexByHash(hash: string | null, kernels: ProcessedKernel[]): number {
  if (!hash) return -1;
  const idx = kernels.findIndex(k => k.metadata?.hash === hash);
  return idx >= 0 ? idx : -1;
}

function listIRTypesForKernel(kernel: ProcessedKernel | undefined): Set<string> {
  const s = new Set<string>();
  if (!kernel) return s;
  for (const key of Object.keys(kernel.irFiles || {})) {
    s.add(getIRType(key));
  }
  if (kernel.pythonSourceInfo?.code) s.add("python");
  return s;
}

function getContentByIRType(kernel: ProcessedKernel | undefined, irType: string): string {
  if (!kernel) return "";
  if (irType === "python") {
    return kernel.pythonSourceInfo?.code || "";
  }
  const keys = Object.keys(kernel.irFiles || {});
  const found = keys.find(k => getIRType(k) === irType);
  return found ? kernel.irFiles[found] : "";
}

const FileDiffView: React.FC<FileDiffViewProps> = ({ kernelsLeft, selectedLeftIndex, leftLoadedUrl }) => {
  const sess = useFileDiffSession();
  // Left source state (URL/local) – overrides props when present
  const [leftKernelsFromUrl, setLeftKernelsFromUrl] = useState<ProcessedKernel[]>([]);
  const [leftKernelsFromLocal, setLeftKernelsFromLocal] = useState<ProcessedKernel[]>([]);
  const [leftLoadedUrlLocal, setLeftLoadedUrlLocal] = useState<string | null>(null);
  const [leftLoadedFromLocal, setLeftLoadedFromLocal] = useState<boolean>(false);
  const [loadingLeft, setLoadingLeft] = useState<boolean>(false);
  const [errorLeft, setErrorLeft] = useState<string | null>(null);

  // Right source state
  const [kernelsRight, setKernelsRight] = useState<ProcessedKernel[]>([]);
  const [rightLoadedUrl, setRightLoadedUrl] = useState<string | null>(null);
  const [loadingRight, setLoadingRight] = useState<boolean>(false);
  const [errorRight, setErrorRight] = useState<string | null>(null);

  // Selection state
  const [leftIdx, setLeftIdx] = useState<number>(Math.max(0, selectedLeftIndex));
  const [rightIdx, setRightIdx] = useState<number>(0);
  const [mode, setMode] = useState<DiffMode>("single");
  const [irType, setIrType] = useState<string>("ttgir");

  // Diff options
  const [ignoreWs, setIgnoreWs] = useState<boolean>(true);
  const [wordLevel, setWordLevel] = useState<boolean>(true);
  const [contextLines, setContextLines] = useState<number>(3);
  const [wordWrap, setWordWrap] = useState<"off" | "on">("on");
  const [onlyChanged, setOnlyChanged] = useState<boolean>(false);

  // Read URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);

    // Right URL
    const bUrl = params.get(PARAM_JSON_B_URL);
    if (bUrl) {
      setRightLoadedUrl(bUrl);
    }

    // Left URL – use json_url only (no json_a_url)
    const aUrl = params.get("json_url");
    if (aUrl) {
      setLeftLoadedUrlLocal(aUrl);
    } else if (leftLoadedUrl) {
      // Prop from App: treat as initial left URL
      setLeftLoadedUrlLocal(leftLoadedUrl);
    }

    // Mode and IR
    const modeParam = (params.get(PARAM_MODE) as DiffMode) || undefined;
    if (modeParam === "single" || modeParam === "all") setMode(modeParam);
    const irParam = params.get(PARAM_IR);
    if (irParam) setIrType(irParam);

    // Options
    setIgnoreWs(params.get(PARAM_IGNORE_WS) === "0" ? false : true);
    setWordLevel(params.get(PARAM_WORD_LEVEL) === "1");
    const ctx = parseInt(params.get(PARAM_CONTEXT) || "");
    if (!Number.isNaN(ctx)) setContextLines(ctx);
    const wrap = params.get(PARAM_WRAP);
    if (wrap === "on" || wrap === "off") setWordWrap(wrap);
    setOnlyChanged(params.get(PARAM_ONLY_CHANGED) === "1");

    // Left/Right kernel index by hash
    const leftHash = params.get(PARAM_KERNEL_HASH_A);
    const rightHash = params.get(PARAM_KERNEL_HASH_B);
    if (leftHash) window.__TRITONPARSE_leftHash = leftHash;
    const li = findKernelIndexByHash(leftHash, kernelsLeft);
    if (li >= 0) setLeftIdx(li);

    // right index will be set after right kernels loaded
    if (rightHash) {
      // store temporarily in dataset
      window.__TRITONPARSE_rightHash = rightHash;
    }
    // Note: leftLoadedUrl is intentionally omitted - we only read URL params on mount
    // and when kernelsLeft changes, not when the prop changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kernelsLeft]);

  // Load left URL when set (internal to FileDiffView)
  useEffect(() => {
    async function loadLeft(url: string) {
      try {
        setLoadingLeft(true);
        setErrorLeft(null);
        const entries = await loadLogData(url);
        const processed = processKernelData(entries);
        setLeftKernelsFromUrl(processed);
        setLeftLoadedFromLocal(false);
        sess.setLeftFromUrl(url, processed);
        // default to first kernel when loading new source
        setLeftIdx(0);
          try { sess.setLeftIdx(0); } catch { /* Session may not be ready */ }
        const leftHash = window.__TRITONPARSE_leftHash;
        if (leftHash) {
          const li = findKernelIndexByHash(leftHash, processed);
          if (li >= 0) setLeftIdx(li);
          try { if (li >= 0) sess.setLeftIdx(li); } catch { /* Session may not be ready */ }
        }
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        setErrorLeft(errorMessage);
      } finally {
        setLoadingLeft(false);
      }
    }
    if (leftLoadedUrlLocal) {
      loadLeft(leftLoadedUrlLocal);
    }
    // Note: sess is stable (from context) and doesn't need to trigger re-runs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [leftLoadedUrlLocal]);

  // Load right URL when set
  useEffect(() => {
    async function loadRight(url: string) {
      try {
        setLoadingRight(true);
        setErrorRight(null);
        const entries = await loadLogData(url);
        const processed = processKernelData(entries);
        setKernelsRight(processed);
        sess.setRightFromUrl(url, processed);
        // set right index by hash if present
        const rightHash = window.__TRITONPARSE_rightHash;
        if (rightHash) {
          const ri = findKernelIndexByHash(rightHash, processed);
          if (ri >= 0) setRightIdx(ri);
        }
      } catch (e: unknown) {
        const errorMessage = e instanceof Error ? e.message : String(e);
        setErrorRight(errorMessage);
      } finally {
        setLoadingRight(false);
      }
    }
    if (rightLoadedUrl) {
      loadRight(rightLoadedUrl);
    }
    // Note: sess is stable (from context) and doesn't need to trigger re-runs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rightLoadedUrl]);

  // Hydrate session left from App props when coming from homepage/top bar load (including local file)
  useEffect(() => {
    try {
      const sessLeftLen = sess.left?.kernels?.length || 0;
      if (sessLeftLen === 0 && kernelsLeft.length > 0) {
        if (leftLoadedUrl) {
          sess.setLeftFromUrl(leftLoadedUrl, kernelsLeft);

        } else {
          sess.setLeftFromLocal(kernelsLeft);

        }
        sess.setLeftIdx(Math.max(0, leftIdx));
      }
    } catch { /* Ignore hydration errors - session state may be stale */ }
  }, [sess, kernelsLeft, leftLoadedUrl, leftIdx]);

  // Hydrate right from session when returning from preview (ensure right side is restored without reloading URL)
  useEffect(() => {
    try {
      if (sess.right?.kernels?.length > 0) {
        setKernelsRight(sess.right.kernels);
        setRightIdx(Math.max(0, sess.right.selectedIdx));
        if (sess.right.sourceType === 'url') {
          setRightLoadedUrl(sess.right.url);
          setRightLoadedFromLocal(false);
        } else if (sess.right.sourceType === 'local') {
          setRightLoadedUrl(null);
          setRightLoadedFromLocal(true);
        }
      }
    } catch { /* Ignore session restore errors */ }
  }, [sess.right]);

  // Compute union ir types and choose default ir if needed
  const unionIrTypes = useMemo(() => {
    const leftArray = leftLoadedFromLocal
      ? leftKernelsFromLocal
      : (leftLoadedUrlLocal ? leftKernelsFromUrl : kernelsLeft);
    const left = leftArray[leftIdx];
    const right = kernelsRight[rightIdx];
    const set = new Set<string>();
    listIRTypesForKernel(left).forEach(t => set.add(t));
    listIRTypesForKernel(right).forEach(t => set.add(t));
    if (set.size === 0) return ["python"] as string[];
    return Array.from(set);
  }, [kernelsLeft, kernelsRight, leftIdx, rightIdx, leftLoadedFromLocal, leftKernelsFromLocal, leftLoadedUrlLocal, leftKernelsFromUrl]);

  useEffect(() => {
    if (!irType && unionIrTypes.length > 0) setIrType(unionIrTypes[0]);
  }, [unionIrTypes, irType]);

  // Update URL on state changes (File Diff owns its params)
  const syncUrl = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    params.set(PARAM_VIEW, "file_diff");
    // left always present
    const leftArray = leftLoadedFromLocal
      ? leftKernelsFromLocal
      : (leftLoadedUrlLocal ? leftKernelsFromUrl : kernelsLeft);
    if (leftArray[leftIdx]?.metadata?.hash) params.set(PARAM_KERNEL_HASH_A, String(leftArray[leftIdx].metadata!.hash));
    else params.delete(PARAM_KERNEL_HASH_A);
    // right url/hash
    if (rightLoadedUrl) params.set(PARAM_JSON_B_URL, rightLoadedUrl);
    else params.delete(PARAM_JSON_B_URL);
    if (kernelsRight[rightIdx]?.metadata?.hash) params.set(PARAM_KERNEL_HASH_B, String(kernelsRight[rightIdx].metadata!.hash));
    else params.delete(PARAM_KERNEL_HASH_B);
    // left url
    if (leftLoadedFromLocal) params.delete("json_url");
    else if (leftLoadedUrlLocal) params.set("json_url", leftLoadedUrlLocal);
    else if (leftLoadedUrl) params.set("json_url", leftLoadedUrl); else params.delete("json_url");
    // mode/ir
    params.set(PARAM_MODE, mode);
    if (mode === "single" && irType) params.set(PARAM_IR, irType);
    else params.delete(PARAM_IR);
    // options
    params.set(PARAM_IGNORE_WS, ignoreWs ? "1" : "0");
    params.set(PARAM_WORD_LEVEL, wordLevel ? "1" : "0");
    params.set(PARAM_CONTEXT, String(contextLines));
    params.set(PARAM_WRAP, wordWrap);
    params.set(PARAM_ONLY_CHANGED, onlyChanged ? "1" : "0");
    const newUrl = new URL(window.location.href);
    newUrl.search = params.toString();
    window.history.replaceState({}, "", newUrl.toString());
  }, [kernelsLeft, kernelsRight, leftIdx, rightIdx, rightLoadedUrl, mode, irType, ignoreWs, wordLevel, contextLines, wordWrap, onlyChanged, leftLoadedFromLocal, leftLoadedUrlLocal, leftLoadedUrl, leftKernelsFromLocal, leftKernelsFromUrl]);

  // Debounce URL updates to reduce history churn
  useEffect(() => {
    const id = setTimeout(() => {
      syncUrl();
    }, 200);
    return () => clearTimeout(id);
  }, [syncUrl]);

  // UI builders
  const leftArrayResolved = (sess.left?.kernels?.length || 0) > 0
    ? sess.left.kernels
    : (leftLoadedFromLocal
      ? leftKernelsFromLocal
      : (leftLoadedUrlLocal ? leftKernelsFromUrl : kernelsLeft));
  const leftKernel = leftArrayResolved[leftIdx];
  const rightKernel = kernelsRight[rightIdx];

  // When navigating away, temporarily hide diff editors to avoid Monaco dispose race
  const [hideDiff, setHideDiff] = useState<boolean>(false);
  useEffect(() => {
    // Reset when view state changes
    setHideDiff(false);
  }, [mode, irType, leftIdx, rightIdx]);

  const renderSingle = () => {
    const leftContent = getContentByIRType(leftKernel, irType);
    const rightContent = getContentByIRType(rightKernel, irType);
    const missingLeft = !leftContent;
    const missingRight = !rightContent;
    return (
      <div>
        <div className="flex items-center justify-between mb-3">
          <div className="text-gray-700 font-medium">IR Type: <span className="text-blue-700">{irType}</span></div>
          <div className="text-sm text-gray-500">
            {missingLeft && <span className="mr-2">Left: Not available</span>}
            {missingRight && <span>Right: Not available</span>}
          </div>
        </div>
        {!hideDiff && (
          <DiffComparisonView
            key={`single-${leftIdx}-${rightIdx}-${irType}`}
            leftContent={leftContent}
            rightContent={rightContent}
            height="calc(100vh - 14rem)"
            language={irType === "python" ? "python" : "plaintext"}
            options={{
              ignoreWhitespace: ignoreWs,
              wordLevel,
              context: contextLines,
              wordWrap,
              onlyChanged,
            }}
          />
        )}
      </div>
    );
  };

  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const toggle = (t: string) => setExpanded(prev => ({ ...prev, [t]: !prev[t] }));

  const renderAll = () => {
    return (
      <div className="space-y-4">
        {unionIrTypes.map((t) => {
          const leftContent = getContentByIRType(leftKernel, t);
          const rightContent = getContentByIRType(rightKernel, t);
          const missingLeft = !leftContent;
          const missingRight = !rightContent;
          const isOpen = !!expanded[t];
          return (
            <div key={t} className="border border-gray-200 rounded bg-white">
              <button
                className="w-full text-left px-4 py-3 flex items-center justify-between hover:bg-gray-50"
                onClick={() => toggle(t)}
              >
                <div className="font-medium text-gray-800">{t}</div>
                <div className="text-sm text-gray-500">
                  {missingLeft && <span className="mr-2">Left: N/A</span>}
                  {missingRight && <span>Right: N/A</span>}
                </div>
              </button>
              {isOpen && (
                <div className="px-2 pb-2">
                  {!hideDiff && (
                    <DiffComparisonView
                      key={`all-${t}-${leftIdx}-${rightIdx}`}
                      leftContent={leftContent}
                      rightContent={rightContent}
                      height="calc(100vh - 14rem)"
                      language={t === "python" ? "python" : "plaintext"}
                      options={{
                        ignoreWhitespace: ignoreWs,
                        wordLevel,
                        context: contextLines,
                        wordWrap,
                        onlyChanged,
                      }}
                    />
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  // Left and Right source loaders
  const [pendingUrl, setPendingUrl] = useState<string>("");
  const [rightLoadedFromLocal, setRightLoadedFromLocal] = useState<boolean>(false);
  const handleLoadRight = async () => {
    if (!pendingUrl) return;
    setRightLoadedFromLocal(false);
    setRightLoadedUrl(pendingUrl);
  };

  const handleLoadRightLocal = async (file: File | null) => {
    if (!file) return;
    try {
      setLoadingRight(true);
      setErrorRight(null);
      const entries = await loadLogDataFromFile(file);
      const processed = processKernelData(entries);
      setKernelsRight(processed);
      setRightLoadedFromLocal(true);
      setRightLoadedUrl(null); // do not persist json_b_url when loaded from local file
      sess.setRightFromLocal(processed);
      // select the first kernel or restore by hash if available
      const rightHash = window.__TRITONPARSE_rightHash;
      if (rightHash) {
        const ri = findKernelIndexByHash(rightHash, processed);
        setRightIdx(ri >= 0 ? ri : 0);
      } else {
        setRightIdx(0);
      }
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setErrorRight(errorMessage);
    } finally {
      setLoadingRight(false);
    }
  };
  const [leftPendingUrlLocal, setLeftPendingUrlLocal] = useState<string>("");
  const handleLoadLeft = async () => {
    if (!leftPendingUrlLocal) return;
    setLeftLoadedFromLocal(false);
    setLeftLoadedUrlLocal(leftPendingUrlLocal);
  };
  const handleLoadLeftLocal = async (file: File | null) => {
    if (!file) return;
    try {
      setLoadingLeft(true);
      setErrorLeft(null);
      const entries = await loadLogDataFromFile(file);
      const processed = processKernelData(entries);
      setLeftKernelsFromLocal(processed);
      setLeftLoadedFromLocal(true);
      setLeftLoadedUrlLocal(null);
      sess.setLeftFromLocal(processed);
      // select first by default
      setLeftIdx(0);
      try { sess.setLeftIdx(0); } catch { /* Session may not be ready */ }
      const leftHash = window.__TRITONPARSE_leftHash;
      if (leftHash) {
        const li = findKernelIndexByHash(leftHash, processed);
        setLeftIdx(li >= 0 ? li : 0);
        try { if (li >= 0) sess.setLeftIdx(li); } catch { /* Session may not be ready */ }
      } else {
        setLeftIdx(0);
      }
    } catch (e: unknown) {
      const errorMessage = e instanceof Error ? e.message : String(e);
      setErrorLeft(errorMessage);
    } finally {
      setLoadingLeft(false);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-xl font-semibold text-gray-800 mb-2">File Diff</h1>

      <div className="bg-white rounded-lg p-3 mb-3 border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <div>
            <div className="text-sm text-gray-500 mb-1">Left Source (json_url)</div>
            <div className="flex items-center gap-2 mb-1">
              <input
                type="url"
                placeholder="https://.../trace.ndjson.gz"
                className="flex-1 border border-gray-300 rounded px-3 py-2"
                value={leftPendingUrlLocal}
                onChange={(e) => setLeftPendingUrlLocal(e.target.value)}
              />
              <button
                className="px-3 py-2 bg-blue-600 text-white rounded"
                onClick={handleLoadLeft}
                disabled={loadingLeft}
              >
                {loadingLeft ? "Loading..." : "Load"}
              </button>
            </div>
            <div className="flex items-center gap-2 mb-1">
              <input
                type="file"
                accept=".ndjson,.ndjson.gz,.gz,.jsonl"
                className="block w-full text-sm text-gray-700"
                onChange={(e) => handleLoadLeftLocal(e.target.files?.[0] || null)}
                disabled={loadingLeft}
              />
            </div>
            {(leftLoadedUrlLocal || leftLoadedUrl || (leftLoadedFromLocal || kernelsLeft.length > 0)) && (
              <div className="text-gray-800 break-all mb-2">
                {leftLoadedUrlLocal || leftLoadedUrl || "(from local file)"}
              </div>
            )}
            {leftLoadedFromLocal && (
              <div className="text-gray-600 text-sm mb-2">(loaded from local file)</div>
            )}
            {errorLeft && (
              <div className="text-red-600 text-sm mb-2">{errorLeft}</div>
            )}
            <div className="flex gap-2 mt-1">
              <button
                className="px-2 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border"
                disabled={leftArrayResolved.length === 0}
                onClick={() => { setHideDiff(true); setTimeout(() => sess.gotoOverview('left'), 0); }}
              >
                Left → Kernel Overview
              </button>
              <button
                className="px-2 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border"
                disabled={leftArrayResolved.length === 0}
                onClick={() => { setHideDiff(true); setTimeout(() => sess.gotoIRCode('left'), 0); }}
              >
                Left → IR Code
              </button>
            </div>
          </div>
          <div>
            <div className="text-sm text-gray-500 mb-1">Right Source (json_b_url)</div>
            <div className="flex items-center gap-2 mb-1">
              <input
                type="url"
                placeholder="https://.../trace.ndjson.gz"
                className="flex-1 border border-gray-300 rounded px-3 py-2"
                value={pendingUrl}
                onChange={(e) => setPendingUrl(e.target.value)}
              />
              <button
                className="px-3 py-2 bg-blue-600 text-white rounded"
                onClick={handleLoadRight}
                disabled={loadingRight}
              >
                {loadingRight ? "Loading..." : "Load"}
              </button>
            </div>
            <div className="flex items-center gap-2 mb-1">
              <input
                type="file"
                accept=".ndjson,.ndjson.gz,.gz,.jsonl"
                className="block w-full text-sm text-gray-700"
                onChange={(e) => handleLoadRightLocal(e.target.files?.[0] || null)}
                disabled={loadingRight}
              />
            </div>
            {rightLoadedUrl && (
              <div className="text-gray-800 break-all mb-2">{rightLoadedUrl}</div>
            )}
            {rightLoadedFromLocal && (
              <div className="text-gray-600 text-sm mb-2">(loaded from local file)</div>
            )}
            {errorRight && (
              <div className="text-red-600 text-sm mb-2">{errorRight}</div>
            )}
            {/* Right kernel select moved to aligned row below */}
            <div className="flex gap-2 mt-1">
              <button
                className="px-2 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border"
                disabled={kernelsRight.length === 0}
                onClick={() => { setHideDiff(true); setTimeout(() => sess.gotoOverview('right'), 0); }}
              >
                Right → Kernel Overview
              </button>
              <button
                className="px-2 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded border"
                disabled={kernelsRight.length === 0}
                onClick={() => { setHideDiff(true); setTimeout(() => sess.gotoIRCode('right'), 0); }}
              >
                Right → IR Code
              </button>
            </div>
          </div>
        </div>

        {/* Aligned kernel selectors row */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Left Kernel</label>
            <select
              className="border border-gray-300 rounded px-3 py-2 bg-white w-full"
              value={leftIdx}
              onChange={(e) => { const v = parseInt(e.target.value); setLeftIdx(v); try { sess.setLeftIdx(v); } catch { /* Ignore */ } }}
              disabled={leftArrayResolved.length === 0}
            >
              {leftArrayResolved.map((k, i) => (
                <option key={`l-${i}`} value={i}>
                  {k.name} {(k.metadata?.hash || "").slice(0, 8)}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Right Kernel</label>
            <select
              className="border border-gray-300 rounded px-3 py-2 bg-white w-full"
              value={rightIdx}
              onChange={(e) => { const v = parseInt(e.target.value); setRightIdx(v); try { sess.setRightIdx(v); } catch { /* Ignore */ } }}
              disabled={kernelsRight.length === 0}
            >
              {kernelsRight.map((k, i) => (
                <option key={`r-${i}`} value={i}>
                  {k.name} {(k.metadata?.hash || "").slice(0, 8)}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Mode</label>
            <div className="flex items-center gap-2">
              <button className={`px-3 py-1 rounded ${mode === "single" ? "bg-blue-600 text-white" : "bg-gray-100"}`} onClick={() => setMode("single")}>Single IR</button>
              <button className={`px-3 py-1 rounded ${mode === "all" ? "bg-blue-600 text-white" : "bg-gray-100"}`} onClick={() => setMode("all")}>All IRs</button>
            </div>
          </div>
          {mode === "single" && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">IR Type</label>
              <select
                className="border border-gray-300 rounded px-3 py-2 bg-white w-full"
                value={irType}
                onChange={(e) => setIrType(e.target.value)}
              >
                {unionIrTypes.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          )}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Diff Options</label>
            <div className="flex flex-wrap gap-2">
              <label className="inline-flex items-center gap-1 text-sm">
                <input type="checkbox" checked={ignoreWs} onChange={(e) => setIgnoreWs(e.target.checked)} />
                Ignore whitespace
              </label>
              <label className="inline-flex items-center gap-1 text-sm">
                <input type="checkbox" checked={onlyChanged} onChange={(e) => setOnlyChanged(e.target.checked)} />
                Only changes
              </label>
              <label className="inline-flex items-center gap-1 text-sm">
                <input type="checkbox" checked={wordLevel} onChange={(e) => setWordLevel(e.target.checked)} />
                Word-level
              </label>
              <label className="inline-flex items-center gap-1 text-sm">
                <span>Context</span>
                <input type="number" className="w-16 border border-gray-300 rounded px-2 py-1" value={contextLines} onChange={(e) => setContextLines(parseInt(e.target.value) || 0)} />
              </label>
              <label className="inline-flex items-center gap-1 text-sm">
                <span>Wrap</span>
                <select className="border border-gray-300 rounded px-2 py-1" value={wordWrap} onChange={(e) => setWordWrap(e.target.value as "off" | "on")}>
                  <option value="off">off</option>
                  <option value="on">on</option>
                </select>
              </label>
            </div>
          </div>
        </div>
      </div>

      {mode === "single" ? renderSingle() : renderAll()}
    </div>
  );
};

export default FileDiffView;
