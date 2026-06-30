import React, { useMemo, useState } from "react";
import type { RooflineData, RooflinePerLaunch } from "../utils/dataLoader";

/**
 * Roofline UI. Roofline is a PER-LAUNCH quantity (one compiled kernel is launched
 * many times with different grids/inputs), so this renders two tiers:
 *   - RooflineSummary: the launch-invariant character + aggregate stats (Overview).
 *   - RooflineLaunchTable: a per-launch breakdown, grouped into distinct profiles
 *     with counts (the "Launch Deep" view).
 * Achieved bandwidth (GB/s) is intentionally absent — it needs per-launch kernel
 * durations the trace does not capture (a profiler / NCU concern).
 */

function formatBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(2)} ${units[i]}`;
}

function formatFlops(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n < 1000) return `${n} FLOP`;
  const units = ["K", "M", "G", "T"];
  let v = n / 1000;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return `${v.toFixed(2)} ${units[i]}FLOP`;
}

function formatCount(n: number | null | undefined): string {
  return n === null || n === undefined ? "—" : n.toLocaleString();
}

function formatGrid(grid: number[] | null | undefined): string {
  return grid && grid.length ? `(${grid.join(", ")})` : "—";
}

function formatIntensity(ai: number | null | undefined): string {
  return ai === null || ai === undefined ? "—" : ai.toFixed(2);
}

/** Compress a list of launch indices into "0-2, 5, 8-9" ranges. */
function formatRanges(indices: number[]): string {
  if (!indices.length) return "";
  const sorted = [...indices].sort((a, b) => a - b);
  const ranges: string[] = [];
  let start = sorted[0];
  let prev = sorted[0];
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] === prev + 1) {
      prev = sorted[i];
      continue;
    }
    ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
    start = sorted[i];
    prev = sorted[i];
  }
  ranges.push(start === prev ? `${start}` : `${start}-${prev}`);
  return ranges.join(", ");
}

const MetricItem: React.FC<{ label: string; value: string; title?: string }> = ({
  label,
  value,
  title,
}) => (
  <div title={title}>
    <span className="text-sm font-medium text-gray-500 block">{label}</span>
    <span className="text-sm text-gray-900 font-mono">{value}</span>
  </div>
);

const BetaTag: React.FC = () => (
  <span className="ml-2 px-2 py-0.5 text-xs bg-blue-100 text-blue-800 rounded border border-blue-300 align-middle">
    Beta
  </span>
);

const NotesList: React.FC<{ notes: string[] }> = ({ notes }) =>
  notes.length === 0 ? null : (
    <ul className="mt-3 list-disc list-inside text-xs text-gray-500 space-y-0.5">
      {notes.map((n, i) => (
        <li key={i}>{n}</li>
      ))}
    </ul>
  );

/** Tier 1 — launch-invariant character + aggregate stats across all launches. */
export const RooflineSummary: React.FC<{ roofline: RooflineData }> = ({
  roofline,
}) => {
  const launches = roofline.per_launch || [];
  const bytesVals = launches
    .map((l) => l.bytes_moved)
    .filter((b): b is number => typeof b === "number");
  const totalBytes = bytesVals.reduce((a, b) => a + b, 0);
  const minBytes = bytesVals.length ? Math.min(...bytesVals) : null;
  const maxBytes = bytesVals.length ? Math.max(...bytesVals) : null;
  const perCtaAI =
    roofline.flops_per_cta && roofline.bytes_moved_per_cta
      ? roofline.flops_per_cta / roofline.bytes_moved_per_cta
      : null;

  return (
    <div className="mb-6">
      <h3 className="text-lg font-medium mb-3 text-gray-800">
        Roofline
        <BetaTag />
      </h3>
      <div className="bg-gray-50 p-4 rounded-md border border-gray-200">
        <div className="grid grid-cols-[repeat(auto-fit,_minmax(180px,_1fr))] gap-3">
          <MetricItem
            label="Kernel type"
            value={roofline.is_gemm ? "GEMM (has tt.dot)" : "Memory-bound"}
          />
          <MetricItem
            label="Arithmetic intensity"
            value={
              !roofline.is_gemm
                ? "N/A (memory-bound)"
                : perCtaAI !== null
                  ? `${perCtaAI.toFixed(2)} FLOP/byte`
                  : "—"
            }
            title="Per-CTA FLOPs / per-CTA bytes — the launch-invariant roofline x-coordinate. FLOPs are only counted for GEMM (tt.dot) kernels."
          />
          <MetricItem
            label="Bytes / CTA"
            value={formatBytes(roofline.bytes_moved_per_cta)}
            title="Global-memory bytes moved by one CTA (from tt.load/tt.store in TTIR)."
          />
          <MetricItem
            label="FLOPs / CTA"
            value={
              roofline.is_gemm
                ? formatFlops(roofline.flops_per_cta)
                : "N/A (memory-bound)"
            }
          />
          <MetricItem label="Launches" value={formatCount(launches.length)} />
          <MetricItem
            label="Total bytes moved"
            value={formatBytes(totalBytes)}
            title="Sum of bytes_moved across all recorded launches."
          />
          <MetricItem
            label="Per-launch bytes range"
            value={
              minBytes !== null
                ? `${formatBytes(minBytes)} – ${formatBytes(maxBytes)}`
                : "—"
            }
          />
        </div>
        <p className="text-xs text-gray-500 mt-3">
          Static estimate from TTIR × each launch&apos;s grid and arg sizes.
          {roofline.is_gemm
            ? " "
            : " FLOPs / arithmetic intensity apply only to GEMM (tt.dot) kernels — this one is memory-bound, so only bytes moved are estimated. "}
          Throughput (GB/s or TFLOP/s) is not shown — that is a rate and needs
          per-launch kernel durations the trace does not capture (use a profiler
          / NCU).
        </p>
        {roofline.notes && <NotesList notes={roofline.notes} />}
      </div>
    </div>
  );
};

interface LaunchProfile {
  key: string;
  grid: number[] | null | undefined;
  num_ctas: number | null | undefined;
  bytes_moved: number | null | undefined;
  bytes_from_ir_x_grid: number | null | undefined;
  bytes_from_args: number | null | undefined;
  flops: number | null | undefined;
  arithmetic_intensity: number | null | undefined;
  count: number;
  indices: number[];
}

function groupLaunches(per: RooflinePerLaunch[]): LaunchProfile[] {
  const map = new Map<string, LaunchProfile>();
  for (const l of per) {
    const key = JSON.stringify([l.grid, l.num_ctas, l.bytes_moved, l.flops]);
    let g = map.get(key);
    if (!g) {
      g = {
        key,
        grid: l.grid,
        num_ctas: l.num_ctas,
        bytes_moved: l.bytes_moved,
        bytes_from_ir_x_grid: l.bytes_from_ir_x_grid,
        bytes_from_args: l.bytes_from_args,
        flops: l.flops,
        arithmetic_intensity: l.arithmetic_intensity,
        count: 0,
        indices: [],
      };
      map.set(key, g);
    }
    g.count++;
    g.indices.push(l.launch_index);
  }
  return Array.from(map.values()).sort(
    (a, b) => (a.bytes_moved ?? 0) - (b.bytes_moved ?? 0)
  );
}

/** Tier 2 — per-launch breakdown, grouped into distinct profiles with counts. */
export const RooflineLaunchTable: React.FC<{ roofline: RooflineData }> = ({
  roofline,
}) => {
  const perLaunch = roofline.per_launch;
  const profiles = useMemo(() => groupLaunches(perLaunch || []), [perLaunch]);
  const [showRanges, setShowRanges] = useState(false);

  if (!perLaunch || perLaunch.length === 0) return null;
  const launchCount = perLaunch.length;
  const isGemm = roofline.is_gemm;

  return (
    <div className="mb-4">
      <h4 className="text-md font-semibold mb-2 text-gray-800">
        Per-launch Roofline{" "}
        <span className="text-sm font-normal text-gray-500">
          ({launchCount} launches, {profiles.length} distinct profile
          {profiles.length === 1 ? "" : "s"})
        </span>
      </h4>
      {!isGemm && (
        <p className="text-xs text-gray-500 mb-2">
          Memory-bound kernel (no <code>tt.dot</code>) — FLOPs / intensity are
          not applicable; only bytes moved are estimated.
        </p>
      )}
      <div className="overflow-x-auto bg-white rounded border border-gray-200">
        <table className="min-w-full text-sm text-right font-mono">
          <thead>
            <tr className="bg-gray-100 text-gray-600 text-xs">
              <th className="px-3 py-2 text-left font-semibold">Grid</th>
              <th className="px-3 py-2 font-semibold">CTAs</th>
              <th className="px-3 py-2 font-semibold">Bytes moved</th>
              <th
                className="px-3 py-2 font-semibold"
                title="bytes_moved is the min of these two estimates"
              >
                IR×grid / args
              </th>
              {isGemm && <th className="px-3 py-2 font-semibold">FLOPs</th>}
              {isGemm && (
                <th
                  className="px-3 py-2 font-semibold"
                  title="FLOPs / bytes_moved for this launch"
                >
                  Intensity
                </th>
              )}
              <th className="px-3 py-2 font-semibold"># launches</th>
              {showRanges && (
                <th className="px-3 py-2 text-left font-semibold">
                  Launch indices
                </th>
              )}
            </tr>
          </thead>
          <tbody>
            {profiles.map((p) => (
              <tr key={p.key} className="border-t border-gray-100">
                <td className="px-3 py-2 text-left">{formatGrid(p.grid)}</td>
                <td className="px-3 py-2">{formatCount(p.num_ctas)}</td>
                <td className="px-3 py-2 font-semibold text-gray-900">
                  {formatBytes(p.bytes_moved)}
                </td>
                <td className="px-3 py-2 text-gray-500 text-xs">
                  {formatBytes(p.bytes_from_ir_x_grid)} /{" "}
                  {formatBytes(p.bytes_from_args)}
                </td>
                {isGemm && (
                  <td className="px-3 py-2">{formatFlops(p.flops)}</td>
                )}
                {isGemm && (
                  <td className="px-3 py-2">
                    {formatIntensity(p.arithmetic_intensity)}
                  </td>
                )}
                <td className="px-3 py-2">{formatCount(p.count)}</td>
                {showRanges && (
                  <td className="px-3 py-2 text-left text-xs text-gray-500">
                    {formatRanges(p.indices)}
                  </td>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        type="button"
        className="mt-2 text-xs text-blue-600 hover:underline"
        onClick={() => setShowRanges((s) => !s)}
      >
        {showRanges ? "Hide launch indices" : "Show launch indices"}
      </button>
    </div>
  );
};
