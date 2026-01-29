/**
 * URL normalization utilities for handling various Meta-internal URL formats
 */

const MANIFOLD_EXPLORER_PREFIX = 'https://www.internalfb.com/manifold/explorer/';
const INTERNCACHE_MANIFOLD_PREFIX = 'https://interncache-all.fbcdn.net/manifold/';

/**
 * Extract json_url parameter from a tritonparse website URL
 * 
 * Example input:
 *   https://interncache-all.fbcdn.net/manifold/tritonparse/tree/index.html?json_url=https%3A%2F%2F...&view=ir_code_comparison
 * 
 * Example output:
 *   https://interncache-all.fbcdn.net/manifold/tlparse_reports/tree/logs/...
 */
function extractJsonUrlParam(url: string): string {
  try {
    const urlObj = new URL(url);
    const jsonUrl = urlObj.searchParams.get('json_url');
    if (jsonUrl) {
      return jsonUrl; // URLSearchParams automatically decodes the value
    }
  } catch {
    // Not a valid URL, return as-is
  }
  return url;
}

/**
 * Convert Manifold Explorer URL to interncache URL
 * 
 * Example input:
 *   https://www.internalfb.com/manifold/explorer/tlparse_reports/tree/logs/tmpy6uh65b8/file.ndjson.gz
 * 
 * Example output:
 *   https://interncache-all.fbcdn.net/manifold/tlparse_reports/tree/logs/tmpy6uh65b8/file.ndjson.gz
 */
function convertManifoldExplorerUrl(url: string): string {
  if (url.startsWith(MANIFOLD_EXPLORER_PREFIX)) {
    const path = url.substring(MANIFOLD_EXPLORER_PREFIX.length);
    return INTERNCACHE_MANIFOLD_PREFIX + path;
  }
  return url;
}

/**
 * Normalize a data URL to a format that can be directly fetched.
 * 
 * Handles:
 * 1. tritonparse website URLs with json_url parameter - extracts the actual data URL
 * 2. Manifold Explorer URLs - converts to interncache format
 * 
 * The order matters: first extract json_url (which may itself be a Manifold Explorer URL),
 * then convert if needed.
 */
export function normalizeDataUrl(inputUrl: string): string {
  let url = inputUrl.trim();
  
  // Step 1: Extract json_url parameter if present
  url = extractJsonUrlParam(url);
  
  // Step 2: Convert Manifold Explorer URL to interncache
  url = convertManifoldExplorerUrl(url);
  
  return url;
}
