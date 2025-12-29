/**
 * Safe dynamic import utility that handles missing modules gracefully
 * This avoids build-time module resolution issues for OSS/internal builds
 */

/**
 * Safely imports a module by path, returning null if it doesn't exist
 * Uses string concatenation to bypass static analysis while maintaining proper path resolution
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function safeImport(modulePath: string): Promise<any | null> {
  try {
    // Adjust path to account for safeImport.ts being in the utils directory
    // When called from App.tsx with './utils/fb/internal_utils',
    // we need to convert it to './fb/internal_utils' since we're already in utils/
    let adjustedPath = modulePath;
    if (modulePath.startsWith('./utils/')) {
      adjustedPath = './' + modulePath.substring('./utils/'.length);
    }

    // Use string concatenation to create dynamic import path
    // This bypasses static analysis while maintaining proper module resolution
    const dynamicPath = '' + adjustedPath;
    const module = await import(/* @vite-ignore */dynamicPath);
    return module;
  } catch (error) {
    console.warn(`Module ${modulePath} not available:`, error);
    return null;
  }
}
