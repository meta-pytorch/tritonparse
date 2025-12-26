/**
 * Utility to detect if the fb directory exists
 * This checks for the existence of the internal_utils file in the fb directory
 */

export const checkFbDirectoryExists = async (): Promise<boolean> => {
  try {
    // Try to fetch the actual internal_utils file in the fb directory
    const response = await fetch('./fb/internal_utils.ts', { method: 'HEAD' });
    
    // Check if we got a successful response AND it's not HTML content
    // Vite dev server returns 200 + HTML (index.html) for missing files as SPA fallback
    const contentType = response.headers.get('content-type') || '';
    const isHtmlFallback = contentType.includes('text/html');
    
    // Only consider it successful if we get 200 AND it's not HTML fallback
    return response.ok && !isHtmlFallback;
  } catch {
    // If the request fails, the directory likely doesn't exist
    return false;
  }
};
