#!/usr/bin/env node
// This script creates a standalone HTML file by inlining all external resources (scripts, styles, images, fonts)
// into a single self-contained HTML file, and also generates a compressed gzipped version of it.
// This is useful for deploying the website as a single file when the hosting service doesn't support serving static files.
// This script is used when we build the website using `npm run build:single`.
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import zlib from 'zlib';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.resolve(__dirname, '../dist');
const outputFile = path.resolve(__dirname, '../dist/standalone.html');
const compressedOutputFile = path.resolve(__dirname, '../dist/standalone.html.gz');

// Read the index.html file
const indexPath = path.join(distDir, 'index.html');
let html = fs.readFileSync(indexPath, 'utf8');

// Regular expressions to find resources
// Input is Vite-generated dist/index.html, not user-supplied HTML — no XSS risk.
const scriptRegex = /<script.*?src="(.*?)".*?><\/script\s*>/g; // lgtm[js/bad-tag-filter]
const styleRegex = /<link.*?rel="stylesheet".*?href="(.*?)".*?>/g;
const faviconRegex = /<link.*?rel="icon".*?href="(.*?)".*?>/g;
const imgRegex = /<img.*?src="(.*?)".*?>/g;
const svgRegex = /<svg.*?<use.*?href="(.*?)".*?>/g;
const fontRegex = /@font-face\s*{[^}]*src\s*:\s*url\(['"]?(.*?)['"]?\)/g;

// Helper function to get file path from a resource URL
function getFilePath(resourceUrl) {
  return path.join(distDir, resourceUrl.startsWith('./') ? resourceUrl.slice(2) : resourceUrl);
}

// Helper function to convert file to data URI
function toDataURI(filePath, mimeType) {
  try {
    const content = fs.readFileSync(filePath);
    const base64 = content.toString('base64');
    return `data:${mimeType};base64,${base64}`;
  } catch (err) {
    console.warn(`Warning: Could not read file ${filePath}. Error: ${err.message}`);
    return null;
  }
}

// Helper function to get MIME type from file extension
function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.html': 'text/html',
    '.css': 'text/css',
    '.js': 'application/javascript',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.woff': 'font/woff',
    '.woff2': 'font/woff2',
    '.ttf': 'font/ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.otf': 'font/otf'
  };

  return mimeTypes[ext] || 'application/octet-stream';
}

// Function to process inline font resources
function processInlineFonts(css) {
  return css.replace(fontRegex, (match, fontUrl) => {
    const fontPath = getFilePath(fontUrl);
    if (fs.existsSync(fontPath)) {
      const mimeType = getMimeType(fontPath);
      const dataUri = toDataURI(fontPath, mimeType);
      if (dataUri) {
        return match.replace(fontUrl, dataUri);
      }
    }
    return match;
  });
}

// Inlining order is load-bearing. The style/favicon/img/svg replacements
// below use broad regexes that could match bundle bytes (e.g. an <img>
// string inside JS), so they run FIRST, while `html` is still the original
// index.html; the CSS font injection inside the style step only rewrites
// @font-face rules, never script bytes. Scripts are inlined next and
// integrity-checked immediately; any later step that rewrites script bytes
// by design must run after the check.

// Replace link tags with inline styles, and process any font references within CSS
html = html.replace(styleRegex, (match, href) => {
  const stylePath = getFilePath(href);
  if (fs.existsSync(stylePath)) {
    try {
      let styleContent = fs.readFileSync(stylePath, 'utf8');
      // Process any font references within CSS
      styleContent = processInlineFonts(styleContent);
      // Keep any relevant attributes
      const crossOrigin = match.includes('crossorigin') ? ' crossorigin' : '';
      return `<style${crossOrigin}>${styleContent}</style>`;
    } catch (err) {
      console.warn(`Warning: Could not inline stylesheet ${stylePath}. Error: ${err.message}`);
    }
  }
  return match;
});

// Handle favicon
html = html.replace(faviconRegex, (match, href) => {
  const faviconPath = getFilePath(href);
  if (fs.existsSync(faviconPath)) {
    const mimeType = getMimeType(faviconPath);
    const dataUri = toDataURI(faviconPath, mimeType);
    if (dataUri) {
      return match.replace(href, dataUri);
    }
  }
  return match;
});

// Replace image references with data URIs
html = html.replace(imgRegex, (match, src) => {
  const imgPath = getFilePath(src);
  if (fs.existsSync(imgPath)) {
    const mimeType = getMimeType(imgPath);
    const dataUri = toDataURI(imgPath, mimeType);
    if (dataUri) {
      return match.replace(src, dataUri);
    }
  }
  return match;
});

// Handle SVG use references
html = html.replace(svgRegex, (match, href) => {
  const svgPath = getFilePath(href.split('#')[0]);
  if (fs.existsSync(svgPath)) {
    const mimeType = 'image/svg+xml';
    const dataUri = toDataURI(svgPath, mimeType);
    if (dataUri) {
      return match.replace(href, `${dataUri}#${href.split('#')[1]}`);
    }
  }
  return match;
});

// Replace script tags with inline scripts. Inlined bundles are collected and
// re-inserted before </body>: `defer` is ignored on src-less inline scripts,
// so a bundle left in <head> would run before <div id="root"> is parsed
// (React createRoot on null). End-of-body placement preserves the original
// deferred timing for classic scripts; inline modules defer implicitly.
const inlinedScripts = [];
html = html.replace(scriptRegex, (match, src) => {
  const scriptPath = getFilePath(src);
  if (fs.existsSync(scriptPath)) {
    try {
      const scriptContent = fs.readFileSync(scriptPath, 'utf8');
      // Preserve type="module" attribute if it exists
      const typeModule = match.includes('type="module"') ? ' type="module"' : '';
      const crossOrigin = match.includes('crossorigin') ? ' crossorigin' : '';
      inlinedScripts.push(`<script${typeModule}${crossOrigin}>${scriptContent}</script>`);
      return '';
    } catch (err) {
      console.warn(`Warning: Could not inline script ${scriptPath}. Error: ${err.message}`);
    }
  }
  return match;
});
if (inlinedScripts.length > 0) {
  const combined = inlinedScripts.join('\n');
  if (!html.includes('</body>')) {
    console.warn('Warning: no </body> tag; appending inlined scripts before </html> or at end of document.');
  }
  // Replacer function, NOT a replacement string: bundle bytes contain `$`
  // sequences (`$$`, `$&`, ... ) that String.replace would substitute.
  html = html.includes('</body>')
    ? html.replace('</body>', () => `${combined}\n</body>`)
    : html.includes('</html>')
      ? html.replace('</html>', () => `${combined}\n</html>`)
      : html + combined;
}

// Integrity self-check: every inlined script must survive byte-identical.
// A corrupt insertion (e.g. `$`-pattern substitution) fails the build here
// instead of shipping a standalone page that throws a SyntaxError.
for (const script of inlinedScripts) {
  if (!html.includes(script)) {
    console.error('Error: inlined script bytes do not match the output HTML; aborting.');
    process.exit(1);
  }
}

// Write the result to a new HTML file
fs.writeFileSync(outputFile, html);
console.log(`✅ Standalone HTML file created at: ${outputFile}`);

// Create a gzipped version
const compressed = zlib.gzipSync(html);
fs.writeFileSync(compressedOutputFile, compressed);
console.log(`✅ Compressed standalone HTML file created at: ${compressedOutputFile}`);

// Print file sizes
const originalSize = (html.length / 1024).toFixed(2);
const compressedSize = (compressed.length / 1024).toFixed(2);
console.log(`Original size: ${originalSize} KB`);
console.log(`Compressed size: ${compressedSize} KB`);
console.log(`Compression ratio: ${(100 - (compressed.length / html.length * 100)).toFixed(2)}%`);
