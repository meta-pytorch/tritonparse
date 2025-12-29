import React, { useState, useEffect, useRef, useCallback } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneLight,
  oneDark,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import type { SourceMapping } from "../utils/dataLoader";
import "./CodeViewer.css";

// Import language support
import llvm from 'react-syntax-highlighter/dist/esm/languages/prism/llvm';
import c from 'react-syntax-highlighter/dist/esm/languages/prism/c';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';

// Register languages with the syntax highlighter
SyntaxHighlighter.registerLanguage('llvm', llvm);
SyntaxHighlighter.registerLanguage('c', c);
SyntaxHighlighter.registerLanguage('python', python);
/**
 * Thresholds for file size optimization:
 * - LARGE_FILE_THRESHOLD: Files larger than this will use virtualized rendering with syntax highlighting
 * - EXTREMELY_LARGE_FILE_THRESHOLD: Files larger than this will use basic rendering without syntax highlighting
 */

const LARGE_FILE_THRESHOLD = 10000000;
const EXTREMELY_LARGE_FILE_THRESHOLD = 10000000;

// Global scroll position storage to persist across re-renders
const scrollPositionStore = new Map<string, number>();

/**
 * Custom hook for managing scroll position and highlight changes
 * Provides common scroll management functionality for code viewers
 */
const useScrollManagement = (
  containerRef: React.RefObject<HTMLDivElement | null>,
  highlightedLines: number[],
  viewerId: string | undefined,
  customScrollToLine?: (lineNumber: number) => void
) => {
  const isRestoringScrollRef = useRef<boolean>(false);
  const previousHighlightedLinesRef = useRef<number[]>([]);
  const scrollKey = viewerId || 'default';

  // Save scroll position on scroll
  const saveScrollPosition = useCallback(() => {
    if (!isRestoringScrollRef.current) {
      const container = containerRef.current;
      if (container) {
        const scrollTop = container.scrollTop;
        scrollPositionStore.set(scrollKey, scrollTop);
      }
    }
  }, [containerRef, scrollKey]);

  // Helper function to reset scroll flag after an operation
  const resetScrollFlag = useCallback((delay: number = 10) => {
    setTimeout(() => {
      isRestoringScrollRef.current = false;
    }, delay);
  }, []);

  // Helper function to perform direct scroll with retry mechanism
  const performDirectScroll = useCallback((container: HTMLDivElement, targetPosition: number) => {
    container.scrollTop = targetPosition;

    // Use requestAnimationFrame for better reliability
    requestAnimationFrame(() => {
      container.scrollTop = targetPosition;
      resetScrollFlag(10);
    });
  }, [resetScrollFlag]);

  // Monitor highlight changes and handle scrolling
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Check if this is a highlight change (not initial render)
    const prevLines = previousHighlightedLinesRef.current;
    const currentLines = highlightedLines;

    const isHighlightChange =
      prevLines.length > 0 || currentLines.length > 0; // Either had highlights before or has now

    // Check if this is truly a NEW highlight change (lines actually changed)
    const linesChanged = JSON.stringify(prevLines) !== JSON.stringify(currentLines);

    if (!isHighlightChange || !linesChanged) {
      // Update previous highlights and exit early
      previousHighlightedLinesRef.current = [...currentLines];
      return;
    }

    // Set flag to prevent scroll position saving during programmatic scrolling
    isRestoringScrollRef.current = true;

    if (currentLines.length > 0) {
      // If we have new highlighted lines, scroll to show the first one
      const firstHighlightedLine = Math.min(...currentLines);

      if (customScrollToLine) {
        // Use custom scroll function (for large files with smooth scrolling)
        customScrollToLine(firstHighlightedLine);
        resetScrollFlag(100);
      } else {
        // ✅ Use scrollIntoView for accurate scrolling without calculating line height
        requestAnimationFrame(() => {
          const targetLine = container.querySelector(
            `[data-line-number="${firstHighlightedLine}"]`
          );

          if (targetLine) {
            targetLine.scrollIntoView({
              behavior: 'auto',
              block: 'start',
              inline: 'nearest'
            });
          }
          resetScrollFlag(10);
        });
      }
    } else {
      // If clearing highlights, restore the saved scroll position
      const savedScrollPosition = scrollPositionStore.get(scrollKey) || 0;
      if (savedScrollPosition > 0) {
        performDirectScroll(container, savedScrollPosition);
      } else {
        resetScrollFlag();
      }
    }

    // Update previous highlights
    previousHighlightedLinesRef.current = [...currentLines];
  }, [highlightedLines, scrollKey, customScrollToLine, resetScrollFlag, performDirectScroll]);

  return {
    saveScrollPosition
  };
};

/**
 * Props for the CodeViewer component
 */
interface CodeViewerProps {
  code: string; // The source code to display
  language?: string; // The programming language for syntax highlighting
  height?: string; // Height of the code viewer container
  theme?: "light" | "dark"; // Color theme
  fontSize?: number; // Font size for the code
  highlightedLines?: number[]; // Array of line numbers to highlight
  onLineClick?: (lineNumber: number) => void; // Callback when a line is clicked
  viewerId?: string; // Unique identifier for this viewer (used for mapping)
  otherViewerId?: string; // Identifier for the paired viewer (for mapping)
  sourceMapping?: Record<string, SourceMapping>; // Source mapping information
  onMappedLinesFound?: (mappedLines: number[]) => void; // Callback when mapped lines are found
  functionStartLine?: number; // Function definition start line (for highlighting in full file mode)
  functionEndLine?: number; // Function definition end line (for highlighting in full file mode)
  initialScrollToLine?: number; // Line number to scroll to on initial render
  onScrollComplete?: () => void; // Callback fired when initial scroll completes
  startingLineNumber?: number; // Starting line number for display (default: 1)
}

/**
 * Maps our internal language names to syntax highlighter languages
 * Note: Exported for use by other components.
 * @param language Internal language identifier
 * @returns Syntax highlighter language identifier
 */
export const mapLanguageToHighlighter = (language: string): string => {
  const lowerCaseLanguage = language.toLowerCase();

  // Handle language types with endsWith for better accuracy
  if (lowerCaseLanguage.endsWith("ttgir") || lowerCaseLanguage.endsWith("ttir")) {
    return 'mlir';
  } else if (lowerCaseLanguage.endsWith("llir")) {
    return 'llvm';
  } else if (lowerCaseLanguage.endsWith("ptx")) {
    return 'ptx';
  } else if (lowerCaseLanguage.endsWith("amdgcn")) {
    return 'amdgcn';
  } else if (lowerCaseLanguage === "python") {
    return 'python';
  }

  return 'plaintext';
};

/**
 * Split code into lines
 * @param code The code content as string
 * @returns Array of code lines
 */
const splitIntoLines = (code: string): string[] => {
  return code.split('\n');
};

/**
 * Creates a debounced function that delays invoking func until after wait milliseconds
 * @param func The function to debounce
 * @param wait The number of milliseconds to delay
 * @returns Debounced function
 */
function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: ReturnType<typeof setTimeout> | null = null;

  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

/**
 * Basic code viewer using pre/code tags for very large files
 * Disables syntax highlighting for extremely large files to maintain performance
 */
const BasicCodeViewer: React.FC<CodeViewerProps> = ({
  code,
  height = "100%",
  fontSize = 14,
  highlightedLines = [],
  onLineClick,
  viewerId,
  initialScrollToLine,
  functionStartLine,
  functionEndLine,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const lines = splitIntoLines(code);
  const hasScrolledRef = useRef(false);

  const handleLineClick = useCallback((lineNumber: number) => {
    if (onLineClick) {
      onLineClick(lineNumber);
    }
  }, [onLineClick]);

  // Initial scroll effect - use scrollIntoView for accuracy
  useEffect(() => {
    if (!initialScrollToLine || hasScrolledRef.current) {
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    // Delay to ensure DOM is rendered
    const timer = setTimeout(() => {
      const targetLine = container.querySelector(
        `[data-line-number="${initialScrollToLine}"]`
      );

      if (targetLine) {
        targetLine.scrollIntoView({
          behavior: 'auto',
          block: 'start',
          inline: 'nearest'
        });
      }

      hasScrolledRef.current = true;
      // Note: onScrollComplete is intentionally not called to avoid triggering parent re-renders
    }, 100);

    return () => clearTimeout(timer);
  }, [initialScrollToLine]);

  return (
    <div
      ref={containerRef}
      style={{
        height,
        overflowY: "auto",
        fontSize: `${fontSize}px`,
        backgroundColor: "#fff"
      }}
      className={`code-viewer ${highlightedLines.length > 0 ? 'has-highlights' : ''}`}
      data-viewer-id={viewerId}
      data-highlighted-lines={highlightedLines.join(',')}
    >
      <pre style={{
        margin: 0,
        padding: "0.5em",
        fontFamily: "Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace"
      }}>
        <code>
          {lines.map((line, index) => {
            const lineNumber = index + 1;
            const isHighlighted = highlightedLines.includes(lineNumber);

            // Check if line is in function range
            const isInFunctionRange =
              functionStartLine !== undefined &&
              functionEndLine !== undefined &&
              lineNumber >= functionStartLine &&
              lineNumber <= functionEndLine;

            const isFunctionStart = lineNumber === functionStartLine;
            const isFunctionEnd = lineNumber === functionEndLine;

            // Build class names
            const classNames = [
              isHighlighted ? 'highlighted-line' : '',
              isInFunctionRange ? 'function-range' : '',
              isFunctionStart ? 'function-range-start' : '',
              isFunctionEnd ? 'function-range-end' : ''
            ].filter(Boolean).join(' ');

            return (
              <div
                key={index}
                style={{
                  paddingLeft: "3.8em",
                  position: "relative",
                  whiteSpace: "pre-wrap",
                  cursor: onLineClick ? "pointer" : "text"
                }}
                onClick={() => handleLineClick(lineNumber)}
                data-line-number={lineNumber}
                className={classNames}
              >
                <span style={{
                  position: "absolute",
                  left: 0,
                  userSelect: "none",
                  opacity: 0.5,
                  width: "3em",
                  textAlign: "right",
                  paddingRight: "0.5em"
                }}>
                  {lineNumber}
                </span>
                {line || " "}
              </div>
            );
          })}
        </code>
      </pre>
    </div>
  );
};

/**
 * Optimized code viewer for large files
 */
const LargeFileViewer: React.FC<CodeViewerProps> = ({
  code,
  language = "plaintext",
  height = "100%",
  theme = "light",
  fontSize = 14,
  highlightedLines = [],
  onLineClick,
  viewerId,
  initialScrollToLine,
  onScrollComplete,
  functionStartLine,
  functionEndLine,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleRange, setVisibleRange] = useState({ start: 0, end: 50 });
  const lines = splitIntoLines(code);
  const lineHeight = Math.ceil(fontSize * 1.5); // Approximate line height based on font size
  const hasScrolledRef = useRef(false);

  // Function to scroll to a specific line
  const scrollToLine = useCallback((lineNumber: number) => {
    const container = containerRef.current;
    if (!container) return;

    // Calculate position to scroll to
    const scrollPosition = (lineNumber - 1) * lineHeight;

    // Scroll with smooth behavior
    container.scrollTo({
      top: scrollPosition,
      behavior: 'smooth'
    });
  }, [lineHeight]);

  // Use the common scroll management hook
  const { saveScrollPosition } = useScrollManagement(
    containerRef,
    highlightedLines,
    viewerId,
    scrollToLine // Pass custom scroll function for large files
  );

  // Initial scroll effect for large files
  // Note: LargeFileViewer uses virtual scrolling, so we use calculated position
  // instead of scrollIntoView (target line may not be rendered yet)
  useEffect(() => {
    if (!initialScrollToLine || hasScrolledRef.current) {
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    // Delay to ensure DOM is rendered
    const timer = setTimeout(() => {
      // For virtual scrolling, we must use calculated position
      // because the target line might not be rendered yet
      const targetScrollTop = (initialScrollToLine - 1) * lineHeight;
      container.scrollTop = Math.max(0, targetScrollTop);

      hasScrolledRef.current = true;
      onScrollComplete?.();
    }, 100);

    return () => clearTimeout(timer);
  }, [initialScrollToLine, lineHeight, onScrollComplete]);

  // Use useCallback to memoize the scroll handler
  const updateVisibleLines = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const scrollTop = container.scrollTop;
    const clientHeight = container.clientHeight;

    // Add buffer before and after visible area for smoother scrolling
    const visibleLines = Math.ceil(clientHeight / lineHeight);
    const bufferLines = visibleLines;

    const startLine = Math.max(0, Math.floor(scrollTop / lineHeight) - bufferLines);
    const endLine = Math.min(
      lines.length - 1,
      Math.ceil((scrollTop + clientHeight) / lineHeight) + bufferLines
    );

    setVisibleRange(prev => {
      // Only update if the range actually changed
      if (prev.start !== startLine || prev.end !== endLine) {
        return { start: startLine, end: endLine };
      }
      return prev;
    });
  }, [lines.length, lineHeight]);

  // Create a debounced version of the update function inside useEffect
  // This avoids the refs-during-render warning
  const debouncedUpdateRef = useRef<((...args: unknown[]) => void) | null>(null);

  // Initialize the debounced function in useEffect
  useEffect(() => {
    debouncedUpdateRef.current = debounce(updateVisibleLines, 10);
  }, [updateVisibleLines]);

  // Combined scroll handler that updates visible range and saves position
  const handleScroll = useCallback(() => {
    saveScrollPosition();
    if (debouncedUpdateRef.current) {
      debouncedUpdateRef.current();
    }
  }, [saveScrollPosition]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Initial update without debounce
    updateVisibleLines();

    // Add scroll listener with both position saving and visible range update
    container.addEventListener('scroll', handleScroll, { passive: true });

    // Cleanup
    return () => {
      container.removeEventListener('scroll', handleScroll);
    };
  }, [updateVisibleLines, handleScroll]);

  // Enhanced line click handler that processes source mapping
  const handleLineClick = useCallback((lineNumber: number) => {

    if (onLineClick) {
      onLineClick(lineNumber);
    }
  }, [onLineClick]);

  // Map the language to the appropriate highlighter language
  const highlighterLanguage = mapLanguageToHighlighter(language);

  // Only render visible lines plus buffer
  const visibleCode = lines.slice(visibleRange.start, visibleRange.end + 1).join('\n');

  // Calculate offsets for line numbers and highlighting
  const lineNumberOffset = visibleRange.start + 1;

  // Note: Scroll management is now handled by the useScrollManagement hook

  return (
    <div
      ref={containerRef}
      style={{
        height,
        overflowY: "auto",
        fontSize: `${fontSize}px`,
        position: "relative"
      }}
      className={`code-viewer ${highlightedLines.length > 0 ? 'has-highlights' : ''}`}
      data-viewer-id={viewerId}
      data-highlighted-lines={highlightedLines.join(',')}
    >
      {/* Container with full height to enable proper scrolling */}
      <div style={{ height: `${lines.length * lineHeight}px`, position: "relative" }}>
        {/* Only render the visible portion */}
        <div style={{
          position: "absolute",
          top: `${visibleRange.start * lineHeight}px`,
          width: "100%"
        }}>
          <SyntaxHighlighter
            language={highlighterLanguage}
            style={theme === "light" ? oneLight : oneDark}
            showLineNumbers
            startingLineNumber={lineNumberOffset}
            wrapLines
            lineProps={(lineNumber) => {
              // Adjust line number based on visible range
              const actualLine = lineNumber + visibleRange.start;
              const isHighlighted = highlightedLines.includes(actualLine);

              // Check if line is in function range
              const isInFunctionRange =
                functionStartLine !== undefined &&
                functionEndLine !== undefined &&
                actualLine >= functionStartLine &&
                actualLine <= functionEndLine;

              const isFunctionStart = actualLine === functionStartLine;
              const isFunctionEnd = actualLine === functionEndLine;

              // Create styles for the line (no colors - handled by CSS classes)
              const style: React.CSSProperties = {
                display: "block",
                cursor: onLineClick ? "pointer" : "text",
              };

              // Build class names
              const classNames = [
                isHighlighted ? 'highlighted-line' : '',
                isInFunctionRange ? 'function-range' : '',
                isFunctionStart ? 'function-range-start' : '',
                isFunctionEnd ? 'function-range-end' : ''
              ].filter(Boolean).join(' ');

              return {
                style,
                onClick: () => handleLineClick(actualLine),
                'data-line-number': actualLine,
                className: classNames,
              };
            }}
            customStyle={{
              margin: 0,
              fontSize: "inherit",
              backgroundColor: theme === "light" ? "#fff" : "#1E1E1E",
              padding: "0.5em",
            }}
          >
            {visibleCode}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
};

/**
 * Standard code viewer for smaller files
 * CSS class toggle optimization: Render once, highlights via direct DOM manipulation
 */
const StandardCodeViewer: React.FC<CodeViewerProps> = ({
  code,
  language = "plaintext",
  height = "100%",
  theme = "light",
  fontSize = 14,
  onLineClick,
  viewerId,
  initialScrollToLine,
  onScrollComplete,
  functionStartLine,
  functionEndLine,
  startingLineNumber = 1,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const hasScrolledRef = useRef(false);

  // Initial scroll effect - use scrollIntoView for accuracy
  useEffect(() => {
    if (!initialScrollToLine || hasScrolledRef.current) {
      return;
    }

    const container = containerRef.current;
    if (!container) return;

    // Delay to ensure DOM is rendered
    const timer = setTimeout(() => {
      const targetLine = container.querySelector(
        `[data-line-number="${initialScrollToLine}"]`
      );

      if (targetLine) {
        targetLine.scrollIntoView({
          behavior: 'auto',
          block: 'start',
          inline: 'nearest'
        });
      }

      hasScrolledRef.current = true;
      onScrollComplete?.();
    }, 100);

    return () => clearTimeout(timer);
  }, [initialScrollToLine, onScrollComplete]);

  // Enhanced line click handler
  const handleLineClick = useCallback((lineNumber: number) => {
    if (onLineClick) {
      onLineClick(lineNumber);
    }
  }, [onLineClick]);

  // Map the language to the appropriate highlighter language
  const highlighterLanguage = mapLanguageToHighlighter(language);

  return (
    <div
      ref={containerRef}
      style={{
        height,
        overflowY: "auto",
        fontSize: `${fontSize}px`
      }}
      className="code-viewer"
      data-viewer-id={viewerId}
    >
      <SyntaxHighlighter
        language={highlighterLanguage}
        style={theme === "light" ? oneLight : oneDark}
        showLineNumbers
        startingLineNumber={startingLineNumber}
        wrapLines
        lineProps={(lineNumber) => {
          // lineNumber from SyntaxHighlighter already includes startingLineNumber offset
          // so it represents the actual/absolute line number to display and use

          // Check if line is in function range
          const isInFunctionRange =
            functionStartLine !== undefined &&
            functionEndLine !== undefined &&
            lineNumber >= functionStartLine &&
            lineNumber <= functionEndLine;

          const isFunctionStart = lineNumber === functionStartLine;
          const isFunctionEnd = lineNumber === functionEndLine;

          // Build class names (excludes highlighted-line, controlled by external DOM manipulation)
          const classNames = [
            isInFunctionRange ? 'function-range' : '',
            isFunctionStart ? 'function-range-start' : '',
            isFunctionEnd ? 'function-range-end' : ''
          ].filter(Boolean).join(' ');

          return {
            style: {
              display: "block",
              cursor: onLineClick ? "pointer" : "text",
            },
            onClick: () => handleLineClick(lineNumber),
            'data-line-number': lineNumber,
            'data-viewer-id': viewerId,
            className: classNames,
          };
        }}
        customStyle={{
          margin: 0,
          fontSize: "inherit",
          backgroundColor: theme === "light" ? "#fff" : "#1E1E1E",
          padding: "0.5em",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

/**
 * CodeViewer component that renders code with syntax highlighting
 * and supports line highlighting and clicking.
 * Automatically chooses between standard, optimized, or basic viewer based on code size.
 */
const CodeViewer: React.FC<CodeViewerProps> = (props) => {
  // Add inline style for highlighted lines to ensure they're visible
  useEffect(() => {
    if (props.highlightedLines && props.highlightedLines.length > 0) {
      const styleId = `highlight-style-${props.viewerId || 'default'}`;
      let styleEl = document.getElementById(styleId) as HTMLStyleElement;

      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = styleId;
        document.head.appendChild(styleEl);
      }

      // Create a style rule for highlighted lines
      const lines = props.highlightedLines.join(', .line-');
      styleEl.innerHTML = `
        .highlighted-line, .line-${lines} {
          background-color: rgba(255, 215, 0, 0.4) !important;
          border-left: 3px solid orange !important;
        }
      `;

    }
  }, [props.highlightedLines, props.viewerId]);

  // Check if there's any code at all
  if (!props.code) {
    return (
      <div className="code-viewer" style={{
        height: props.height || "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: props.theme === "light" ? "#fff" : "#1E1E1E",
        color: props.theme === "light" ? "#333" : "#eee",
      }}>
        No code to display
      </div>
    );
  }

  // Use optimized viewer for large files
  if (props.code.length > EXTREMELY_LARGE_FILE_THRESHOLD) {
    return <BasicCodeViewer {...props} />;
  } else if (props.code.length > LARGE_FILE_THRESHOLD) {
    return <LargeFileViewer {...props} />;
  } else {
    return <StandardCodeViewer {...props} />;
  }
};

// Use React.memo to prevent unnecessary re-renders
export default React.memo(CodeViewer);
