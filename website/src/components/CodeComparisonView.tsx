// (c) Meta Platforms, Inc. and affiliates.

import React, { useCallback, useMemo, useRef } from "react";
import { Panel, PanelGroup, PanelResizeHandle } from "react-resizable-panels";
import CodeViewer from "./CodeViewer";
import CopyCodeButton from "./CopyCodeButton";
import {
    IRFile,
    PythonSourceCodeInfo,
    SourceMapping,
    getIRType,
} from "../utils/dataLoader";
import { getDisplayLanguage } from "./TritonIRs";

/**
 * Props for a single code panel
 */
interface PanelProps {
    code?: IRFile;
    content?: string;
    language?: string;
    title?: string;
}

/**
 * Props for the CodeComparisonView component
 */
interface CodeComparisonViewProps {
    leftPanel: PanelProps;
    rightPanel: PanelProps;
    py_code_info?: PythonSourceCodeInfo;
    showPythonSource?: boolean;
    pythonMapping?: Record<string, SourceMapping>;
}

/**
 * Unified highlight state interface
 */
interface HighlightState {
    left: number[];
    right: number[];
    python: number[];
}

/**
 * Panel data interface for cached computations
 */
interface PanelData {
    title: string;
    content: string;
    sourceMapping: Record<string, SourceMapping>;
    displayLanguage: string;
}

/**
 * Python info interface for cached computations
 */
interface PythonInfo {
    code: string;
    file_path: string;
    start_line: number;
    isFullFileMode: boolean;
    function_start_line?: number;
    function_end_line?: number;
}

/**
 * CodeComparisonView component that renders two or three code panels side by side
 * with optional line highlighting and synchronization between panels
 *
 * Performance optimizations:
 * - Single state object for all highlights (reduces renders from 3 to 1)
 * - useMemo for panel data (avoids unnecessary object recreations)
 * - Pure functions with empty dependencies (functions never recreated)
 */
const CodeComparisonView: React.FC<CodeComparisonViewProps> = ({
    leftPanel,
    rightPanel,
    py_code_info,
    showPythonSource = false,
    pythonMapping,
}) => {
    // ==================== State Management ====================

    /**
     * CSS class toggle optimization: Use Ref instead of State
     * Avoids component re-renders by implementing highlights via direct DOM manipulation
     */
    const highlightedLinesRef = useRef<HighlightState>({
        left: [],
        right: [],
        python: []
    });

    /**
     * Smart scrolling: only scroll container when element is not visible
     * Never scrolls the entire page, only scrolls the code container
     * @param container Scroll container (CodeViewer)
     * @param element Target element (code line)
     */
    const scrollToElementIfNeeded = useCallback((container: HTMLElement, element: HTMLElement) => {
        const containerRect = container.getBoundingClientRect();
        const elementRect = element.getBoundingClientRect();

        // Ensure container has valid dimensions
        if (containerRect.height === 0) return;

        // Calculate element position relative to container
        const elementTop = elementRect.top - containerRect.top;
        const elementBottom = elementRect.bottom - containerRect.top;

        // Container visible height
        const containerHeight = containerRect.height;

        // Check if element is within visible range (with 20px margin)
        const margin = 20;
        const isVisible =
            elementTop >= margin &&
            elementBottom <= containerHeight - margin;

        if (isVisible) {
            // Element is already visible, no scrolling needed
            return;
        }

        // Element is not visible, scroll container (not the page)
        if (elementTop < margin) {
            // Element is above viewport: scroll up to show it at top (with margin)
            container.scrollTop += elementTop - margin;
        } else if (elementBottom > containerHeight - margin) {
            // Element is below viewport: scroll down to show it at bottom (with margin)
            container.scrollTop += elementBottom - containerHeight + margin;
        }
    }, []);

    /**
     * Direct DOM manipulation to update highlights
     * Uses requestAnimationFrame to ensure DOM elements are rendered before updating
     * @param viewerId Panel ID ('left', 'right', 'python')
     * @param lineNumbers Array of line numbers to highlight
     */
    const updateHighlights = useCallback((
        viewerId: 'left' | 'right' | 'python',
        lineNumbers: number[]
    ) => {
        // Use requestAnimationFrame to ensure CodeViewer components have rendered
        requestAnimationFrame(() => {
            const container = document.querySelector(`[data-viewer-id="${viewerId}"]`) as HTMLElement;
            if (!container) {
                // CodeViewer not yet rendered, skip update
                return;
            }

            // Remove old highlights
            const oldLines = highlightedLinesRef.current[viewerId];
            oldLines.forEach(lineNum => {
                const element = container.querySelector(
                    `[data-line-number="${lineNum}"]`
                );
                element?.classList.remove('highlighted-line');
            });

            // Add new highlights
            lineNumbers.forEach(lineNum => {
                const element = container.querySelector(
                    `[data-line-number="${lineNum}"]`
                );
                element?.classList.add('highlighted-line');
            });

            // Update ref (does not trigger re-render)
            highlightedLinesRef.current[viewerId] = lineNumbers;

            // Smart scrolling: only scroll when necessary, only scroll container
            if (lineNumbers.length > 0) {
                const firstLine = Math.min(...lineNumbers);
                const element = container.querySelector(
                    `[data-line-number="${firstLine}"]`
                ) as HTMLElement;

                if (element) {
                    scrollToElementIfNeeded(container, element);
                }
            }
        });
    }, [scrollToElementIfNeeded]);

    // ==================== Memoized Computations ====================

    /**
     * Memoized left panel data
     * Only recomputes when actual panel props change
     */
    const leftPanel_data = useMemo<PanelData>(() => ({
        title: leftPanel.title || "TTGIR",
        content: leftPanel.content || leftPanel.code?.content || "",
        sourceMapping: leftPanel.code?.source_mapping || {},
        displayLanguage: getDisplayLanguage(leftPanel.title || "TTGIR")
    }), [leftPanel.title, leftPanel.content, leftPanel.code]);

    /**
     * Memoized right panel data
     * Only recomputes when actual panel props change
     */
    const rightPanel_data = useMemo<PanelData>(() => ({
        title: rightPanel.title || "PTX",
        content: rightPanel.content || rightPanel.code?.content || "",
        sourceMapping: rightPanel.code?.source_mapping || {},
        displayLanguage: getDisplayLanguage(rightPanel.title || "PTX")
    }), [rightPanel.title, rightPanel.content, rightPanel.code]);

    /**
     * Memoized Python source info
     * Extracts and caches all Python-related data
     */
    const pythonInfo = useMemo<PythonInfo>(() => ({
        code: py_code_info?.code || "",
        file_path: py_code_info?.file_path || "",
        start_line: py_code_info?.start_line || 1,
        isFullFileMode: py_code_info?.start_line === 1 &&
                       py_code_info?.function_start_line !== undefined,
        function_start_line: py_code_info?.function_start_line,
        function_end_line: py_code_info?.function_end_line,
    }), [py_code_info]);

    // ==================== Pure Utility Functions ====================

    /**
     * Pure function: Calculate mapped lines from source to target IR
     * No external dependencies - only uses parameters
     * @param sourceMappings Source mapping record
     * @param lineNumber Line number in source
     * @param targetTitle Target panel title (to determine IR type)
     * @returns Array of mapped line numbers
     */
    const calculateMappedLines = useCallback(
        (
            sourceMappings: Record<string, SourceMapping>,
            lineNumber: number,
            targetTitle: string
        ): number[] => {
            const lineKey = lineNumber.toString();
            if (!sourceMappings[lineKey]) return [];

            const sourceMapping = sourceMappings[lineKey];
            const targetIRType = getIRType(targetTitle);

            const irTypesToCheck = [
                { type: "ttgir", property: "ttgir_lines" },
                { type: "ttir", property: "ttir_lines" },
                { type: "ptx", property: "ptx_lines" },
                { type: "llir", property: "llir_lines" },
                { type: "amdgcn", property: "amdgcn_lines" }
            ];

            for (const { type, property } of irTypesToCheck) {
                if (targetIRType === type &&
                    sourceMapping[property as keyof SourceMapping] !== undefined) {
                    const lines = sourceMapping[property as keyof SourceMapping] as number[];
                    return lines.map(line =>
                        typeof line === "string" ? parseInt(line, 10) : line
                    );
                }
            }

            return [];
        },
        [] // Empty deps - pure function, never recreated
    );

    /**
     * Pure function: Calculate Python lines from IR source mapping
     * No external dependencies - all info passed via parameters
     * @param sourceMapping Source mapping record
     * @param lineNumber Line number in IR
     * @param pythonInfo Python file information
     * @returns Array of Python line numbers
     */
    const calculatePythonLines = useCallback(
        (
            sourceMapping: Record<string, SourceMapping>,
            lineNumber: number,
            pythonInfo: PythonInfo
        ): number[] => {
            if (!sourceMapping || !pythonInfo.code) return [];

            const lineKey = lineNumber.toString();
            const mapping = sourceMapping[lineKey];
            if (!mapping || !mapping.file || !mapping.line) return [];

            // Check if file path matches
            if (!mapping.file.includes(pythonInfo.file_path)) return [];

            const absoluteLine = Number(mapping.line);

            if (pythonInfo.isFullFileMode) {
                // Full file mode: use absolute line numbers directly
                return [absoluteLine];
            } else {
                // Function mode: convert absolute to relative line numbers
                const adjustedLine = absoluteLine - pythonInfo.start_line + 1;
                const pythonLines = pythonInfo.code.split("\n");

                if (adjustedLine >= 1 && adjustedLine <= pythonLines.length) {
                    return [adjustedLine];
                } else {
                    console.error(
                        `Adjusted line ${adjustedLine} is out of range (1-${pythonLines.length})`
                    );
                }
            }

            return [];
        },
        [] // Empty deps - pure function, never recreated
    );

    // ==================== Event Handlers ====================

    /**
     * Handle line click in left or right panel
     * Calculates all required highlights in one go, then updates state once
     * @param lineNumber Line number clicked
     * @param panel Which panel was clicked ('left' or 'right')
     */
    const handlePanelLineClick = useCallback(
        (lineNumber: number, panel: 'left' | 'right') => {
            const isLeftPanel = panel === 'left';

            // Get source and target panel data
            const sourceData = isLeftPanel ? leftPanel_data : rightPanel_data;
            const targetData = isLeftPanel ? rightPanel_data : leftPanel_data;

            // Step 1: Calculate target panel's mapped lines
            const targetLines = calculateMappedLines(
                sourceData.sourceMapping,
                lineNumber,
                targetData.title
            );

            // Step 2: Calculate Python panel's mapped lines
            const pythonLines = showPythonSource && py_code_info
                ? calculatePythonLines(
                    sourceData.sourceMapping,
                    lineNumber,
                    pythonInfo
                )
                : [];

            // Step 3: Update highlights via direct DOM manipulation (no re-render)
            updateHighlights('left', isLeftPanel ? [lineNumber] : targetLines);
            updateHighlights('right', isLeftPanel ? targetLines : [lineNumber]);
            updateHighlights('python', pythonLines);
        },
        [
            leftPanel_data,
            rightPanel_data,
            calculateMappedLines,
            calculatePythonLines,
            showPythonSource,
            py_code_info,
            pythonInfo
        ]
    );

    /**
     * Handle line click in Python panel
     * Finds corresponding lines in both IR panels
     * @param lineNumber Line number clicked in Python panel
     */
    const handlePythonLineClick = useCallback(
        (lineNumber: number) => {
            if (!pythonMapping) {
                return;
            }

            // Convert display line number to absolute line number
            const absoluteLineNumber = pythonInfo.isFullFileMode
                ? lineNumber
                : lineNumber + pythonInfo.start_line - 1;

            const mapping = pythonMapping[absoluteLineNumber.toString()];
            if (!mapping) {
                return;
            }

            // Find corresponding lines in left panel
            const leftLines = calculateMappedLines(
                { [absoluteLineNumber.toString()]: mapping },
                absoluteLineNumber,
                leftPanel_data.title
            );

            // Find corresponding lines in right panel
            const rightLines = calculateMappedLines(
                { [absoluteLineNumber.toString()]: mapping },
                absoluteLineNumber,
                rightPanel_data.title
            );

            // Update highlights via direct DOM manipulation (no re-render)
            updateHighlights('left', leftLines);
            updateHighlights('right', rightLines);
            updateHighlights('python', [lineNumber]);
        },
        [
            pythonMapping,
            pythonInfo,
            calculateMappedLines,
            leftPanel_data.title,
            rightPanel_data.title
        ]
    );

    /**
     * Curried function for left panel line clicks
     */
    const handleLeftLineClick = useCallback(
        (lineNumber: number) => handlePanelLineClick(lineNumber, 'left'),
        [handlePanelLineClick]
    );

    /**
     * Curried function for right panel line clicks
     */
    const handleRightLineClick = useCallback(
        (lineNumber: number) => handlePanelLineClick(lineNumber, 'right'),
        [handlePanelLineClick]
    );

    // ==================== Render ====================

    return (
        <PanelGroup direction="horizontal" style={{ height: '100%' }}>
            {/* Left Panel */}
            <Panel defaultSize={33} minSize={20}>
                <div style={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    position: "relative"
                }}>
                    <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
                        <span>{leftPanel_data.title}</span>
                        <div className="flex items-center gap-2">
                            <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                                {leftPanel_data.displayLanguage}
                            </span>
                            <CopyCodeButton
                                code={leftPanel_data.content}
                                className="text-sm bg-blue-700 px-2 py-1 rounded"
                            />
                        </div>
                    </div>
                    <div style={{ flex: 1, overflow: "hidden" }}>
                        <CodeViewer
                            code={leftPanel_data.content}
                            language={leftPanel_data.displayLanguage}
                            height="100%"
                            highlightedLines={[]}
                            onLineClick={handleLeftLineClick}
                            viewerId="left"
                            sourceMapping={leftPanel_data.sourceMapping}
                        />
                    </div>
                </div>
            </Panel>

            <PanelResizeHandle style={{
                width: "4px",
                backgroundColor: "#ddd",
                cursor: "col-resize"
            }} />

            {/* Right Panel */}
            <Panel defaultSize={33} minSize={20}>
                <div style={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    position: "relative"
                }}>
                    <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
                        <span>{rightPanel_data.title}</span>
                        <div className="flex items-center gap-2">
                            <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                                {rightPanel_data.displayLanguage}
                            </span>
                            <CopyCodeButton
                                code={rightPanel_data.content}
                                className="text-sm bg-blue-700 px-2 py-1 rounded"
                            />
                        </div>
                    </div>
                    <div style={{ flex: 1, overflow: "hidden" }}>
                        <CodeViewer
                            code={rightPanel_data.content}
                            language={rightPanel_data.displayLanguage}
                            height="100%"
                            highlightedLines={[]}
                            onLineClick={handleRightLineClick}
                            viewerId="right"
                            sourceMapping={rightPanel_data.sourceMapping}
                        />
                    </div>
                </div>
            </Panel>

            {/* Python Source Panel (Optional) */}
            {showPythonSource && py_code_info && (
                <>
                    <PanelResizeHandle style={{
                        width: "4px",
                        backgroundColor: "#ddd",
                        cursor: "col-resize"
                    }} />
                    <Panel defaultSize={34} minSize={20}>
                        <div style={{
                            height: "100%",
                            display: "flex",
                            flexDirection: "column",
                            position: "relative"
                        }}>
                            <div className="bg-blue-600 text-white p-2 font-medium flex justify-between items-center">
                                <span>
                                    {pythonInfo.isFullFileMode
                                        ? "Python Source (Full File)"
                                        : "Python Source"}
                                </span>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm bg-blue-700 px-2 py-1 rounded">
                                        python
                                    </span>
                                    <CopyCodeButton
                                        code={pythonInfo.code}
                                        className="text-sm bg-blue-700 px-2 py-1 rounded"
                                    />
                                </div>
                            </div>
                            <div style={{ flex: 1, overflow: "hidden" }}>
                                <CodeViewer
                                    code={pythonInfo.code}
                                    language="python"
                                    height="100%"
                                    highlightedLines={[]}
                                    onLineClick={handlePythonLineClick}
                                    viewerId="python"
                                    sourceMapping={pythonMapping}
                                    functionStartLine={pythonInfo.function_start_line}
                                    functionEndLine={pythonInfo.function_end_line}
                                    initialScrollToLine={
                                        pythonInfo.isFullFileMode
                                            ? pythonInfo.function_start_line
                                            : undefined
                                    }
                                />
                            </div>
                        </div>
                    </Panel>
                </>
            )}
        </PanelGroup>
    );
};

export default React.memo(CodeComparisonView);
