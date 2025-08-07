/**
 * Source mapping information that connects lines in IR code to source code
 */
export interface SourceMapping {
    line: number;
    file?: string;
    column?: number;
    // The {ir_type}_line fields are the line numbers in the current IR file that corresponds to
    // the current line in the source code. It should be same with the key in the source_mapping.
    ttgir_line?: number;
    ttir_line?: number;
    ptx_line?: number;
    amdgcn_line?: number;
    llir_line?: number;
    ptx_lines?: number[]; // Array of corresponding PTX lines
    ttir_lines?: number[]; // Array of corresponding TTIR lines
    ttgir_lines?: number[]; // Array of corresponding TTGIR lines
    llir_lines?: number[]; // Array of corresponding LLIR lines
    amdgcn_lines?: number[]; // Array of corresponding AMDGCN lines
}

/**
 * Get IR type from file name
 * @param fileName - The name of the file
 * @returns The type of IR file (without the dot)
 */
export function getIRType(fileName: string): string {
    // 1. Extract the file extension
    const extMatch = fileName.toLowerCase().match(/\.([^.]+)$/);
    if (extMatch) return extMatch[1];

    // 2. If there is no extension or the format does not match, return the original value
    return fileName.toLowerCase();
}

/**
 * Represents an IR file with content and source mapping information
 */
export interface IRFile {
    content: string;
    source_mapping?: Record<string, SourceMapping>;
}


/**
 * Represents a stack trace entry with line, function name, file info
 */
export interface StackEntry {
    line: number;
    name: string;
    filename: string | number | [string, number]; // Can be index, array [filepath, index], or filepath string
    loc: string;
}

/**
 * Represents kernel compilation metadata
 */
export interface KernelMetadata {
    hash?: string;
    target?: {
        backend?: string;
        arch?: number;
        warp_size?: number;
    };
    num_warps?: number;
    num_ctas?: number;
    num_stages?: number;
    maxnreg?: number | null;
    cluster_dims?: number[];
    ptx_version?: number | null;
    enable_fp_fusion?: boolean;
    launch_cooperative_grid?: boolean;
    supported_fp8_dtypes?: string[];
    [key: string]: any; // For other metadata properties
}

/**
 * Launch range information
 */
export interface LaunchRange {
    start: number;
    end: number;
}

/**
 * Distribution value with count and launch information
 */
export interface DistributionValue<T = any> {
    value: T;
    count: number;
    launches: LaunchRange[];
}

/**
 * Different types of diff structures
 */
export interface SummaryDiff {
    diff_type: "summary";
    summary_text: string;
}

export interface DistributionDiff<T = any> {
    diff_type: "distribution";
    values: DistributionValue<T>[];
}

export interface ArgumentDiff {
    diff_type: "argument_diff";
    sames?: Record<string, any>;
    diffs?: Record<string, SummaryDiff | DistributionDiff>;
}

/**
 * Union type for all diff types
 */
export type DiffData = SummaryDiff | DistributionDiff | ArgumentDiff;

/**
 * Launch diff data structure
 */
export interface LaunchDiffData {
    function?: DiffData;
    stack?: DiffData;
    extracted_args?: Record<string, ArgumentDiff>;
    [key: string]: DiffData | Record<string, ArgumentDiff> | undefined;
}

/**
 * Compilation metadata for launch events
 */
export interface CompilationMetadata {
    allowed_dot_input_precisions?: string[];
    arch?: string;
    backend_name?: string;
    cluster_dims?: number[];
    debug?: boolean;
    default_dot_input_precision?: string;
    deprecated_fp8_dot_operand_dtypes?: string[];
    enable_fp_fusion?: boolean;
    extern_libs?: [string, string][];
    global_scratch_align?: number;
    global_scratch_size?: number;
    hash?: string;
    ir_override?: any;
    launch_cooperative_grid?: boolean;
    launch_pdl?: boolean;
    max_num_imprecise_acc_default?: number;
    maxnreg?: number | null;
    name?: string;
    num_ctas?: number;
    num_stages?: number;
    num_warps?: number;
    ptx_options?: any;
    ptx_version?: number | null;
    sanitize_overflow?: boolean;
    shared?: number;
    supported_fp8_dtypes?: string[];
    target?: {
        backend?: string;
        arch?: number;
        warp_size?: number;
    };
    tensordesc_meta?: any[];
    tmem_size?: number;
    triton_version?: string;
    warp_size?: number;
    [key: string]: any; // Allow additional unknown fields
}

/**
 * Extracted argument information
 */
export interface ExtractedArg {
    type: string;
    value?: any;
    length?: number;
    [key: string]: any; // Allow additional unknown fields
}

/**
 * Launch sames data structure
 */
export interface LaunchSamesData {
    event_type?: string;
    pid?: number;
    name?: string;
    stream?: number;
    grid?: number[];
    compilation_metadata?: CompilationMetadata;
    timestamp?: string;
    extracted_args?: Record<string, ExtractedArg>;
    [key: string]: any;
}

/**
 * Python source code information
 */
export interface PythonSourceCodeInfo {
    file_path: string;
    start_line: number;
    end_line?: number; // End line number (inclusive)
    code?: string;
}

/**
 * Raw log entry from the Triton trace log
 */
export interface LogEntry {
    event_type: string;
    pid?: number;
    stack?: StackEntry[];
    timestamp?: string; // Format: "2025-03-25T13:22:04.%fZ"
    payload?: {
        metadata?: KernelMetadata;
        file_path?: Record<string, string>; // Mapping from filename to filepath
        file_content?: Record<string, string>; // Mapping from filename to content
        source_mappings?: Record<string, Record<string, SourceMapping>>; // Alternative field name for source_mapping
        python_source?: PythonSourceCodeInfo;
    };
    // Fields for launch_diff event type
    hash?: string;
    name?: string;
    total_launches?: number;
    launch_index_map?: LaunchRange[];
    diffs?: LaunchDiffData;
    sames?: LaunchSamesData;
}

/**
 * Processed kernel data structure for rendering in the UI
 */
export interface ProcessedKernel {
    name: string; // Inferred from filename
    stack: StackEntry[];
    irFiles: Record<string, string>; // IR file contents
    filePaths: Record<string, string>; // IR file paths
    sourceMappings?: Record<string, Record<string, SourceMapping>>; // Source mappings for each IR file
    pythonSource?: PythonSourceCodeInfo;
    metadata?: KernelMetadata; // Compilation metadata
    launchDiff?: any;
    autotuneAnalyses: AutotuneAnalysisEvent[]; // New field for autotune data
}

/**
 * Parses NDJSON text data (Newline Delimited JSON)
 * @param textData - The NDJSON text data to parse
 * @returns Array of LogEntry objects
 */
export function parseLogData(textData: string): LogEntry[] {
    console.log("Starting to parse NDJSON data...");
    if (typeof textData !== 'string') {
        throw new Error("Input must be a string in NDJSON format");
    }

    try {
        const lines = textData.split('\n').filter((line: string) => line.trim() !== '');
        const entries: LogEntry[] = [];

        for (const line of lines) {
            try {
                const parsedLine: LogEntry = JSON.parse(line);
                if (parsedLine && typeof parsedLine === 'object') {
                    entries.push(parsedLine);
                }
            } catch (e) {
                console.warn(`Failed to parse line as JSON: ${line.substring(0, 100)}...`);
                // Continue processing other lines even if one fails
            }
        }

        if (entries.length === 0) {
            console.error("No valid JSON entries found in NDJSON data");
            throw new Error("No valid JSON entries found in NDJSON data");
        }

        console.log(`Successfully parsed ${entries.length} log entries.`);
        return entries;
    } catch (error) {
        console.error("Error parsing NDJSON data:", error);
        throw error;
    }
}

/**
 * Detects if a file is in gzip format by checking its header bytes
 * @param buffer - ArrayBuffer containing the file data
 * @returns Boolean indicating if the file is in gzip format
 */
function isGzipFile(buffer: ArrayBuffer): boolean {
    // Check for gzip magic number (first two bytes should be 0x1F, 0x8B)
    const header = new Uint8Array(buffer.slice(0, 2));
    return header[0] === 0x1F && header[1] === 0x8B;
}

/**
 * Decompresses a gzip file using CompressionStream API
 * @param buffer - ArrayBuffer containing the gzip data
 * @returns Promise resolving to decompressed text
 */
async function decompressGzip(buffer: ArrayBuffer): Promise<string> {
    try {
        // Check if CompressionStream is supported
        if (!('DecompressionStream' in window)) {
            throw new Error('DecompressionStream API is not supported in this browser');
        }

        // Create a decompression stream
        const ds = new DecompressionStream('gzip');
        const decompressedStream = new Blob([buffer]).stream().pipeThrough(ds);
        const decompressedBlob = await new Response(decompressedStream).blob();
        return await decompressedBlob.text();
    } catch (error) {
        console.error('Error decompressing gzip file:', error);
        throw new Error(`Failed to decompress gzip file: ${error instanceof Error ? error.message : String(error)}`);
    }
}

/**
 * Processes ArrayBuffer data, handling gzip decompression if needed
 * @param buffer - ArrayBuffer containing the data
 * @returns Promise resolving to text string
 */
async function processArrayBuffer(buffer: ArrayBuffer): Promise<string> {
    // Check if file is gzip compressed
    if (isGzipFile(buffer)) {
        try {
            const textData = await decompressGzip(buffer);
            return textData;
        } catch (error) {
            console.error("Error decompressing gzip data:", error);
            throw new Error(`Failed to decompress gzip data: ${error instanceof Error ? error.message : String(error)}`);
        }
    } else {
        // Convert ArrayBuffer to string if it's not compressed
        const decoder = new TextDecoder();
        const textData = decoder.decode(buffer);
        return textData;
    }
}

/**
 * Loads log data from a URL and parses it as NDJSON
 * @param url - The URL of the log file to load
 * @returns Promise resolving to an array of LogEntry objects
 */
export async function loadLogData(url: string): Promise<LogEntry[]> {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load log data: ${response.statusText}`);
        }

        const buffer = await response.arrayBuffer();
        const textData = await processArrayBuffer(buffer);

        return parseLogData(textData);
    } catch (error) {
        console.error("Error loading log data:", error);
        throw error;
    }
}


/**
 * Loads log data from a local file using FileReader
 * @param file - The File object to load
 * @returns Promise resolving to an array of LogEntry objects
 */
export function loadLogDataFromFile(file: File): Promise<LogEntry[]> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = async (event) => {
            try {
                if (!event.target || !event.target.result) {
                    throw new Error("Failed to read file");
                }

                const result = event.target.result;
                if (!(result instanceof ArrayBuffer)) {
                    throw new Error("Expected ArrayBuffer from FileReader");
                }

                const textData = await processArrayBuffer(result);
                resolve(parseLogData(textData));
            } catch (error) {
                console.error("Error parsing data from file:", error);
                reject(error);
            }
        };

        reader.onerror = () => {
            reject(new Error("Error reading file"));
        };

        reader.readAsArrayBuffer(file);
    });
}

/**
 * Process raw log entries to extract kernel information
 * @param logEntries - Array of log entries from the trace file
 * @returns Array of processed kernel objects ready for display
 */
export function processKernelData(logEntries: LogEntry[]): ProcessedKernel[] {
    const kernelsByHash: Map<string, ProcessedKernel> = new Map();
    const autotuneEvents: AutotuneAnalysisEvent[] = [];

    // First pass: process all compilation events
    for (const entry of logEntries) {
        if (entry.event_type === "compilation" && entry.payload) {
            const hash = entry.payload.metadata?.hash;
            if (!hash) {
                console.warn("Compilation event missing hash", entry);
                continue;
            }

            if (!entry.payload.file_path || !entry.payload.file_content) {
                continue;
            }

            const irFileNames = Object.keys(entry.payload.file_path);
            let kernelName = "unknown_kernel";
            if (irFileNames.length > 0) {
                const fileName = irFileNames[0];
                const nameParts = fileName.split(".");
                kernelName =
                    nameParts.length > 1
                        ? nameParts.slice(0, -1).join(".")
                        : fileName;
            }

            const newKernel: ProcessedKernel = {
                name: kernelName,
                stack: entry.stack || [],
                irFiles: entry.payload.file_content,
                filePaths: entry.payload.file_path,
                sourceMappings: entry.payload.source_mappings,
                pythonSource: entry.payload.python_source,
                metadata: entry.payload.metadata,
                autotuneAnalyses: [], // Initialize with an empty array
            };
            kernelsByHash.set(hash, newKernel);
        } else if (entry.event_type === "autotune_analysis") {
            autotuneEvents.push(entry);
        }
    }

    // Second pass: attach launch_diff events
    for (const entry of logEntries) {
        if (entry.event_type === "launch_diff") {
            const hash = entry.hash;
            if (hash && kernelsByHash.has(hash)) {
                const kernel = kernelsByHash.get(hash)!;
                kernel.launchDiff = entry; // Attach the entire event object
            } else {
                console.warn(
                    `Could not find matching kernel for launch_diff hash: ${hash}`
                );
            }
        }
    }

    // Third pass: Link autotune analysis events to kernels
    for (const autotuneEvent of autotuneEvents) {
        if (autotuneEvent.compilation_hashes) {
            for (const hash of autotuneEvent.compilation_hashes) {
                const kernel = kernelsByHash.get(hash);
                if (kernel) {
                    kernel.autotuneAnalyses.push(autotuneEvent);
                }
            }
        }
    }

    return Array.from(kernelsByHash.values());
}

export interface AutotuneAnalysisEvent {
  event_type: "autotune_analysis";
  session_id: string;
  name: string;
  selected_hash: string | null;
  compilation_hashes: string[];
  common_info: {
    stack: any[];
    python_source: any;
  };
  configs: {
    config_params: Record<string, any>;
    compilation_hash: string;
  }[];
}
