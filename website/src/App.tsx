import { useState, useEffect } from "react";
import "./App.css";
import {
  loadLogData,
  loadLogDataFromFile,
  ProcessedKernel,
  processKernelData,
  getIRType,
} from "./utils/dataLoader";
import { checkFbDirectoryExists } from "./utils/fbDetection";
import CodeView from "./pages/CodeView";
import FileDiffView from "./pages/FileDiffView";
import SingleCodeViewer from "./components/SingleCodeViewer";
import KernelOverview from "./pages/KernelOverview";
import IRAnalysis from "./pages/IRAnalysis";
import DataSourceSelector from "./components/DataSourceSelector";
import WelcomeScreen from "./components/WelcomeScreen";
import ExternalLink from "./components/ExternalLink";
import { mapLanguageToHighlighter } from "./components/CodeViewer";
import { useFileDiffSession } from "./context/FileDiffSession";
import { GitHubIcon, BookOpenIcon, ShareIcon } from "./components/icons";

/**
 * Main application component that handles data loading,
 * state management, and rendering different views.
 */
function App() {
  const sess = useFileDiffSession();
  // Store processed kernel data from log file
  const [kernels, setKernels] = useState<ProcessedKernel[]>([]);
  // Track loading state for displaying loading indicator
  const [loading, setLoading] = useState<boolean>(false);
  // Store error message if data loading fails
  const [error, setError] = useState<string | null>(null);
  // Track active tab (overview or code comparison)
  const [activeTab, setActiveTab] = useState<string>("overview");
  // Track which IR file is selected for viewing
  const [selectedIR, setSelectedIR] = useState<string | null>(null);
  // Track which kernel is currently selected
  const [selectedKernel, setSelectedKernel] = useState<number>(-1);
  // Track if data has been loaded
  const [dataLoaded, setDataLoaded] = useState<boolean>(false);
  // Track the loaded data source URL
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);
  // Track if fb directory exists for internal wiki link
  const [fbDirectoryExists, setFbDirectoryExists] = useState<boolean>(false);

  // Internal wiki URL - will be loaded dynamically if fb directory exists
  const [internalWikiUrl, setInternalWikiUrl] = useState<string>('');

  /**
   * Helper function to find a kernel by its hash
   */
  const findKernelByHash = (hash: string, kernels: ProcessedKernel[]): number => {
    return kernels.findIndex(kernel =>
      kernel.metadata?.hash === hash
    );
  };

  // Check URL parameters and fb directory when component mounts
  useEffect(() => {
    // Parse URL parameters
    const params = new URLSearchParams(window.location.search);
    const jsonUrl = params.get("json_url");
    const view = params.get("view");
    const kernelHash = params.get("kernel_hash");

    // If json_url parameter exists, load the data from it
    if (jsonUrl) {
      handleUrlSelected(jsonUrl, view, kernelHash);
      // Update the browser URL to include the json_url parameter
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("json_url", jsonUrl);
      window.history.replaceState({}, "", newUrl.toString());
    } else if (view === "file_diff") {
      // Allow direct navigation to File Diff even without json_url
      setActiveTab("file_diff");
    }

    // Check if fb directory exists and load internal utils if available
    checkFbDirectoryExists().then(async (exists) => {
      setFbDirectoryExists(exists);
      if (exists) {
        try {
          // Dynamically import internal utils only if fb directory exists
          // Use safe import utility to completely bypass static analysis
          const { safeImport } = await import('./utils/safeImport');
          const module = await safeImport('./utils/fb/internal_utils');
          if (module && module.getInternalWikiUrl) {
            const url = module.getInternalWikiUrl();
            setInternalWikiUrl(url);
          }
        } catch (error) {
          console.warn('Failed to load internal utils:', error);
        }
      }
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps -- handleUrlSelected is stable but ESLint cannot verify it
  }, []);

  /**
   * Generic data loading function that handles both URLs and local files
   * @param source - Either a URL string or a File object
   */
  const loadData = async (source: string | File) => {
    try {
      setLoading(true);
      setError(null);

      // Get URL parameters for view and kernel hash
      const params = new URLSearchParams(window.location.search);
      const view = params.get("view");
      const kernelHash = params.get("kernel_hash");

      // Load log entries based on source type
      const logEntries = typeof source === 'string'
        ? await loadLogData(source)
        : await loadLogDataFromFile(source);

      // Process raw log entries into kernel data structures
      const processedKernels = processKernelData(logEntries);

      if (processedKernels.length > 0) {
        setKernels(processedKernels);

        // First, determine which kernel to select
        let kernelIndex = 0; // Default to first kernel
        if (kernelHash) {
          const foundIndex = findKernelByHash(kernelHash, processedKernels);
          if (foundIndex >= 0) {
            kernelIndex = foundIndex;
          }
        }

        // Set the selected kernel
        setSelectedKernel(kernelIndex);

        // Then, determine which view to show
        if (view === "ir_code_comparison") {
          setActiveTab("comparison");
        } else if (view === "file_diff") {
          setActiveTab("file_diff");
        }

        setDataLoaded(true);
      } else {
        console.warn("No kernels found in the processed data");
        const errorMsg = typeof source === 'string'
          ? "No kernels found in the default data file. Please try loading a different file."
          : "No kernels found in the selected file. Please check the file format.";
        setError(errorMsg);
      }
    } catch (err) {
      console.error(`Error loading data from ${typeof source === 'string' ? 'URL' : 'file'}:`, err);
      const errorMsg = typeof source === 'string'
        ? `Failed to load default data: ${err instanceof Error ? err.message : String(err)}`
        : `Failed to load from local file: ${err instanceof Error ? err.message : String(err)}`;
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Loads the default data file from public directory
   */
  const loadDefaultData = async () => {
    const logFile = "./dedicated_log_triton_trace_findhao__mapped.ndjson.gz";
    await loadData(logFile);
  };

  /**
   * Handles loading data from a local file
   */
  const handleFileSelected = async (file: File) => {
    await loadData(file);
  };

  /**
   * Handles loading data from a custom URL
   */
  const handleUrlSelected = async (url: string, initialView?: string | null, kernelHash?: string | null) => {
    try {
      setLoading(true);
      setError(null);

      const logEntries = await loadLogData(url);

      // Process raw log entries into kernel data structures
      const processedKernels = processKernelData(logEntries);

      if (processedKernels.length > 0) {
        setKernels(processedKernels);

        // First, determine which kernel to select
        let kernelIndex = 0; // Default to first kernel
        if (kernelHash) {
          const foundIndex = findKernelByHash(kernelHash, processedKernels);
          if (foundIndex >= 0) {
            kernelIndex = foundIndex;
          } else {
            console.log(`Kernel hash ${kernelHash} not found, selected first kernel`);
          }
        }

        // Set the selected kernel
        setSelectedKernel(kernelIndex);

        // Then, determine which view to show
        if (initialView === "ir_code_comparison") {
          setActiveTab("comparison");
        } else if (initialView === "file_diff") {
          setActiveTab("file_diff");
        }
        setDataLoaded(true);
        setLoadedUrl(url);

        // Update URL parameters
        const newUrl = new URL(window.location.href);
        newUrl.searchParams.set("json_url", url);

        // Add view and kernel_hash parameters if applicable
        if (initialView === "ir_code_comparison") {
          newUrl.searchParams.set("view", "ir_code_comparison");
        } else if (initialView === "file_diff") {
          newUrl.searchParams.set("view", "file_diff");
        }

        if (kernelHash) {
          const foundIndex = findKernelByHash(kernelHash, processedKernels);
          if (foundIndex >= 0) {
            newUrl.searchParams.set("kernel_hash", kernelHash);
          }
        }

        window.history.replaceState({}, "", newUrl.toString());
      } else {
        console.warn("No kernels found in the URL data");
        setError("No kernels found in the URL data. Please check the file format.");
      }
    } catch (err) {
      console.error("Error loading data from URL:", err);
      setError(`Failed to load from URL: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Opens the URL input field
   */
  const openUrlInput = () => {
    // Find the URL button and click it to activate the input field
    const urlButton = document.querySelector('button:has(svg path[d*="12.586 4.586"])');
    if (urlButton && urlButton instanceof HTMLElement) {
      urlButton.click();
      // Focus on the input field
      setTimeout(() => {
        const urlInput = document.querySelector('input[type="url"]');
        if (urlInput && urlInput instanceof HTMLInputElement) {
          urlInput.focus();
        }
      }, 100);
    }
  };

  /**
   * Handles selection of an IR file for detailed viewing
   */
  const handleViewSingleIR = (irType: string) => {
    setSelectedIR(irType);
  };

  /**
   * Handles returning from single IR view to overview/comparison
   */
  const handleBackFromIRView = () => {
    setSelectedIR(null);
  };

  /**
   * Handles selection of a kernel from the list
   */
  const handleSelectKernel = (index: number) => {
    setSelectedKernel(index);
    setSelectedIR(null);

    // Update URL with the selected kernel hash
    if (loadedUrl && kernels[index]?.metadata?.hash) {
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.set("kernel_hash", kernels[index].metadata.hash);
      window.history.replaceState({}, "", newUrl.toString());
    }
  };

  // Register app controls for FileDiffSession navigation
  useEffect(() => {
    sess.registerAppControls({
      setActiveTab,
    });
  }, [sess, setActiveTab]);

  // Clear FileDiff preview when entering/returning to File Diff to avoid preview intercepting other views
  useEffect(() => {
    if (activeTab === 'file_diff' && sess.preview.active) {
      sess.clearPreview();
    }
  }, [activeTab, sess]);

  // Show loading indicator while data is being fetched
  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen w-full bg-gray-50 text-gray-800">
        <div className="text-center">
          <div className="text-xl mb-4">Loading Triton compilation data...</div>
          <div className="animate-pulse">Please wait...</div>
        </div>
      </div>
    );
  }

  /**
   * Renders the main content based on current state (selected IR, kernel, and active tab)
   */
  const renderContent = () => {
    if (selectedIR && selectedKernel >= 0 && kernels.length > 0) {
      // Display single IR view
      const kernel = kernels[selectedKernel];
      if (!kernel) {
        console.error(`Selected kernel index ${selectedKernel} not found in kernels array of length ${kernels.length}`);
        return <div className="text-red-600">Error: Selected kernel not found</div>;
      }

      const irContent = kernel.irFiles[selectedIR];
      if (!irContent) {
        console.error(`IR file ${selectedIR} not found in kernel ${kernel.name}`);
        return <div className="text-red-600">Error: Selected IR file not found in kernel</div>;
      }

      // Create IRFile object with content and potential source mapping
      const irFile = {
        content: irContent,
        // Add source mapping if available in the kernel data
        source_mapping: kernel.sourceMappings?.[getIRType(selectedIR)],
      };

      return (
        <SingleCodeViewer
          irFile={irFile}
          title={selectedIR}
          language={mapLanguageToHighlighter(selectedIR)}
          onBack={handleBackFromIRView}
        />
      );
    } else if (sess.preview.active) {
      const side = sess.preview.side === 'left' ? sess.left : sess.right;
      const idx = side.selectedIdx ?? 0;
      const kernelsSrc = side.kernels ?? [];
      if (!kernelsSrc || kernelsSrc.length === 0) {
        return <div className="text-red-600">Error: Preview kernel data not available.</div>;
      }
      if (sess.preview.view === 'overview') {
        return (
          <KernelOverview
            key={`preview-overview-${sess.preview.side}-${idx}`}
            kernels={kernelsSrc}
            onViewIR={handleViewSingleIR}
            selectedKernel={idx}
            onSelectKernel={() => {}}
          />
        );
      }
      if (sess.preview.view === 'ir') {
        return (
          <CodeView
            key={`preview-ir-${sess.preview.side}-${idx}`}
            kernels={kernelsSrc}
            selectedKernel={idx}
          />
        );
      }
      return null;
    } else if (!dataLoaded) {
      // Show welcome screen if no data is loaded, but allow File Diff tab to render its view when selected
      if (activeTab === 'file_diff') {
        return (
          <FileDiffView
            kernelsLeft={kernels}
            selectedLeftIndex={Math.max(0, selectedKernel)}
            leftLoadedUrl={loadedUrl}
          />
        );
      }
      return (
        <WelcomeScreen
          loadDefaultData={loadDefaultData}
          handleFileSelected={handleFileSelected}
          openUrlInput={openUrlInput}
        />
      );
    } else if (kernels.length === 0) {
      // Show message when no kernels are found
      return (
        <div className="flex items-center justify-center p-8">
          <div className="bg-yellow-50 p-6 rounded-lg border border-yellow-200 max-w-2xl">
            <h2 className="text-xl font-semibold text-yellow-800 mb-3">No Kernel Data Found</h2>
            <p className="text-yellow-700">
              No Triton kernels were found in the log file. Please make sure you're using the correct log file and that
              it contains Triton kernel information.
            </p>
          </div>
        </div>
      );
    } else {
      // Show either overview, IR code, IR analysis, or file diff based on active tab
      if (activeTab === "overview") {
        return (
          <KernelOverview
            kernels={kernels}
            onViewIR={handleViewSingleIR}
            selectedKernel={selectedKernel}
            onSelectKernel={handleSelectKernel}
          />
        );
      }
      if (activeTab === "ir_analysis") {
        return (
          <IRAnalysis
            kernels={kernels}
            selectedKernel={selectedKernel}
          />
        );
      }
      if (activeTab === "comparison") {
        return (
          <CodeView
            key={`codeview-main-${selectedKernel}`}
            kernels={kernels}
            selectedKernel={selectedKernel}
          />
        );
      }
      if (activeTab === "file_diff") {
        return (
          <FileDiffView
            kernelsLeft={kernels}
            selectedLeftIndex={Math.max(0, selectedKernel)}
            leftLoadedUrl={loadedUrl}
          />
        );
      }
      return null;
    }
  };

  return (
    <div className="min-h-screen w-full bg-gray-50 flex flex-col">
      {/* Header with navigation */}
      <header className="bg-white border-b border-gray-200">
        <div className="w-full px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-6">
              <h1
                className="text-gray-800 text-2xl font-bold cursor-pointer hover:text-indigo-600 transition-colors"
                onClick={() => {
                  // Reset app state to show welcome page
                  if (dataLoaded) {
                    setDataLoaded(false);
                    setSelectedIR(null);
                    setSelectedKernel(-1);
                    setError(null);
                  }
                }}
                title="Back to home"
              >
                TritonParse
              </h1>

              {/* GitHub and Wiki Links */}
              <div className="flex items-center space-x-4">
                <ExternalLink
                  href="https://github.com/meta-pytorch/tritonparse"
                  title="View on GitHub"
                  text="GitHub"
                  icon={<GitHubIcon className="w-5 h-5" />}
                />

                <ExternalLink
                  href="https://github.com/meta-pytorch/tritonparse/wiki"
                  title="View Wiki"
                  text="Wiki"
                  icon={<BookOpenIcon className="w-5 h-5" />}
                />

                {/* Internal Wiki Link - only show if fb directory exists */}
                {fbDirectoryExists && (
                  <ExternalLink
                    href={internalWikiUrl}
                    title="View Internal Wiki"
                    text="Internal Wiki"
                    icon={<BookOpenIcon className="w-5 h-5" />}
                  />
                )}
              </div>
            </div>

            {/* Data source selector (only show when data is loaded or when on the welcome screen) */}
            <DataSourceSelector
              onFileSelected={handleFileSelected}
              onUrlSelected={handleUrlSelected}
              isLoading={loading}
            />

            {/* Tab navigation: File Diff button placed as the last (rightmost) button */}
            <div className="flex space-x-4">
              {dataLoaded && kernels.length > 0 && !selectedIR && (
                <>
                  <button
                    className={`px-3 py-2 text-sm font-medium rounded-md ${activeTab === "overview" ? "bg-blue-700 text-white shadow-md" : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                      }`}
                    onClick={() => {
                      if (sess.preview?.active) sess.clearPreview();
                      setActiveTab("overview");

                      if (loadedUrl) {
                        const newUrl = new URL(window.location.href);
                        newUrl.searchParams.delete("view");
                        window.history.replaceState({}, "", newUrl.toString());
                      }
                    }}
                  >
                    Kernel Overview
                  </button>
                  <button
                    className={`px-3 py-2 text-sm font-medium rounded-md ${activeTab === "comparison" ? "bg-blue-700 text-white shadow-md" : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                      }`}
                    onClick={() => {
                      if (sess.preview?.active) sess.clearPreview();
                      setActiveTab("comparison");

                      if (loadedUrl) {
                        const newUrl = new URL(window.location.href);
                        newUrl.searchParams.set("view", "ir_code_comparison");
                        window.history.replaceState({}, "", newUrl.toString());
                      }
                    }}
                  >
                    IR Code
                  </button>
                </>
              )}
              <button
                className={`px-3 py-2 text-sm font-medium rounded-md ${activeTab === "file_diff" ? "bg-blue-700 text-white shadow-md" : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                  }`}
                onClick={() => {
                  if (sess.preview?.active) sess.clearPreview();
                  setActiveTab("file_diff");
                }}
              >
                File Diff
              </button>
              {dataLoaded && kernels.length > 0 && (
              <button
                    className={`px-3 py-2 text-sm font-medium rounded-md ${activeTab === "ir_analysis" ? "bg-blue-700 text-white shadow-md" : "bg-blue-100 text-blue-700 hover:bg-blue-200"
                      }`}
                    onClick={() => {
                      if (sess.preview?.active) sess.clearPreview();
                      setActiveTab("ir_analysis");

                      if (loadedUrl) {
                        const newUrl = new URL(window.location.href);
                        newUrl.searchParams.set("view", "ir_analysis");
                        window.history.replaceState({}, "", newUrl.toString());
                      }
                    }}
                  >
                    IR Analysis (Beta)
                  </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="w-full p-6 flex-grow">
        {/* Show error message if data loading failed */}
        {error && <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">{error}</div>}

        {/* Show loaded URL if available */}
        {loadedUrl && dataLoaded && (
          <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded mb-6 flex items-center justify-between">
            <div>
              <span className="font-medium">Loaded from: </span>
              <span className="break-all">{loadedUrl}</span>
            </div>
            <button
              onClick={() => {
                // Create a shareable URL with the current json_url
                const shareableUrl = new URL(window.location.href);

                // Add or remove view parameter
                if (activeTab === "comparison") {
                  shareableUrl.searchParams.set("view", "ir_code_comparison");
                } else {
                  shareableUrl.searchParams.delete("view");
                }

                // Add or remove kernel_hash parameter
                if (selectedKernel >= 0 && kernels[selectedKernel]?.metadata?.hash) {
                  shareableUrl.searchParams.set("kernel_hash", kernels[selectedKernel].metadata.hash);
                } else {
                  shareableUrl.searchParams.delete("kernel_hash");
                }

                navigator.clipboard
                  .writeText(shareableUrl.toString())
                  .then(() => alert("Shareable link copied to clipboard!"))
                  .catch((err) => console.error("Failed to copy link:", err));
              }}
              className="ml-4 bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-sm flex items-center"
            >
              <ShareIcon className="h-4 w-4 mr-1" />
              Share
            </button>
          </div>
        )}

        {renderContent()}
      </main>

      {/* Footer with version info */}
      <footer className="w-full py-1 px-6 border-t border-gray-200 bg-white mt-auto">
        <div className="container mx-auto flex justify-between items-center text-sm">
          <div className="text-gray-500">
            Copyright © {new Date().getFullYear()} Meta Platforms, Inc
            <span className="mx-2">|</span>
            <a href="https://opensource.fb.com/legal/terms" className="hover:underline">Terms of Use</a>
            <span className="mx-2">|</span>
            <a href="https://opensource.fb.com/legal/privacy" className="hover:underline">Privacy Policy</a>
          </div>
          <div className="text-gray-500">
            {(() => {
              const buildDate = new Date(import.meta.env.PACKAGE_BUILD_DATE)
              const localized = isNaN(buildDate.getTime())
                ? import.meta.env.PACKAGE_BUILD_DATE
                : buildDate.toLocaleString()
              const sha = import.meta.env.GIT_COMMIT_SHA_SHORT
              return `Version ${import.meta.env.PACKAGE_VERSION} · Built ${localized} · ${sha}`
            })()}
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
