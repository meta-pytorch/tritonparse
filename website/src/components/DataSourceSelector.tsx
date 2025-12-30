import { useState } from "react";
import { PaperClipIcon, LinkIcon } from "./icons";

interface DataSourceSelectorProps {
  onFileSelected: (file: File) => void;
  onUrlSelected: (url: string) => void;
  isLoading: boolean;
}

/**
 * Component for selecting data sources - either local file or URL
 */
const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({ onFileSelected, onUrlSelected, isLoading }) => {
  const [showUrlInput, setShowUrlInput] = useState(false);
  const [url, setUrl] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Support NDJSON and compressed files only
      const fileName = file.name.toLowerCase();
      const isValidFile =
        fileName.endsWith(".ndjson") || fileName.endsWith(".gz") || file.type === "application/x-ndjson";

      if (isValidFile) {
        setError(null);
        onFileSelected(file);
      } else {
        setError("Please select an NDJSON or compressed file");
      }
    }
  };

  const handleUrlSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) {
      setError("Please enter a URL");
      return;
    }

    try {
      // Basic URL validation
      new URL(url);
      setError(null);
      onUrlSelected(url);
    } catch {
      setError("Please enter a valid URL");
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-sm mb-4">
      <div className="flex flex-wrap items-center gap-3">
        {/* Local file input */}
        <div>
          <label
            htmlFor="fileInput"
            className={`inline-flex items-center px-4 py-2 border border-gray-300 rounded-md font-medium text-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 cursor-pointer ${
              isLoading ? "opacity-50 cursor-not-allowed" : ""
            }`}
          >
            <PaperClipIcon className="h-5 w-5 mr-2 text-gray-500" />
            Open Local File
          </label>
          <input
            type="file"
            id="fileInput"
            accept=".ndjson,.gz,application/x-ndjson,application/gzip"
            onChange={handleFileChange}
            disabled={isLoading}
            className="hidden"
          />
        </div>

        {/* URL input toggle button */}
        <button
          type="button"
          onClick={() => setShowUrlInput(!showUrlInput)}
          className={`inline-flex items-center px-4 py-2 border border-gray-300 rounded-md font-medium text-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
            isLoading ? "opacity-50 cursor-not-allowed" : ""
          }`}
          disabled={isLoading}
        >
          <LinkIcon className="h-5 w-5 mr-2 text-gray-500" />
          Load from URL
        </button>
      </div>

      {/* URL input form */}
      {showUrlInput && (
        <form onSubmit={handleUrlSubmit} className="mt-3">
          <div className="flex items-center">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="Enter NDJSON file URL"
              className="flex-1 p-2 border border-gray-300 rounded-l-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm"
              disabled={isLoading}
            />
            <button
              type="submit"
              className={`inline-flex items-center px-4 py-2 border border-transparent rounded-r-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 ${
                isLoading ? "opacity-50 cursor-not-allowed" : ""
              }`}
              disabled={isLoading}
            >
              Load
            </button>
          </div>
        </form>
      )}

      {/* Error message */}
      {error && <div className="mt-2 text-sm text-red-600">{error}</div>}
    </div>
  );
};

export default DataSourceSelector;
