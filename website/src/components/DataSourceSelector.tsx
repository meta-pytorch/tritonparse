import { useState } from "react";

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
            {/* SVG icon representing a paperclip/attachment for the file upload button */}
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5 mr-2 text-gray-500"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M8 4a3 3 0 00-3 3v4a5 5 0 0010 0V7a1 1 0 112 0v4a7 7 0 11-14 0V7a5 5 0 0110 0v4a3 3 0 11-6 0V7a1 1 0 012 0v4a1 1 0 102 0V7a3 3 0 00-3-3z"
                clipRule="evenodd"
              />
            </svg>
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
          {/* SVG icon representing a link/chain for the URL input button */}
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5 mr-2 text-gray-500"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z"
              clipRule="evenodd"
            />
          </svg>
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
