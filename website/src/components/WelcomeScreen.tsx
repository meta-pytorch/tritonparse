import React, { useState } from "react";
import { DocumentDuplicateIcon, DocumentPlusIcon, LinkIcon } from "./icons";

interface WelcomeScreenProps {
  loadDefaultData: () => void;
  handleFileSelected: (file: File) => void;
  openUrlInput: () => void;
}

/**
 * Welcome screen component shown when no data is loaded
 */
const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ loadDefaultData, handleFileSelected, openUrlInput }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  // Handle file input change
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Support NDJSON and compressed files only
      const fileName = file.name.toLowerCase();
      const isValidFile = fileName.endsWith(".ndjson") || fileName.endsWith(".gz");

      if (isValidFile) {
        handleFileSelected(file);
      }
    }
  };

  // Handle drag and drop events
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Support NDJSON and compressed files only
      const fileName = file.name.toLowerCase();
      const isValidFile = fileName.endsWith(".ndjson") || fileName.endsWith(".gz");

      if (isValidFile) {
        handleFileSelected(file);
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center px-4 py-16 max-w-4xl mx-auto text-center">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Welcome to TritonParse</h2>
      <p className="mb-8 text-gray-600">
        Load a Triton log file to analyze compiled kernels and their IR representations. Supports NDJSON and compressed
        (.gz) files.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-4xl mb-10">
        {/* Default Example Card */}
        <div
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
          onClick={loadDefaultData}
          style={{ cursor: "pointer" }}
        >
          <div className="bg-blue-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            <DocumentDuplicateIcon className="h-6 w-6 text-blue-500" />
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Default Example</h3>
          <p className="text-sm text-gray-600">Load the included example Triton log file</p>
        </div>

        {/* Local File Card */}
        <div
          className={`bg-white p-6 rounded-lg shadow-sm border border-gray-200 relative h-52 transition-all duration-200 ${isDragOver ? 'border-green-400 bg-green-200' : ''
            }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="bg-green-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            <DocumentPlusIcon className="h-6 w-6 text-green-500" />
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Local File</h3>
          <p className="text-sm text-gray-600 mb-4">
            {isDragOver ? 'Drop Triton log file here' : 'Open or drag a Triton log file (NDJSON or .gz)'}
          </p>
          <label htmlFor="welcomeFileInput" className="absolute inset-0 cursor-pointer" aria-label="Open local file" />
          <input
            type="file"
            id="welcomeFileInput"
            accept=".ndjson,.gz,application/x-ndjson,application/gzip"
            onChange={handleFileChange}
            className="hidden"
          />
        </div>

        {/* Remote URL Card */}
        <div
          className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-md transition-shadow"
          onClick={openUrlInput}
          style={{ cursor: "pointer" }}
        >
          <div className="bg-purple-50 p-3 rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4">
            <LinkIcon className="h-6 w-6 text-purple-500" />
          </div>
          <h3 className="text-lg font-medium text-gray-800 mb-2">Remote URL</h3>
          <p className="text-sm text-gray-600">Load a Triton log file from a URL</p>
        </div>
      </div>

      <div className="text-sm text-gray-500 max-w-2xl">
        <h4 className="font-medium mb-2">About TritonParse</h4>
        <p>
          TritonParse helps you analyze Triton GPU kernels by visualizing the compilation process across different IR
          stages.
        </p>
      </div>
    </div>
  );
};

export default WelcomeScreen;
