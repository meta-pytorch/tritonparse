import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ChevronRightIcon } from './icons';

/**
 * Stack frame in a stack trace
 */
interface StackFrame {
  filename: string;
  line: number;
  name: string;
  line_code?: string;
}

/**
 * Launch range for distribution values
 */
interface LaunchRange {
  start: number;
  end: number;
}

/**
 * Distribution value item
 */
interface StackDistributionValue {
  value: StackFrame[];
  count: number;
  launches: LaunchRange[];
}

/**
 * Stack diff with distribution type
 */
interface StackDiff {
  diff_type: string;
  values: StackDistributionValue[];
}

// A single frame of a stack trace
const StackTraceFrame: React.FC<{ frame: StackFrame }> = ({ frame }) => (
  <div className="font-mono text-xs break-all">
    <span className="text-gray-500">{frame.filename}</span>:
    <span className="font-semibold text-blue-600">{frame.line}</span> in{" "}
    <span className="font-semibold text-green-700">{frame.name}</span>
    {frame.line_code && (
       <div className="pl-6 mt-1 bg-gray-100 rounded">
        <SyntaxHighlighter 
            language="python" 
            style={oneLight} 
            customStyle={{ 
                margin: 0, 
                padding: '0.25em 0.5em', 
                fontSize: '0.75rem',
                background: 'transparent'
             }}
        >
            {frame.line_code}
        </SyntaxHighlighter>
    </div>
    )}
  </div>
);


// eslint-disable-next-line @typescript-eslint/no-explicit-any -- stackDiff contains dynamic data from trace
const StackDiffViewer: React.FC<{ stackDiff: StackDiff | null | undefined }> = ({ stackDiff }) => {
  const [isCollapsed, setIsCollapsed] = useState(true);

  if (!stackDiff || stackDiff.diff_type !== 'distribution') {
    return null;
  }

  return (
    <div>
      <h5 
        className="text-md font-semibold mb-2 text-gray-700 cursor-pointer flex items-center"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        Stack Traces
        <ChevronRightIcon
          className={`w-4 h-4 ml-2 transform transition-transform ${isCollapsed ? '' : 'rotate-90'}`}
          strokeWidth={2}
        />
      </h5>
      {!isCollapsed && (
         <div className="space-y-2">
          {stackDiff.values.map((item: StackDistributionValue, index: number) => {
            const launchRanges = item.launches
              .map((r: LaunchRange) => (r.start === r.end ? `${r.start}` : `${r.start}-${r.end}`))
              .join(", ");
            
            return (
              <div key={index} className="bg-white p-2 rounded border border-gray-200">
                <p className="text-xs font-semibold text-gray-600 mb-1">
                  Variant seen {item.count} times (in launches: {launchRanges})
                </p>
                <div className="space-y-1 bg-gray-50 p-1 rounded">
                   {Array.isArray(item.value) ? item.value.map((frame: StackFrame, frameIndex: number) => (
                    <StackTraceFrame key={frameIndex} frame={frame} />
                  )) : <p>Invalid stack format</p>}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default StackDiffViewer; 