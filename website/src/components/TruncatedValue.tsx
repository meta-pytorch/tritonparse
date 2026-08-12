import React, { useId, useState } from "react";
import CopyCodeButton from "./CopyCodeButton";
import { ChevronRightIcon } from "./icons";

/**
 * Above this length the value is never expanded into the DOM. Trace fields can
 * reach several MB (an opaque `function` handle that some backends report as
 * the whole compiled binary), and painting that as a single text node freezes
 * the tab — an "expand" button there would be a trap rather than a feature.
 */
const NOT_EXPANDABLE_CHARS = 100_000;

interface TruncatedValueProps {
  /** The already-formatted value to display. */
  text: string;
  /** Length above which the value is collapsed behind a toggle. */
  maxChars: number;
  /** Classes applied to the rendered value. */
  className?: string;
}

/**
 * Renders a value, collapsing it behind a "Show more" toggle when it is long
 * enough to distort the surrounding layout.
 *
 * Only the preview is handed to React while collapsed, so a multi-MB value
 * costs nothing until the user asks for it. Structured viewers (arguments,
 * diffs, stacks) render their own markup and are deliberately left alone —
 * this is for values that are displayed as plain text.
 */
const TruncatedValue: React.FC<TruncatedValueProps> = ({
  text,
  maxChars,
  className = "",
}) => {
  const [expanded, setExpanded] = useState(false);
  const [expandedFor, setExpandedFor] = useState(text);
  const regionId = useId();

  // Callers key these by metadata field name, not by value, so switching
  // kernels reuses the same instance with different text. Without this the
  // field would stay open showing an unrelated value. React's documented
  // "adjust state during render" pattern; cheaper than an effect, which would
  // paint the stale expanded state first.
  if (expandedFor !== text) {
    setExpandedFor(text);
    setExpanded(false);
  }

  if (text.length <= maxChars) {
    return <span className={className}>{text}</span>;
  }

  // Too large to ever paint: describe it and offer the clipboard instead.
  // Root is a div, not a span: CopyCodeButton renders a block-level wrapper.
  if (text.length > NOT_EXPANDABLE_CHARS) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <span className="text-gray-500 italic">
          {text.length.toLocaleString()} characters — too large to display
        </span>
        <CopyCodeButton
          code={text}
          className="text-gray-500 hover:text-gray-700"
        />
      </div>
    );
  }

  if (!expanded) {
    return (
      <span className={className}>
        {text.slice(0, maxChars)}
        <span className="text-gray-500">…</span>
        {/* No aria-controls while collapsed: the region is genuinely absent
            from the DOM (that is the point), so referencing its id would be a
            dangling reference. */}
        <button
          onClick={() => setExpanded(true)}
          className="ml-2 inline-flex items-center text-xs text-blue-600 hover:text-blue-800 whitespace-nowrap"
          aria-expanded={false}
        >
          <ChevronRightIcon className="w-3 h-3 mr-0.5" />
          Show more ({text.length.toLocaleString()} chars)
        </button>
      </span>
    );
  }

  // Root is a div: <pre> is block-level and may not sit inside a <span>.
  return (
    <div className={className}>
      <button
        onClick={() => setExpanded(false)}
        className="mb-1 inline-flex items-center text-xs text-blue-600 hover:text-blue-800"
        aria-expanded={true}
        aria-controls={regionId}
      >
        <ChevronRightIcon className="w-3 h-3 mr-0.5 rotate-90" />
        Show less
      </button>
      {/* Bounded height: even an in-limit value can be thousands of lines. */}
      <pre
        id={regionId}
        className="max-h-96 overflow-auto whitespace-pre-wrap break-all bg-gray-50 p-2 rounded border border-gray-200"
      >
        {text}
      </pre>
    </div>
  );
};

export default TruncatedValue;
