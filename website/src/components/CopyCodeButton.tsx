import React, { useState } from "react";
import { CheckIcon, ClipboardDocumentIcon } from "./icons";

/**
 * Props for the CopyCodeButton component.
 */
interface CopyCodeButtonProps {
    /** Code passed in so it can be copied */
    code: string;
    /** To modify styling on the fly (so the component can be used anywhere)*/
    className?: string;
}

/**
 * A reusable copy code button component with styling that can be modified.
 */
const CopyCodeButton: React.FC<CopyCodeButtonProps> = ({
    code,
    className = "",
}) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(code);
            setCopied(true);
            setTimeout(() => setCopied(false), 3000);
        } catch (err) {
            console.error("Failed to copy code:", err);
        }
    };

    return (
        <div className="flex items-center">
            <button
                onClick={handleCopy}
                className={`${className}`}
                title={copied ? "Copied!" : "Copy code"}
                aria-label={copied ? "Code copied to clipboard" : "Copy code to clipboard"}
            >
                {copied ? (
                    <span>
                        <CheckIcon className="size-5" />
                    </span>
                ) : (
                    <span>
                        <ClipboardDocumentIcon className="size-5" />
                    </span>
                )}
            </button>
        </div>
    );
};

export default CopyCodeButton;
