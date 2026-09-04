import React, { useCallback, useEffect, useRef, useState } from "react";
import { ShareIcon } from "./icons";

interface ShareButtonProps {
  /** Builds the shareable URL at click time (so it is never stale). */
  getShareUrl: () => string;
  disabled?: boolean;
  /** Tooltip shown when disabled. */
  disabledTitle?: string;
  label?: string;
}

interface ToastState {
  kind: "success" | "error";
  text: string;
}

/**
 * Copies text to the clipboard, with a fallback for contexts where the
 * async clipboard API is unavailable (e.g. non-secure contexts).
 */
async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  ta.setSelectionRange(0, text.length);
  try {
    if (!document.execCommand("copy")) {
      throw new Error("execCommand copy failed");
    }
  } finally {
    document.body.removeChild(ta);
  }
}

const TOAST_DURATION_MS = 2500;

/**
 * A share button that copies the current shareable URL and confirms with a
 * transient toast instead of a blocking dialog.
 */
const ShareButton: React.FC<ShareButtonProps> = ({
  getShareUrl,
  disabled = false,
  disabledTitle,
  label = "Share",
}) => {
  const [toast, setToast] = useState<ToastState | null>(null);
  const timerRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    return () => {
      window.clearTimeout(timerRef.current);
    };
  }, []);

  const showToast = useCallback((next: ToastState) => {
    window.clearTimeout(timerRef.current);
    setToast(next);
    timerRef.current = window.setTimeout(() => setToast(null), TOAST_DURATION_MS);
  }, []);

  const handleClick = useCallback(async () => {
    if (disabled) return;
    let url: string;
    try {
      url = getShareUrl();
    } catch (err) {
      showToast({
        kind: "error",
        text: `Failed to build share link: ${err instanceof Error ? err.message : String(err)}`,
      });
      return;
    }
    try {
      await copyText(url);
      showToast({ kind: "success", text: "Shareable link copied to clipboard" });
    } catch (err) {
      console.error("Failed to copy link:", err);
      showToast({ kind: "error", text: "Failed to copy link to clipboard" });
    }
  }, [disabled, getShareUrl, showToast]);

  return (
    // Tooltip lives on the wrapper so it still surfaces when the button is
    // natively disabled (disabled buttons don't fire hover/title in browsers).
    <span
      className="relative inline-flex"
      title={disabled ? disabledTitle ?? label : "Copy shareable link to clipboard"}
    >
      <button
        type="button"
        onClick={handleClick}
        disabled={disabled}
        className={`px-3 py-1 rounded text-sm flex items-center ${
          disabled
            ? "bg-blue-100 text-blue-900 cursor-not-allowed"
            : "bg-blue-600 text-white hover:bg-blue-700"
        }`}
      >
        <ShareIcon className="h-4 w-4 mr-1" />
        {label}
      </button>
      {/* Persistent live regions so screen readers announce copy results;
          only the text inside changes. Errors assert, successes politely note. */}
      <span role="status" aria-live="polite" className="sr-only">
        {toast?.kind === "success" ? toast.text : ""}
      </span>
      <span role="alert" className="sr-only">
        {toast?.kind === "error" ? toast.text : ""}
      </span>
      {toast && (
        <span
          aria-hidden="true"
          className={`fixed bottom-6 left-1/2 -translate-x-1/2 z-50 whitespace-nowrap rounded-md px-3 py-2 text-sm text-white shadow-lg ${
            toast.kind === "success" ? "bg-gray-900" : "bg-red-700"
          }`}
        >
          {toast.text}
        </span>
      )}
    </span>
  );
};

export default ShareButton;
