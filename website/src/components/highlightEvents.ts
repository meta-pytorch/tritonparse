export const HIGHLIGHT_LINES_EVENT = "tritonparse:highlight-lines";

export interface HighlightLinesEventDetail {
  viewerId: string;
  lineNumbers: number[];
}

export const notifyCodeViewerHighlights = (
  viewerId: string,
  lineNumbers: number[]
) => {
  window.dispatchEvent(new CustomEvent<HighlightLinesEventDetail>(
    HIGHLIGHT_LINES_EVENT,
    { detail: { viewerId, lineNumbers } }
  ));
};
