export const HIGHLIGHT_LINES_EVENT = "tritonparse:highlight-lines";

export interface HighlightLinesEventDetail {
  viewerId: string;
  lineNumbers: number[];
}

const currentHighlights = new Map<string, number[]>();

export const getCodeViewerHighlights = (
  viewerId: string
): number[] | undefined => currentHighlights.get(viewerId);

export const notifyCodeViewerHighlights = (
  viewerId: string,
  lineNumbers: number[]
) => {
  currentHighlights.set(viewerId, lineNumbers);
  window.dispatchEvent(new CustomEvent<HighlightLinesEventDetail>(
    HIGHLIGHT_LINES_EVENT,
    { detail: { viewerId, lineNumbers } }
  ));
};
