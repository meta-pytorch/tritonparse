/**
 * Helpers for comparing the compilation metadata of two kernels, used to
 * highlight what changed when the user switches between kernels that share a
 * name.
 */

/**
 * Formats a value for display in the metadata section.
 * @param value - The value to format
 * @returns Formatted string representation
 */
export const formatMetadataValue = (value: unknown): string => {
  if (value === null) {
    return "null";
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
};

/**
 * Returns the metadata keys whose displayed value differs between two kernels.
 * A key present on only one side counts as changed. Returns an empty map when
 * the kernels do not share a name, because a comparison between different
 * kernels is not meaningful.
 *
 * @param previous - Name and metadata of the kernel that was selected before
 * @param current - Name and metadata of the kernel that is selected now
 * @returns Map from changed key to the value it had on the previous kernel
 *   ("" when the previous kernel did not have the key)
 */
export const changedMetadataKeys = (
  previous: { name: string; metadata?: Record<string, unknown> } | undefined,
  current: { name: string; metadata?: Record<string, unknown> } | undefined
): Map<string, string> => {
  const changed = new Map<string, string>();
  if (!previous?.metadata || !current?.metadata) return changed;
  if (previous.name !== current.name) return changed;
  for (const key of Object.keys(current.metadata)) {
    const before = key in previous.metadata ? formatMetadataValue(previous.metadata[key]) : null;
    const after = formatMetadataValue(current.metadata[key]);
    if (before !== after) changed.set(key, before ?? "");
  }
  return changed;
};
