# Regression bisection

## LLVM descriptors

After finding a Triton regression commit, the workflow compares its LLVM descriptor with its first parent's descriptor. Only a change in the LLVM source hash proceeds to LLVM pair testing and, if appropriate, LLVM source bisection.

`CommitDetector.read_llvm_descriptor(revision)` reads `cmake/llvm-info.json` when present, otherwise `cmake/llvm-hash.txt`. It returns an `LLVMDescriptor` containing the resolved Triton revision, LLVM source hash, descriptor path, and optional artifact build number and platform checksums. Invalid Git revisions, failed reads and malformed descriptors raise `CommitDetectorError`; they are not treated as unchanged LLVM.

`CommitDetector.detect(revision, parent=...)` allows an explicit comparison revision instead of the first parent. Source comparison requires full 40-character LLVM hashes. Legacy abbreviated hashes remain supported by `get_llvm_hash_at_commit()` for lookup.

`LLVMBumpInfo` reports source and artifact changes separately:

| Field | Meaning |
| --- | --- |
| `is_llvm_bump` | The normalized LLVM source hashes differ. |
| `artifact_changed=True` | Known build numbers differ, or the selected platform checksum differs. |
| `artifact_changed=False` | The selected platform checksums, or both complete nonempty checksum maps, match and known build numbers do not differ. |
| `artifact_changed=None` | The available metadata cannot establish artifact equality. |

Pass `artifact_platform` to `CommitDetector` to select a checksum key; the default is the explicit `TRITON_LLVM_SYSTEM_SUFFIX` environment override, if set. Without a selected platform, different checksum maps alone do not prove that the local artifact changed. A format migration or a change only to another platform's checksum is not an LLVM source bump.

The full workflow saves the complete comparison in `llvm_comparison` in its state and JSON report, including artifact-only changes.
