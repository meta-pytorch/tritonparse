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

## Bisect outcomes and logs

`TritonBisector.run()`, `LLVMBisector.run()` and `TorchBisector.run()` return a commit string only when Git identifies a unique first bad commit. Their `result` attribute records the Git bisect attempt as a `BisectResult`; failures from that attempt raise the existing component-specific `BisectError` subclass with the same object in `error.result`. Preliminary LLVM endpoint validation still precedes the Git bisect attempt.

| Status | Meaning |
| --- | --- |
| `found` | Git completed successfully with a unique first bad commit in `culprit`. |
| `ambiguous` | Skipped commits prevented a unique result. All reported candidates are in `candidates`, and `culprit` is unset. |
| `aborted` | The test told Git to abort, or execution was interrupted. |
| `error` | Setup, execution, result parsing or evidence capture failed. |

Each Git bisect attempt writes a separate `*_bisect_result.json` with its status, candidate set, endpoint revisions, exit code and artifact paths. Before resetting its own bisect, the executor saves `git bisect log` to a separate `*_git_bisect.log`. The command log retains the build/test output and any skip reasons printed by the scripts. An existing bisect is left untouched, including in a linked worktree.

Result JSON and replay log files are published atomically after a complete write; write failures do not leave a result pointing to partial files. A failed `git bisect start` is still cleaned up if it created state. Process termination waits are bounded: if a command cannot be stopped after SIGKILL, the attempt records the cleanup failure and preserves the checkout and replay log instead of resetting files that a surviving process may still use. An interrupted command retains its aborted status, output and command-log footer even when cleanup also fails. A crash signal delivered to Git itself is an execution error; fatal per-commit test exits still follow Git's abort protocol.

On interruption, the executor gives the process group five seconds after SIGTERM, sends SIGKILL, and then waits up to five seconds for the leader. It keeps the leader unreaped until all group signals have been sent, preventing reuse of its process group ID during cleanup. If the leader has already been reaped, cleanup reports an error instead of signalling an unverified group.

The full workflow records `triton_bisect_result` and `llvm_bisect_result` in its state and report. An ambiguous or aborted attempt stops the workflow with that status and a nonzero CLI exit; it does not enter LLVM detection with an unconfirmed Triton candidate. `--status` displays the retained candidates. Resuming a stopped attempt reports its saved error; start a new attempt after resolving the cause.
