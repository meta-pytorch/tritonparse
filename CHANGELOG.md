# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-11-03

### TritonParse Release Notes (last 24 commits)

- **Date range**: 2025-10-14 — 2025-11-03
- **Scope**: IR Analysis enhancements (beta), Reproducer template extensions, code viewer improvements, bug fixes.

### Highlights

- **📊 IR Analysis (Beta)**: New analysis capabilities for visualizing Software Pipelining (SWP), BufferOps statistics, and loop schedules in Triton IR. **Note: This is a beta feature.**
- **🏷️ Variable Location Tracking**: Complete location alias tracking system for mapping IR locations back to source code with frontend visualization.
- **🔧 TritonBench Template**: New reproducer template for easy TritonBench integration and kernel benchmarking.
- **🎨 Code Viewer Enhancements**: Full Python source extraction, function highlighting, and performance optimizations.
- **🔄 Reproducer Refactoring**: AST-based function extraction eliminates code duplication and simplifies template maintenance.

### Changes by area

#### 📊 **IR Analysis (Beta)**
- **Software Pipelining (SWP) visualization** (PR #189):
  - Analyzes inner `scf.for` loops and identifies prologue, loop_body, and epilogue stages
  - Tracks `tt.load` and `tt.dot` operations through TTIR → TTGIR → Python source mappings
  - Frontend displays simplified source code with SWP stage information
  - **Limitations**: Does not support Warp Specialization or Blackwell operators yet
- **BufferOps backend information** (PR #181):
  - Statistical analysis of buffer operations (tt.load/store, amdgpu.buffer_load/store, global_load/store) at TTGIR and AMDGCN levels
  - Useful for AMD GPU backend optimization analysis
- **Web frontend IR Analysis page** (PR #184):
  - New dedicated page at `/ir-analysis` route with integrated display for loop schedules and BufferOps statistics

#### 🏷️ **Variable Location Tracking**
Complete three-part implementation (PR #186, #187, #188):
- Fixed #loc storage key conflict in IR parser
- Added location alias parsing support in `ir_parser.py` and `trace_processor.py`
- Frontend visualization with CSS styling and interactive location display in Code Viewer

#### 🔄 **Reproducer System**
- **TritonBench template support** (commit 3493ac8):
  - New template: `tritonparse/reproducer/templates/tritonbench.py`
  - CLI option: `--template tritonbench` for TritonBench-compatible reproducers
  - Integrates with TritonBench's `BenchmarkOperator` and benchmark harness
- **AST-based refactoring** (PR #178):
  - New module: `tritonparse/reproducer/function_extractor.py` using Python AST
  - Simplified `example.py` template from ~370 lines to ~20 lines
- **Bug fixes**:
  - Fixed 1-based to 0-based line number conversion (PR #185)
  - Corrected output key typo: `repo_*` → `repro_*` (PR #175)
  - CUDA device normalization to `cuda:0` format (PR #177)

#### 📝 **Callsite Location Support**
- **TTIR/TTGIR callsite location** (PR #190):
  - Extended IR parser to extract callsite location information
  - Better debugging with call graph information and test coverage

#### 💻 **Code Viewer & Frontend**
- **Full Python source extraction** (commit 2976887):
  - Enhanced `structured_logging.py` to extract complete Python source files
- **Full file display with function highlighting** (commit 220d5a4):
  - CodeViewer now supports displaying entire source files with function-level highlighting
- **CodeComparisonView performance optimization** (commit c17e584):
  - Significant rendering performance improvements for large files
  - Reduced re-renders and improved memory efficiency

#### 🌐 **Website & Maintenance**
- **Dependency updates** (PR #179): Added automation script `website/scripts/update_deps.sh`
- **Copyright updates** (PR #183): Updated copyright headers across source files

### Compatibility notes

- **No breaking changes**: All updates are backward compatible with v0.3.0.
- **IR Analysis (Beta)**: New optional feature accessible through web UI.
- **TritonBench template**: Optional, does not impact existing reproducer generation.

### Upgrade guidance

1. **Using IR Analysis (Beta)**:
   - Open web UI and navigate to IR Analysis page after parsing
   - View SWP stage information (prologue/loop_body/epilogue) and BufferOps statistics
   - Note: Beta feature with some limitations on advanced pipelining patterns

2. **Generating TritonBench reproducers**:
   ```bash
   tritonparseoss reproduce trace.ndjson.gz --line <N> --template tritonbench --out-dir <output>
   ```

3. **Code viewer enhancements**: Automatically enabled with full source display and function highlighting

## [0.3.0] - 2025-10-14

### TritonParse Release Notes (last 44 commits)

- **Date range**: 2025-09-19 — 2025-10-14
- **Scope**: Major feature release - Reproducer system, tensor storage, SASS support, enhanced context manager, CLI improvements.

### Highlights

- **🔄 Reproducer System (Complete)**: Full-featured standalone kernel script generation with template support, tensor reconstruction, and multiple import modes. Extract any traced kernel into a self-contained Python script for debugging, testing, and sharing.
- **💾 TensorBlobManager**: Production-ready content-addressed tensor storage with automatic compression, deduplication, quota management, and efficient disk usage. Enables high-fidelity kernel reproduction with actual tensor data.
- **🔧 SASS Disassembly Support**: Optional NVIDIA SASS disassembly during compilation tracing for low-level debugging and performance analysis. Toggle via `enable_sass_dump` parameter or `TRITONPARSE_DUMP_SASS` environment variable.
- **🎯 Enhanced Context Manager**: Configurable `TritonParseManager` context manager with support for trace launch control, inductor compilation splitting, and flexible parsing parameters.
- **⚡ CLI Modernization**: Refactored to subcommand structure (`tritonparse parse`, `tritonparse reproduce`) with unified entry point and improved argument handling.
- **📊 Auto-enable Inductor Launch Tracing**: Automatic detection and tracing of PyTorch Inductor-compiled kernels without manual configuration.
- **🌐 Website Improvements**: Light mode color scheme, improved stack display in Launch Analysis, and better file diff navigation.

### Changes by area

#### 🔄 **Reproducer System**
- **Complete reproducer infrastructure** (PR #117-127):
  - CLI subcommand structure: `tritonparse reproduce <ndjson_file> [options]`
  - NDJSON ingestion layer with IR preservation
  - Context bundle system for kernel metadata and parameters
  - Standardized output paths: `repro_output/<kernel_name>/repro_<timestamp>.py`
  - Template support with placeholder system for custom generation
  - Example templates for tensor loading and kernel invocation
  - Dynamic import generation for kernel dependencies
  - Kernel signature parsing and integration
  - Kernel invocation snippet generation with grid/block configuration
- **Kernel import modes** (PR #165, #166):
  - `--kernel-import direct`: Import kernel from source file
  - `--kernel-import override-ttir`: Override and inject TTIR for advanced debugging
  - Flexible kernel loading strategies for different debugging workflows
- **Enhanced tensor handling** (PR #141):
  - Improved tensor metadata logging (shape, dtype, stride, storage offset, device)
  - Better tensor reconstruction quality in generated reproducers
  - Support for non-contiguous tensors (commit 12f1d1b)
- **Extensible placeholder system** (PR #149):
  - Refactored placeholder replacement with class-based design
  - Support for: `{{KERNEL_IMPORT_PLACEHOLDER}}`, `{{KERNEL_INVOCATION_PLACEHOLDER}}`, `{{KERNEL_SYSPATH_PLACEHOLDER}}`, `{{JSON_FILE_NAME_PLACEHOLDER}}`
  - Easy extension for future template needs
- **Documentation**: Comprehensive reproducer section in README (PR #161) and Usage Guide in Wiki

#### 💾 **TensorBlobManager & Storage**
- **Production-ready blob storage** (PR #156):
  - Content-addressed storage using BLAKE2b hashing
  - Automatic gzip compression for large tensors (>1MB)
  - Two-level directory structure (`xx/hash.bin.gz`) to avoid filesystem limits
  - Automatic deduplication: identical tensors stored only once
  - Storage quota enforcement (default: 100GB)
  - Per-tensor size limit (default: 10GB) to prevent OOM
  - Real-time statistics: saved count, dedup hits, compression ratio
  - Graceful degradation with warning logs when quota exceeded
- **Compression support** (PR #157):
  - Configurable compression level (default: 4)
  - Atomic writes using temporary files + rename for safety
  - Hash verification for data integrity
- **Comprehensive testing** (PR #162):
  - Unit tests for compression, deduplication, quota management
  - Edge case handling and cleanup verification

#### 🔧 **SASS Disassembly**
- **SASS extraction support** (PR #137):
  - New tool: `tritonparse/tools/disasm.py` for CUBIN disassembly
  - Integration into structured logging behind opt-in flag
  - Uses `nvdisasm -c -gp -g -gi` for detailed disassembly
  - Parses output to find function blocks with preserved labels and source mapping
- **Configuration**:
  - Environment variable: `TRITONPARSE_DUMP_SASS=1`
  - API parameter: `enable_sass_dump=True` in `structured_logging.init()`
  - API parameter takes precedence over environment variable
- **Robustness**:
  - Error handling for subprocess failures, missing nvdisasm, and generic exceptions
  - Writes marker messages instead of failing the trace
  - Requires NVIDIA CUDA Binary Utilities (nvdisasm)
- **CUDA testing** (PR #138):
  - Strengthened tests to validate SASS extraction and persistence

#### 🎯 **Context Manager & API**
- **Enhanced context manager** (PR #144, #159):
  - Added `__init__` method with configurable parameters:
    - `enable_trace_launch`: Control trace launch logging
    - `split_inductor_compilations`: Control inductor compilation splitting
    - `**parse_kwargs`: Additional arguments for `unified_parse`
  - Updated `__exit__` to pass parameters through to parsing pipeline
  - More flexible for different use cases and workflows
- **Split inductor compilations control**:
  - Parameter threading through: `unified_parse()` → `oss_run()` → `parse_logs()` → `parse_single_file()`
  - Renamed from `split_by_frame_id_and_compile_id` to `split_inductor_compilations` for clarity
  - Default `True`: splits by frame_id, frame_compile_id, attempt_id, compiled_autograd_id
  - When `False`: groups all inductor compilations together
  - Follows tlparse's convention
- **Unit tests** (commit a5338ce):
  - Tests for enhanced context manager behavior
  - Validation of split inductor compilation modes

#### ⚡ **CLI & Entry Points**
- **Subcommand structure** (PR #117):
  - Refactored from single-command to modern subcommand architecture
  - `tritonparse parse <source> [options]` - Run structured log parser
  - `tritonparse reproduce <ndjson_file> [options]` - Generate reproducers
  - Breaking change: old `python run.py <source>` no longer works
  - Extract parser flags into `tritonparse.utils._add_parse_args()`
  - Remove `unified_parse_from_cli` (programmatic `unified_parse()` remains)
- **Unified entry point** (PR #133):
  - Added proper CLI entry point in package configuration
  - Unified argument handling across commands
- **CLI entry point fix** (PR #154):
  - Fixed `ModuleNotFoundError` for tritonparse CLI entry point
  - Improved package installation and command availability

#### 📊 **Logging & Tracing**
- **Auto-enable Inductor Launch Tracing** (PR #142):
  - Automatically detect and trace PyTorch Inductor-compiled kernels
  - No manual configuration required for Inductor workflows
  - Seamless integration with existing tracing infrastructure
- **Kernel source path output** (commit 03bc1e1):
  - Output `kernel_src_path` in trace metadata for better debugging
- **NDJSON prettifier improvements** (PR #135):
  - Renamed and inverted flag to default-filter IRs
  - More intuitive filtering behavior
- **Debug flag deprecation** (PR #132):
  - Removed unused debugging flags
  - Cleaner configuration surface

#### 🌐 **Website & UI**
- **Upgraded to Tailwind CSS v4** (commit 6c42d8a):
  - Migrated from PostCSS plugin to `@tailwindcss/vite` for improved performance
  - Updated CSS import syntax from `@tailwind` directives to `@import "tailwindcss"`
  - Removed `tailwind.config.js` and `postcss.config.js` (now CSS-based configuration)
  - Updated `shadow` class naming to v4 convention (`shadow` → `shadow-sm`)
  - Cleaned up global CSS to prevent interference with Tailwind utility classes
- **Upgraded all frontend dependencies**:
  - Vite: 6.3.5 → 7.1.10
  - React ecosystem: Updated to latest versions (React 19+)
  - TypeScript: 5.7.2 → 5.7.3
  - Added `@types/node` for Node.js type definitions
  - Fixed dompurify security vulnerability (3.1.7 → 3.3.0) via npm overrides
- **Light mode color scheme** (PR #139):
  - Updated `index.css` to support only light mode
  - Consistent, professional appearance
- **Improved stack display** (PR #151):
  - Better stack trace visualization in Launch Analysis
  - Clearer debugging information
- **Documentation cleanup** (PR #172):
  - Removed redundant docs directory and screenshots
  - Streamlined repository structure

#### 🔧 **Bug Fixes & Maintenance**
- **General bug fixes** (PR #153):
  - Multiple stability and reliability improvements
  - Better error handling throughout codebase
- **Deserialization fix** (commit d4d7a20):
  - Fixed unhandled types in deserialization
  - More robust data loading
- **README improvements** (PR #158, #164):
  - Refactored and cleaned up README
  - Fixed command typos in reproducer generation examples
  - Clearer installation and usage instructions
- **Test cleanup** (PR #160):
  - Removed deprecated test for triton_kernels Tensor functionality
  - Updated test suite for current codebase

### Compatibility notes

- **Breaking Change**: CLI now uses subcommand structure. Old usage `python run.py <source>` must be updated to `tritonparse parse <source>` or `python run.py parse <source>`.
- **New Dependencies**: SASS disassembly requires NVIDIA CUDA Binary Utilities (`nvdisasm`). This is optional and only needed if `enable_sass_dump=True`.
- **Storage**: TensorBlobManager introduces new blob storage directory structure. Default quota is 100GB; configure via `TensorBlobManager` initialization if needed.
- **Context Manager API**: Enhanced with new parameters. Fully backward compatible with sensible defaults.

### Upgrade guidance

1. **Update CLI commands**: Change `python run.py <source>` to `tritonparse parse <source>` or use the new `tritonparse` command if installed via pip.
2. **Reproducer usage**: Use `tritonparse reproduce ./parsed_output/trace.ndjson.gz --line <N> --out-dir <output>` to generate standalone kernel scripts.
3. **SASS disassembly**: Opt-in by setting `TRITONPARSE_DUMP_SASS=1` or passing `enable_sass_dump=True` to `structured_logging.init()`. Requires `nvdisasm` in PATH.
4. **Tensor storage**: Enable high-fidelity reproduction by using TensorBlobManager (enabled by default when `enable_trace_launch=True`).
5. **Context manager**: Use enhanced `TritonParseManager` for more control over tracing and parsing behavior.

## [0.2.3] - 2025-09-19

### TritonParse Release Notes (last 15 commits)

- **Date range**: 2025-09-13 — 2025-09-18
- **Scope**: Website UI/UX, core library, CI/CD & packaging, documentation & testing.

### Highlights
- **Website File Diff tooling**: Introduced a new Diff Comparison view and File Diff page, preserved diff sessions across navigation, integrated Monaco editor, added preview mode, and shipped a round of UI polish with a URL redirect fix for File Diff navigation.
- **Kernel Overview**: Added a tiled kernel view toggle to improve dense overviews.
- **Core**: Added lazy-import support for Triton repo `triton_kernels` custom types, attribution check for `torch._utils_internal`, and safer file mapping cleanup in the log parser.
- **CI/Packaging**: Refactored dependencies in `pyproject.toml`, removed a legacy Triton install script, and updated GitHub Actions workflows.
- **Docs & tests**: Improved README guidance; added tests and example outputs; minor UI bug fix in `CopyCodeButton` SVG attributes.

### Changes by area
- **Website UI/UX**
  - Introduce `DiffComparisonView` and `FileDiffView`; maintain diff session state; integrate Monaco editor; preview mode; UI polish and navigation fixes.
  - Add tiled kernel view toggle in `KernelOverview`.

- **Core library**
  - Lazy-import support for `triton_kernels` custom types; extend tensor handling in tests.
  - Add attribution check for `torch._utils_internal`.
  - Refactor file mapping cleanup in `parse_logs`.

- **CI/CD & packaging**
  - Refactor dependencies in `pyproject.toml`; remove `.ci/install-triton-pip.sh`.
  - Update GitHub Actions workflows; add helper for `triton_kernels` in CI.

- **Docs & testing**
  - Clarify tool purpose and installation in `README.md`.
  - Add tests and sample outputs; small UI component fixes.

### Compatibility notes
- No breaking changes expected. `triton_kernels` support is optional via lazy import.

### Upgrade guidance
- Reinstall website dependencies if developing the UI to pick up the Monaco editor.

## [0.2.0] - 2025-09-11

### TritonParse Release Notes (last 27 commits)

- **Date range**: 2025-07-25 — 2025-09-11
- **Scope**: Core library, website UI/UX, performance & scalability, CI/CD & packaging, documentation & maintenance.

### Highlights
- **Website usability**: Drag-and-drop to open logs; one-click copy in code viewers; sticky, compact kernel selector; footer shows app version, localized build date, and Git short SHA; tensor arguments in Launch Analysis now display concise summaries with expandable details.
- **Large-file parsing**: Streaming NDJSON parsing and robust gzip handling significantly reduce memory usage and improve stability for files >100 MB.
- **Core & integrations**: Persist Inductor kernel config into `inductor_metadata` and pass to JIT hooks; ensure Inductor path invokes `jit_post_compile_hook`; new `init_with_env` for environment-based initialization; move compilation timing `times` into `metadata` for automatic frontend rendering.
- **Releases & versioning**: Adopt setuptools-scm dynamic versioning; add Nightly PyPI publishing; enable stable publishing on tag push; fix nightly version potentially being older than stable; correct packaging license metadata.
- **CI stability**: Ubuntu 24.04 compatibility; improved CUDA/cuDNN setup and detection; parallelize jobs; add parallel CI for pip-installed Triton; better error visibility in install scripts; upgrade libstdc++.

### Changes by area
- **Core library**
  - Save Inductor kernel params to `inductor_metadata` and forward to JIT hooks.
  - Manually invoke `jit_post_compile_hook` in the Inductor Triton compile path.
  - Add `init_with_env` that reads `TRITON_TRACE_FOLDER` and `TRITON_TRACE_LAUNCH`.
  - Move compilation `times` into `metadata` so the frontend auto-renders it.
  - Use cached source in compile listener for stability.
  - Refactor source-mapping pipeline into modular units for maintainability.

- **Website UI/UX**
  - Drag-and-drop to open supported log files.
  - Copy button in code viewer panels.
  - Sticky/collapsible/compact kernel selector in Kernel Overview; resizable compilation stack trace vertically.
  - Launch Analysis: tensor args show concise summaries with expandable details.
  - Footer displays version, localized build date, and Git short SHA.
  - Streaming NDJSON parsing and improved error handling for large logs.

- **Performance & scalability**
  - Use streaming path for files >100 MB to reduce memory peaks and improve robustness.

- **CI/CD & packaging**
  - Enable setuptools-scm and nightly PyPI publishing.
  - Publish stable releases on tag push; improve version computation and tag detection.
  - Fix nightly version possibly lagging behind stable; add clear error on missing tags.
  - Add parallel CI for pip-installed Triton; recommend pip installation in docs.
  - Improve Ubuntu 24.04 setup, CUDA/cuDNN handling, and job parallelism.
  - Increase error visibility in install scripts and upgrade libstdc++.
  - Define lower bounds for prerequisites in `pyproject.toml`.

- **Docs & maintenance**
  - Move repository to `meta-pytorch` org; update links and guidance; add AI assistant context.
  - Update/restore CONTRIBUTING docs to avoid breaking downstream consumers.

- **Testing**
  - Preserve test outputs when `TEST_KEEP_OUTPUT=1` to aid debugging.

### Compatibility notes
- Versioning & publishing: setuptools-scm with tag-based stable releases and nightly dev versions. Ensure `PYPI_API_TOKEN` is configured in CI if publishing is intended.
- Data format: compilation timing `times` moved under `metadata`; update any downstream scripts that referenced the old location.
- Build metadata: footer shows localized build date and Git short SHA; restart dev server to refresh these values.

### Upgrade guidance
- Prefer Triton from PyPI (≥ 3.4.0) and adhere to the lower bounds declared in `pyproject.toml`.
- For deterministic build metadata in the website, set `BUILD_DATE` and `GIT_COMMIT_SHA_SHORT` in the environment when running dev/build.


## [0.1.1] - 2025-07-25

### Added

- **Launch Difference Analysis**: A new `launch_diff` event is automatically generated for each kernel, providing a concise summary of how launch parameters vary across different calls. This helps to quickly identify changes in kernel arguments, grid dimensions, and other metadata.
- **Enhanced Web UI for Launch Analysis**: The web interface now visualizes the `launch_diff` data, offering an interactive way to explore how kernel launches differ. It includes a detailed breakdown of constant vs. varying parameters and their value distributions.
- **Kernel-Centric Event Grouping**: The parser now intelligently groups compilation and launch events by kernel, making it easier to analyze the entire lifecycle of a specific kernel.
- **Launch Event Tracing Control**: Added an `enable_trace_launch` parameter to `tritonparse.structured_logging.init` to give users explicit control over whether to trace kernel launch events.
- **Enhanced Logging and Testing**: Improved the structured logging initialization and expanded test coverage to verify the correctness of `launch` and `compilation` event counts.

## [0.1.0] - 2025-07-21

This is the initial public release of TritonParse.

### Added

- **Interactive Web Interface**: A rich, client-side web UI for exploring, comparing, and understanding Triton IRs. Features side-by-side code views, synchronized highlighting, and detailed metadata panels.
- **Structured Logging Backend**: A powerful Python backend to capture detailed information from the Triton compiler and runtime, including IRs (TTIR, TTGIR, PTX, AMDGCN), metadata, timings, and Python source code, and outputs it as structured NDJSON logs.
- **Source-to-Source Mapping**: Automatic generation of bidirectional mappings between Python code and all intermediate representations (IRs), allowing you to trace a line of Python code all the way down to the generated assembly and back.
- **Kernel Launch Tracing**: Capability to trace each kernel launch, capturing the grid dimensions, kernel arguments (with detailed tensor information), and other runtime metadata.
- **Flexible Log Parsing CLI**: A command-line interface (`run.py`) to parse logs from local files or directories, and from single or multiple ranks in a distributed training job.
- **Prerequisites Documentation**: Clear requirements for Python (>=3.10), PyTorch, and Triton (>3.3.1, compiled from source).
- **Getting Started Guide**: A step-by-step workflow for generating, parsing, and visualizing traces.
- **Configuration via Environment Variables**: Support for `TRITON_TRACE`, `TRITON_TRACE_LAUNCH`, `TRITONPARSE_KERNEL_ALLOWLIST`, and `TRITON_TRACE_GZIP`.
