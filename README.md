# TritonParse

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deploy-brightgreen)](https://meta-pytorch.org/tritonparse/)

**A comprehensive visualization and analysis tool for Triton kernel compilation and launch** — helping developers analyze, debug, and understand Triton kernel compilation processes.

🌐 **[Try it online →](https://meta-pytorch.org/tritonparse/?json_url=https://meta-pytorch.org/tritonparse/dedicated_log_triton_trace_findhao__mapped.ndjson.gz)**

## ✨ Key Features

### 🔍 Visualization & Analysis
- **🚀 Launch Difference Analysis** - Detect and visualize kernel launch parameter variations
- **📊 IR Code View** - Side-by-side IR viewing with synchronized highlighting and line mapping
- **🔄 File Diff View** - Compare kernels across different trace files side-by-side
- **📝 Multi-format IR Support** - View TTGIR, TTIR, LLIR, PTX, and AMDGCN
- **🎯 Interactive Code Views** - Click-to-highlight corresponding lines across IR stages

### 🔧 Reproducer & Debugging Tools
- **🔄 Standalone Script Generation** - Extract any kernel into a self-contained Python script
- **💾 Tensor Data Reconstruction** - Preserve actual tensor data or use statistical approximation
- **🎯 Custom Templates** - Flexible reproducer templates for different workflows
- **🐛 Bug Isolation** - Share reproducible test cases for debugging and collaboration

### 📊 Structured Logging & Analysis
- **📝 Compilation & Launch Tracing** - Capture detailed events with source mapping
- **🔍 Stack Trace Integration** - Full Python stack traces for debugging
- **📈 Metadata Extraction** - Comprehensive kernel statistics

### 🛠️ Developer Tools
- **🌐 Browser-based Interface** - No installation required, works in your browser
- **🔒 Privacy-first** - All processing happens locally, no data uploaded

## 🚀 Quick Start

### 1. Installation

**Four options to install:**
```bash
# install nightly version
pip install -U --pre tritonparse
# install stable version
pip install tritonparse
# install from source
git clone https://github.com/meta-pytorch/tritonparse.git
cd tritonparse
pip install -e .
# pip install the latest version from github
pip install git+https://github.com/meta-pytorch/tritonparse.git
```

**Prerequisites:** Python ≥ 3.10, Triton ≥ 3.4.0, GPU required (NVIDIA/AMD)

TritonParse relies on new features in Triton. If you're using nightly PyTorch, Triton is already included. Otherwise, install the latest Triton:
```bash
pip install triton
```

### 2. Generate Traces

```python
import tritonparse.structured_logging
import tritonparse.parse.utils

# Initialize logging with full tracing options
tritonparse.structured_logging.init(
    "./logs/",
    enable_trace_launch=True,                 # Capture kernel launch events (enables torch.compile tracing automatically)
    enable_more_tensor_information=True,      # Optional: collect tensor statistics (min/max/mean/std)
)

# Your Triton/PyTorch code here
# ... your kernels ...

# Parse and generate trace files
tritonparse.parse.utils.unified_parse("./logs/", out="./parsed_output")
```

> **💡 Note**: `enable_trace_launch=True` automatically enables tracing for both native Triton kernels (`@triton.jit`) and `torch.compile` / TorchInductor kernels.

<details>
<summary>📝 Example output (click to expand)</summary>

```bash
================================================================================
📁 TRITONPARSE PARSING RESULTS
================================================================================
📂 Parsed files directory: /scratch/findhao/tritonparse/tests/parsed_output
📊 Total files generated: 2

📄 Generated files:
   1. 📝 dedicated_log_triton_trace_findhao__mapped.ndjson.gz (7.2KB)
   2. 📝 log_file_list.json (181B)
================================================================================
✅ Parsing completed successfully!
================================================================================
```
</details>

### 3. Visualize Results

**Visit [https://meta-pytorch.org/tritonparse/](https://meta-pytorch.org/tritonparse/?json_url=https://meta-pytorch.org/tritonparse/dedicated_log_triton_trace_findhao__mapped.ndjson.gz)** and open your local trace files (.ndjson.gz format).

> **🔒 Privacy Note**: Your trace files are processed entirely in your browser - nothing is uploaded to any server!

### 4. Generate Reproducers (Optional)

Extract any kernel into a standalone, executable Python script for debugging or testing:

```bash
# Generate reproducer for the first launch event
# (--line is 0-based: line 0 is compilation event, line 1 is first launch event)
tritonparseoss reproduce ./parsed_output/trace.ndjson.gz --line 1 --out-dir repro_output

# Run the generated reproducer
cd repro_output/<kernel_name>/
python repro_*.py
```

**Python API:**
```python
from tritonparse.reproducer.orchestrator import reproduce

result = reproduce(
    input_path="./parsed_output/trace.ndjson.gz",
    line_index=0,           # 0-based index (first event is 0)
    out_dir="repro_output"
)
```

<details>
<summary>🎯 Common Reproducer Use Cases (click to expand)</summary>

- **🐛 Bug Isolation**: Extract a failing kernel into a minimal standalone script
- **⚡ Performance Testing**: Benchmark specific kernels without running the full application
- **🤝 Team Collaboration**: Share reproducible test cases with colleagues or in bug reports
- **📊 Regression Testing**: Compare kernel behavior and performance across different versions
- **🔍 Deep Debugging**: Modify and experiment with kernel parameters in isolation

</details>

## Virtual-target capture

Capture a controlled eager script for a different GPU without executing its
Triton kernels:

```bash
python -m tritonparse.capture --target hip:gfx950 --out ./virtual-traces capture_mm.py
# The same arguments are available as: tritonparseoss capture ...
```

Run each capture in a **fresh process** and use a new output directory. The
initial profiles are H100 SXM5 (`cuda:90`), B200 (`cuda:100`), MI300X (`hip:gfx942`)
and MI350X (`hip:gfx950`). These names fix processor counts and shared-memory budgets as
well as the ISA; they are not defaults for every device with that ISA.

A real host GPU and its working PyTorch runtime are still required for tensor
allocation and initialization. The target GPU is not required. The capture
runtime currently supports Triton 3.9 and needs the full compiler tools for the
selected backend, including AMD codegen and the linker for HIP targets, and a
Blackwell-capable `ptxas` for `cuda:100`. For example, an H100 host can capture a
B200 workload with `--target cuda:100`: allocations and streams remain on H100,
while compilation and the supported target queries describe B200. Missing
compiler tools are errors; capture never substitutes placeholder binaries.

The script must use eager dispatch with deterministic heuristics, a singleton
configuration, or an explicitly selected configuration. Host Python and ordinary
PyTorch operations still run. Triton outputs are unchanged by skipped launches,
so the script must not inspect them to choose later dispatches or claim numerical
correctness. This is not an emulator and does not detect arbitrary output-dependent
Python. Benchmark-driven autotuning, CUDA graphs, `torch.compile`, in-process
target switching, and GPU-free/FakeTensor capture are outside this initial scope.

Compilation and launch events carry `virtual_target` and
`runtime_validated: false`; launch events additionally carry `executed: false`.
The runtime preserves real argument binding and specialization, including nested
tuples and integer grids. Register counts, occupancy and launch-time resource
validity are not inferred. Tensor statistics, tensor blobs, external autotune
caches and Manifold trace upload are disabled. Compiler caches are private to the
capture and removed on exit. Runtime patches last until process exit.

Parse these traces normally. Parsing does not label virtual invocations as
measured autotune winners. Offline parsing and static analysis do not require a
GPU, but any later execution or reproducer run requires a suitable real target.

## 📚 Complete Documentation

| 📖 Guide | Description |
|----------|-------------|
| **[🏠 Wiki Home](https://github.com/meta-pytorch/tritonparse/wiki)** | Complete documentation and quick navigation |
| **[📦 Installation](https://github.com/meta-pytorch/tritonparse/wiki/01.-Installation)** | Setup guide for all scenarios |
| **[📋 Usage Guide](https://github.com/meta-pytorch/tritonparse/wiki/02.-Usage-Guide)** | Complete workflow, reproducer generation, and examples |
| **[🌐 Web Interface](https://github.com/meta-pytorch/tritonparse/wiki/03.-Web-Interface-Guide)** | Master the visualization interface |
| **[🔧 Developer Guide](https://github.com/meta-pytorch/tritonparse/wiki/04.-Developer-Guide)** | Contributing and architecture overview |
| **[📝 Code Formatting](https://github.com/meta-pytorch/tritonparse/wiki/05.-Code-Formatting)** | Formatting standards and tools |
| **[❓ FAQ](https://github.com/meta-pytorch/tritonparse/wiki/06.-FAQ)** | Quick answers and troubleshooting |
| **[⚙️ Environment Variables](https://github.com/meta-pytorch/tritonparse/wiki/07.-Environment-Variables-Reference)** | Complete environment variable reference |
| **[📖 Python API Reference](https://github.com/meta-pytorch/tritonparse/wiki/08.-Python-API-Reference)** | Full API documentation |
| **[🔄 Reproducer Guide](https://github.com/meta-pytorch/tritonparse/wiki/09.-Reproducer-Guide)** | Comprehensive kernel reproducer guide |

## 📊 Understanding Triton Compilation

TritonParse visualizes the complete Triton compilation pipeline:

**Python Source** → **TTIR** → **TTGIR** → **LLIR** → **PTX/AMDGCN**

Each stage can be inspected and compared to understand optimization transformations.

## 🤝 Contributing

We welcome contributions! Please see our **[Developer Guide](https://github.com/meta-pytorch/tritonparse/wiki/04.-Developer-Guide)** for:
- Development setup and prerequisites
- Code formatting standards (**[Formatting Guide](https://github.com/meta-pytorch/tritonparse/wiki/05.-Code-Formatting)**)
- Pull request and code review process
- Testing guidelines
- Architecture overview

## 📞 Support & Community

- **🐛 Report Issues**: [GitHub Issues](https://github.com/meta-pytorch/tritonparse/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/meta-pytorch/tritonparse/discussions)
- **📚 Documentation**: [TritonParse Wiki](https://github.com/meta-pytorch/tritonparse/wiki)

## 📄 License

This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details.

---

**✨ Ready to get started?** Visit our **[Installation Guide](https://github.com/meta-pytorch/tritonparse/wiki/01.-Installation)** or try the **[online tool](https://meta-pytorch.org/tritonparse/)** directly!
