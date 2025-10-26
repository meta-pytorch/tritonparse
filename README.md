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
import tritonparse.utils

# Initialize logging
tritonparse.structured_logging.init("./logs/", enable_trace_launch=True)

# Your Triton/PyTorch code here
# ... your kernels ...

# Parse and generate trace files
tritonparse.utils.unified_parse("./logs/", out="./parsed_output")
```

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
# Generate reproducer from first launch event
tritonparseoss reproduce ./parsed_output/trace.ndjson.gz --line 2 --out-dir repro_output

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
