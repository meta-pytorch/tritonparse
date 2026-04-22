# Welcome to TritonParse Wiki 🚀

**TritonParse** is a comprehensive visualization and analysis tool for Triton IR files, designed to help developers analyze, debug, and understand Triton kernel compilation processes.

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Deploy-brightgreen)](https://meta-pytorch.org/tritonparse/)

## 🎯 Quick Navigation

### 📚 Getting Started
- **[Installation](01.-Installation.md)** - Complete setup instructions
- **[Quick Start Tutorial](#-quick-start)** - Your first TritonParse experience
- **[System Requirements](01.-Installation.md#-prerequisites)** - Prerequisites and compatibility

### 📖 User Guide
- **[Usage Guide](02.-Usage-Guide.md)** - Generate traces and analyze kernels
- **[Web Interface Guide](03.-Web-Interface-Guide.md)** - Master the visualization interface
- **[File Diff View](03.-Web-Interface-Guide.md#-file-diff-view---compare-kernels-across-traces)** - Compare kernels across different traces
- **[Reproducer Guide](09.-Reproducer-Guide.md)** - Generate standalone kernel scripts (comprehensive)
- **[File Formats](02.-Usage-Guide.md#supported-file-formats)** - Understanding input/output formats
- **[Troubleshooting](01.-Installation.md#-troubleshooting)** - Common issues and solutions

### 🔧 Developer Guide
- **[Architecture Overview](04.-Developer-Guide.md#-architecture-overview)** - System design and components
- **[Contributing](04.-Developer-Guide.md#-contributing-guidelines)** - Development setup and guidelines
- **[Code Formatting](05.-Code-Formatting.md)** - Formatting standards and tools
- **[Testing](04.-Developer-Guide.md#-testing)** - Test structure and running tests

### 🎓 Advanced Topics
- **[IR Code View](03.-Web-Interface-Guide.md#-ir-code-view-tab)** - Side-by-side IR viewing with mapping
- **[Source Mapping](02.-Usage-Guide.md#-understanding-the-results)** - IR stage mapping explained
- **[Environment Variables](07.-Environment-Variables-Reference.md)** - Complete configuration reference
- **[Python API Reference](08.-Python-API-Reference.md)** - Full API documentation
- **[Performance Optimization](03.-Web-Interface-Guide.md#-browser--performance)** - Tips for large traces

### 📝 Quick Reference
- **[FAQ](06.-FAQ.md)** - Frequently asked questions
- **[Basic Examples](02.-Usage-Guide.md#example-complete-triton-kernel)** - Simple usage scenarios
- **[Advanced Examples](02.-Usage-Guide.md#-advanced-features)** - Complex use cases
- **[Troubleshooting](01.-Installation.md#-troubleshooting)** - Common issues and solutions

## 🌟 Key Features

### 🔍 Visualization & Analysis
- **Interactive Kernel Explorer** - Browse kernel information and stack traces
- **Multi-format IR Support** - View TTGIR, TTIR, LLIR, PTX, and AMDGCN
- **IR Code View** - Side-by-side IR viewing with synchronized highlighting and line mapping
- **Interactive Code Views** - Click-to-highlight corresponding lines across IR stages
- **Launch Diff Analysis** - Compare kernel launch events
- **File Diff View** - Compare kernels across different trace files side-by-side

### 📊 Structured Logging
- **Compilation Tracing** - Capture detailed Triton compilation events
- **Launch Tracing** - Capture detailed kernel launch events
- **Stack Trace Integration** - Full Python stack traces for debugging
- **Metadata Extraction** - Comprehensive kernel metadata and statistics
- **NDJSON Output** - Structured logging format for easy processing

### 🔧 Reproducer Generation
- **Standalone Scripts** - Generate self-contained Python scripts to reproduce kernels
- **Tensor Reconstruction** - Rebuild tensors from statistical data or saved blobs
- **Template System** - Customize reproducer output with flexible templates
- **Minimal Dependencies** - Scripts run independently for debugging and testing
- **[Full Reproducer Guide](09.-Reproducer-Guide.md)** - Comprehensive documentation

### 🔬 Diff & Bisect
- **Compilation Diff** - Compare compilation events within or across trace files
- **Tensor Value Comparison** - Compare tensor values with configurable tolerance
- **Triton/LLVM/PyTorch Bisect** - Find regression-causing commits automatically
- **LLVM Compat Builder** - Build compatibility maps for LLVM bumps with AI-assisted fixes

### 🌐 Deployment Options
- **GitHub Pages** - Ready-to-use online interface
- **Local Development** - Full development environment
- **Standalone HTML** - Self-contained deployments

## ⚡ Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/meta-pytorch/tritonparse.git
cd tritonparse

# Install dependencies
pip install -e .
```

### 2. Generate Traces
```python
import tritonparse.structured_logging

# Initialize logging
tritonparse.structured_logging.init("./logs/", enable_trace_launch=True)

# Your Triton/PyTorch code here
...

# Parse logs
import tritonparse.parse.utils
tritonparse.parse.utils.unified_parse(source="./logs/", out="./parsed_output")
```

### 3. Analyze Results
Visit **[https://meta-pytorch.org/tritonparse/](https://meta-pytorch.org/tritonparse/)** and load your trace files!

### 4. (Optional) Generate Reproducer
```bash
# Generate standalone reproducer script
tritonparseoss reproduce ./parsed_output/trace.ndjson --line 1 --out-dir repro_output
```

> 💡 **See [Reproducer Guide](09.-Reproducer-Guide.md)** for comprehensive documentation on reproducer generation.

## 🔗 Important Links

- **Live Tool**: [https://meta-pytorch.org/tritonparse/](https://meta-pytorch.org/tritonparse/)
- **GitHub Repository**: [https://github.com/meta-pytorch/tritonparse](https://github.com/meta-pytorch/tritonparse)
- **Issues**: [GitHub Issues](https://github.com/meta-pytorch/tritonparse/issues)
- **Discussions**: [GitHub Discussions](https://github.com/meta-pytorch/tritonparse/discussions)

## 🤝 Contributing

We welcome contributions! Please see our **[Contributing Guide](04.-Developer-Guide.md#-contributing-guidelines)** for details on:
- Development setup and prerequisites
- Code formatting standards ([Code Formatting Guide](05.-Code-Formatting.md))
- Pull request and code review process
- Issue reporting guidelines

## 📄 License

This project is licensed under the BSD-3 License. See the [LICENSE](https://github.com/meta-pytorch/tritonparse/blob/main/LICENSE) file for details.

---

**Note**: This tool is designed for developers working with Triton kernels and GPU computing. Basic familiarity with GPU programming concepts (CUDA for NVIDIA or ROCm/HIP for AMD), and the Triton language is recommended for effective use.
