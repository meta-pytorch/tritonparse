The tool scripts in this folder are used separately. They are not part of the main tritonparse functionality.

## Available Tools

### extract_irs.py
Extract IR (Intermediate Representation) files from NDJSON trace logs.

**Usage:**
```bash
python extract_irs.py -i <input.ndjson> --line <line_number> -o <output_folder>
```

**Arguments:**
- `-i, --input`: Path to the input NDJSON file
- `--line`: Line number to extract (0-based indexing, where 0 = first line)
- `-o, --output`: Output directory to save extracted IR files
- `--kernel-name`: (Optional) Custom kernel name for output files

**Examples:**
```bash
# Extract IRs from the first line (line 0)
python extract_irs.py -i logs.ndjson --line 0 -o output_folder

# Extract from line 5
python extract_irs.py -i logs.ndjson --line 5 -o ./irs

# Specify custom kernel name
python extract_irs.py -i logs.ndjson --line 0 -o ./irs --kernel-name my_kernel
```

**Extracted Files:**
- `*.ttir` - Triton IR
- `*.ttgir` - Triton GPU IR
- `*.llir` - LLVM IR
- `*.ptx` - PTX assembly
- `*.json` - Kernel metadata
- `*_source.py` - Python source code (if available)
