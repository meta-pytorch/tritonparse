#!/usr/bin/env python3
"""
Simple test script for SASS parsing functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tritonparse'))

from tritonparse.ir_parser import extract_sass_mappings

# Sample SASS content similar to what we saw in the actual data
SAMPLE_SASS_CONTENT = """Function:test_kernel
\t//## File "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 188
\t//## File ".nv_debug_ptx_txt", line 19
        /*0000*/                   MOV R1, c[0x0][0x28] ;
\t//## File "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 191
\t//## File ".nv_debug_ptx_txt", line 37
        /*0010*/                   S2R R0, SR_TID.X ;
\t//## File "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 194
\t//## File ".nv_debug_ptx_txt", line 55
        /*0020*/                   IADD3 R2, R0, 0x4, RZ ;
        /*0030*/                   ISETP.LT.AND P0, PT, R0, R1, PT ;
"""

def test_sass_parsing():
    print("Testing SASS parsing functionality...")
    
    # Test the extract_sass_mappings function
    mappings = extract_sass_mappings(SAMPLE_SASS_CONTENT)
    
    print(f"Generated {len(mappings)} mappings:")
    for line_num, mapping in mappings.items():
        print(f"  Line {line_num}: {mapping}")
    
    # Expected mappings:
    # Line 4 should map to "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 188
    # Line 7 should map to "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 191  
    # Line 10 should map to "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 194
    # Line 11 should also map to "/scratch/findhao/tritonparse/tests/test_tritonparse.py", line 194
    
    expected_mappings = {
        "4": {
            "file": "/scratch/findhao/tritonparse/tests/test_tritonparse.py",
            "line": 188,
            "column": 0,
            "sass_line": 4
        },
        "7": {
            "file": "/scratch/findhao/tritonparse/tests/test_tritonparse.py", 
            "line": 191,
            "column": 0,
            "sass_line": 7
        },
        "10": {
            "file": "/scratch/findhao/tritonparse/tests/test_tritonparse.py",
            "line": 194,
            "column": 0, 
            "sass_line": 10
        },
        "11": {
            "file": "/scratch/findhao/tritonparse/tests/test_tritonparse.py",
            "line": 194,
            "column": 0,
            "sass_line": 11
        }
    }
    
    # Verify results
    success = True
    for expected_line, expected_mapping in expected_mappings.items():
        if expected_line not in mappings:
            print(f"ERROR: Missing mapping for line {expected_line}")
            success = False
            continue
            
        actual_mapping = mappings[expected_line]
        for key, expected_value in expected_mapping.items():
            if actual_mapping.get(key) != expected_value:
                print(f"ERROR: Line {expected_line}, key {key}: expected {expected_value}, got {actual_mapping.get(key)}")
                success = False
    
    # Check that .nv_debug_ptx_txt lines were skipped
    for line_num, mapping in mappings.items():
        if ".nv_debug_ptx_txt" in mapping["file"]:
            print(f"ERROR: .nv_debug_ptx_txt line was not skipped: {mapping}")
            success = False
    
    if success:
        print("✅ All SASS parsing tests passed!")
    else:
        print("❌ Some SASS parsing tests failed!")
    
    return success

if __name__ == "__main__":
    success = test_sass_parsing()
    sys.exit(0 if success else 1)