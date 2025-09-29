#!/usr/bin/env python3
"""
Integration test for SASS parsing in trace processing.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tritonparse'))

from tritonparse.trace_processor import parse_single_trace_content

# Sample trace content with SASS included
SAMPLE_TRACE_WITH_SASS = {
    "event_type": "compilation",
    "pid": 12345,
    "payload": {
        "file_content": {
            "test_kernel.sass": """Function:test_kernel
\t//## File "/path/to/source.py", line 100
\t//## File ".nv_debug_ptx_txt", line 19
        /*0000*/                   MOV R1, c[0x0][0x28] ;
\t//## File "/path/to/source.py", line 105
\t//## File ".nv_debug_ptx_txt", line 37
        /*0010*/                   S2R R0, SR_TID.X ;
        /*0020*/                   IADD3 R2, R0, 0x4, RZ ;""",
            "test_kernel.ttir": """#loc = loc("/path/to/source.py":100:0)
module {
  tt.func public @test_kernel() {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    tt.return loc(#loc2)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/path/to/source.py":105:24)
#loc2 = loc("/path/to/source.py":106:4)"""
        },
        "file_path": {
            "test_kernel.sass": "/tmp/test_kernel.sass",
            "test_kernel.ttir": "/tmp/test_kernel.ttir"
        }
    }
}

def test_sass_integration():
    print("Testing SASS integration in trace processing...")
    
    # Convert to JSON string format as expected by parse_single_trace_content
    trace_json = json.dumps(SAMPLE_TRACE_WITH_SASS)
    
    try:
        # Process the trace content
        result_json = parse_single_trace_content(trace_json)
        result = json.loads(result_json)
        
        # Check if SASS mappings were generated
        source_mappings = result.get("payload", {}).get("source_mappings", {})
        
        if "sass" not in source_mappings:
            print("❌ ERROR: No SASS mappings found in result")
            return False
        
        sass_mappings = source_mappings["sass"]
        print(f"✅ Found SASS mappings: {len(sass_mappings)} entries")
        
        # Print SASS mappings for verification
        for line_num, mapping in sass_mappings.items():
            print(f"  SASS Line {line_num}: {mapping}")
        
        # Check if TTIR mappings are still working 
        if "ttir" not in source_mappings:
            print("❌ ERROR: No TTIR mappings found in result")
            return False
            
        ttir_mappings = source_mappings["ttir"]
        print(f"✅ Found TTIR mappings: {len(ttir_mappings)} entries")
        
        # Verify that SASS mappings contain expected content
        found_valid_mapping = False
        for line_num, mapping in sass_mappings.items():
            if (mapping.get("file") == "/path/to/source.py" and 
                mapping.get("line") in [100, 105] and
                "sass_line" in mapping):
                found_valid_mapping = True
                break
        
        if not found_valid_mapping:
            print("❌ ERROR: No valid SASS mapping found with expected content")
            return False
        
        print("✅ SASS integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR during integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sass_integration()
    sys.exit(0 if success else 1)