# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for IR parsing functionality."""

import unittest

from tests.test_utils import get_sass_test_file
from tritonparse.parse.ir_parser import extract_loc_definitions, extract_sass_mappings
from tritonparse.parse.mapper import create_bidirectional_mapping, create_ir_mapping
from tritonparse.parse.trace_processor import generate_source_mappings


class TestIRParser(unittest.TestCase):
    """Tests for IR parsing functions."""

    def test_callsite_parsing(self):
        """Test parsing of callsite locations in TTIR/TTGIR"""

        # Test MLIR callsite location definitions
        ir_with_callsite = """
module {
  #loc7 = loc("/tmp/test.py":1091:8)
  #loc57 = loc("/tmp/test.py":421:16)
  #loc58 = loc("/tmp/test.py":853:16)
  #loc190 = loc(callsite(#loc58 at #loc7))
  #loc220 = loc(callsite(#loc57 at #loc190))
  %0 = tt.load %ptr loc(#loc220)
}
"""
        # Extract loc definitions
        locs = extract_loc_definitions(ir_with_callsite)

        # Verify loc220 (nested callsite)
        self.assertIn("220", locs)
        self.assertEqual(locs["220"]["file"], "/tmp/test.py")
        self.assertEqual(locs["220"]["line"], 421)  # Inherited from callee loc57
        self.assertEqual(locs["220"]["column"], 16)
        self.assertTrue(locs["220"].get("is_callsite"))
        self.assertEqual(locs["220"]["callsite_callee"], "57")
        self.assertEqual(locs["220"]["callsite_caller"], "190")

        # Verify loc190 (simple callsite)
        self.assertIn("190", locs)
        self.assertEqual(locs["190"]["line"], 853)  # Inherited from callee loc58
        self.assertTrue(locs["190"].get("is_callsite"))
        self.assertEqual(locs["190"]["callsite_callee"], "58")
        self.assertEqual(locs["190"]["callsite_caller"], "7")

        # Test source mappings generation
        mappings = generate_source_mappings(ir_with_callsite, "ttir")

        # Find the line with tt.load
        line_with_load = None
        for line_num, content in enumerate(ir_with_callsite.split("\n"), start=1):
            if "tt.load" in content:
                line_with_load = str(line_num)
                break

        self.assertIsNotNone(line_with_load)
        self.assertIn(line_with_load, mappings)

        mapping = mappings[line_with_load]
        self.assertEqual(mapping["file"], "/tmp/test.py")
        self.assertEqual(mapping["line"], 421)  # From loc220 -> loc57
        self.assertTrue(mapping.get("is_callsite"))
        self.assertEqual(mapping["callsite_callee"], "57")
        self.assertEqual(mapping["callsite_caller"], "190")

        print("✓ Callsite parsing tests passed")

    def test_loc_alias_parsing(self):
        """Test parsing of location aliases in TTIR/TTGIR"""

        # Test case 1: Bare #loc reference (no number)
        ir_with_bare_loc = """
module {
  #loc = loc("/tmp/test.py":10:5)
  #loc13 = loc("x_ptr"(#loc))
  func @kernel(%arg0: !tt.ptr<f32> loc(#loc13)) {
    return loc(#loc)
  }
}
"""
        locs = extract_loc_definitions(ir_with_bare_loc)
        # Main #loc should be stored with "" key
        assert "" in locs, "Main #loc not found"
        assert locs[""]["file"] == "/tmp/test.py"
        assert locs[""]["line"] == 10
        # Alias #loc13 should resolve to same location
        assert "13" in locs, "#loc13 not found"
        assert locs["13"]["file"] == "/tmp/test.py"
        assert locs["13"]["line"] == 10
        assert locs["13"]["alias_name"] == "x_ptr"
        assert locs["13"]["alias_of"] == ""

        # Test case 2: Named alias with numbered reference
        ir_with_numbered_alias = """
#loc = loc("/tmp/test.py":5:0)
#loc2 = loc("/tmp/test.py":20:28)
#loc16 = loc("pid"(#loc2))
%0 = tt.get_program_id x : i32 loc(#loc16)
"""
        locs = extract_loc_definitions(ir_with_numbered_alias)
        assert "2" in locs
        assert locs["2"]["line"] == 20
        assert "16" in locs
        assert locs["16"]["file"] == "/tmp/test.py"
        assert locs["16"]["line"] == 20
        assert locs["16"]["alias_name"] == "pid"
        assert locs["16"]["alias_of"] == "2"

        # Test case 3: Simple alias (no name)
        ir_with_simple_alias = """
#loc = loc("/tmp/test.py":1:1)
#loc1 = loc("/tmp/test.py":15:10)
#loc20 = loc(#loc1)
%1 = arith.constant 0 : i32 loc(#loc20)
"""
        locs = extract_loc_definitions(ir_with_simple_alias)
        assert "1" in locs
        assert "20" in locs
        assert locs["20"]["file"] == "/tmp/test.py"
        assert locs["20"]["line"] == 15
        assert locs["20"]["alias_of"] == "1"
        assert "alias_name" not in locs["20"]

        # Test case 4: Definition line tracking
        assert "def_line" in locs[""]
        assert "def_line" in locs["1"]
        assert "def_line" in locs["20"]

        print("✓ All loc alias parsing tests passed")

    def test_extract_sass_mappings(self):
        """Test SASS source mapping extraction from real SASS file."""
        sass_content = get_sass_test_file("test_kernel.sass").read_text()

        mappings = extract_sass_mappings(sass_content)

        # Basic validation: should have mapping results
        self.assertGreater(len(mappings), 0, "Should extract at least one mapping")

        # Verify first SASS instruction (line 4: /*0000*/ LDC R1...)
        # Maps to line 140 in the source file
        self.assertIn("4", mappings)
        self.assertIn("file", mappings["4"])
        self.assertEqual(
            mappings["4"]["file"],
            "/home/test/tritonparse/tests/gpu/test_structured_logging.py",
        )
        self.assertEqual(mappings["4"]["line"], 140)
        self.assertEqual(mappings["4"]["column"], 0)  # SASS has no column info

        # Verify second SASS instruction (line 7: /*0010*/ S2R R0...)
        # Maps to line 143 in the source file
        self.assertIn("7", mappings)
        self.assertEqual(mappings["7"]["line"], 143)

        # Verify .nv_debug_ptx_txt lines are skipped
        for line_num, mapping in mappings.items():
            self.assertNotIn(
                ".nv_debug_ptx_txt",
                mapping["file"],
                f"Line {line_num} should not map to .nv_debug_ptx_txt",
            )

        # Verify SASS instruction line format is correctly identified (/*hexaddr*/ format)
        for line_num in mappings.keys():
            self.assertTrue(
                line_num.isdigit(), f"Line number should be integer: {line_num}"
            )

        print("✓ SASS parsing tests passed")

    def test_sass_fuzzy_matching(self):
        """Test that ignore_column parameter enables fuzzy matching."""
        # Simulate SASS (column=0) and PTX (column=24) mappings
        sass_map = {
            "10": {"file": "/test.py", "line": 100, "column": 0, "sass_line": 10}
        }
        ptx_map = {"5": {"file": "/test.py", "line": 100, "column": 24, "ptx_line": 5}}

        # Without ignore_column: should have no match (columns differ)
        result_strict = create_ir_mapping(sass_map, ptx_map, ignore_column=False)
        self.assertEqual(
            len(result_strict), 0, "Strict matching should fail when columns differ"
        )

        # With ignore_column: should match successfully
        result_fuzzy = create_ir_mapping(sass_map, ptx_map, ignore_column=True)
        self.assertIn("10", result_fuzzy)
        self.assertEqual(result_fuzzy["10"], [5])

        print("✓ SASS fuzzy matching tests passed")

    def test_sass_bidirectional_mapping(self):
        """Test automatic fuzzy matching when source or target is SASS."""
        sass_map = {
            "10": {"file": "/test.py", "line": 100, "column": 0, "sass_line": 10}
        }
        ptx_map = {"5": {"file": "/test.py", "line": 100, "column": 24, "ptx_line": 5}}

        # Call bidirectional mapping (source_type="sass" should auto-enable ignore_column)
        create_bidirectional_mapping(sass_map, ptx_map, "sass", "ptx")

        # Verify forward mapping (sass -> ptx)
        self.assertIn("ptx_lines", sass_map["10"])
        self.assertIn(5, sass_map["10"]["ptx_lines"])

        # Verify reverse mapping (ptx -> sass)
        self.assertIn("sass_lines", ptx_map["5"])
        self.assertIn(10, ptx_map["5"]["sass_lines"])

        print("✓ SASS bidirectional mapping tests passed")

    def test_sass_integration_with_trace_processor(self):
        """Test SASS integration in full trace processing pipeline."""
        sass_content = get_sass_test_file("test_kernel.sass").read_text()

        # Directly test generate_source_mappings
        mappings = generate_source_mappings(sass_content, "sass")

        self.assertIsInstance(mappings, dict)
        self.assertGreater(len(mappings), 0)

        # Verify mapping structure
        first_key = next(iter(mappings))
        first_mapping = mappings[first_key]
        self.assertIn("file", first_mapping)
        self.assertIn("line", first_mapping)
        self.assertEqual(first_mapping["column"], 0)

        print("✓ SASS integration tests passed")


if __name__ == "__main__":
    unittest.main()
