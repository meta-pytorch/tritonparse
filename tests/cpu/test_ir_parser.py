"""Tests for IR parsing functionality."""

import unittest

from tritonparse.parse.ir_parser import extract_loc_definitions
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


if __name__ == "__main__":
    unittest.main()
