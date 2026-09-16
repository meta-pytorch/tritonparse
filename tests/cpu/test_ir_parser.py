# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Tests for IR parsing functionality."""

import unittest

from tests.test_utils import get_sass_test_file
from tritonparse.parse.ir_parser import (
    extract_loc_definitions,
    extract_sass_mappings,
    extract_sass_pc_mappings,
)
from tritonparse.parse.mapper import (
    create_bidirectional_mapping,
    create_ir_mapping,
    create_python_mapping,
)
from tritonparse.parse.trace_processor import generate_source_mappings


class TestInlinedCallSiteMapping(unittest.TestCase):
    """Inlined code must map back to the line the user actually wrote.

    For anything inlined, `file`/`line` describe the callee -- a Triton library
    file. Filing that under the python map puts a library line number where a
    kernel line number belongs, so the UI highlights unrelated source. Every
    parser now also records the outermost frame in `inlined_at_*`, and
    create_python_mapping keys on it.
    """

    def test_mlir_callsite_resolves_to_outermost_frame(self):
        ir = """
module {
  #loc7 = loc("/tmp/test.py":1091:8)
  #loc57 = loc("/tmp/test.py":421:16)
  #loc58 = loc("/tmp/test.py":853:16)
  #loc190 = loc(callsite(#loc58 at #loc7))
  #loc220 = loc(callsite(#loc57 at #loc190))
  %0 = tt.load %ptr loc(#loc220)
}
"""
        locs = extract_loc_definitions(ir)
        # #loc220 -> caller #loc190 -> caller #loc7, which is not a callsite
        self.assertEqual(locs["220"]["line"], 421)  # callee, unchanged
        self.assertEqual(locs["220"]["callsite_caller"], "190")  # immediate
        self.assertEqual(locs["220"]["inlined_at_line"], 1091)  # root

    def test_broken_chain_records_no_call_site(self):
        """A walk that cannot reach a non-callsite root must record nothing.

        Breaking out on a missing, cyclic or unresolvable caller leaves the
        cursor on a callsite, whose file/line are a callee -- a library line.
        Recording that as `inlined_at_*` is the exact mis-attribution this pass
        removes, so an incomplete chain leaves the entry plain instead. Matches
        the PTX path when a `.file` index will not resolve.
        """
        ir = """
module {
  #loc57 = loc("/tmp/standard.py":43:16)
  #loc190 = loc(callsite(#loc57 at #loc999))
  #loc220 = loc(callsite(#loc57 at #loc190))
  %0 = tt.load %ptr loc(#loc220)
}
"""
        locs = extract_loc_definitions(ir)
        # the callee is still reported, as for any callsite
        self.assertEqual(locs["220"]["line"], 43)
        # but standard.py:43 must not be passed off as the call site
        self.assertNotIn("inlined_at_line", locs["220"])
        self.assertNotIn("inlined_at_file", locs["220"])
        self.assertNotIn("inlined_at_line", locs["190"])

    def test_sass_multiframe_block_records_the_call_site(self):
        """nvdisasm orders frames innermost-first; the last is the call site."""
        sass = (
            "Function:kernel\n"
            '\t//## File "/lib/standard.py", line 43 inlined at "/work/k.py", line 288\n'
            '\t//## File "/work/k.py", line 288\n'
            "        /*0000*/                   MOV R1, c[0x0][0x28] ;\n"
        )
        m = extract_sass_mappings(sass)
        instr = m["4"]
        self.assertEqual(instr["file"], "/lib/standard.py")  # innermost, unchanged
        self.assertEqual(instr["line"], 43)
        self.assertTrue(instr["is_callsite"])
        self.assertEqual(instr["inlined_at_line"], 288)

    def test_single_frame_block_is_not_a_callsite(self):
        sass = (
            "Function:kernel\n"
            '\t//## File "/work/k.py", line 12\n'
            "        /*0000*/                   MOV R1, c[0x0][0x28] ;\n"
        )
        instr = extract_sass_mappings(sass)["3"]
        self.assertEqual(instr["line"], 12)
        self.assertNotIn("is_callsite", instr)
        self.assertNotIn("inlined_at_line", instr)

    def test_python_mapping_keys_on_the_call_site(self):
        """The whole point: an inlined entry lands on the kernel, not the library."""
        ir_maps = [
            (
                "llir",
                {
                    "5": {
                        "file": "/lib/standard.py",
                        "line": 43,
                        "llir_line": 5,
                        "is_callsite": True,
                        "inlined_at_file": "/work/k.py",
                        "inlined_at_line": 288,
                    },
                    "6": {"file": "/work/k.py", "line": 290, "llir_line": 6},
                },
            ),
        ]
        py = create_python_mapping(ir_maps)
        self.assertIn(288, py, "inlined entry should file under the call site")
        self.assertNotIn(43, py, "library line must not appear as a kernel line")
        self.assertEqual(py[288]["llir_lines"], ["5"])
        self.assertEqual(py[290]["llir_lines"], ["6"])


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

    def test_extract_sass_pc_mappings(self):
        """Test SASS PC-offset-keyed source mapping extraction."""
        sass_content = get_sass_test_file("test_kernel.sass").read_text()

        pc_mappings = extract_sass_pc_mappings(sass_content)

        # All keys must be int (PC offsets)
        for pc in pc_mappings:
            self.assertIsInstance(pc, int, f"PC key should be int, got {type(pc)}")

        # Verify first SASS instruction: /*0000*/ LDC R1...
        # Inherits source from "//## File ...test_structured_logging.py", line 140
        self.assertIn(0x0000, pc_mappings)
        self.assertEqual(
            pc_mappings[0x0000]["file"],
            "/home/test/tritonparse/tests/gpu/test_structured_logging.py",
        )
        self.assertEqual(pc_mappings[0x0000]["line"], 140)
        self.assertEqual(pc_mappings[0x0000]["column"], 0)

        # Verify second SASS instruction: /*0010*/ S2R R0...
        # Inherits source from "//## File ...test_structured_logging.py", line 143
        self.assertIn(0x0010, pc_mappings)
        self.assertEqual(pc_mappings[0x0010]["line"], 143)

        # Verify .nv_debug_ptx_txt never appears as a file value
        for pc, mapping in pc_mappings.items():
            self.assertNotIn(
                ".nv_debug_ptx_txt",
                mapping["file"],
                f"PC 0x{pc:04x} should not map to .nv_debug_ptx_txt",
            )

        # //## File comment lines (pc_hex=None) must NOT appear in pc_mappings
        # The fixture has //## File lines at text lines 2, 5, 8, ... which have
        # no PC offset — they should be absent from the result.
        line_based = extract_sass_mappings(sass_content)
        self.assertGreater(
            len(line_based),
            len(pc_mappings),
            "Line-based mappings include //## File comment lines, "
            "PC mappings should have fewer entries",
        )

        # Each value must contain the expected keys
        for _pc, mapping in pc_mappings.items():
            self.assertIn("file", mapping)
            self.assertIn("line", mapping)
            self.assertIn("column", mapping)
            self.assertIn("sass_line", mapping)

        print("✓ SASS PC-offset mapping tests passed")

    def test_sass_inlined_call_stack_uses_innermost_frame(self):
        """Inlined SASS instructions map to the innermost frame, not the call site.

        nvdisasm emits one ``//## File`` comment per inlined frame, ordered
        innermost-first. The instruction's true source is the innermost frame
        (first comment), not the outer call site (last comment). Regression test
        for instructions previously attributed to the outermost call site.
        """
        sass_content = (
            "Function:_attn_bwd_ws\n"
            # 4-level inline stack: innermost standard.py:170 ... outer kernel.py:593
            '\t//## File "/p/standard.py", line 170 inlined at "/p/standard.py", line 194\n'
            '\t//## File "/p/standard.py", line 194 inlined at "/p/kernel.py", line 227\n'
            '\t//## File "/p/kernel.py", line 227 inlined at "/p/kernel.py", line 593\n'
            '\t//## File "/p/kernel.py", line 593\n'
            '\t//## File ".nv_debug_ptx_txt", line 99\n'
            "        /*0bc0*/                   USHF.R.S32.HI UR4, URZ, 0x1f, UR5 ;\n"
            # single-level inline: innermost kernel.py:2005, call site kernel.py:2452
            '\t//## File "/p/kernel.py", line 2005 inlined at "/p/kernel.py", line 2452\n'
            '\t//## File "/p/kernel.py", line 2452\n'
            '\t//## File ".nv_debug_ptx_txt", line 841\n'
            "        /*11b0*/                   LDS.128 R80, [R88+UR8+0x24440] ;\n"
        )

        pc_mappings = extract_sass_pc_mappings(sass_content)

        # 4-level inline: attribute to innermost standard.py:170, not kernel.py:593
        self.assertEqual(pc_mappings[0x0BC0]["file"], "/p/standard.py")
        self.assertEqual(pc_mappings[0x0BC0]["line"], 170)

        # 1-level inline (the user-reported case): innermost line 2005, not 2452
        self.assertEqual(pc_mappings[0x11B0]["file"], "/p/kernel.py")
        self.assertEqual(pc_mappings[0x11B0]["line"], 2005)

        # Comment lines still map to their own literal location.
        line_mappings = extract_sass_mappings(sass_content)
        # SASS text line 2 is the innermost frame comment (standard.py:170).
        self.assertEqual(line_mappings["2"]["line"], 170)
        # SASS text line 5 is the outermost call-site comment (kernel.py:593).
        self.assertEqual(line_mappings["5"]["line"], 593)

        print("✓ SASS inlined call stack tests passed")

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
