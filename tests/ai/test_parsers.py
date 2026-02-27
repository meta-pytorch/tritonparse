#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for ai/parsers.py and ai/utils.py
"""

import unittest

from tritonparse.ai.client import Message
from tritonparse.ai.parsers import extract_code_block, extract_diff_patch, extract_json
from tritonparse.ai.utils import format_messages, truncate_context


class ExtractJsonTest(unittest.TestCase):
    """Tests for extract_json function."""

    def test_direct_json(self):
        """Test parsing direct JSON text."""
        text = '{"key": "value", "number": 42}'
        result = extract_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})

    def test_json_in_markdown_block(self):
        """Test extracting JSON from markdown code block."""
        text = """Here is the result:
```json
{"status": "success", "count": 10}
```
"""
        result = extract_json(text)
        self.assertEqual(result, {"status": "success", "count": 10})

    def test_json_in_plain_code_block(self):
        """Test extracting JSON from code block without language."""
        text = """Result:
```
{"items": ["a", "b", "c"]}
```
"""
        result = extract_json(text)
        self.assertEqual(result, {"items": ["a", "b", "c"]})

    def test_empty_string(self):
        """Test with empty string."""
        self.assertIsNone(extract_json(""))

    def test_whitespace_only(self):
        """Test with whitespace only."""
        self.assertIsNone(extract_json("   \n\t  "))

    def test_invalid_json(self):
        """Test with invalid JSON."""
        self.assertIsNone(extract_json("not json at all"))

    def test_json_with_nested_objects(self):
        """Test parsing nested JSON."""
        text = '{"outer": {"inner": {"deep": true}}}'
        result = extract_json(text)
        self.assertEqual(result, {"outer": {"inner": {"deep": True}}})


class ExtractCodeBlockTest(unittest.TestCase):
    """Tests for extract_code_block function."""

    def test_python_code_block(self):
        """Test extracting Python code block."""
        text = """Here's the code:
```python
def hello():
    print("Hello, World!")
```
"""
        result = extract_code_block(text, "python")
        self.assertEqual(result, 'def hello():\n    print("Hello, World!")')

    def test_any_language_block(self):
        """Test extracting any code block without language filter."""
        text = """```javascript
console.log("test");
```
"""
        result = extract_code_block(text)
        self.assertEqual(result, 'console.log("test");')

    def test_no_matching_language(self):
        """Test when requested language not found."""
        text = """```python
print("python")
```
"""
        result = extract_code_block(text, "rust")
        self.assertIsNone(result)

    def test_empty_text(self):
        """Test with empty text."""
        self.assertIsNone(extract_code_block(""))
        self.assertIsNone(extract_code_block(None))

    def test_multiline_code_block(self):
        """Test extracting multiline code block."""
        text = """```python
import os
import sys

def main():
    pass
```
"""
        result = extract_code_block(text, "python")
        self.assertIn("import os", result)
        self.assertIn("def main():", result)

    def test_code_block_without_trailing_newline(self):
        """Test extracting code block when no newline before closing backticks."""
        text = "```python\nprint('hello')```"
        result = extract_code_block(text, "python")
        self.assertEqual(result, "print('hello')")

    def test_any_language_without_trailing_newline(self):
        """Test extracting any code block without trailing newline."""
        text = "```js\nconsole.log('test')```"
        result = extract_code_block(text)
        self.assertEqual(result, "console.log('test')")


class ExtractDiffPatchTest(unittest.TestCase):
    """Tests for extract_diff_patch function."""

    def test_markdown_diff_block(self):
        """Test extracting diff from markdown block."""
        text = """Here's the patch:
```diff
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
 line1
+new line
 line2
```
"""
        result = extract_diff_patch(text)
        self.assertIn("--- a/file.py", result)
        self.assertIn("+++ b/file.py", result)
        self.assertIn("+new line", result)

    def test_raw_git_diff(self):
        """Test extracting raw git diff format."""
        text = """Some text before
diff --git a/file.py b/file.py
index abc123..def456 100644
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
        result = extract_diff_patch(text)
        self.assertIn("diff --git", result)
        self.assertIn("-old", result)
        self.assertIn("+new", result)

    def test_raw_unified_diff(self):
        """Test extracting raw unified diff starting with ---."""
        text = """--- original.py
+++ modified.py
@@ -1 +1 @@
-old content
+new content
"""
        result = extract_diff_patch(text)
        self.assertIn("--- original.py", result)
        self.assertIn("+new content", result)

    def test_empty_text(self):
        """Test with empty text."""
        self.assertIsNone(extract_diff_patch(""))
        self.assertIsNone(extract_diff_patch(None))

    def test_no_diff_content(self):
        """Test with text containing no diff."""
        result = extract_diff_patch("Just some regular text")
        self.assertIsNone(result)

    def test_raw_diff_trims_trailing_text(self):
        """Test that trailing non-diff text is trimmed from raw diffs."""
        text = """Some intro text
diff --git a/file.py b/file.py
index abc..def 100644
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 context
-old line
+new line

This explanation should be trimmed.
Make sure to rebuild after applying.
"""
        result = extract_diff_patch(text)
        self.assertIn("+new line", result)
        self.assertNotIn("explanation", result)
        self.assertNotIn("rebuild", result)

    def test_raw_diff_trims_trailing_blank_lines(self):
        """Test that trailing blank lines after diff are trimmed."""
        text = """--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new


"""
        result = extract_diff_patch(text)
        self.assertTrue(result.endswith("+new"))


class FormatMessagesTest(unittest.TestCase):
    """Tests for format_messages function."""

    def test_single_message(self):
        """Test formatting a single message."""
        messages = [Message(role="user", content="Hello")]
        result = format_messages(messages)
        self.assertEqual(result, "[user]\nHello")

    def test_multiple_messages(self):
        """Test formatting multiple messages."""
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello!"),
        ]
        result = format_messages(messages)
        expected = "[system]\nYou are helpful.\n\n[user]\nHi\n\n[assistant]\nHello!"
        self.assertEqual(result, expected)

    def test_empty_list(self):
        """Test with empty message list."""
        result = format_messages([])
        self.assertEqual(result, "")

    def test_multiline_content(self):
        """Test message with multiline content."""
        messages = [Message(role="user", content="Line 1\nLine 2\nLine 3")]
        result = format_messages(messages)
        self.assertIn("Line 1\nLine 2\nLine 3", result)


class TruncateContextTest(unittest.TestCase):
    """Tests for truncate_context function."""

    def test_no_truncation_needed(self):
        """Test when text is under limit."""
        text = "short text"
        result = truncate_context(text, max_chars=100)
        self.assertEqual(result, text)

    def test_head_strategy(self):
        """Test truncation keeping head."""
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = truncate_context(text, max_chars=25, strategy="head")
        self.assertTrue(result.startswith("0123456"))
        self.assertTrue(result.endswith("... [truncated]"))
        self.assertEqual(len(result), 25)

    def test_tail_strategy(self):
        """Test truncation keeping tail."""
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = truncate_context(text, max_chars=25, strategy="tail")
        self.assertTrue(result.startswith("... [truncated]"))
        self.assertTrue(result.endswith("VWXYZ"))
        self.assertEqual(len(result), 25)

    def test_middle_strategy(self):
        """Test truncation keeping head and tail."""
        text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        result = truncate_context(text, max_chars=25, strategy="middle")
        self.assertTrue(result.startswith("0123"))
        self.assertIn("... [truncated]", result)
        self.assertTrue(result.endswith("XYZ"))

    def test_empty_text(self):
        """Test with empty text."""
        result = truncate_context("", max_chars=100)
        self.assertEqual(result, "")

    def test_invalid_strategy(self):
        """Test with invalid strategy raises error."""
        with self.assertRaises(ValueError) as ctx:
            truncate_context("some text", max_chars=5, strategy="invalid")
        self.assertIn("Invalid strategy", str(ctx.exception))

    def test_custom_ellipsis(self):
        """Test with custom ellipsis."""
        text = "0123456789ABCDEF"
        result = truncate_context(text, max_chars=15, strategy="head", ellipsis="...")
        self.assertTrue(result.endswith("..."))
        self.assertEqual(len(result), 15)


if __name__ == "__main__":
    unittest.main()
