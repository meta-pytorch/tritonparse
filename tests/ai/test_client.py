#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for AI module client classes."""

import unittest

from tritonparse.ai import LLMClient, Message, MockClient, Response, ToolCall


class TestMessage(unittest.TestCase):
    """Tests for Message dataclass."""

    def test_message_creation(self):
        """Test creating a Message with role and content."""
        msg = Message(role="user", content="Hello, world!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello, world!")

    def test_message_system_role(self):
        """Test creating a system message."""
        msg = Message(role="system", content="You are a helpful assistant.")
        self.assertEqual(msg.role, "system")
        self.assertEqual(msg.content, "You are a helpful assistant.")

    def test_message_assistant_role(self):
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="I can help with that.")
        self.assertEqual(msg.role, "assistant")


class TestToolCall(unittest.TestCase):
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall with name and arguments."""
        tool = ToolCall(name="read_file", arguments={"path": "/tmp/test.txt"})
        self.assertEqual(tool.name, "read_file")
        self.assertEqual(tool.arguments, {"path": "/tmp/test.txt"})

    def test_tool_call_default_arguments(self):
        """Test that arguments defaults to empty dict."""
        tool = ToolCall(name="list_files")
        self.assertEqual(tool.name, "list_files")
        self.assertEqual(tool.arguments, {})


class TestResponse(unittest.TestCase):
    """Tests for Response dataclass."""

    def test_response_with_content_only(self):
        """Test creating a Response with only content."""
        resp = Response(content="Hello!")
        self.assertEqual(resp.content, "Hello!")
        self.assertIsNone(resp.session_id)
        self.assertIsNone(resp.cost_usd)
        self.assertIsNone(resp.tool_calls)
        self.assertIsNone(resp.raw)

    def test_response_with_all_fields(self):
        """Test creating a Response with all fields."""
        tool_calls = [ToolCall(name="test", arguments={})]
        raw_data = {"key": "value"}
        resp = Response(
            content="Response text",
            session_id="session-123",
            cost_usd=0.05,
            tool_calls=tool_calls,
            raw=raw_data,
        )
        self.assertEqual(resp.content, "Response text")
        self.assertEqual(resp.session_id, "session-123")
        self.assertEqual(resp.cost_usd, 0.05)
        self.assertEqual(resp.tool_calls, tool_calls)
        self.assertEqual(resp.raw, raw_data)


class TestLLMClientABC(unittest.TestCase):
    """Tests for LLMClient abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that LLMClient cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            LLMClient()


class TestMockClient(unittest.TestCase):
    """Tests for MockClient implementation."""

    def test_chat_returns_predefined_response(self):
        """Test that chat returns the first predefined response."""
        client = MockClient(responses=["Hello from mock!"])
        messages = [Message(role="user", content="Hi")]
        resp = client.chat(messages)
        self.assertEqual(resp.content, "Hello from mock!")

    def test_chat_cycles_through_responses(self):
        """Test that chat returns responses in order."""
        client = MockClient(responses=["First", "Second", "Third"])

        self.assertEqual(client.chat([]).content, "First")
        self.assertEqual(client.chat([]).content, "Second")
        self.assertEqual(client.chat([]).content, "Third")

    def test_chat_returns_default_after_exhausting_responses(self):
        """Test fallback to default response after exhausting list."""
        client = MockClient(responses=["Only one"])

        self.assertEqual(client.chat([]).content, "Only one")
        self.assertEqual(client.chat([]).content, "Mock response")
        self.assertEqual(client.chat([]).content, "Mock response")

    def test_chat_with_no_predefined_responses(self):
        """Test default response when no responses provided."""
        client = MockClient()
        resp = client.chat([])
        self.assertEqual(resp.content, "Mock response")

    def test_chat_records_last_messages(self):
        """Test that last_messages is updated after each call."""
        client = MockClient()
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        client.chat(messages)
        self.assertEqual(client.last_messages, messages)

    def test_chat_increments_call_count(self):
        """Test that call_count tracks number of calls."""
        client = MockClient(responses=["A", "B"])
        self.assertEqual(client.call_count, 0)

        client.chat([])
        self.assertEqual(client.call_count, 1)

        client.chat([])
        self.assertEqual(client.call_count, 2)

        # Even after exhausting responses, count continues
        client.chat([])
        self.assertEqual(client.call_count, 3)

    def test_chat_ignores_temperature_and_max_tokens(self):
        """Test that temperature and max_tokens are accepted but ignored."""
        client = MockClient(responses=["Response"])
        resp = client.chat([], temperature=0.7, max_tokens=100)
        self.assertEqual(resp.content, "Response")

    def test_chat_stream_yields_content(self):
        """Test that chat_stream yields the response content."""
        client = MockClient(responses=["Streamed content"])
        result = list(client.chat_stream([]))
        self.assertEqual(result, ["Streamed content"])

    def test_chat_stream_increments_call_count(self):
        """Test that chat_stream also increments call count."""
        client = MockClient(responses=["A", "B"])
        list(client.chat_stream([]))  # Consume the iterator
        self.assertEqual(client.call_count, 1)


if __name__ == "__main__":
    unittest.main()
