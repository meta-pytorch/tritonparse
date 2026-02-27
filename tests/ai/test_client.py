#  Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for AI module client classes."""

import unittest
from unittest.mock import MagicMock, patch

from tritonparse.ai import (
    ClaudeCodeClient,
    LLMClient,
    Message,
    MockClient,
    Response,
    ToolCall,
)


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


class TestClaudeCodeClient(unittest.TestCase):
    """Tests for ClaudeCodeClient implementation."""

    def test_init_default_values(self):
        """Test default initialization values."""
        client = ClaudeCodeClient()
        self.assertEqual(client.model, "opus")
        self.assertEqual(client.retry_count, 3)
        self.assertEqual(client.timeout, 600)
        self.assertEqual(client.allowed_tools, ["Read", "Grep", "Glob"])
        self.assertIsNone(client.cwd)
        self.assertIsNone(client.session_id)

    def test_init_custom_values(self):
        """Test custom initialization values."""
        client = ClaudeCodeClient(
            allowed_tools=["Read", "Write"],
            retry_count=5,
            timeout=300,
            model="sonnet",
            cwd="/tmp/test",
        )
        self.assertEqual(client.model, "sonnet")
        self.assertEqual(client.retry_count, 5)
        self.assertEqual(client.timeout, 300)
        self.assertEqual(client.allowed_tools, ["Read", "Write"])
        self.assertEqual(client.cwd, "/tmp/test")

    def test_extract_prompts_system_and_user(self):
        """Test extracting system and user prompts."""
        client = ClaudeCodeClient()
        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
        ]
        system, user = client._extract_prompts(messages)
        self.assertEqual(system, "Be helpful")
        self.assertEqual(user, "Hello")

    def test_extract_prompts_user_only(self):
        """Test extracting with only user message."""
        client = ClaudeCodeClient()
        messages = [Message(role="user", content="Hello")]
        system, user = client._extract_prompts(messages)
        self.assertEqual(system, "")
        self.assertEqual(user, "Hello")

    def test_extract_prompts_multiple_user_uses_last(self):
        """Test that multiple user messages use the last one."""
        client = ClaudeCodeClient()
        messages = [
            Message(role="user", content="First"),
            Message(role="user", content="Last"),
        ]
        _, user = client._extract_prompts(messages)
        self.assertEqual(user, "Last")

    def test_parse_response_json(self):
        """Test parsing JSON output."""
        client = ClaudeCodeClient()
        stdout = '{"result": "Hello", "session_id": "abc123", "total_cost_usd": 0.05}'
        response = client._parse_response(stdout)
        self.assertEqual(response.content, "Hello")
        self.assertEqual(response.session_id, "abc123")
        self.assertEqual(response.cost_usd, 0.05)
        self.assertEqual(client.session_id, "abc123")

    def test_parse_response_non_json(self):
        """Test parsing non-JSON output."""
        client = ClaudeCodeClient()
        stdout = "Plain text response"
        response = client._parse_response(stdout)
        self.assertEqual(response.content, "Plain text response")
        self.assertIsNone(response.session_id)

    def test_parse_response_with_whitespace(self):
        """Test parsing output with whitespace."""
        client = ClaudeCodeClient()
        stdout = "  Response with spaces  \n"
        response = client._parse_response(stdout)
        self.assertEqual(response.content, "Response with spaces")

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_success(self, mock_unlink, mock_run):
        """Test successful chat call."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout='{"result": "Success", "session_id": "test123"}'
        )
        client = ClaudeCodeClient()
        response = client.chat([Message(role="user", content="Hi")])
        self.assertEqual(response.content, "Success")
        self.assertEqual(response.session_id, "test123")
        mock_run.assert_called_once()

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_retry_on_failure(self, mock_unlink, mock_run):
        """Test retry on failure."""
        mock_run.side_effect = [
            MagicMock(returncode=1, stderr="Error"),
            MagicMock(returncode=1, stderr="Error"),
            MagicMock(returncode=0, stdout='{"result": "Success"}'),
        ]
        client = ClaudeCodeClient(retry_count=3)
        response = client.chat([Message(role="user", content="Hi")])
        self.assertEqual(response.content, "Success")
        self.assertEqual(mock_run.call_count, 3)

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_all_retries_fail(self, mock_unlink, mock_run):
        """Test all retries failing."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        client = ClaudeCodeClient(retry_count=2)
        with self.assertRaises(RuntimeError) as context:
            client.chat([Message(role="user", content="Hi")])
        self.assertIn("failed after 2 attempts", str(context.exception))

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_with_cwd(self, mock_unlink, mock_run):
        """Test chat uses cwd from constructor."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient(cwd="/tmp/test")
        client.chat([Message(role="user", content="Hi")])
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs["cwd"], "/tmp/test")

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_without_cwd(self, mock_unlink, mock_run):
        """Test chat passes None cwd when not configured."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient()
        client.chat([Message(role="user", content="Hi")])
        call_kwargs = mock_run.call_args[1]
        self.assertIsNone(call_kwargs["cwd"])

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_auto_resumes_session(self, mock_unlink, mock_run):
        """Test that chat automatically resumes session when session_id exists."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient()
        client.session_id = "existing-session-123"
        client.chat([Message(role="user", content="Hi")])
        cmd = mock_run.call_args[0][0]
        self.assertIn('--resume "existing-session-123"', cmd)

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_command_includes_model(self, mock_unlink, mock_run):
        """Test that command includes model parameter."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient(model="sonnet")
        client.chat([Message(role="user", content="Hi")])
        cmd = mock_run.call_args[0][0]
        self.assertIn('--model "sonnet"', cmd)

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_command_includes_allowed_tools(self, mock_unlink, mock_run):
        """Test that command includes allowed tools."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient(allowed_tools=["Read", "Write"])
        client.chat([Message(role="user", content="Hi")])
        cmd = mock_run.call_args[0][0]
        self.assertIn('--allowedTools "Read,Write"', cmd)

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_cleans_up_temp_files(self, mock_unlink, mock_run):
        """Test that temp files are cleaned up."""
        mock_run.return_value = MagicMock(returncode=0, stdout='{"result": "OK"}')
        client = ClaudeCodeClient()
        client.chat([Message(role="user", content="Hi")])
        # Verify both user and system temp files are cleaned up
        # Note: tempfile.mkstemp() may call os.unlink internally on first use
        # to verify the temp directory, so we check for our specific files
        unlinked_paths = [call[0][0] for call in mock_unlink.call_args_list]
        self.assertTrue(
            any("claude_user_" in p for p in unlinked_paths),
            f"Expected claude_user_ temp file to be cleaned up, got: {unlinked_paths}",
        )
        self.assertTrue(
            any("claude_system_" in p for p in unlinked_paths),
            f"Expected claude_system_ temp file to be cleaned up, got: {unlinked_paths}",
        )

    @patch("tritonparse.ai.client.subprocess.run")
    @patch("tritonparse.ai.client.os.unlink")
    def test_chat_cleans_up_on_failure(self, mock_unlink, mock_run):
        """Test that temp files are cleaned up even on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="Error")
        client = ClaudeCodeClient(retry_count=1)
        with self.assertRaises(RuntimeError):
            client.chat([Message(role="user", content="Hi")])
        # Verify both user and system temp files are cleaned up even on failure
        # Note: tempfile.mkstemp() may call os.unlink internally on first use
        # to verify the temp directory, so we check for our specific files
        unlinked_paths = [call[0][0] for call in mock_unlink.call_args_list]
        self.assertTrue(
            any("claude_user_" in p for p in unlinked_paths),
            f"Expected claude_user_ temp file to be cleaned up, got: {unlinked_paths}",
        )
        self.assertTrue(
            any("claude_system_" in p for p in unlinked_paths),
            f"Expected claude_system_ temp file to be cleaned up, got: {unlinked_paths}",
        )


if __name__ == "__main__":
    unittest.main()
