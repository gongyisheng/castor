from castor.types import (
    Message, UserMessage, AssistantMessage, ToolMessage,
    ToolCall, ToolResult,
)


def test_user_message():
    msg = UserMessage(content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_user_message_with_interruption():
    msg = UserMessage.interrupted("do X", "do Y")
    assert "[user interrupted:" in msg.content
    assert "do Y" in msg.content


def test_assistant_message_text_only():
    msg = AssistantMessage(content="sure", tool_calls=[])
    assert msg.role == "assistant"
    assert msg.tool_calls == []


def test_assistant_message_with_tool_calls():
    tc = ToolCall(id="tc_1", name="read_file", arguments={"path": "foo.py"})
    msg = AssistantMessage(content="let me read", tool_calls=[tc])
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].name == "read_file"


def test_tool_message():
    msg = ToolMessage(tool_call_id="tc_1", content="file contents here")
    assert msg.role == "tool"
    assert msg.tool_call_id == "tc_1"


def test_tool_message_cancelled():
    msg = ToolMessage.cancelled(tool_call_id="tc_1")
    assert "[cancelled by user]" in msg.content


def test_tool_message_error():
    msg = ToolMessage(tool_call_id="tc_1", content="not found", is_error=True)
    assert msg.is_error is True


def test_tool_call():
    tc = ToolCall(id="tc_1", name="bash", arguments={"command": "ls"})
    assert tc.id == "tc_1"
    assert tc.name == "bash"
    assert tc.arguments == {"command": "ls"}


def test_tool_result():
    tr = ToolResult(tool_call_id="tc_1", output="file.py", is_error=False)
    assert tr.tool_call_id == "tc_1"
    assert tr.is_error is False
