from castor.tool import Tool, ToolRegistry
from castor.types import ToolCall, ToolResult


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}}

    async def execute(self, arguments: dict) -> str:
        return arguments["text"]


class FailTool(Tool):
    name = "fail"
    description = "Always fails"
    parameters = {}

    async def execute(self, arguments: dict) -> str:
        raise RuntimeError("boom")


async def test_tool_execute():
    tool = EchoTool()
    result = await tool.run(ToolCall(id="tc_1", name="echo", arguments={"text": "hi"}))
    assert result.output == "hi"
    assert result.is_error is False
    assert result.tool_call_id == "tc_1"


async def test_tool_execute_error():
    tool = FailTool()
    result = await tool.run(ToolCall(id="tc_2", name="fail", arguments={}))
    assert result.is_error is True
    assert "boom" in result.output


async def test_registry_register_and_get():
    reg = ToolRegistry()
    reg.register(EchoTool())
    assert reg.get("echo") is not None
    assert reg.get("nonexistent") is None


async def test_registry_execute():
    reg = ToolRegistry()
    reg.register(EchoTool())
    tc = ToolCall(id="tc_1", name="echo", arguments={"text": "hi"})
    result = await reg.execute(tc)
    assert result.output == "hi"


async def test_registry_execute_unknown_tool():
    reg = ToolRegistry()
    tc = ToolCall(id="tc_1", name="nope", arguments={})
    result = await reg.execute(tc)
    assert result.is_error is True
    assert "unknown tool" in result.output.lower()


def test_registry_schemas():
    reg = ToolRegistry()
    reg.register(EchoTool())
    schemas = reg.schemas()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "echo"
