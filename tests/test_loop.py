import asyncio
from castor.loop import agent_loop
from castor.provider import Provider
from castor.tool import Tool, ToolRegistry
from castor.types import (
    AssistantMessage, ToolCall, ToolMessage, UserMessage,
)


class FakeProvider(Provider):
    def __init__(self, responses: list[AssistantMessage]):
        self._responses = list(responses)
        self._idx = 0

    async def complete(self, messages, tools=None):
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class EchoTool(Tool):
    name = "echo"
    description = "Echoes input"
    parameters = {}

    async def execute(self, arguments):
        return arguments["text"]


class SlowTool(Tool):
    name = "slow"
    description = "Sleeps then returns"
    parameters = {}

    async def execute(self, arguments):
        await asyncio.sleep(arguments.get("delay", 0.5))
        return "done"


def make_registry(*tools):
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


async def test_simple_text_response():
    """LLM returns text only, loop exits after one iteration."""
    provider = FakeProvider([AssistantMessage(content="hello!")])
    messages = [UserMessage(content="hi")]
    cancel = asyncio.Event()

    result = await agent_loop(messages, provider, make_registry(), cancel)

    assert len(result) == 2
    assert result[0].role == "user"
    assert result[1].role == "assistant"
    assert result[1].content == "hello!"


async def test_tool_call_and_response():
    """LLM calls a tool, gets result, then responds with text."""
    tc = ToolCall(id="tc_1", name="echo", arguments={"text": "world"})
    provider = FakeProvider([
        AssistantMessage(content="let me check", tool_calls=[tc]),
        AssistantMessage(content="got it: world"),
    ])
    messages = [UserMessage(content="test")]
    cancel = asyncio.Event()

    result = await agent_loop(messages, provider, make_registry(EchoTool()), cancel)

    assert len(result) == 4  # user, assistant+tc, tool result, assistant
    assert result[1].tool_calls[0].name == "echo"
    assert result[2].role == "tool"
    assert result[2].content == "world"
    assert result[3].content == "got it: world"


async def test_max_iterations():
    """Loop exits after max_iter even if LLM keeps calling tools."""
    tc = ToolCall(id="tc_1", name="echo", arguments={"text": "loop"})
    responses = [AssistantMessage(content="again", tool_calls=[tc])] * 5
    provider = FakeProvider(responses)
    messages = [UserMessage(content="go")]
    cancel = asyncio.Event()

    result = await agent_loop(
        messages, provider, make_registry(EchoTool()), cancel, max_iter=3,
    )

    # 3 iterations: each adds assistant + tool = 6, plus original user = 7
    assert provider._idx == 3


async def test_interrupt_during_tool_execution():
    """Cancel during tool execution: completed tools keep results, pending cancelled."""
    tc1 = ToolCall(id="tc_1", name="echo", arguments={"text": "fast"})
    tc2 = ToolCall(id="tc_2", name="slow", arguments={"delay": 5})
    provider = FakeProvider([
        AssistantMessage(content="doing both", tool_calls=[tc1, tc2]),
    ])
    messages = [UserMessage(content="do stuff")]
    cancel = asyncio.Event()

    async def interrupt_soon():
        await asyncio.sleep(0.1)
        cancel.set()

    asyncio.create_task(interrupt_soon())
    result = await agent_loop(
        messages, provider, make_registry(EchoTool(), SlowTool()), cancel,
    )

    # user + assistant + tool results (some completed, some cancelled)
    tool_msgs = [m for m in result if m.role == "tool"]
    assert len(tool_msgs) == 2
    contents = {m.content for m in tool_msgs}
    assert "[cancelled by user]" in contents


async def test_interrupt_before_llm_call():
    """Cancel before LLM call: loop exits immediately."""
    provider = FakeProvider([AssistantMessage(content="never")])
    messages = [UserMessage(content="hi")]
    cancel = asyncio.Event()
    cancel.set()  # already cancelled

    result = await agent_loop(messages, provider, make_registry(), cancel)

    assert len(result) == 1  # only original user message
    assert provider._idx == 0  # LLM never called
