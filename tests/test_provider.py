import asyncio
from castor.provider import Provider
from castor.types import UserMessage, AssistantMessage, ToolCall


class FakeProvider(Provider):
    def __init__(self, responses: list[AssistantMessage]):
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, messages, tools=None):
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


async def test_provider_complete():
    provider = FakeProvider([AssistantMessage(content="hello")])
    msgs = [UserMessage(content="hi")]
    result = await provider.complete(msgs)
    assert result.content == "hello"
    assert result.tool_calls == []


async def test_provider_with_tool_calls():
    tc = ToolCall(id="tc_1", name="bash", arguments={"cmd": "ls"})
    provider = FakeProvider([AssistantMessage(content="let me check", tool_calls=[tc])])
    result = await provider.complete([UserMessage(content="list files")])
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "bash"


async def test_provider_is_abstract():
    try:
        Provider()
        assert False, "Should not instantiate abstract class"
    except TypeError:
        pass
