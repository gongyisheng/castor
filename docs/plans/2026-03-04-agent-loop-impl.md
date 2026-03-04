# Agent Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the core async agent loop with types, provider/tool interfaces, and user-priority interruption.

**Architecture:** `types.py` defines data types, `provider.py` and `tool.py` define abstract interfaces, `loop.py` implements the core loop. All async, tested with pytest-asyncio.

**Tech Stack:** Python 3.12, dataclasses, asyncio, abc, pytest + pytest-asyncio

---

### Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/castor/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "castor"
version = "0.1.0"
description = "A coding agent"
requires-python = ">=3.12"

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

**Step 2: Create package init**

```python
# src/castor/__init__.py
```

**Step 3: Install in dev mode and verify**

Run: `cd /home/yisheng/Documents/castor && python3 -m pip install -e ".[dev]"`
Expected: installs successfully, `pytest --co` runs without error

**Step 4: Commit**

```bash
git add pyproject.toml src/castor/__init__.py
git commit -m "[build] Add project setup with pytest"
```

---

### Task 2: Data Types

**Files:**
- Create: `tests/test_types.py`
- Create: `src/castor/types.py`

**Step 1: Write the failing tests**

```python
# tests/test_types.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_types.py -v`
Expected: FAIL — cannot import castor.types

**Step 3: Write implementation**

```python
# src/castor/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    output: str
    is_error: bool = False


@dataclass
class UserMessage:
    content: str
    role: str = field(default="user", init=False)

    @classmethod
    def interrupted(cls, original: str, new: str) -> UserMessage:
        return cls(content=f"[user interrupted: '{original}'] {new}")


@dataclass
class AssistantMessage:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: str = field(default="assistant", init=False)


@dataclass
class ToolMessage:
    tool_call_id: str
    content: str
    is_error: bool = False
    role: str = field(default="tool", init=False)

    @classmethod
    def cancelled(cls, tool_call_id: str) -> ToolMessage:
        return cls(tool_call_id=tool_call_id, content="[cancelled by user]")


Message = UserMessage | AssistantMessage | ToolMessage
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_types.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tests/test_types.py src/castor/types.py
git commit -m "[feat] Add core data types"
```

---

### Task 3: Provider Interface

**Files:**
- Create: `tests/test_provider.py`
- Create: `src/castor/provider.py`

**Step 1: Write the failing tests**

```python
# tests/test_provider.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_provider.py -v`
Expected: FAIL — cannot import castor.provider

**Step 3: Write implementation**

```python
# src/castor/provider.py
from __future__ import annotations
from abc import ABC, abstractmethod

from castor.types import AssistantMessage, Message, ToolCall


class Provider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> AssistantMessage: ...
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_provider.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tests/test_provider.py src/castor/provider.py
git commit -m "[feat] Add LLM provider interface"
```

---

### Task 4: Tool Interface

**Files:**
- Create: `tests/test_tool.py`
- Create: `src/castor/tool.py`

**Step 1: Write the failing tests**

```python
# tests/test_tool.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tool.py -v`
Expected: FAIL — cannot import castor.tool

**Step 3: Write implementation**

```python
# src/castor/tool.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

from castor.types import ToolCall, ToolResult


class Tool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> str: ...

    async def run(self, tool_call: ToolCall) -> ToolResult:
        try:
            output = await self.execute(tool_call.arguments)
            return ToolResult(tool_call_id=tool_call.id, output=output)
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, output=str(e), is_error=True)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        tool = self.get(tool_call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=f"Unknown tool: {tool_call.name}",
                is_error=True,
            )
        return await tool.run(tool_call)

    def schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tool.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tests/test_tool.py src/castor/tool.py
git commit -m "[feat] Add tool interface and registry"
```

---

### Task 5: Agent Loop — Basic Flow (no interruption)

**Files:**
- Create: `tests/test_loop.py`
- Create: `src/castor/loop.py`

**Step 1: Write the failing tests**

```python
# tests/test_loop.py
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_loop.py -v`
Expected: FAIL — cannot import castor.loop

**Step 3: Write implementation**

```python
# src/castor/loop.py
from __future__ import annotations
import asyncio

from castor.provider import Provider
from castor.tool import ToolRegistry
from castor.types import (
    AssistantMessage, Message, ToolMessage, UserMessage,
)


async def agent_loop(
    messages: list[Message],
    provider: Provider,
    registry: ToolRegistry,
    cancel: asyncio.Event,
    max_iter: int = 20,
) -> list[Message]:
    for _ in range(max_iter):
        if cancel.is_set():
            return messages

        response = await provider.complete(messages, registry.schemas())
        if cancel.is_set():
            return messages

        messages.append(response)

        if not response.tool_calls:
            return messages

        results = await _execute_tools(response, registry, cancel)
        messages.extend(results)

    return messages


async def _execute_tools(
    response: AssistantMessage,
    registry: ToolRegistry,
    cancel: asyncio.Event,
) -> list[ToolMessage]:
    results: list[ToolMessage] = []

    tasks = {
        tc.id: asyncio.create_task(registry.execute(tc))
        for tc in response.tool_calls
    }

    for tc in response.tool_calls:
        if cancel.is_set():
            # cancel remaining, mark as cancelled
            for remaining_tc in response.tool_calls:
                if remaining_tc.id not in {r.tool_call_id for r in results}:
                    tasks[remaining_tc.id].cancel()
                    results.append(ToolMessage.cancelled(remaining_tc.id))
            break

        try:
            tool_result = await tasks[tc.id]
            results.append(
                ToolMessage(
                    tool_call_id=tool_result.tool_call_id,
                    content=tool_result.output,
                    is_error=tool_result.is_error,
                )
            )
        except asyncio.CancelledError:
            results.append(ToolMessage.cancelled(tc.id))

    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_loop.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add tests/test_loop.py src/castor/loop.py
git commit -m "[feat] Add core agent loop with tool execution"
```

---

### Task 6: Agent Loop — Interruption

**Files:**
- Modify: `tests/test_loop.py` (add interruption tests)
- Modify: `src/castor/loop.py` (if adjustments needed)

**Step 1: Add failing interruption tests**

Append to `tests/test_loop.py`:

```python
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
```

**Step 2: Run tests to verify behavior**

Run: `pytest tests/test_loop.py -v`
Expected: all PASS (the loop implementation from Task 5 should already handle these cases)

If any fail, adjust `_execute_tools` to properly handle cancellation timing.

**Step 3: Commit**

```bash
git add tests/test_loop.py src/castor/loop.py
git commit -m "[feat] Add interruption handling to agent loop"
```

---

### Task 7: Package Exports and Final Verification

**Files:**
- Modify: `src/castor/__init__.py`
- Create: `tests/__init__.py`

**Step 1: Update package exports**

```python
# src/castor/__init__.py
from castor.types import (
    Message, UserMessage, AssistantMessage, ToolMessage,
    ToolCall, ToolResult,
)
from castor.provider import Provider
from castor.tool import Tool, ToolRegistry
from castor.loop import agent_loop

__all__ = [
    "Message", "UserMessage", "AssistantMessage", "ToolMessage",
    "ToolCall", "ToolResult",
    "Provider",
    "Tool", "ToolRegistry",
    "agent_loop",
]
```

**Step 2: Create tests init**

```python
# tests/__init__.py
```

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 4: Commit**

```bash
git add src/castor/__init__.py tests/__init__.py
git commit -m "[chore] Add package exports and finalize structure"
```
