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
