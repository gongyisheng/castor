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
